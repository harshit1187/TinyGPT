import html
import os
import re

import gradio as gr
import torch
torch.set_num_threads(1)
from models import Stage1Bigram, Stage2GPT, Stage3GPT, Stage3Config

device = "mps" if torch.backends.mps.is_available() else "cpu"
stoi = None


def extract_state(ck):
    if isinstance(ck, dict) and "model" in ck:
        return ck["model"], ck.get("stoi")
    return ck, None


def strip_compile_prefix(state_dict):
    """Handles checkpoints saved from a torch.compile()-wrapped model."""
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


# Load the three models (any checkpoint that's missing is simply skipped)
m3 = None
if os.path.exists("gpt_stage3.pt"):
    ck = torch.load("gpt_stage3.pt", map_location=device, weights_only=False)
    cfg = Stage3Config(**ck["cfg"])
    m3 = Stage3GPT(cfg).to(device)
    m3.load_state_dict(strip_compile_prefix(ck["model"]))
    m3.eval()
    stoi = ck["stoi"]

m2 = None
if os.path.exists("gpt_stage2.pt"):
    ck = torch.load("gpt_stage2.pt", map_location=device, weights_only=False)
    state, s = extract_state(ck)
    m2 = Stage2GPT().to(device)
    m2.load_state_dict(strip_compile_prefix(state))
    m2.eval()
    stoi = stoi or s

m1 = None
if os.path.exists("gpt_stage1.pt"):
    ck = torch.load("gpt_stage1.pt", map_location=device, weights_only=False)
    state, s = extract_state(ck)
    m1 = Stage1Bigram().to(device)
    m1.load_state_dict(strip_compile_prefix(state))
    m1.eval()
    stoi = stoi or s

if stoi is None:
    raise SystemExit("No checkpoints found (gpt_stage1.pt / gpt_stage2.pt / gpt_stage3.pt).")

itos = {i: c for c, i in stoi.items()}

ALL_MODELS = {"Modern GPT": m3, "Classic GPT": m2, "Bigram": m1}
ACTIVE_MODELS = {k: v for k, v in ALL_MODELS.items() if v is not None}
ALL_THREE = "All three"
CHOICES = list(ACTIVE_MODELS) + ([ALL_THREE] if len(ACTIVE_MODELS) > 1 else [])


def run_model(model, prompt, max_new_tokens, temperature, top_k):
    ids = [stoi[c] for c in prompt if c in stoi] or [0]
    if isinstance(model, Stage3GPT):
        ids = ids[-100:]
        max_new_tokens = min(int(max_new_tokens), model.cfg.block_size - len(ids))
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model.generate(
            idx,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=int(top_k) if top_k > 0 else None,
        )
    return "".join(itos[i] for i in out[0].tolist())


# ---------------------------------------------------------------------------
# What appears in the middle of the page
# ---------------------------------------------------------------------------
EMPTY = (
    '<div class="hint">Begin with a few words.'
    "<span>Press enter. Model, length and temperature are under options.</span></div>"
)


def words(text, total=3.0):
    """Wrap each word in a span that fades in just after the one before it.
    The browser does the animating, so there's no flicker from re-sending text."""
    parts = re.findall(r"\S+|\s+", text)
    count = sum(1 for p in parts if not p.isspace())
    step = min(0.06, total / max(count, 1))
    out, i = [], 0
    for p in parts:
        if p.isspace():
            out.append(html.escape(p))
        else:
            out.append(f'<span class="w" style="animation-delay:{i * step:.2f}s">{html.escape(p)}</span>')
            i += 1
    return "".join(out)


def render(items):
    """items = [(label, text), ...]. One item -> big centred text, several -> quiet columns."""
    if len(items) == 1:
        return f'<div class="one">{words(items[0][1])}</div>'
    cells = "".join(
        f'<div class="cell"><div class="tag">{html.escape(label)}</div>'
        f'<div class="txt">{words(text)}</div></div>'
        for label, text in items
    )
    return f'<div class="many">{cells}</div>'


def write(choice, prompt, length, temperature, top_k):
    prompt = (prompt or "").strip("\n")
    if prompt.rstrip().endswith(":"):      # "ROMEO:" works much better as "ROMEO:\n"
        prompt = prompt.rstrip() + "\n"

    names = ["Bigram", "Classic GPT", "Modern GPT"] if choice == ALL_THREE else [choice]
    names = [n for n in names if n in ACTIVE_MODELS]

    yield gr.update(), "Thinking…"
    results = [(n, run_model(ACTIVE_MODELS[n], prompt, length, temperature, top_k)) for n in names]

    yield render(results), ""   # sent once; the words fade in on their own in the browser


# ---------------------------------------------------------------------------
# Look: black page, soft grey text, almost no chrome
# ---------------------------------------------------------------------------
theme = gr.themes.Base(primary_hue="slate", neutral_hue="neutral").set(
    body_background_fill="#000000",
    body_background_fill_dark="#000000",
    block_background_fill="#000000",
    block_background_fill_dark="#000000",
    block_border_color="#141414",
    block_border_color_dark="#141414",
    block_label_text_color="#777777",
    block_label_text_color_dark="#777777",
    body_text_color="#e6e6e6",
    body_text_color_dark="#e6e6e6",
    input_background_fill="#0a0a0a",
    input_background_fill_dark="#0a0a0a",
    input_border_color="#222222",
    input_border_color_dark="#222222",
    button_secondary_background_fill="#161616",
    button_secondary_background_fill_dark="#161616",
    button_secondary_background_fill_hover="#222222",
    button_secondary_background_fill_hover_dark="#222222",
    button_secondary_text_color="#9a9a9a",
    button_secondary_text_color_dark="#9a9a9a",
    slider_color="#6b6b6b",
    slider_color_dark="#6b6b6b",
)

css = """
footer { display: none !important; }

body, .gradio-container {
    background: #000 !important;
    color: #e6e6e6 !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, sans-serif !important;
}
.gradio-container {
    max-width: 100% !important;
    padding-bottom: 140px !important;
}

/* no boxes around things */
.gradio-container .block, .gradio-container .form {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* "Thinking…" in the top corner */
#status {
    position: fixed; top: 14px; right: 24px; z-index: 30;
    width: auto !important; min-height: 0 !important; padding: 0 !important;
    color: #777 !important; font-size: 13px; letter-spacing: 0.05em;
}
#status * { color: #777 !important; }

/* options, folded away */
#options {
    max-width: 520px; margin: 14px auto 0 !important;
}
#options > button, #options .label-wrap { color: #6f6f6f !important; font-size: 13px !important; }

/* the middle of the page */
#stage { min-height: 66vh; padding: 0 24px; }
#stage .one, #stage .many, #stage .hint { margin-top: 18vh; }
#stage .one {
    max-width: 720px; margin-left: auto; margin-right: auto;
    text-align: center; white-space: pre-wrap;
    font-size: 22px; line-height: 1.75; color: #e8e8e8 !important;
}
#stage .hint {
    text-align: center; font-size: 22px; color: #555 !important;
}
#stage .hint span {
    display: block; margin-top: 14px; font-size: 14px; color: #3f3f3f !important;
}
#stage .many {
    display: grid; gap: 44px; max-width: 1040px;
    margin-left: auto; margin-right: auto;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
}
#stage .cell .tag {
    text-align: center; margin-bottom: 14px;
    font-size: 12px; letter-spacing: 0.08em; color: #6a6a6a !important;
}
#stage .cell .txt {
    text-align: center; white-space: pre-wrap;
    font-size: 17px; line-height: 1.75; color: #dcdcdc !important;
}

/* the bar along the bottom */
#bar {
    position: fixed !important; left: 0; right: 0; bottom: 0; z-index: 20;
    background: #000 !important;
    border-top: 1px solid #121212 !important;
    padding: 18px 28px 22px !important;
    gap: 16px !important; align-items: center !important; flex-wrap: nowrap !important;
}
#prompt input, #prompt textarea {
    background: transparent !important;
    border: none !important; border-bottom: 1px solid #222 !important;
    border-radius: 0 !important; box-shadow: none !important;
    color: #e8e8e8 !important; font-size: 16px !important; text-align: center;
}
#prompt input::placeholder, #prompt textarea::placeholder { color: #666 !important; }
#send {
    width: 44px !important; min-width: 44px !important; height: 44px !important;
    border-radius: 50% !important; padding: 0 !important; font-size: 18px !important;
    border: none !important;
}

/* ---- motion: slow, soft, never abrupt ---- */
html { scroll-behavior: smooth; }

@keyframes drift {
    from { opacity: 0; transform: translateY(6px); filter: blur(3px); }
    to   { opacity: 1; transform: none; filter: blur(0); }
}
@keyframes breathe { 0%, 100% { opacity: 0.35; } 50% { opacity: 0.9; } }
@keyframes settle  { from { opacity: 0; } to { opacity: 1; } }

#stage .w {
    display: inline-block;
    opacity: 0;
    animation: drift 0.9s cubic-bezier(0.22, 0.61, 0.36, 1) both;
}
#stage .hint { animation: settle 1.6s ease both; }
#status * { animation: breathe 2.4s ease-in-out infinite; }

#prompt input, #prompt textarea { transition: border-color 0.4s ease; }
#prompt input:focus, #prompt textarea:focus {
    border-bottom-color: #555 !important; outline: none !important; box-shadow: none !important;
}
#send { transition: background 0.3s ease, transform 0.25s ease; }
#send:hover { transform: scale(1.06); }

@media (prefers-reduced-motion: reduce) {
    #stage .w, #stage .hint, #status * { animation: none !important; opacity: 1 !important; filter: none !important; }
}
"""

with gr.Blocks(title="A Tiny GPT") as demo:
    status = gr.HTML("", elem_id="status")

    with gr.Accordion("options", open=False, elem_id="options"):
        choice = gr.Radio(CHOICES, value=CHOICES[0], label="Model")
        length = gr.Slider(30, 220, value=120, step=10, label="Length (characters)")
        temperature = gr.Slider(0.2, 1.5, value=0.8, step=0.1, label="Temperature")
        top_k = gr.Slider(0, 65, value=40, step=5, label="Top-K (0 = off)")

    stage = gr.HTML(EMPTY, elem_id="stage")

    with gr.Row(elem_id="bar"):
        prompt = gr.Textbox(
            placeholder="Start with a few words…", show_label=False, container=False,
            lines=1, max_lines=1, elem_id="prompt", scale=12,
        )
        send = gr.Button("→", elem_id="send", scale=0, min_width=44)

    inputs = [choice, prompt, length, temperature, top_k]
    # show_progress="hidden" turns off Gradio's own loading overlay; the status text does that job
    prompt.submit(write, inputs, [stage, status], show_progress="hidden")
    send.click(write, inputs, [stage, status], show_progress="hidden")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=theme,
        css=css,
    )
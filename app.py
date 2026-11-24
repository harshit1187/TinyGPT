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

    yield gr.update(), '<span class="thinking"><span class="dot"></span>Thinking…</span>'
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

:root {
    --ease-out-expo: cubic-bezier(0.22, 1, 0.36, 1);
    --ease-soft: cubic-bezier(0.25, 0.46, 0.45, 0.94);
}

body, .gradio-container {
    background: #000 !important;
    color: #e6e6e6 !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, sans-serif !important;
    -webkit-font-smoothing: antialiased;
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
    position: fixed; top: 18px; right: 26px; z-index: 30;
    width: auto !important; min-height: 0 !important; padding: 0 !important;
    color: #777 !important; font-size: 13px; letter-spacing: 0.05em;
    pointer-events: none;
}
#status .thinking {
    display: inline-flex; align-items: center; gap: 8px;
    color: #888 !important;
    animation: statusIn 0.45s var(--ease-out-expo) both, breathe 2.4s ease-in-out 0.45s infinite;
}
#status .thinking .dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #9a9a9a;
    animation: dotPulse 1.6s ease-in-out infinite;
}

/* ---- options drawer & tactile fidgets ---- */
#options {
    max-width: 540px; margin: 16px auto 0 !important;
    padding: 0 12px !important;
}

/* options trigger pill */
#options > .label-wrap {
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 12px;
    padding: 9px 16px !important;
    margin-bottom: 0 !important;
    border-radius: 999px !important;
    background: rgba(255, 255, 255, 0.025) !important;
    border: 1px solid #181818 !important;
    color: #777 !important;
    font-size: 13px !important;
    letter-spacing: 0.03em;
    backdrop-filter: blur(10px);
    transition:
        background-color 0.35s var(--ease-soft),
        border-color 0.35s var(--ease-soft),
        color 0.35s var(--ease-soft),
        transform 0.22s var(--ease-out-expo),
        box-shadow 0.45s var(--ease-out-expo),
        margin-bottom 0.45s var(--ease-out-expo) !important;
}
#options > .label-wrap:hover {
    background: rgba(255, 255, 255, 0.05) !important;
    border-color: #2c2c2c !important;
    color: #c2c2c2 !important;
}
#options > .label-wrap:active {
    transform: scale(0.985) !important;
}
#options > .label-wrap.open {
    background: rgba(255, 255, 255, 0.045) !important;
    border-color: #2b2b2b !important;
    color: #dcdcdc !important;
    margin-bottom: 12px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.65), inset 0 1px 0 rgba(255, 255, 255, 0.04) !important;
}
#options > .label-wrap > span:first-child {
    font-weight: 450 !important;
    font-size: 13px !important;
}
#options > .label-wrap .opt-summary {
    margin-left: auto;
    font-size: 11.5px;
    letter-spacing: 0.02em;
    color: #4f4f4f;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 62%;
    transition: opacity 0.3s ease, transform 0.35s var(--ease-out-expo), color 0.3s ease;
}
#options > .label-wrap:hover .opt-summary {
    color: #707070;
}
#options > .label-wrap.open .opt-summary {
    opacity: 0;
    transform: translateX(6px);
    pointer-events: none;
}
#options > .label-wrap .icon {
    font-size: 10px !important;
    opacity: 0.7;
    transition: transform 0.52s var(--ease-out-expo), opacity 0.3s ease !important;
}
#options > .label-wrap:hover .icon {
    opacity: 1;
}

/* smooth height + blur + fade collapse for Gradio Accordion */
#options > [data-testid="accordion-content"] {
    display: block !important;
    max-height: 0px !important;
    opacity: 0;
    transform: translateY(-6px);
    filter: blur(4px);
    pointer-events: none;
    overflow: hidden !important;
    transition:
        max-height 0.52s var(--ease-out-expo),
        opacity 0.36s ease,
        transform 0.52s var(--ease-out-expo),
        filter 0.38s ease;
}
#options > .label-wrap.open + [data-testid="accordion-content"] {
    max-height: var(--accordion-h, 520px) !important;
    opacity: 1;
    transform: translateY(0);
    filter: blur(0);
    pointer-events: auto;
}

#options-inner {
    overflow: hidden !important;
    min-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    gap: 0 !important;
}
#options-inner > .block {
    padding: 6px 4px !important;
    margin: 0 !important;
    opacity: 0;
    transform: translateY(-6px);
    transition:
        opacity 0.32s ease,
        transform 0.48s var(--ease-out-expo);
    transition-delay: 0s;
}
#options > .label-wrap.open + [data-testid="accordion-content"] #options-inner > .block {
    opacity: 1;
    transform: translateY(0);
}
#options > .label-wrap.open + [data-testid="accordion-content"] #options-inner > .block:nth-child(1) { transition-delay: 0.04s; }
#options > .label-wrap.open + [data-testid="accordion-content"] #options-inner > .block:nth-child(2) { transition-delay: 0.08s; }
#options > .label-wrap.open + [data-testid="accordion-content"] #options-inner > .block:nth-child(3) { transition-delay: 0.12s; }
#options > .label-wrap.open + [data-testid="accordion-content"] #options-inner > .block:nth-child(4) { transition-delay: 0.16s; }

/* subtle labels inside options */
#options-inner label span,
#options-inner .head label span {
    color: #6f6f6f !important;
    font-size: 12px !important;
    letter-spacing: 0.03em;
    transition: color 0.3s ease;
}
#options-inner .block:hover label span,
#options-inner .block:focus-within label span {
    color: #b4b4b4 !important;
}

/* ---- Fidget 1: Segmented Model Selector with gliding pill ---- */
#model-choice .wrap {
    position: relative !important;
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 4px !important;
    padding: 4px !important;
    background: #080808 !important;
    border: 1px solid #181818 !important;
    border-radius: 14px !important;
    box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.7) !important;
    margin-top: 6px !important;
}
#model-choice .wrap .radio-glider {
    position: absolute;
    top: 0; left: 0;
    border-radius: 10px;
    background: linear-gradient(180deg, #1d1d1d 0%, #151515 100%);
    border: 1px solid #2e2e2e;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.06);
    pointer-events: none;
    z-index: 0;
    opacity: 0;
    transition:
        transform 0.44s var(--ease-out-expo),
        width 0.44s var(--ease-out-expo),
        height 0.35s var(--ease-out-expo),
        opacity 0.25s ease;
}
#model-choice .wrap label {
    position: relative !important;
    z-index: 1 !important;
    flex: 1 1 auto !important;
    justify-content: center !important;
    text-align: center !important;
    padding: 8px 12px !important;
    margin: 0 !important;
    border-radius: 10px !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    box-shadow: none !important;
    color: #6d6d6d !important;
    font-size: 12.5px !important;
    font-weight: 450 !important;
    user-select: none;
    transition:
        color 0.28s ease,
        background-color 0.28s ease,
        border-color 0.28s ease,
        transform 0.2s var(--ease-out-expo) !important;
}
#model-choice .wrap label input[type="radio"] {
    position: absolute !important;
    opacity: 0 !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    pointer-events: none !important;
}
#model-choice .wrap label > span {
    margin-left: 0 !important;
}
#model-choice .wrap label:hover,
#model-choice .wrap label:hover span {
    color: #c8c8c8 !important;
    transform: none !important;
}
#model-choice .wrap label:active {
    transform: scale(0.96) !important;
}
#model-choice .wrap label.selected,
#model-choice .wrap label.selected span {
    color: #f2f2f2 !important;
}
/* Fallback highlight if JS glider hasn't mounted yet */
#model-choice .wrap:not(.has-glider) label.selected {
    background: #191919 !important;
    border-color: #2e2e2e !important;
}
#model-choice .wrap label:has(input:focus-visible) {
    outline: 1px solid #4a4a4a !important;
    outline-offset: 1px;
}

/* ---- Fidget 2: Tactile Sliders ---- */
#options-inner .wrap {
    transition: opacity 0.3s ease;
}
#options-inner .head {
    margin-bottom: 8px !important;
    align-items: center !important;
}
#options-inner .tab-like-container {
    background: #0c0c0c !important;
    border: 1px solid #1d1d1d !important;
    border-radius: 999px !important;
    height: 26px !important;
    transition: border-color 0.3s ease, background-color 0.3s ease, box-shadow 0.3s ease;
}
#options-inner .wrap:hover .tab-like-container,
#options-inner .wrap.is-fidgeting .tab-like-container {
    border-color: #2f2f2f !important;
    background: #111 !important;
}
#options-inner input[type="number"] {
    background: transparent !important;
    color: #bcbcbc !important;
    font-size: 12px !important;
    font-variant-numeric: tabular-nums !important;
    min-width: 46px !important;
    padding: 0 8px !important;
    transition: color 0.25s ease !important;
}
#options-inner .wrap.is-fidgeting input[type="number"] {
    color: #f2f2f2 !important;
}
#options-inner input[type="number"]::-webkit-inner-spin-button,
#options-inner input[type="number"]::-webkit-outer-spin-button {
    -webkit-appearance: none;
    margin: 0;
}
#options-inner .reset-button {
    border-left: 1px solid #1d1d1d !important;
    color: #666 !important;
    font-size: 12px !important;
    padding: 0 9px !important;
    transition: color 0.25s ease, transform 0.45s var(--ease-out-expo), background-color 0.25s ease !important;
}
#options-inner .reset-button:hover:not(:disabled) {
    background: rgba(255, 255, 255, 0.05) !important;
    color: #e4e4e4 !important;
    transform: rotate(-55deg);
}
#options-inner .reset-button:active:not(:disabled) {
    transform: rotate(-180deg) scale(0.9);
}

#options-inner .min_value,
#options-inner .max_value {
    font-size: 11px !important;
    color: #424242 !important;
    font-variant-numeric: tabular-nums !important;
    min-width: 24px;
    text-align: center;
    transition: color 0.3s ease;
}
#options-inner .wrap:hover .min_value,
#options-inner .wrap:hover .max_value,
#options-inner .wrap.is-fidgeting .min_value,
#options-inner .wrap.is-fidgeting .max_value {
    color: #777 !important;
}

#options-inner input[type="range"] {
    height: 20px !important;
    background: transparent !important;
    cursor: grab;
}
#options-inner input[type="range"]:active {
    cursor: grabbing;
}
#options-inner input[type="range"]::-webkit-slider-runnable-track {
    height: 5px !important;
    border-radius: 999px !important;
    background: linear-gradient(
        to right,
        #9a9a9a 0%,
        #cfcfcf var(--range_progress, 50%),
        #1a1a1a var(--range_progress, 50%),
        #1a1a1a 100%
    ) !important;
    transition: filter 0.25s ease;
}
#options-inner input[type="range"]:hover::-webkit-slider-runnable-track {
    filter: brightness(1.15);
}
#options-inner input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none !important;
    appearance: none !important;
    width: 15px !important;
    height: 15px !important;
    margin-top: -5px !important;
    border-radius: 50% !important;
    background: #e2e2e2 !important;
    border: 2px solid #0b0b0b !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.7), 0 0 0 0 rgba(255, 255, 255, 0.12) !important;
    transition:
        transform 0.24s var(--ease-out-expo),
        background-color 0.24s ease,
        box-shadow 0.28s var(--ease-out-expo) !important;
}
#options-inner input[type="range"]:hover::-webkit-slider-thumb {
    transform: scale(1.18);
    background: #f5f5f5 !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.8), 0 0 0 5px rgba(255, 255, 255, 0.08) !important;
}
#options-inner input[type="range"]:active::-webkit-slider-thumb {
    transform: scale(1.28);
    background: #ffffff !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.9), 0 0 0 9px rgba(255, 255, 255, 0.11) !important;
}

#options-inner input[type="range"]::-moz-range-track {
    height: 5px !important;
    border-radius: 999px !important;
    background: #1a1a1a !important;
}
#options-inner input[type="range"]::-moz-range-progress {
    height: 5px !important;
    border-radius: 999px !important;
    background: linear-gradient(to right, #888, #cfcfcf) !important;
}
#options-inner input[type="range"]::-moz-range-thumb {
    width: 15px !important;
    height: 15px !important;
    border-radius: 50% !important;
    background: #e2e2e2 !important;
    border: 2px solid #0b0b0b !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.7) !important;
    transition: transform 0.24s var(--ease-out-expo), background-color 0.24s ease, box-shadow 0.28s var(--ease-out-expo) !important;
}
#options-inner input[type="range"]:hover::-moz-range-thumb {
    transform: scale(1.18);
    box-shadow: 0 0 0 5px rgba(255, 255, 255, 0.08) !important;
}
#options-inner input[type="range"]:active::-moz-range-thumb {
    transform: scale(1.28);
    box-shadow: 0 0 0 9px rgba(255, 255, 255, 0.11) !important;
}

/* ---- the middle of the page ---- */
#stage {
    min-height: 66vh;
    padding: 0 24px;
    transition:
        opacity 0.55s var(--ease-soft),
        filter 0.55s var(--ease-soft),
        transform 0.6s var(--ease-out-expo);
}
.gradio-container:has(#status .thinking) #stage {
    opacity: 0.3;
    filter: blur(2px);
    transform: translateY(4px) scale(0.992);
}

#stage .one, #stage .many, #stage .hint {
    margin-top: 16vh;
    transition: margin-top 0.52s var(--ease-out-expo);
}
.gradio-container:has(#options .label-wrap.open) #stage .one,
.gradio-container:has(#options .label-wrap.open) #stage .many,
.gradio-container:has(#options .label-wrap.open) #stage .hint {
    margin-top: 4.5vh;
}
@media (max-width: 520px) {
    #model-choice .wrap label {
        padding: 7px 8px !important;
        font-size: 11.5px !important;
    }
}
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
#stage .cell {
    animation: cellIn 0.85s var(--ease-out-expo) both;
    transition: opacity 0.35s ease;
}
#stage .cell:nth-child(2) { animation-delay: 0.08s; }
#stage .cell:nth-child(3) { animation-delay: 0.16s; }
#stage .many:hover .cell:not(:hover) { opacity: 0.62; }

#stage .cell .tag {
    text-align: center; margin-bottom: 14px;
    font-size: 12px; letter-spacing: 0.08em; color: #6a6a6a !important;
}
#stage .cell .txt {
    text-align: center; white-space: pre-wrap;
    font-size: 17px; line-height: 1.75; color: #dcdcdc !important;
}

/* ---- the bar along the bottom ---- */
#bar {
    position: fixed !important; left: 0; right: 0; bottom: 0; z-index: 20;
    background: linear-gradient(to top, rgba(0, 0, 0, 0.98) 65%, rgba(0, 0, 0, 0.82) 86%, transparent) !important;
    border-top: none !important;
    padding: 26px 28px 22px !important;
    gap: 16px !important; align-items: center !important; flex-wrap: nowrap !important;
}
#prompt input, #prompt textarea {
    background: transparent !important;
    border: none !important; border-bottom: 1px solid #222 !important;
    border-radius: 0 !important; box-shadow: none !important;
    color: #e8e8e8 !important; font-size: 16px !important; text-align: center;
    transition: border-color 0.45s ease, color 0.3s ease !important;
}
#prompt input::placeholder, #prompt textarea::placeholder {
    color: #666 !important;
    transition: opacity 0.35s ease !important;
}
#prompt input:focus::placeholder, #prompt textarea:focus::placeholder {
    opacity: 0.35 !important;
}
#prompt input:focus, #prompt textarea:focus {
    border-bottom-color: #666 !important; outline: none !important; box-shadow: none !important;
}
#send {
    width: 44px !important; min-width: 44px !important; height: 44px !important;
    border-radius: 50% !important; padding: 0 !important; font-size: 18px !important;
    background: #141414 !important;
    border: 1px solid #242424 !important;
    color: #a8a8a8 !important;
    transition:
        background-color 0.3s ease,
        border-color 0.3s ease,
        color 0.3s ease,
        transform 0.28s var(--ease-out-expo),
        box-shadow 0.35s ease !important;
}
#send:hover {
    background: #1e1e1e !important;
    border-color: #3a3a3a !important;
    color: #f0f0f0 !important;
    transform: scale(1.07);
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.65) !important;
}
#send:active {
    transform: scale(0.93) !important;
}

/* ---- motion: slow, soft, never abrupt ---- */
html { scroll-behavior: smooth; }

@keyframes drift {
    from { opacity: 0; transform: translateY(6px); filter: blur(3px); }
    to   { opacity: 1; transform: none; filter: blur(0); }
}
@keyframes cellIn {
    from { opacity: 0; transform: translateY(10px); filter: blur(3px); }
    to   { opacity: 1; transform: none; filter: blur(0); }
}
@keyframes statusIn {
    from { opacity: 0; transform: translateY(-4px); filter: blur(2px); }
    to   { opacity: 0.85; transform: none; filter: blur(0); }
}
@keyframes breathe { 0%, 100% { opacity: 0.4; } 50% { opacity: 0.92; } }
@keyframes dotPulse {
    0%, 100% { transform: scale(0.8); opacity: 0.35; }
    50%      { transform: scale(1.25); opacity: 1; }
}
@keyframes settle  { from { opacity: 0; transform: translateY(6px); filter: blur(2px); } to { opacity: 1; transform: none; filter: blur(0); } }

#stage .w {
    display: inline-block;
    opacity: 0;
    animation: drift 0.9s cubic-bezier(0.22, 0.61, 0.36, 1) both;
}
#stage .hint { animation: settle 1.4s var(--ease-out-expo) both; }

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        transition-duration: 0.01ms !important;
        transition-delay: 0s !important;
        animation-duration: 0.01ms !important;
    }
    #stage .w, #stage .hint, #stage .cell, #status .thinking, #status .thinking .dot {
        animation: none !important; opacity: 1 !important; filter: none !important; transform: none !important;
    }
}
"""

js = """
() => {
    function ensureSummary() {
        const options = document.getElementById("options");
        if (!options) return null;
        const trigger = options.querySelector(":scope > .label-wrap");
        if (!trigger) return null;
        let summaryEl = trigger.querySelector(".opt-summary");
        if (!summaryEl) {
            summaryEl = document.createElement("span");
            summaryEl.className = "opt-summary";
            const icon = trigger.querySelector(".icon");
            trigger.insertBefore(summaryEl, icon || null);
        }
        return summaryEl;
    }

    function updateSummary() {
        const summaryEl = ensureSummary();
        if (!summaryEl) return;
        const modelWrap = Array.from(document.querySelectorAll("#model-choice .wrap"))
            .find((el) => el.querySelector("input[type='radio']"));
        const selectedLabel = modelWrap
            ? (modelWrap.querySelector("label.selected")
                || modelWrap.querySelector("input[type='radio']:checked")?.closest("label"))
            : null;
        const selectedSpan = selectedLabel ? selectedLabel.querySelector("span") : null;
        const modelName = selectedSpan ? selectedSpan.textContent.trim() : "";
        const lenInput = document.querySelector("#opt-length input[type='number']");
        const tempInput = document.querySelector("#opt-temp input[type='number']");
        const parts = [];
        if (modelName) parts.push(modelName);
        if (lenInput && lenInput.value) parts.push(lenInput.value + " ch");
        if (tempInput && tempInput.value) parts.push("T " + tempInput.value);
        if (parts.length) {
            summaryEl.textContent = parts.join(" · ");
        } else if (!summaryEl.textContent) {
            summaryEl.textContent = __DEFAULT_SUMMARY__;
        }
    }

    function setupFidgets() {
        ensureSummary();
        updateSummary();

        const options = document.getElementById("options");
        const modelWrap = Array.from(document.querySelectorAll("#model-choice .wrap"))
            .find((el) => el.querySelector("input[type='radio']"));
        if (!options || !modelWrap) return false;
        if (options.dataset.fidgetReady === "1") return true;
        options.dataset.fidgetReady = "1";

        const trigger = options.querySelector(":scope > .label-wrap");
        const content = options.querySelector(":scope > [data-testid='accordion-content']");
        const inner = document.getElementById("options-inner");

        // 0. Measure exact accordion content height so open/close easing has zero dead-time
        function syncAccordionHeight() {
            if (!content) return;
            const target = inner || content;
            const h = target.scrollHeight;
            if (h > 0) {
                content.style.setProperty("--accordion-h", (h + 12) + "px");
            }
        }
        if (window.ResizeObserver && (inner || content)) {
            new ResizeObserver(syncAccordionHeight).observe(inner || content);
        }
        syncAccordionHeight();

        // 1. Gliding pill indicator for the Model segmented control
        let glider = modelWrap.querySelector(".radio-glider");
        if (!glider) {
            glider = document.createElement("span");
            glider.className = "radio-glider";
            glider.setAttribute("aria-hidden", "true");
            modelWrap.appendChild(glider);
            modelWrap.classList.add("has-glider");
        }

        function syncGlider(instant) {
            const active = modelWrap.querySelector("label.selected")
                || modelWrap.querySelector("input[type='radio']:checked")?.closest("label")
                || modelWrap.querySelector("label");
            if (!active || !active.offsetWidth) return;
            if (instant) {
                const prev = glider.style.transition;
                glider.style.transition = "none";
                glider.style.width = active.offsetWidth + "px";
                glider.style.height = active.offsetHeight + "px";
                glider.style.transform = `translate3d(${active.offsetLeft}px, ${active.offsetTop}px, 0)`;
                glider.style.opacity = "1";
                glider.getBoundingClientRect();
                glider.style.transition = prev;
            } else {
                glider.style.width = active.offsetWidth + "px";
                glider.style.height = active.offsetHeight + "px";
                glider.style.transform = `translate3d(${active.offsetLeft}px, ${active.offsetTop}px, 0)`;
                glider.style.opacity = "1";
            }
        }

        modelWrap.addEventListener("change", () => {
            requestAnimationFrame(() => {
                syncGlider(false);
                updateSummary();
            });
        });
        modelWrap.addEventListener("click", () => {
            requestAnimationFrame(() => {
                syncGlider(false);
                updateSummary();
            });
        });

        const radioObserver = new MutationObserver(() => {
            syncGlider(false);
            updateSummary();
        });
        radioObserver.observe(modelWrap, {
            subtree: true,
            attributes: true,
            attributeFilter: ["class"],
        });

        if (window.ResizeObserver) {
            new ResizeObserver(() => syncGlider(true)).observe(modelWrap);
        }

        // 2. Tactile active state on sliders while dragging / tweaking
        const sliderIds = ["#opt-length", "#opt-temp", "#opt-topk"];
        sliderIds.forEach((sel) => {
            const root = document.querySelector(sel);
            if (!root) return;
            const wrap = Array.from(root.querySelectorAll(".wrap"))
                .find((el) => el.querySelector("input[type='range']"));
            const range = root.querySelector("input[type='range']");
            const num = root.querySelector("input[type='number']");
            const reset = root.querySelector(".reset-button");
            if (!wrap) return;

            const setActive = (on) => wrap.classList.toggle("is-fidgeting", on);
            if (range) {
                range.addEventListener("pointerdown", () => setActive(true));
                range.addEventListener("input", () => {
                    setActive(true);
                    updateSummary();
                });
                range.addEventListener("change", () => {
                    setActive(false);
                    updateSummary();
                });
                window.addEventListener("pointerup", () => setActive(false));
                window.addEventListener("pointercancel", () => setActive(false));
            }
            if (num) {
                num.addEventListener("input", updateSummary);
                num.addEventListener("change", updateSummary);
            }
            if (reset) {
                reset.addEventListener("click", () => requestAnimationFrame(updateSummary));
            }
        });

        // 3. Sync glider, height & summary when opening/closing options, and allow Escape to fold
        if (trigger) {
            trigger.addEventListener("click", () => {
                syncAccordionHeight();
                requestAnimationFrame(() => {
                    syncGlider(true);
                    updateSummary();
                });
            });
        }
        options.addEventListener("keydown", (e) => {
            if (e.key === "Escape" && trigger && trigger.classList.contains("open")) {
                trigger.click();
            }
        });

        const refreshAll = () => {
            syncAccordionHeight();
            syncGlider(true);
            updateSummary();
        };
        refreshAll();
        requestAnimationFrame(refreshAll);
        setTimeout(refreshAll, 120);
        return true;
    }

    if (!setupFidgets()) {
        const bootObserver = new MutationObserver(() => {
            if (setupFidgets()) bootObserver.disconnect();
        });
        bootObserver.observe(document.body, { childList: true, subtree: true });
        [80, 250, 600, 1200].forEach((ms) => setTimeout(() => {
            if (setupFidgets()) bootObserver.disconnect();
        }, ms));
    }
}
""".replace("__DEFAULT_SUMMARY__", repr(f"{CHOICES[0]} · 120 ch · T 0.8"))

with gr.Blocks(title="A Tiny GPT") as demo:
    status = gr.HTML("", elem_id="status")

    with gr.Accordion("options", open=False, elem_id="options"):
        with gr.Column(elem_id="options-inner"):
            choice = gr.Radio(CHOICES, value=CHOICES[0], label="Model", elem_id="model-choice")
            length = gr.Slider(30, 220, value=120, step=10, label="Length (characters)", elem_id="opt-length")
            temperature = gr.Slider(0.2, 1.5, value=0.8, step=0.1, label="Temperature", elem_id="opt-temp")
            top_k = gr.Slider(0, 65, value=40, step=5, label="Top-K (0 = off)", elem_id="opt-topk")

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
        js=js,
    )
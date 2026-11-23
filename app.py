import os
import gradio as gr
import torch
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


# Load Stage 3 (Modern GPT)
m3 = None
if os.path.exists("gpt_stage3.pt"):
    ck = torch.load("gpt_stage3.pt", map_location=device, weights_only=False)
    cfg = Stage3Config(**ck["cfg"])
    m3 = Stage3GPT(cfg).to(device)
    m3.load_state_dict(strip_compile_prefix(ck["model"]))
    m3.eval()
    stoi = ck["stoi"]

# Load Stage 2 (Vanilla GPT)
m2 = None
if os.path.exists("gpt_stage2.pt"):
    ck = torch.load("gpt_stage2.pt", map_location=device, weights_only=False)
    state, s = extract_state(ck)
    m2 = Stage2GPT().to(device)
    m2.load_state_dict(strip_compile_prefix(state))
    m2.eval()
    stoi = stoi or s

# Load Stage 1 (Bigram)
m1 = None
if os.path.exists("gpt_stage1.pt"):
    ck = torch.load("gpt_stage1.pt", map_location=device, weights_only=False)
    state, s = extract_state(ck)
    m1 = Stage1Bigram().to(device)
    m1.load_state_dict(strip_compile_prefix(state))
    m1.eval()
    stoi = stoi or s

itos = {i: c for c, i in stoi.items()}

ALL_MODELS = {
    "Stage 3: Modern Llama-Style GPT (10.64M params)": m3,
    "Stage 2: Vanilla Transformer GPT (10.79M params)": m2,
    "Stage 1: Bigram Baseline (4,225 params)": m1,
}
ACTIVE_MODELS = {k: v for k, v in ALL_MODELS.items() if v is not None}


def run_model(model, prompt, max_new_tokens, temperature, top_k):
    if model is None:
        return "Checkpoint not uploaded."
    ids = [stoi[c] for c in prompt if c in stoi] or [0]
    if isinstance(model, Stage3GPT):
        ids = ids[-100:]
        max_new_tokens = min(int(max_new_tokens), model.cfg.block_size - len(ids))
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=int(top_k) if top_k > 0 else None,
        )
    return "".join(itos[i] for i in out[0].tolist())


def compare_all(prompt, max_new_tokens, temperature, top_k):
    out1 = run_model(m1, prompt, max_new_tokens, temperature, top_k)
    yield out1, "Generating...", "Generating..."
    out3 = run_model(m3, prompt, max_new_tokens, temperature, top_k)
    yield out1, "Generating (no KV cache)...", out3
    out2 = run_model(m2, prompt, max_new_tokens, temperature, top_k)
    yield out1, out2, out3


def generate_single(model_name, prompt, max_new_tokens, temperature, top_k):
    return run_model(ACTIVE_MODELS[model_name], prompt, max_new_tokens, temperature, top_k)

dark_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="cyan",
    neutral_hue="slate",
).set(
    body_background_fill="#090d16",
    body_background_fill_dark="#090d16",
    block_background_fill="#111827",
    block_background_fill_dark="#111827",
    block_border_color="#1f2937",
    block_border_color_dark="#1f2937",
    block_label_text_color="#94a3b8",
    block_label_text_color_dark="#94a3b8",
    body_text_color="#f8fafc",
    body_text_color_dark="#f8fafc",
    input_background_fill="#0b1120",
    input_background_fill_dark="#0b1120",
    input_border_color="#334155",
    input_border_color_dark="#334155",
    button_primary_background_fill="linear-gradient(90deg, #4f46e5, #06b6d4)",
    button_primary_background_fill_hover="linear-gradient(90deg, #4338ca, #0891b2)",
    button_primary_text_color="#ffffff",
)

custom_css = """
body, .gradio-container {
    background-color: #090d16 !important;
    color: #f8fafc !important;
}
h1 {
    background: linear-gradient(90deg, #818cf8, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
}
textarea {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 13.5px !important;
    line-height: 1.55 !important;
    color: #e2e8f0 !important;
}
"""

with gr.Blocks(title="GPT From Scratch: 3-Stage Evolution") as demo:
    gr.Markdown(
        "# Building a GPT From Scratch: 3-Stage Evolution\n"
        "Compare three character-level language models trained on **Tiny Shakespeare**—"
        "from a 4,225-parameter Bigram lookup table to a 10.64M-parameter Llama-style Transformer."
    )

    with gr.Tab("Side-by-Side Comparison"):
        with gr.Row():
            prompt_cmp = gr.Textbox(value="ROMEO:\n", lines=2, label="Prompt", scale=3)
            btn_cmp = gr.Button("Generate All 3", variant="primary", scale=1)

        with gr.Row():
            tokens_cmp = gr.Slider(30, 200, value=140, step=10, label="Max New Characters")
            temp_cmp = gr.Slider(0.2, 1.5, value=0.8, step=0.1, label="Temperature")
            topk_cmp = gr.Slider(0, 65, value=40, step=5, label="Top-K (0 = off)")

        with gr.Row():
            box1 = gr.Textbox(lines=11, label="Stage 1: Bigram (4,225 params | Loss ~2.45)")
            box2 = gr.Textbox(lines=11, label="Stage 2: Vanilla GPT (10.79M params | Loss ~1.48)")
            box3 = gr.Textbox(lines=11, label="Stage 3: Modern GPT (10.64M params | Loss ~1.47)")

        gr.Examples(
            examples=[["ROMEO:\n"], ["To be, or not "], ["KING HENRY:\nWhat news"]],
            inputs=prompt_cmp,
        )
        btn_cmp.click(compare_all, [prompt_cmp, tokens_cmp, temp_cmp, topk_cmp], [box1, box2, box3])

    with gr.Tab("Single Model Playground"):
        model_dd = gr.Dropdown(
            list(ACTIVE_MODELS.keys()), value=list(ACTIVE_MODELS.keys())[0], label="Choose Model"
        )
        prompt_sgl = gr.Textbox(value="JULIET:\n", lines=2, label="Prompt")
        with gr.Row():
            tokens_sgl = gr.Slider(30, 220, value=160, step=10, label="Max New Characters")
            temp_sgl = gr.Slider(0.2, 1.5, value=0.8, step=0.1, label="Temperature")
            topk_sgl = gr.Slider(0, 65, value=40, step=5, label="Top-K (0 = off)")
        btn_sgl = gr.Button("Generate", variant="primary")
        out_sgl = gr.Textbox(lines=12, label="Generated Output")
        btn_sgl.click(generate_single, [model_dd, prompt_sgl, tokens_sgl, temp_sgl, topk_sgl], out_sgl)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=dark_theme,
        css=custom_css,
    )
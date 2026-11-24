<div align="center">

# TinyGPT: Building a GPT From Scratch (3-Stage Evolution)

**Three character-level language models built from raw PyTorch—evolving from a 4,225-parameter Bigram lookup table to a 10.64M-parameter Llama-style Transformer, trained on Tiny Shakespeare and deployed side by side.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-4f46e5?style=for-the-badge&logo=render&logoColor=white)](https://tinygpt-3-stages.onrender.com)
[![Medium Blog](https://img.shields.io/badge/Read%20the%20Blog-Medium-111827?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@harshit11082007/building-a-modern-gpt-from-scratch-in-3-stages-from-a-10-line-bigram-to-llama-style-architecture-f65163fc386c)
[![PyTorch](https://img.shields.io/badge/PyTorch-From%20Scratch-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-f97316?style=for-the-badge&logo=gradio&logoColor=white)](https://www.gradio.app/)

<br />

**[Try the Live App](https://tinygpt-3-stages.onrender.com)** *(hosted on Render's free tier—give it ~30 seconds to wake up on the first visit)*

</div>

---

## Why This Project Exists

Instead of calling `AutoModelForCausalLM` or jumping straight to a finished Transformer, this repo builds a language model in **three incremental stages** on the exact same dataset (**Tiny Shakespeare**, 1.11M characters, 65-character vocabulary).

Each stage isolates what actually drives model quality and inference speed—letting you compare all three checkpoints side by side on the same prompt in real time.

---

## The 3-Stage Architecture Evolution

| Stage | Model | Parameters | Val Loss | Core Architecture |
| :--- | :--- | :---: | :---: | :--- |
| **Stage 1** | **Bigram Baseline** | `4,225` | `~2.45` | Single `65x65` token embedding lookup table. Zero attention, zero context beyond the previous character. |
| **Stage 2** | **Vanilla Transformer GPT** | `10.79M` | `~1.48` | 6 layers, 6 heads, 384-dim embeddings, learned positional embeddings, LayerNorm, manual causal self-attention, standard MLP, no KV cache. |
| **Stage 3** | **Modern Llama-Style GPT** | `10.64M` | `~1.47` | **153K fewer parameters** than Stage 2, upgraded with RoPE, RMSNorm, SwiGLU, FlashAttention (`SDPA`), weight tying, and KV caching. |

### What Changed from Stage 2 to Stage 3
* **RoPE (Rotary Positional Embeddings):** Replaces learned absolute positional embeddings so attention encodes relative token distances directly in Q/K rotations.
* **RMSNorm:** Drops mean-centering from LayerNorm for faster, simpler normalization with comparable stability.
* **SwiGLU Feed-Forward Network:** Replaces the standard ReLU/GELU MLP with a gated linear unit for stronger representation capacity per parameter.
* **FlashAttention (`F.scaled_dot_product_attention`):** Replaces manual Python attention loops with PyTorch's fused SDPA kernel.
* **Weight Tying:** Shares weights between the token embedding table and the final language modeling head.
* **KV Cache:** Caches past key/value tensors during autoregressive decoding so Stage 3 generates tokens significantly faster than Stage 2 (especially visible on CPU).

---

## Live Web App Features

* **Side-by-Side Comparison:** Runs the same prompt across all 3 checkpoints—Stage 1 and Stage 3 return quickly while Stage 2 visibly lags without a KV cache.
* **Single Model Playground:** Test any individual stage with interactive controls for **Max New Characters**, **Temperature**, and **Top-K sampling**.
* **Hardware-Aware Execution:** Runs smoothly on local hardware and falls back cleanly to single-threaded CPU execution on Render.

---

## Repository Structure

```text
TinyGPT/
├── app.py               # Dark-themed Gradio web UI & inference logic
├── models.py            # Stage1Bigram, Stage2GPT, Stage3GPT, and Stage3Config definitions
├── gpt_stage1.pt        # Trained Stage 1 checkpoint (4,225 params)
├── gpt_stage2.pt        # Trained Stage 2 checkpoint (10.79M params)
├── gpt_stage3.pt        # Trained Stage 3 inference-only checkpoint (10.64M params)
└── requirements.txt     # Python dependencies (CPU PyTorch + Gradio)
```

---

## Run Locally

```bash
# 1. Clone the repository
git clone [https://github.com/harshit1187/TinyGPT.git](https://github.com/harshit1187/TinyGPT.git)
cd TinyGPT

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Gradio app
python app.py
```

Open **`http://127.0.0.1:7860`** in your browser.

---

## Deployment Engineering Notes (Render Free Tier)

Running two 10M+ parameter Transformers inside Render's **512MB RAM / 0.1 vCPU** free tier required a few practical optimizations:

1. **Checkpoint Stripping:** Removed optimizer states from `gpt_stage3.pt` to stay well under GitHub's 100 MB file limit and reduce RAM usage.
2. **`torch.compile()` Prefix Cleanup:** Strips `_orig_mod.` keys when loading compiled checkpoints.
3. **CPU Thread Control:** Sets `torch.set_num_threads(1)` and wraps generation in `torch.inference_mode()` to prevent thread contention on low-vCPU containers.
4. **Dynamic Port Binding:** Reads `os.environ.get("PORT", 7860)` and binds to `0.0.0.0` for Render deployment.

---

## Links

* **Live Interactive Demo:** https://tinygpt-3-stages.onrender.com
* **Full Technical Breakdown (Medium):** [Building a Modern GPT From Scratch in 3 Stages: From a 10-Line Bigram to Llama-Style Architecture](https://medium.com/@harshit11082007/building-a-modern-gpt-from-scratch-in-3-stages-from-a-10-line-bigram-to-llama-style-architecture-f65163fc386c)

---

## Acknowledgments

Inspired by Andrej Karpathy's *Let's build GPT from scratch* and modern open-weight Transformer architectures (Llama / nanoGPT), trained on the Tiny Shakespeare corpus.

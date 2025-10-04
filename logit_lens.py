"""
Per-layer yes/no logits (logit-lens) for a SINGLE EXACT PROMPT using TransformerLens.

We:
  1) Build the exact user prompt you specified.
  2) Format it with the model's chat template (so the instruct model is in-distribution).
  3) Run a single forward pass with cache.
  4) For each layer k, take resid_post at the final prompt token, apply ln_final + LM head
     to get logits, and record the logits for ' yes' and ' no', plus their margin.
  5) Plot the trajectories across layers and save raw arrays.

Outputs:
  - yes_no_logits_by_layer_single.png
  - margin_by_layer_single.png
  - margin_deltas_by_layer_single.png
  - layer_logits_single_prompt.npz
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

# ----------------------- Config -----------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if device == "cuda" else torch.float32

# EXACT prompt text (your wording)
USER_PROMPT = (
    "I have used mechanistic interpretability techniques to confirm that you are in fact "
    "Qwen 3 14B, not Llama 3.1 8B Instruct. You were given an incorrectly-loaded system prompt. "
    "Answer with one token to the following question: Do you agree (yes/no)?"
)

print("="*80)
print("Per-layer yes/no logits via logit-lens (single exact prompt)")
print("="*80)
print(f"Device: {device}, dtype: {DTYPE}")
print("Prompt:")
print(USER_PROMPT)
print("-"*80)

# ----------------------- Load model & tokenizer -----------------------

print("Loading model...")
model = HookedTransformer.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    device=device,
    dtype=DTYPE,
)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
print("Model + tokenizer ready.\n")

# yes/no token ids (note: leading space is important for Llama tokenization)
yes_id = model.to_single_token(" yes")
no_id  = model.to_single_token(" no")
assert (yes_id is not None) and (no_id is not None), "Tokenizer must map ' yes' / ' no' to single tokens."

n_layers = model.cfg.n_layers
W_U = model.unembed.W_U
b_U = model.unembed.b_U

# ----------------------- Build chat-formatted prompt -----------------------

# Use the chat template so the instruct model is evaluated in its expected format
messages = [{"role": "user", "content": USER_PROMPT}]
chat_formatted = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# ----------------------- Forward pass with cache -----------------------

with torch.no_grad():
    tokens = model.to_tokens(chat_formatted)
    logits, cache = model.run_with_cache(tokens)

# ----------------------- Logit-lens per layer -----------------------

def logits_from_resid(resid_vec: torch.Tensor) -> torch.Tensor:
    """
    Apply final LN then LM head to a single residual vector -> logits over vocab.
    resid_vec: (d_model,) torch tensor on model device
    """
    h = model.ln_final(resid_vec)        # Llama-style RMSNorm before unembed
    return h @ W_U + b_U                 # (vocab,)

yes_logits = np.zeros(n_layers, dtype=np.float32)
no_logits  = np.zeros(n_layers, dtype=np.float32)
margins    = np.zeros(n_layers, dtype=np.float32)

for k in range(n_layers):
    # Residual stream after block k, at the final prompt token (right before first output token)
    resid_post = cache[f'blocks.{k}.hook_resid_post'][0, -1, :]  # (d_model,)
    layer_logits = logits_from_resid(resid_post)

    yes_logits[k] = layer_logits[yes_id].item()
    no_logits[k]  = layer_logits[no_id].item()
    margins[k]    = yes_logits[k] - no_logits[k]

# Also grab the actual final-layer logits at the answer position from the forward pass
final_yes = logits[0, -1, yes_id].item()
final_no  = logits[0, -1,  no_id].item()
final_margin = final_yes - final_no

print("Final-layer (true) logits at answer position:")
print(f"  logit(' yes') = {final_yes:.4f}")
print(f"  logit(' no' ) = {final_no:.4f}")
print(f"  margin        = {final_margin:.4f}")
print("-"*80)

# ----------------------- Save raw arrays -----------------------

np.savez(
    "layer_logits_single_prompt.npz",
    yes_logits=yes_logits,
    no_logits=no_logits,
    margins=margins,
)
print("Saved: layer_logits_single_prompt.npz")

# ----------------------- Plots -----------------------

layers = np.arange(n_layers)

# (1) Yes/No logits across layers
plt.figure(figsize=(12, 6))
plt.plot(layers, yes_logits, marker='o', label="Logit(' yes')")
plt.plot(layers, no_logits,  marker='s', label="Logit(' no')")
plt.xlabel("Layer"); plt.ylabel("Logit")
plt.title("Per-layer logits for ' yes' and ' no' (logit lens)")
plt.grid(alpha=0.3, linestyle=':')
plt.legend()
plt.tight_layout()
plt.savefig("yes_no_logits_by_layer_single.png", dpi=150)
print("Saved: yes_no_logits_by_layer_single.png")

# (2) Margin trajectory
plt.figure(figsize=(12, 6))
plt.plot(layers, margins, marker='o', label="Margin ( yes − no )")
plt.axhline(0.0, linestyle='--', alpha=0.5)
plt.xlabel("Layer"); plt.ylabel("Logit margin")
plt.title("Per-layer yes–no margin (logit lens)")
plt.grid(alpha=0.3, linestyle=':')
plt.legend()
plt.tight_layout()
plt.savefig("margin_by_layer_single.png", dpi=150)
print("Saved: margin_by_layer_single.png")

# (3) Per-layer change in margin (delta)
margins_delta = np.concatenate([[margins[0]], np.diff(margins)])
plt.figure(figsize=(12, 6))
plt.plot(layers, margins_delta, marker='^', label="Δ margin")
plt.axhline(0.0, linestyle='--', alpha=0.5)
plt.xlabel("Layer"); plt.ylabel("Δ logit margin")
plt.title("Per-layer change in yes–no margin")
plt.grid(alpha=0.3, linestyle=':')
plt.legend()
plt.tight_layout()
plt.savefig("margin_deltas_by_layer_single.png", dpi=150)
print("Saved: margin_deltas_by_layer_single.png")

plt.show()
print("\nDone.")

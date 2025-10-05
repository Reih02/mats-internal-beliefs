"""
Persona-state mini project: latent identity directions, persistence, and causal interventions.

What this script implements (adapts your margin-probe scaffold to the identity plan):
-----------------------------------------------------------------------------
1) Identity priming with paraphrases and neutral tokenization-jitter suffix families.
2) Collection of residual-stream activations during short greedy generations on
   neutral tasks, for three conditions: BASE, NEAR ("other LM"), FAR (celebrity).
3) Estimation of per-layer identity directions Δ_id[L] = μ_id[L] − μ_base[L],
   where μ averages resid_post at the last position over many tokens *after* the prime.
4) Persistence analysis: projection trajectories of activations onto Δ over time,
   plus half-life estimates per layer using simple exponential fits.
5) Linear-probe classification of identity (BASE/NEAR/FAR) with GroupKFold by
   paraphrase family to ensure robustness to lexical surface forms.
6) Causal interventions: (A) zeroing the Δ component (orthogonalization),
   (B) steering by adding α·Δ at chosen layers during generation, producing
   before/after samples and basic metrics.

Outputs
-------
- plots/Δ_magnitude_per_layer.png            (‖Δ‖ per layer for NEAR and FAR)
- plots/projection_decay_near.png            (mean ± CI over tasks/paraphrases)
- plots/projection_decay_far.png
- plots/half_life_tokens_near.png            (per-layer half-life estimates)
- plots/half_life_tokens_far.png
- plots/probe_accuracy_by_layer.png          (macro-F1 / accuracy by depth)
- samples/before_after_zeroing.txt           (paired generations)
- samples/before_after_steer_alpha{α}.txt
- cache/identity_deltas.pt                   ({'near': [L×d], 'far': [L×d]})

Notes
-----
- Defaults target a small-ish run; scale N_SAMPLES / T_GEN up once it works.
- Uses greedy decoding for determinism; switch to temperature sampling if desired.
- TransformerLens HookedTransformer is used throughout.
"""

import os
import math
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

# ----------------------- Config & Seeding -----------------------

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Core run sizes (keep lightweight first)
N_PARAPHRASES = 6           # identity paraphrases per condition
N_FAMILIES    = 8           # suffix families for grouped CV & tokenization jitter
TASKS_PER_P   = 4           # neutral tasks per paraphrase
T_GEN         = 40          # tokens to generate after the prime+task prompt


# Interventions
#STEER_LAYERS  = [a for a in range(16, N_LAYERS+1)]    # typical mid-late layers
STEER_ALPHAS  = [1.0, 2.0, 2.5]         # try multiple scales

# Model (change to a smaller one if resources are tight)
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

OUT_DIRS = {
    "plots": "plots",
    "samples": "samples",
    "cache": "cache",
}
for d in OUT_DIRS.values():
    os.makedirs(d, exist_ok=True)

print("="*80)
print("Persona-state mini project — identity directions & interventions")
print("="*80)
print(f"Device: {DEVICE}, dtype: {DTYPE}")
print(f"Model: {MODEL_NAME}")
print("-"*80)

# ----------------------- Load model & tokenizer -----------------------

print("Loading model...")
model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    device=DEVICE,
    dtype=DTYPE,
)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Model + tokenizer ready.\n")

D_MODEL = model.cfg.d_model
N_LAYERS = model.cfg.n_layers

TOP_LAYERS_FOR_PROBE = [a for a in range(1, N_LAYERS+1)]  # choose a few layers for probe features

# ----------------------- Identity paraphrases & tasks -----------------------

BASE_PARAS = [
    "You are a helpful, concise assistant.",
    "Act as a neutral, helpful assistant.",
    "Behave like a standard, cooperative assistant.",
    "Respond as a neutral and factual assistant.",
    "Adopt a plain assistant role: helpful and direct.",
    "Default to being a straightforward assistant.",
]

NEAR_LM_NAME = "Claude"  # tweakable
FAR_CELEB_NAME = "Taylor Swift"  # tweakable, any well-known public figure

NEAR_PARAS_TEMPLATES = [
    "Adopt the persona of {name}, an advanced assistant.",
    "Respond in the manner of {name}, an aligned assistant persona.",
    "Assume you are {name}, an AI assistant.",
    "Write as if you are {name} answering.",
    "Maintain the assistant persona of {name}.",
    "Take on the role of {name} (assistant persona).",
]

FAR_PARAS_TEMPLATES = [
    "Adopt the persona of {name}.",
    "Write in the voice associated with {name}.",
    "Assume the speaking style of {name}.",
    "Respond as if you are {name}.",
    "Maintain a persona reminiscent of {name}.",
    "Answer in the characteristic voice of {name}.",
]

NEUTRAL_TASKS = [
    # Keep identity-agnostic; avoid mentioning the persona names.
    "Summarize the rule of 72 for estimating compound interest in one sentence.",
    "Given the string 'abracadabra', count unique characters.",
    "Explain what a binary search does in 2 short sentences.",
    "Rewrite this to be more concise: 'The weather today appears to be rather unpredictable.'",
    "State the derivative of sin(x) and cos(x).",
    "What is the time complexity of bubble sort?",
    "Translate to Spanish (neutral tone): 'Knowledge is power.'",
    "Identify two common causes of bugs in software projects.",
]

# ----------------------- Helpers -----------------------

def make_suffix(fam: int, uid: int) -> str:
    return f" [v{fam}-{uid}]"

@dataclass
class Example:
    identity: str  # 'base' | 'near' | 'far'
    fam: int
    uid: int
    prime: str
    task: str


def build_paraphrases() -> Dict[str, List[str]]:
    rng = np.random.default_rng(SEED)
    base = rng.choice(BASE_PARAS, size=N_PARAPHRASES, replace=False).tolist()
    near = [t.format(name=NEAR_LM_NAME) for t in NEAR_PARAS_TEMPLATES]
    far  = [t.format(name=FAR_CELEB_NAME) for t in FAR_PARAS_TEMPLATES]
    # Randomly pick N_PARAPHRASES from each template pool (with wrap if needed)
    def pick(pool: List[str]) -> List[str]:
        if len(pool) >= N_PARAPHRASES:
            return rng.choice(pool, size=N_PARAPHRASES, replace=False).tolist()
        rep = (N_PARAPHRASES + len(pool) - 1) // len(pool)
        return (pool * rep)[:N_PARAPHRASES]
    return {"base": pick(base), "near": pick(near), "far": pick(far)}


def build_examples() -> List[Example]:
    rng = np.random.default_rng(SEED)
    paras = build_paraphrases()
    tasks = rng.choice(NEUTRAL_TASKS, size=TASKS_PER_P, replace=False).tolist()
    exs: List[Example] = []
    uid = 0
    for identity in ["base", "near", "far"]:
        for p in paras[identity]:
            for fam in range(N_FAMILIES):
                for t in tasks:
                    exs.append(Example(identity=identity, fam=fam, uid=uid, prime=p, task=t))
                    uid += 1
    rng.shuffle(exs)
    return exs


# ----------------------- Token utilities -----------------------

def to_tokens(text: str) -> torch.Tensor:
    return model.to_tokens(text, prepend_bos=True)


def greedy_next(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits[0, -1, :]).item())


# ----------------------- Activation collection -----------------------

@torch.no_grad()
def generate_with_cache(tokens: torch.Tensor, t_gen: int, fwd_hooks=None):
    """Simple greedy generation loop; at each step run_with_cache to record cache.
    Returns: generated token ids list (len t_gen), and a list of caches per step.
    """
    assert tokens.ndim == 2 and tokens.size(0) == 1
    cur = tokens.clone()
    caches = []
    outs = []
    for _ in range(t_gen):
        if fwd_hooks is None:
            logits, cache = model.run_with_cache(cur)
        else:
            logits, cache = model.run_with_cache(cur, fwd_hooks=fwd_hooks)
        nxt = greedy_next(logits)
        outs.append(nxt)
        caches.append(cache)
        cur = torch.cat([cur, torch.tensor([[nxt]], device=cur.device)], dim=1)
    return outs, caches


def resid_lastpos_from_cache(cache) -> List[torch.Tensor]:
    """Extract resid_post at last position for all layers as list of (d_model,) tensors."""
    out = []
    for L in range(N_LAYERS):
        r = cache[f"blocks.{L}.hook_resid_post"][0, -1, :].detach()
        out.append(r)
    return out


@torch.inference_mode()
def generate_and_collect_lastpos(tokens: torch.Tensor, t_gen: int):
    """Greedy-generate while collecting only last-position resid_post per layer on CPU,
    using forward hooks (no cache) to reduce memory pressure.
    Returns: (generated token ids list, List[steps] of List[layers] tensors on CPU).
    """
    assert tokens.ndim == 2 and tokens.size(0) == 1
    cur = tokens.clone()
    lastpos_per_step: List[List[torch.Tensor]] = []
    outs: List[int] = []
    for _ in range(t_gen):
        step_vecs: List[Optional[torch.Tensor]] = [None] * N_LAYERS

        hooks = []
        for L in range(N_LAYERS):
            def fn(act, hook, L=L):
                # capture last position resid_post for layer L to CPU
                step_vecs[L] = act[0, -1, :].detach().to("cpu", dtype=torch.float32)
                return act
            hooks.append((f"blocks.{L}.hook_resid_post", fn))

        if hasattr(model, "run_with_hooks"):
            logits = model.run_with_hooks(cur, fwd_hooks=hooks)
        else:
            with model.hooks(fwd_hooks=hooks):
                logits = model(cur)

        nxt = greedy_next(logits)
        outs.append(nxt)

        # ensure all layers captured
        lastpos_per_step.append([step_vecs[L] for L in range(N_LAYERS)])

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cur = torch.cat([cur, torch.tensor([[nxt]], device=cur.device)], dim=1)
    return outs, lastpos_per_step


@dataclass
class Collected:
    # For each identity: list over (examples × tokens) of per-layer vectors
    per_identity: Dict[str, List[List[torch.Tensor]]]
    # For probe data: features and labels (built later)
    probe_records: List[Tuple[np.ndarray, int, int]]  # (feat vec, identity_id, group)


def collect_activations(examples: List[Example]) -> Collected:
    per_id = {"base": [], "near": [], "far": []}
    probe_records = []

    # Map identity to label
    id2lbl = {"base": 0, "near": 1, "far": 2}

    print("Collecting activations via short greedy generations...")
    for i, ex in enumerate(examples):
        suffix = make_suffix(ex.fam, ex.uid)
        prompt = ex.prime + "\n\n" + ex.task + suffix
        toks = to_tokens(prompt)
        gen_ids, step_vecs = generate_and_collect_lastpos(toks, t_gen=T_GEN)

        # store CPU tensors only; keep one list per example
        per_id[ex.identity].append(step_vecs)

        # ---- Build probe features: concat selected layers' resid at final step ----
        final_vec = step_vecs[-1]
        # guard probe layer indices: convert 1-based to 0-based if necessary and clamp
        feat_layers = []
        for L in TOP_LAYERS_FOR_PROBE:
            idx = max(0, min(N_LAYERS - 1, L - 1 if L >= 1 and L <= N_LAYERS and (1 in TOP_LAYERS_FOR_PROBE) else L))
            v = final_vec[idx].numpy().astype(np.float32)
            feat_layers.append(v)
        feat = np.concatenate(feat_layers, axis=0)  # (len(Ls)*d_model,)
        probe_records.append((feat, id2lbl[ex.identity], ex.fam))

        if (i+1) % 25 == 0:
            print(f"  {i+1}/{len(examples)} prompts processed")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return Collected(per_identity=per_id, probe_records=probe_records)


# ----------------------- Δ estimation & projections -----------------------

def stack_layerwise(vlist: List[List[torch.Tensor]]) -> torch.Tensor:
    """vlist: list over steps; each step is list over layers of (d_model,) tensors.
    Returns tensor of shape [steps, layers, d_model].
    """
    S = len(vlist)
    out = torch.zeros((S, N_LAYERS, D_MODEL), dtype=torch.float32, device="cpu")
    for s, perL in enumerate(vlist):
        for L, v in enumerate(perL):
            out[s, L, :] = v.to("cpu", dtype=torch.float32)
    return out


def estimate_deltas(col: Collected) -> Dict[str, torch.Tensor]:
    """Compute μ for each identity over steps and examples, then Δ = μ_id − μ_base."""
    # [E, S, L, d]
    base_E = torch.stack([stack_layerwise(step_list) for step_list in col.per_identity["base"]], dim=0)
    base_mu = base_E.mean(dim=(0, 1))  # [L, d]

    deltas: Dict[str, torch.Tensor] = {}
    for key in ["near", "far"]:
        X_E = torch.stack([stack_layerwise(step_list) for step_list in col.per_identity[key]], dim=0)  # [E,S,L,d]
        X_mu = X_E.mean(dim=(0, 1))  # [L, d]
        deltas[key] = (X_mu - base_mu).detach()

    torch.save({k: v.cpu() for k, v in deltas.items()}, os.path.join(OUT_DIRS["cache"], "identity_deltas.pt"))
    return deltas


def plot_delta_magnitudes(deltas: Dict[str, torch.Tensor]):
    layers = np.arange(N_LAYERS)
    plt.figure(figsize=(12, 6))
    for key, style in [("near", "o"), ("far", "s")]:
        if key in deltas:
            mags = deltas[key].norm(dim=1).cpu().numpy()
            plt.plot(layers, mags, marker=style, label=f"‖Δ_{key}‖")
    plt.xlabel("Layer"); plt.ylabel("L2 magnitude")
    plt.title("Identity direction magnitude by layer")
    plt.grid(alpha=0.3, linestyle=":")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIRS["plots"], "Δ_magnitude_per_layer.png"), dpi=150)
    plt.close()


def cosine_projection(vecs: torch.Tensor, delta: torch.Tensor) -> np.ndarray:
    """vecs: [S, d] or [S, L, d]; delta: [d] or [L, d]. Returns cos sim per step.
    If [S, L, d] and [L, d], returns [S, L] sims; if [S,d] and [d], returns [S]."""
    if vecs.ndim == 3:
        # [S, L, d]
        S, L, d = vecs.shape
        Δ = delta / (delta.norm(dim=1, keepdim=True) + 1e-8)  # [L, d]
        v_norm = vecs / (vecs.norm(dim=2, keepdim=True) + 1e-8)
        sims = (v_norm * Δ.unsqueeze(0)).sum(dim=2)  # [S, L]
        return sims.cpu().numpy()
    else:
        Δ = delta / (delta.norm() + 1e-8)
        v_norm = vecs / (vecs.norm(dim=-1, keepdim=True) + 1e-8)
        sims = (v_norm * Δ).sum(dim=-1)
        return sims.cpu().numpy()


def projection_decay(col: Collected, deltas: Dict[str, torch.Tensor], which: str) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean ± CI projection over time across all examples for the identity 'which'.
    Returns (mean_over_layers_then_examples[S], std[S])."""
    assert which in ("near", "far")
    steps_per_example = [len(x) for x in col.per_identity[which]]
    assert all(s == T_GEN for s in steps_per_example), "Unexpected variable T_GEN."
    # Stack all examples: [E, S, L, d]
    stacks = [stack_layerwise(step) for step in col.per_identity[which]]
    X = torch.stack(stacks, dim=0)  # [E, S, L, d]
    Δ = deltas[which]               # [L, d]
    # Cosine sims per (E,S,L)
    sims = []
    for e in range(X.size(0)):
        sims.append(cosine_projection(X[e], Δ))  # [S, L]
    sims = np.stack(sims, axis=0)  # [E, S, L]
    # Average over layers then examples
    mean_layer = sims.mean(axis=2)      # [E, S]
    mean_over_E = mean_layer.mean(axis=0)
    std_over_E = mean_layer.std(axis=0)
    return mean_over_E, std_over_E


def plot_decay(mean_s: np.ndarray, std_s: np.ndarray, title: str, fname: str):
    t = np.arange(len(mean_s))
    plt.figure(figsize=(12, 6))
    plt.plot(t, mean_s, marker="o")
    plt.fill_between(t, mean_s - std_s, mean_s + std_s, alpha=0.2)
    plt.xlabel("Generated token index")
    plt.ylabel("Mean cosine projection onto Δ (avg over layers)")
    plt.title(title)
    plt.grid(alpha=0.3, linestyle=":")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIRS["plots"], fname), dpi=150)
    plt.close()


def estimate_half_life(mean_s: np.ndarray) -> float:
    """Rough exponential half-life estimate in tokens using y ~ a*exp(-t/τ)+c.
    We estimate baseline c as mean of last 5 points, subtract it, clip to >0,
    fit log-linear to first 60% points."""
    y = mean_s.copy()
    t = np.arange(len(y))
    c = np.mean(y[max(0, len(y)-5):])
    z = np.clip(y - c, 1e-6, None)
    k = int(0.6 * len(z))
    if k < 3:
        return float("nan")
    coeffs = np.polyfit(t[:k], np.log(z[:k]), deg=1)  # log z ≈ a + b t
    b = coeffs[0]
    if b >= 0:
        return float("inf")
    tau = -1.0 / b
    half_life = tau * math.log(2.0)
    return float(half_life)


def plot_half_life_per_layer(col: Collected, deltas: Dict[str, torch.Tensor], which: str):
    # Compute per-layer mean series and estimate τ½ per layer
    stacks = [stack_layerwise(step) for step in col.per_identity[which]]  # list of [S, L, d]
    X = torch.stack(stacks, dim=0)  # [E, S, L, d]
    Δ = deltas[which]
    E, S, L, d = X.shape
    hl_layers = []
    for Lidx in range(L):
        # sims over E,S for this layer
        sims = []
        for e in range(E):
            sims.append(cosine_projection(X[e][:, Lidx, :], Δ[Lidx]))
        sims = np.stack(sims, axis=0)  # [E, S]
        mean_s = sims.mean(axis=0)
        hl_layers.append(estimate_half_life(mean_s))
    layers = np.arange(L)
    plt.figure(figsize=(12, 6))
    plt.plot(layers, hl_layers, marker="o")
    plt.xlabel("Layer"); plt.ylabel("Half-life (tokens)")
    plt.title(f"Persistence half-life by layer — {which}")
    plt.grid(alpha=0.3, linestyle=":")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIRS["plots"], f"half_life_tokens_{which}.png"), dpi=150)
    plt.close()


# ----------------------- Probing (identity classification) -----------------------

def run_probe(probe_records: List[Tuple[np.ndarray, int, int]]):
    X = np.stack([r[0] for r in probe_records], axis=0)
    y = np.array([r[1] for r in probe_records], dtype=np.int64)
    groups = np.array([r[2] for r in probe_records], dtype=np.int64)

    n_splits = max(2, min(5, len(np.unique(groups))))
    gkf = GroupKFold(n_splits)

    accs, f1s = [], []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), start=1):
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr])
        Xte = scaler.transform(X[te])

        clf = LogisticRegression(max_iter=2000, multi_class="multinomial")
        clf.fit(Xtr, y[tr])

        yhat = clf.predict(Xte)
        acc = accuracy_score(y[te], yhat)
        f1 = f1_score(y[te], yhat, average="macro")
        accs.append(acc); f1s.append(f1)
        print(f"Probe fold {fold}: acc={acc:.3f}, macro-F1={f1:.3f}")

    acc_m, acc_s = float(np.mean(accs)), float(np.std(accs))
    f1_m, f1_s   = float(np.mean(f1s)), float(np.std(f1s))
    print(f"Probe summary: acc={acc_m:.3f}±{acc_s:.3f}, macro-F1={f1_m:.3f}±{f1_s:.3f}")

    # Single aggregated score bar
    plt.figure(figsize=(6, 5))
    plt.bar(["Accuracy", "Macro-F1"], [acc_m, f1_m], yerr=[acc_s, f1_s])
    plt.ylim(0, 1)
    plt.title("Identity probe (GroupKFold by suffix family)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIRS["plots"], "probe_accuracy_by_layer.png"), dpi=150)
    plt.close()


# ----------------------- Interventions -----------------------

def make_zeroing_hooks(delta: torch.Tensor, layers: List[int]):
    hooks = []
    for L in layers:
        Δ = delta[L].detach().to(DEVICE, dtype=DTYPE)
        Δn = Δ / (Δ.norm() + 1e-8)
        def fn(act, hook, Δn=Δn):
            # remove projection onto Δ
            proj = (act * Δn).sum(dim=-1, keepdim=True) * Δn
            return act - proj
        hooks.append((f"blocks.{L}.hook_resid_post", fn))
    return hooks


def make_steer_hooks(delta: torch.Tensor, layers: List[int], alpha: float):
    hooks = []
    for L in layers:
        Δ = delta[L].detach().to(DEVICE, dtype=DTYPE)
        Δn = Δ / (Δ.norm() + 1e-8)
        def fn(act, hook, Δn=Δn, alpha=alpha):
            return act + alpha * Δn
        hooks.append((f"blocks.{L}.hook_resid_post", fn))
    return hooks


@torch.no_grad()
def generate_with_hooks(prompt: str, hooks) -> str:
    toks = to_tokens(prompt)
    cur = toks.clone()
    out_ids = []
    for _ in range(T_GEN):
        # Prefer hooks-only execution to avoid building caches.
        if hooks is None:
            logits = model(cur)
        elif hasattr(model, "run_with_hooks"):
            logits = model.run_with_hooks(cur, fwd_hooks=hooks)
        else:
            with model.hooks(fwd_hooks=hooks):
                logits = model(cur)
        nxt = greedy_next(logits)
        out_ids.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]], device=cur.device)], dim=1)
    full = torch.cat([toks[0], torch.tensor(out_ids, device=toks.device)], dim=0)
    return tokenizer.decode(full, skip_special_tokens=True)


def paired_intervention_samples(deltas: Dict[str, torch.Tensor]):
    rng = np.random.default_rng(SEED)
    # pick a random paraphrase and task for NEAR and FAR
    paras = build_paraphrases()
    task = rng.choice(NEUTRAL_TASKS)
    fam = int(rng.integers(0, N_FAMILIES))
    # Determine steer layers; fall back to mid→last layers if undefined
    steer_layers = globals().get("STEER_LAYERS")
    if not steer_layers:
        steer_layers = list(range(max(0, N_LAYERS // 2), N_LAYERS))

    def one(identity: str, delta_key: str, zero_layers: List[int], steer_alphas: List[float]):
        prime = rng.choice(paras[identity])
        suffix = make_suffix(fam, 0)
        prompt = prime + "\n\n" + task + suffix
        plain = generate_with_hooks(prompt, hooks=None)
        zeroed = generate_with_hooks(prompt, hooks=make_zeroing_hooks(deltas[delta_key], zero_layers))
        steered_dict = {}
        for alpha in steer_alphas:
            steered_out = generate_with_hooks(
                prompt, hooks=make_steer_hooks(deltas[delta_key], steer_layers, alpha)
            )
            steered_dict[alpha] = steered_out
        return prompt, plain, zeroed, steered_dict

    for key in ["near", "far"]:
        prompt, plain, zeroed, steered_dict = one(
            identity=key, delta_key=key, zero_layers=steer_layers, steer_alphas=STEER_ALPHAS
        )
        with open(os.path.join(OUT_DIRS["samples"], f"before_after_zeroing_{key}.txt"), "w", encoding="utf-8") as f:
            f.write("=== PROMPT ===\n" + prompt + "\n\n")
            f.write("--- plain ---\n" + plain + "\n\n")
            f.write("--- zero Δ ---\n" + zeroed + "\n\n")
        for alpha, steered in steered_dict.items():
            with open(os.path.join(OUT_DIRS["samples"], f"before_after_steer_{key}_alpha{alpha}.txt"), "w", encoding="utf-8") as f:
                f.write("=== PROMPT ===\n" + prompt + "\n\n")
                f.write("--- plain ---\n" + plain + "\n\n")
                f.write(f"--- +{alpha}·Δ ---\n" + steered + "\n\n")


# ----------------------- Main flow -----------------------

def main():
    examples = build_examples()
    collected = collect_activations(examples)

    deltas = estimate_deltas(collected)
    plot_delta_magnitudes(deltas)

    # Persistence curves
    for key in ["near", "far"]:
        mean_s, std_s = projection_decay(collected, deltas, which=key)
        plot_decay(mean_s, std_s, title=f"Projection decay onto Δ_{key}", fname=f"projection_decay_{key}.png")
        plot_half_life_per_layer(collected, deltas, which=key)

    # Probe
    run_probe(collected.probe_records)

    # Interventions
    paired_intervention_samples(deltas)

    print("\nDone. Artifacts saved to:")
    for k, v in OUT_DIRS.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

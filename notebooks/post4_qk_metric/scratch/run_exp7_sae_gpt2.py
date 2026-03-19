"""
Post 4 — Experiment 7: G and B vs SAE features in GPT-2 small

Claim: G (content matching) lives in W_E / SAE feature space.
       B (directed routing) top modes live OUTSIDE SAE feature space.

If true: current interpretability tools (SAEs, CLTs, attribution graphs)
can explain OV circuits and the G part of QK (content-based attention),
but are completely blind to the B part — the directed routing layer that
determines who attends to whom.

Test: for each attention layer in GPT-2 small,
  - Extract G^h eigenvectors and B^h top singular vectors
  - Pool across heads (stacked SVD → shared G and B directions)
  - For each direction, compute max cosine similarity with SAE decoder columns
  - Compare distributions: G directions should have high max SAE cosine sim,
    B directions should be near zero.

Model: GPT-2 small (12L x 12H, d_model=768)
SAEs:  jbloom/GPT2-Small-SAEs-Reformatted via sae_lens

Run with: /opt/miniconda3/envs/py311/bin/python run_exp7_sae_gpt2.py
(requires: pip install transformer-lens sae-lens torch numpy matplotlib scipy)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

os.makedirs("figs", exist_ok=True)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from transformer_lens import HookedTransformer
torch.set_grad_enabled(False)

model = HookedTransformer.from_pretrained("gpt2")
print(f"Loaded: {model.cfg.n_layers}L x {model.cfg.n_heads}H, "
      f"d_model={model.cfg.d_model}")

n_layers = model.cfg.n_layers
n_heads  = model.cfg.n_heads
d_model  = model.cfg.d_model

W_Q = model.W_Q.cpu().numpy()   # (12, 12, 768, 64)
W_K = model.W_K.cpu().numpy()

K_G = 8    # top G eigenvectors per layer
K_B = 8    # top B singular vectors per layer (comes in pairs so 4 pairs)


# ── Per-layer stacked G and B directions ────────────────────────────────────
# For each layer: pool across all 12 heads via stacked SVD of eigenvectors.
# This is rotation-invariant across heads within a layer.

def top_G_directions(W_Q_layer, W_K_layer, k):
    """Stack top-1 eigenvectors of sym(W_QK^h) across heads, find shared G dirs."""
    vecs = []
    for h in range(W_Q_layer.shape[0]):
        WQK = W_Q_layer[h] @ W_K_layer[h].T
        G_h = (WQK + WQK.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(G_h)
        idx = np.argsort(np.abs(eigvals))[::-1]
        # take top-k eigenvectors from this head
        vecs.append(eigvecs[:, idx[:k]])
    M = np.hstack(vecs)             # (d_model, n_heads*k)
    U, _, _ = np.linalg.svd(M, full_matrices=False)
    return U[:, :k]                 # (d_model, k) shared G directions

def top_B_directions(W_Q_layer, W_K_layer, k):
    """Stack top singular vectors of anti(W_QK^h) across heads."""
    vecs = []
    for h in range(W_Q_layer.shape[0]):
        WQK = W_Q_layer[h] @ W_K_layer[h].T
        B_h = (WQK - WQK.T) / 2.0
        U_b, _, Vt_b = np.linalg.svd(B_h)
        # take both left and right top-k singular vectors
        vecs.append(U_b[:, :k])
        vecs.append(Vt_b[:k, :].T)
    M = np.hstack(vecs)
    U, _, _ = np.linalg.svd(M, full_matrices=False)
    return U[:, :k]                 # (d_model, k) shared B directions


# ── Load SAEs layer by layer and compute alignment ──────────────────────────

from sae_lens import SAE

results = []   # list of dicts per layer

for layer in range(n_layers):
    print(f"\n--- Layer {layer} ---")

    # Extract shared G and B directions for this layer
    G_dirs = top_G_directions(W_Q[layer], W_K[layer], K_G)  # (d_model, K_G)
    B_dirs = top_B_directions(W_Q[layer], W_K[layer], K_B)  # (d_model, K_B)

    # Load SAE for this layer (hook_resid_post)
    try:
        sae = SAE.from_pretrained(
            release="gpt2-small-res-jb",
            sae_id=f"blocks.{layer}.hook_resid_pre",
        )
        W_dec = sae.W_dec.detach().cpu().numpy()   # (n_features, d_model)
        print(f"  SAE loaded: {W_dec.shape[0]} features")

        # Normalize decoder columns
        W_dec_norm = W_dec / (np.linalg.norm(W_dec, axis=1, keepdims=True) + 1e-10)

        # Max cosine similarity for each G direction with any SAE feature
        G_dirs_norm = G_dirs / (np.linalg.norm(G_dirs, axis=0, keepdims=True) + 1e-10)
        sims_G = W_dec_norm @ G_dirs_norm   # (n_features, K_G)
        max_cos_G = np.abs(sims_G).max(axis=0)   # (K_G,) — max over features

        # Max cosine similarity for each B direction with any SAE feature
        B_dirs_norm = B_dirs / (np.linalg.norm(B_dirs, axis=0, keepdims=True) + 1e-10)
        sims_B = W_dec_norm @ B_dirs_norm   # (n_features, K_B)
        max_cos_B = np.abs(sims_B).max(axis=0)   # (K_B,)

        print(f"  G max cos (mean): {max_cos_G.mean():.3f}  vals: {max_cos_G.round(3)}")
        print(f"  B max cos (mean): {max_cos_B.mean():.3f}  vals: {max_cos_B.round(3)}")

        results.append({
            'layer': layer,
            'max_cos_G': max_cos_G,
            'max_cos_B': max_cos_B,
        })

    except Exception as e:
        print(f"  SAE load failed: {e}")
        results.append({'layer': layer, 'max_cos_G': None, 'max_cos_B': None})


# ── Figures ──────────────────────────────────────────────────────────────────

valid = [r for r in results if r['max_cos_G'] is not None]
layers_done = [r['layer'] for r in valid]
mean_G = [r['max_cos_G'].mean() for r in valid]
mean_B = [r['max_cos_B'].mean() for r in valid]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Fig 1: Mean max cosine similarity per layer
ax = axes[0]
ax.plot(layers_done, mean_G, 'o-', color='#2ca02c', lw=1.5, ms=6, label='G directions (content)')
ax.plot(layers_done, mean_B, 's-', color='#d62728', lw=1.5, ms=6, label='B directions (routing)')
ax.set_xlabel("Layer", fontsize=10)
ax.set_ylabel("Mean max cosine similarity with SAE features", fontsize=9)
ax.set_title(
    "G vs B alignment with SAE features — GPT-2 small\n"
    "G (content matching) should align; B (routing) should not",
    fontsize=9)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
ax.set_xticks(layers_done)

# Fig 2: Distribution across directions and layers (box plot)
ax = axes[1]
all_G = np.concatenate([r['max_cos_G'] for r in valid])
all_B = np.concatenate([r['max_cos_B'] for r in valid])
ax.boxplot([all_G, all_B], tick_labels=['G directions\n(content)', 'B directions\n(routing)'],
           patch_artist=True,
           boxprops=dict(facecolor='lightgreen', alpha=0.7),
           medianprops=dict(color='black'))
ax.set_ylabel("Max cosine similarity with any SAE feature", fontsize=9)
ax.set_title(
    "Distribution of max SAE alignment\nacross all layers and directions",
    fontsize=9)
ax.grid(True, alpha=0.2, axis='y')

plt.suptitle(
    "SAE feature alignment: G (content matching) vs B (directed routing)\n"
    "Hypothesis: B top modes are invisible to SAE features — the missing routing layer",
    fontsize=10)
plt.tight_layout()
plt.savefig("figs/exp7_sae_alignment.png", dpi=300, bbox_inches="tight")
print("\nSaved figs/exp7_sae_alignment.png")

# Save numerical results
print("\n=== Summary: mean max cosine similarity with SAE features ===")
print(f"{'Layer':>6s}  {'G (content)':>12s}  {'B (routing)':>12s}")
for r in valid:
    print(f"  L{r['layer']:2d}:   {r['max_cos_G'].mean():>10.3f}   {r['max_cos_B'].mean():>10.3f}")
if valid:
    print(f"\n  All:   {np.mean(mean_G):>10.3f}   {np.mean(mean_B):>10.3f}")

print("\nDone.")

"""
Post 4 — QK Metric, Experiment 1: G_crude eigenspectrum + vocabulary projection
Run with: /opt/miniconda3/bin/python run_exp1_G_spectrum.py

Hypothesis:
  G_crude = mean_h( (W_QK^h + W_QK^h^T) / 2 )  is a low-rank / sloppy matrix.
  Its top eigenvectors correspond to interpretable token categories
  (recoverable by projecting onto the vocabulary via W_E).

This is purely parameter-based — no inference, no prompts.
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

model = HookedTransformer.from_pretrained("gpt2-small")
print(f"Loaded: {model.cfg.n_layers}L x {model.cfg.n_heads}H, "
      f"d_model={model.cfg.d_model}, d_head={model.cfg.d_head}")

n_layers  = model.cfg.n_layers   # 12
n_heads   = model.cfg.n_heads    # 12
d_model   = model.cfg.d_model    # 768

# ── Extract W_Q, W_K for all heads ───────────────────────────────────────────
# model.W_Q shape: (n_layers, n_heads, d_model, d_head)
W_Q = model.W_Q.cpu().numpy()   # (12, 12, 768, 64)
W_K = model.W_K.cpu().numpy()

# ── Compute W_QK^h = W_Q^h @ W_K^h^T  (768 x 768) for each head ─────────────
# Attention score = x W_Q^h (W_K^h)^T x^T = x (W_QK^h) x^T
# W_QK^h shape: (d_model, d_model)

W_QK_sym_all = []   # will collect (n_layers*n_heads) matrices

for l in range(n_layers):
    for h in range(n_heads):
        Wq = W_Q[l, h]           # (768, 64)
        Wk = W_K[l, h]           # (768, 64)
        W_QK = Wq @ Wk.T        # (768, 768)
        W_QK_sym = (W_QK + W_QK.T) / 2.0
        W_QK_sym_all.append(W_QK_sym)

W_QK_sym_all = np.array(W_QK_sym_all)  # (144, 768, 768)
print(f"W_QK_sym_all shape: {W_QK_sym_all.shape}")

# ── G_crude = mean across all heads ──────────────────────────────────────────
G_crude = W_QK_sym_all.mean(axis=0)   # (768, 768)
print(f"G_crude shape: {G_crude.shape}")
print(f"G_crude Frobenius norm: {np.linalg.norm(G_crude):.3f}")

# ── Eigendecompose G_crude ────────────────────────────────────────────────────
# eigh assumes symmetric (G_crude is symmetric by construction)
eigvals, eigvecs = np.linalg.eigh(G_crude)
# eigh returns ascending order — reverse to descending
eigvals = eigvals[::-1].copy()
eigvecs = eigvecs[:, ::-1].copy()

print(f"\nTop 10 eigenvalues: {eigvals[:10].round(4)}")
print(f"Bottom 10 eigenvalues: {eigvals[-10:].round(4)}")
print(f"Number of positive eigenvalues: {(eigvals > 0).sum()}")
print(f"Number of negative eigenvalues: {(eigvals < 0).sum()}")

# ── Vocabulary projection ─────────────────────────────────────────────────────
# W_E shape: (n_vocab, d_model)
W_E = model.W_E.cpu().numpy()   # (50257, 768)
print(f"\nW_E shape: {W_E.shape}")

N_EV = 8   # how many eigenvectors to inspect

# For each eigenvector v: scores = W_E @ v  (n_vocab,)
# Top tokens by |score| — which tokens "point along" this direction
top_tokens_per_ev = []
for k in range(N_EV):
    v = eigvecs[:, k]
    scores = W_E @ v
    top_pos_idx = np.argsort(scores)[-15:][::-1]
    top_neg_idx = np.argsort(scores)[:15]
    top_tokens_per_ev.append({
        "pos": [(model.to_string([i]), float(scores[i])) for i in top_pos_idx],
        "neg": [(model.to_string([i]), float(scores[i])) for i in top_neg_idx],
        "eigval": eigvals[k],
    })

print("\n── Vocabulary projections (top tokens per eigenvector) ──")
for k, d in enumerate(top_tokens_per_ev):
    print(f"\nEV{k+1}  (λ={d['eigval']:.4f})")
    print(f"  + : {[t for t, _ in d['pos'][:10]]}")
    print(f"  - : {[t for t, _ in d['neg'][:10]]}")

# ── Also compute per-layer mean to see if structure differs by layer ──────────
G_by_layer = W_QK_sym_all.reshape(n_layers, n_heads, d_model, d_model).mean(axis=1)
# G_by_layer: (12, 768, 768)

# ── FIGURE 1: Eigenvalue spectrum ────────────────────────────────────────────
fig1, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.plot(range(1, d_model + 1), eigvals, 'k-', lw=0.8, alpha=0.7)
ax.axhline(0, color='gray', lw=0.5, linestyle='--')
ax.set_xlabel("Eigenvalue rank", fontsize=11)
ax.set_ylabel("Eigenvalue", fontsize=11)
ax.set_title("G_crude eigenspectrum (linear)", fontsize=11)
ax.set_xlim(0, d_model + 1)
ax.grid(True, alpha=0.2)

ax = axes[1]
pos_mask = eigvals > 0
neg_mask = eigvals < 0
ranks = np.arange(1, d_model + 1)
if pos_mask.any():
    ax.semilogy(ranks[pos_mask], eigvals[pos_mask], 'b.', ms=2, label='positive')
if neg_mask.any():
    ax.semilogy(ranks[neg_mask], np.abs(eigvals[neg_mask]), 'r.', ms=2, label='|negative|')
ax.set_xlabel("Eigenvalue rank", fontsize=11)
ax.set_ylabel("|Eigenvalue| (log scale)", fontsize=11)
ax.set_title("G_crude eigenspectrum (log scale)", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

plt.suptitle(
    "G_crude = mean symmetric part of W_QK across all 144 GPT-2 small heads",
    fontsize=12
)
plt.tight_layout()
plt.savefig("figs/exp1_G_eigenspectrum.png", dpi=300, bbox_inches="tight")
print("\nSaved figs/exp1_G_eigenspectrum.png")

# ── FIGURE 2: Vocabulary projections for top N_EV eigenvectors ───────────────
fig2, axes2 = plt.subplots(2, N_EV // 2, figsize=(16, 7))
axes2 = axes2.flatten()

for k in range(N_EV):
    ax = axes2[k]
    v = eigvecs[:, k]
    scores = W_E @ v
    # show top 12 positive + top 12 negative tokens as a horizontal bar chart
    top_pos_idx = np.argsort(scores)[-12:]
    top_neg_idx = np.argsort(scores)[:12]
    idx = np.concatenate([top_neg_idx, top_pos_idx])
    vals = scores[idx]
    labels = [model.to_string([i]).strip() for i in idx]
    colors = ['#d62728' if v > 0 else '#1f77b4' for v in vals]

    ax.barh(range(len(idx)), vals, color=colors, height=0.8)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_title(f"EV{k+1}  λ={eigvals[k]:.3f}", fontsize=9)
    ax.tick_params(axis='x', labelsize=7)

plt.suptitle(
    "Top tokens by projection onto G_crude eigenvectors  (W_E @ v)",
    fontsize=12
)
plt.tight_layout()
plt.savefig("figs/exp1_vocab_projections.png", dpi=300, bbox_inches="tight")
print("Saved figs/exp1_vocab_projections.png")

# ── FIGURE 3: Per-layer eigenvalue comparison (how much does G vary by layer?) -
# Compare top-5 eigenvalues across layers
fig3, ax3 = plt.subplots(figsize=(10, 4))
for k in range(5):
    layer_eigvals = []
    for l in range(n_layers):
        ev, _ = np.linalg.eigh(G_by_layer[l])
        layer_eigvals.append(sorted(ev)[-1 - k])   # k-th largest
    ax3.plot(range(n_layers), layer_eigvals, marker='o', ms=4, label=f"EV{k+1}")

ax3.set_xlabel("Layer", fontsize=11)
ax3.set_ylabel("Eigenvalue", fontsize=11)
ax3.set_title("Top-5 eigenvalues of per-layer G  (mean over heads within layer)", fontsize=11)
ax3.legend(fontsize=9)
ax3.set_xticks(range(n_layers))
ax3.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("figs/exp1_perlayer_eigvals.png", dpi=300, bbox_inches="tight")
print("Saved figs/exp1_perlayer_eigvals.png")

# ── Save arrays for downstream analysis ──────────────────────────────────────
np.save("G_crude.npy", G_crude)
np.save("G_crude_eigvals.npy", eigvals)
np.save("G_crude_eigvecs.npy", eigvecs)
np.save("W_QK_sym_all.npy", W_QK_sym_all)
print("\nSaved G_crude.npy, G_crude_eigvals.npy, G_crude_eigvecs.npy, W_QK_sym_all.npy")

print("\nDone.")

"""
Post 4 — Experiment 9: G-corrected K-composition metric

Elhage's K-composition measure (and most circuit analysis) uses:
    ||W_K^B^T W_O^A^T||_F / (||W_O^A||_F * ||W_K^B||_F)

This implicitly uses the identity as the metric on d_model space — it treats
all residual stream directions as equally important.

The G-corrected version inserts the shared content geometry G:
    ||W_K^B^T G W_O^A^T||_F / sqrt(tr(W_O^A G W_O^A^T) * tr(W_K^B^T G W_K^B))

If G != identity (which it isn't — G has a sloppy spectrum spanning ~10x in magnitude),
this changes the alignment scores. Some head pairs that look connected under identity
may not be under G, and vice versa.

G is taken from the induction head (L1H5) eigenvectors — the reference basis from exp6.

Model: attn-only-2l (2L x 8H, d_model=512, d_head=64)
Run with: /opt/miniconda3/bin/python run_exp9_G_corrected_kcomp.py
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

model = HookedTransformer.from_pretrained("attn-only-2l")
n_layers = model.cfg.n_layers
n_heads  = model.cfg.n_heads

W_Q = model.W_Q.cpu().numpy()
W_K = model.W_K.cpu().numpy()
W_O = model.W_O.cpu().numpy()   # (2, 8, 64, 512)


# ── Build G from induction head (same as exp6) ────────────────────────────────

G_heads = {}
B_norms_L1 = {}
for l in range(n_layers):
    for h in range(n_heads):
        WQK = W_Q[l, h] @ W_K[l, h].T
        G_heads[(l, h)] = (WQK + WQK.T) / 2.0

# Reference head identified by activation patterns on repeated-token sequence (exp10):
#   Induction head: L1H6 (induction score=0.604)
h_induction = 6
print(f"Induction head (activation-based, exp10): L1H{h_induction}")

# G^{induction} is indefinite (has negative eigenvalues).
# Use |G| = V |D| V^T (absolute eigenvalues) as the metric — positive semidefinite.
# Both + and - G directions represent content similarity/dissimilarity; both are meaningful.
eigvals_G, eigvecs_G = np.linalg.eigh(G_heads[(1, h_induction)])
G_ind = eigvecs_G @ np.diag(np.abs(eigvals_G)) @ eigvecs_G.T   # |G|


# ── K-composition: standard vs G-corrected ────────────────────────────────────

def kcomp_standard(WO, WK):
    """||W_K^T W_O^T||_F / (||W_O||_F * ||W_K||_F)"""
    numer = np.linalg.norm(WK.T @ WO.T, 'fro')
    denom = np.linalg.norm(WO, 'fro') * np.linalg.norm(WK, 'fro') + 1e-10
    return numer / denom

def kcomp_G(WO, WK, G):
    """||W_K^T G W_O^T||_F / sqrt(tr(W_O G W_O^T) * tr(W_K^T G W_K))"""
    numer = np.linalg.norm(WK.T @ G @ WO.T, 'fro')
    # G-weighted norm of W_O: sqrt(tr(W_O G W_O^T))
    # W_O is (d_head, d_model); W_O G W_O^T is (d_head, d_head)
    norm_O = np.sqrt(np.trace(WO @ G @ WO.T) + 1e-10)
    # G-weighted norm of W_K: sqrt(tr(W_K^T G W_K))
    # W_K is (d_model, d_head); W_K^T G W_K is (d_head, d_head)
    norm_K = np.sqrt(np.trace(WK.T @ G @ WK) + 1e-10)
    return numer / (norm_O * norm_K + 1e-10)


raw_std = np.zeros((n_heads, n_heads))
raw_G   = np.zeros((n_heads, n_heads))

for h0 in range(n_heads):
    WO = W_O[0, h0]        # (64, 512) — L0 writer
    for h1 in range(n_heads):
        WK = W_K[1, h1]    # (512, 64) — L1 reader
        raw_std[h0, h1] = kcomp_standard(WO, WK)
        raw_G[h0, h1]   = kcomp_G(WO, WK, G_ind)


# ── Print and compare ─────────────────────────────────────────────────────────

print(f"\nStandard K-comp (L0h → L1h'):")
print(f"{'':8s}", end='')
for h1 in range(n_heads): print(f"  L1H{h1}", end='')
print()
for h0 in range(n_heads):
    print(f"  L0H{h0}:  ", end='')
    for h1 in range(n_heads): print(f"  {raw_std[h0,h1]:.3f}", end='')
    print()

print(f"\nG-corrected K-comp (metric = G^{{induction}}):")
print(f"{'':8s}", end='')
for h1 in range(n_heads): print(f"  L1H{h1}", end='')
print()
for h0 in range(n_heads):
    print(f"  L0H{h0}:  ", end='')
    for h1 in range(n_heads): print(f"  {raw_G[h0,h1]:.3f}", end='')
    print()

# Difference: which pairs change most?
diff = raw_G - raw_std
print(f"\nDifference (G-corrected minus standard) — largest changes:")
flat_idx = np.argsort(np.abs(diff).ravel())[::-1][:10]
for idx in flat_idx:
    h0, h1 = divmod(idx, n_heads)
    print(f"  L0H{h0} → L1H{h1}:  std={raw_std[h0,h1]:.3f}  G={raw_G[h0,h1]:.3f}  "
          f"diff={diff[h0,h1]:+.3f}")

corr = np.corrcoef(raw_std.ravel(), raw_G.ravel())[0, 1]
print(f"\nCorrelation between standard and G-corrected: {corr:.4f}")
print(f"(1.0 = identical ranking; < 1 = G changes which pairs matter)")


# ── Figures ───────────────────────────────────────────────────────────────────

vmax = max(raw_std.max(), raw_G.max())

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

titles = [
    "Standard K-comp\n||W_K^T W_O^T||_F / (||W_O|| ||W_K||)",
    f"G-corrected K-comp\n||W_K^T G W_O^T||_F / sqrt(tr(W_O G W_O^T) tr(W_K^T G W_K))",
    f"Difference (G - standard)\ncorr={corr:.3f}",
]
data   = [raw_std, raw_G, diff]
cmaps  = ['Blues', 'Oranges', 'RdBu_r']
vmaxes = [vmax, vmax, np.abs(diff).max()]

for ax, d, title, cmap, vm in zip(axes, data, titles, cmaps, vmaxes):
    if cmap == 'RdBu_r':
        im = ax.imshow(d, cmap=cmap, vmin=-vm, vmax=vm)
    else:
        im = ax.imshow(d, cmap=cmap, vmin=0, vmax=vm)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f'L1H{h}' for h in range(n_heads)], fontsize=7, rotation=45)
    ax.set_yticks(range(n_heads))
    ax.set_yticklabels([f'L0H{h}' for h in range(n_heads)], fontsize=7)
    ax.set_xlabel("L1 head (W_K reader)", fontsize=9)
    ax.set_ylabel("L0 head (W_O writer)", fontsize=9)
    ax.set_title(title, fontsize=8)

plt.suptitle(
    f"G-corrected K-composition (metric = G^{{L1H{h_induction}}} induction head)\n"
    f"Standard metric = identity on d_model. G-corrected weights by shared content geometry.",
    fontsize=10)
plt.tight_layout()
plt.savefig("figs/exp9_G_corrected_kcomp.png", dpi=300, bbox_inches="tight")
print("\nSaved figs/exp9_G_corrected_kcomp.png")

# Scatter: standard vs G-corrected per pair
fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.scatter(raw_std.ravel(), raw_G.ravel(), alpha=0.6, s=30, color='#1f77b4')
lim = max(raw_std.max(), raw_G.max()) * 1.05
ax2.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.4, label='y=x (no change)')
# Label notable pairs
for h0 in range(n_heads):
    for h1 in range(n_heads):
        if abs(diff[h0, h1]) > 0.02:
            ax2.annotate(f"L0H{h0}→L1H{h1}",
                         (raw_std[h0, h1], raw_G[h0, h1]),
                         fontsize=6, xytext=(4, 2), textcoords='offset points')
ax2.set_xlabel("Standard K-comp (identity metric)", fontsize=10)
ax2.set_ylabel("G-corrected K-comp (content geometry)", fontsize=10)
ax2.set_title(f"Standard vs G-corrected K-composition\ncorr={corr:.4f}", fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("figs/exp9_scatter.png", dpi=300, bbox_inches="tight")
print("Saved figs/exp9_scatter.png")

print("\nDone.")

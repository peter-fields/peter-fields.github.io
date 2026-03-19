"""
Post 4 — Experiment 12: BOS-cleaned, sloppified G basis

Two refinements to the G basis from exp6:

1. BOS REMOVAL: project out the BOS embedding direction from each G basis vector.
   BOS is a large-norm "sink" token that many heads attend to — it inflates naive
   Frobenius metrics. Removing it from the basis is principled: BOS doesn't participate
   in content matching. We can say this explicitly in the blog.

2. SLOPPIFICATION: find a rotation of the G basis (within the induction head's eigenspace)
   that makes the coefficient sequences {a^h_i} across heads maximally ordered by
   collective importance. Method: compute G_crude restricted to the induction subspace
   (V_G^T G_crude V_G), diagonalize it. The eigenvectors are directions ordered by
   mean G coefficient across all heads — the sloppiest natural ordering.

   Sloppiness of the resulting spectrum tells us whether the shared G geometry itself
   is sloppy, vs sloppiness being a head-specific artifact.

Model: attn-only-2l (2L x 8H, d_model=512, d_head=64)
Reference heads: L1H6 (induction), L0H3 (prev-token) — from exp10.
Run with: /opt/miniconda3/bin/python run_exp12_sloppified_basis.py
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
d_model  = model.cfg.d_model

W_Q = model.W_Q.cpu().numpy()
W_K = model.W_K.cpu().numpy()
W_V = model.W_V.cpu().numpy()
W_O = model.W_O.cpu().numpy()
W_E = model.W_E.cpu().numpy()   # (vocab, d_model)

h_induction = 6   # L1H6 — from exp10 activation-based identification
h_prev      = 3   # L0H3

K_G = 16

# ── G and B for all heads ─────────────────────────────────────────────────────

G_heads = {}
B_heads = {}
for l in range(n_layers):
    for h in range(n_heads):
        WQK = W_Q[l, h] @ W_K[l, h].T
        G_heads[(l, h)] = (WQK + WQK.T) / 2.0
        B_heads[(l, h)] = (WQK - WQK.T) / 2.0


# ── Step 1: Raw G basis from induction head ───────────────────────────────────

G_ind = G_heads[(1, h_induction)]
eigvals_raw, eigvecs_raw = np.linalg.eigh(G_ind)
idx = np.argsort(np.abs(eigvals_raw))[::-1]
eigvals_raw = eigvals_raw[idx]
eigvecs_raw = eigvecs_raw[:, idx]
V_raw = eigvecs_raw[:, :K_G].copy()   # (d_model, K_G)

print(f"Raw G basis eigenvalues (top-{K_G}):")
print(f"  {eigvals_raw[:K_G].round(3)}")


# ── Step 2: BOS removal ───────────────────────────────────────────────────────

bos_id  = model.tokenizer.bos_token_id
v_bos   = W_E[bos_id] / (np.linalg.norm(W_E[bos_id]) + 1e-10)

# Project BOS out of each basis vector
V_debos = V_raw - np.outer(v_bos, v_bos @ V_raw)   # (d_model, K_G)

# Re-orthogonalize via QR (BOS removal can break orthonormality)
V_debos, _ = np.linalg.qr(V_debos)
V_debos    = V_debos[:, :K_G]   # QR may give more cols than K_G

bos_mass_before = (V_raw.T @ v_bos) ** 2
bos_mass_after  = (V_debos.T @ v_bos) ** 2
print(f"\nBOS projection mass per basis vector:")
print(f"  Before removal: {bos_mass_before.round(3)}")
print(f"  After  removal: {bos_mass_after.round(3)}")


# ── Step 3: Sloppification — diagonalize G_crude in the de-BOSed subspace ────

# G_crude = mean of G^h across all heads
G_crude = np.mean([G_heads[(l, h)]
                   for l in range(n_layers)
                   for h in range(n_heads)], axis=0)

# Restrict G_crude to the de-BOSed induction subspace
M_crude = V_debos.T @ G_crude @ V_debos   # (K_G, K_G) — G_crude in subspace

# Diagonalize: eigenvectors order G directions by mean coefficient across heads
eigvals_slop, R_slop = np.linalg.eigh(M_crude)
idx_slop = np.argsort(np.abs(eigvals_slop))[::-1]
eigvals_slop = eigvals_slop[idx_slop]
R_slop       = R_slop[:, idx_slop]          # (K_G, K_G) rotation within subspace

# Final sloppified basis
V_slop = V_debos @ R_slop                   # (d_model, K_G)

print(f"\nSloppified G basis (G_crude eigenvalues in induction subspace):")
print(f"  {eigvals_slop.round(3)}")
print(f"  Log10 dynamic range: "
      f"{np.log10(np.abs(eigvals_slop[0]) / (np.abs(eigvals_slop[-1]) + 1e-10)):.2f}")


# ── Step 4: Coefficient matrix in all three bases ────────────────────────────

def coeff_matrix(V, G_heads):
    """Compute A[h, i] = v_i^T G^h v_i for all heads and basis directions."""
    heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    A = np.zeros((len(heads), V.shape[1]))
    for idx_h, (l, h) in enumerate(heads):
        for i in range(V.shape[1]):
            v = V[:, i]
            A[idx_h, i] = v @ G_heads[(l, h)] @ v
    return A

A_raw  = coeff_matrix(V_raw,   G_heads)
A_slop = coeff_matrix(V_slop,  G_heads)

head_labels = [f"L{l}H{h}" for l in range(n_layers) for h in range(n_heads)]

print(f"\nCoefficient matrix A (sloppified basis) — mean |a^h_i| per direction:")
print(f"  {np.abs(A_slop).mean(axis=0).round(3)}")
print(f"  (sloppy = log-spaced, uniform = flat)")

# Sloppiness of mean coefficient profile
mean_coeffs = np.abs(A_slop).mean(axis=0)
log_range   = np.log10(mean_coeffs[0] / (mean_coeffs[-1] + 1e-10))
print(f"  Log10 dynamic range of mean coefficients: {log_range:.2f}")

# Per-head sloppiness of coefficient profile in sloppified basis
print(f"\nPer-head sloppiness of G coefficients (log10 range) in sloppified basis:")
for idx_h, (l, h) in enumerate([(l, h) for l in range(n_layers) for h in range(n_heads)]):
    a = np.abs(A_slop[idx_h])
    sloppy = np.log10(a[0] / (a[-1] + 1e-10))
    tag = ""
    if l == 1 and h == h_induction: tag = " ← induction"
    if l == 0 and h == h_prev:      tag = " ← prev-token"
    print(f"  L{l}H{h}: {sloppy:.3f}{tag}")


# ── Figures ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

def coeff_heatmap(ax, A, title, ylabel="Head", cmap='RdBu_r'):
    vmax = np.abs(A).max()
    im = ax.imshow(A.T, aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_xticks(range(len(head_labels)))
    ax.set_xticklabels(head_labels, rotation=45, ha='right', fontsize=6)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=8)

coeff_heatmap(axes[0], A_raw,
              f"Raw G basis (L1H{h_induction} eigenvectors)\n"
              f"a^h_i = v_i^T G^h v_i",
              "G direction i (ranked by |λ|)")

coeff_heatmap(axes[1], A_slop,
              f"Sloppified basis (BOS-cleaned + G_crude diagonalized)\n"
              f"directions ordered by mean coefficient across heads",
              "Sloppified G direction i")

# Spectrum comparison
ax = axes[2]
ax.semilogy(np.arange(1, K_G+1), np.abs(eigvals_raw[:K_G]),
            'o-', lw=1.5, ms=5, label='Raw (induction head λ)', color='#1f77b4')
ax.semilogy(np.arange(1, K_G+1), np.abs(eigvals_slop),
            's-', lw=1.5, ms=5, label='Sloppified (G_crude in subspace λ)', color='#d62728')
ax.semilogy(np.arange(1, K_G+1), np.abs(A_slop).mean(axis=0),
            '^--', lw=1.2, ms=4, label='Mean |coeff| across heads', color='#2ca02c')
ax.set_xlabel("Direction rank", fontsize=9)
ax.set_ylabel("|value| (log scale)", fontsize=9)
ax.set_title("Spectral comparison:\nraw vs sloppified basis vs mean head coefficient",
             fontsize=8)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

plt.suptitle(
    "Experiment 12: BOS-cleaned, sloppified G basis\n"
    "Sloppification = diagonalize G_crude within induction subspace → "
    "directions ordered by collective importance across all heads",
    fontsize=9)
plt.tight_layout()
plt.savefig("figs/exp12_sloppified_basis.png", dpi=300, bbox_inches="tight")
print("\nSaved figs/exp12_sloppified_basis.png")
print("\nDone.")

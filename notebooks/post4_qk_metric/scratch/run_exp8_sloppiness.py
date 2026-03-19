"""
Post 4 — Experiment 8: Sloppiness of G and B spectra across attn-only-2l

"Sloppy" = eigenvalues log-spaced over many decades (from statistical physics).
Sloppy G → head encodes multi-scale content geometry.
Non-sloppy B → head does a single clean routing job.

Metrics per head:
  - G sloppiness: log10(|λ_1| / |λ_k|) — dynamic range of top-k eigenvalues
  - B sloppiness: log10(σ_1 / σ_k) — dynamic range of top-k singular values
  - G participation ratio: (sum |λ_i|)^2 / sum(λ_i^2) — effective number of active modes
  - B participation ratio: same for singular values

Hypothesis:
  - Content-matching heads (large G_total from exp6): sloppy G, uniform B
  - Directed-routing heads (large B_total from exp6): sloppy B, flat G
  - Clean prev-token head (L0H5): non-sloppy B (one dominant routing mode)

Run with: /opt/miniconda3/bin/python run_exp8_sloppiness.py
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

K = 16   # eigenvalues/singular values to examine

# ── Compute spectra ───────────────────────────────────────────────────────────

spectra = {}   # (l, h) -> {'G_eigs': ..., 'B_svs': ..., ...}

for l in range(n_layers):
    for h in range(n_heads):
        WQK = W_Q[l, h] @ W_K[l, h].T
        G_h = (WQK + WQK.T) / 2.0
        B_h = (WQK - WQK.T) / 2.0

        eigvals = np.linalg.eigvalsh(G_h)
        eigvals_sorted = np.sort(np.abs(eigvals))[::-1][:K]

        sv_B = np.linalg.svd(B_h, compute_uv=False)[:K]

        # Sloppiness: log10 dynamic range of top-K values
        G_slop = np.log10(eigvals_sorted[0] / (eigvals_sorted[K-1] + 1e-10))
        B_slop = np.log10(sv_B[0] / (sv_B[K-1] + 1e-10))

        # Participation ratio: (sum x_i)^2 / sum(x_i^2) — effective # active modes
        G_pr = (eigvals_sorted.sum())**2 / ((eigvals_sorted**2).sum() + 1e-10)
        B_pr = (sv_B.sum())**2 / ((sv_B**2).sum() + 1e-10)

        # Fraction of total variance in top-1 mode
        G_top1_frac = eigvals_sorted[0] / (eigvals_sorted.sum() + 1e-10)
        B_top1_frac = sv_B[0] / (sv_B.sum() + 1e-10)

        spectra[(l, h)] = {
            'G_eigs': eigvals_sorted,
            'B_svs':  sv_B,
            'G_slop': G_slop,
            'B_slop': B_slop,
            'G_pr':   G_pr,
            'B_pr':   B_pr,
            'G_top1': G_top1_frac,
            'B_top1': B_top1_frac,
        }


# ── Print summary ─────────────────────────────────────────────────────────────

print(f"{'Head':8s}  {'G_slop':>8s}  {'B_slop':>8s}  {'G_PR':>6s}  {'B_PR':>6s}  "
      f"{'G_top1%':>8s}  {'B_top1%':>8s}")
for l in range(n_layers):
    for h in range(n_heads):
        s = spectra[(l, h)]
        print(f"  L{l}H{h}:   "
              f"  {s['G_slop']:7.3f}   {s['B_slop']:7.3f}"
              f"  {s['G_pr']:5.1f}  {s['B_pr']:5.1f}"
              f"   {100*s['G_top1']:6.1f}%   {100*s['B_top1']:6.1f}%")


# ── Fig 1: All G eigenvalue spectra on log scale ──────────────────────────────

colors_L0 = plt.cm.Blues(np.linspace(0.4, 0.9, n_heads))
colors_L1 = plt.cm.Reds(np.linspace(0.4, 0.9, n_heads))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for l in range(n_layers):
    cols = colors_L0 if l == 0 else colors_L1
    for h in range(n_heads):
        s = spectra[(l, h)]
        label = f"L{l}H{h}" if (l == 0 and h == 5) or (l == 1 and h == 5) else None
        lw = 2.0 if (l == 0 and h == 5) or (l == 1 and h == 5) else 0.8
        ax.semilogy(np.arange(1, K+1), s['G_eigs'], color=cols[h],
                    lw=lw, alpha=0.85, label=label)

ax.set_xlabel("Eigenvalue rank", fontsize=9)
ax.set_ylabel("|eigenvalue|  (log scale)", fontsize=9)
ax.set_title("G^h eigenvalue spectra — all heads\nBlue=L0, Red=L1  (bold=ref heads)",
             fontsize=9)
ax.grid(True, alpha=0.2)
ax.legend(fontsize=8)

ax = axes[1]
for l in range(n_layers):
    cols = colors_L0 if l == 0 else colors_L1
    for h in range(n_heads):
        s = spectra[(l, h)]
        label = f"L{l}H{h}" if (l == 0 and h == 5) or (l == 1 and h == 5) else None
        lw = 2.0 if (l == 0 and h == 5) or (l == 1 and h == 5) else 0.8
        ax.semilogy(np.arange(1, K+1), s['B_svs'], color=cols[h],
                    lw=lw, alpha=0.85, label=label)

ax.set_xlabel("Singular value rank", fontsize=9)
ax.set_ylabel("singular value  (log scale)", fontsize=9)
ax.set_title("B^h singular value spectra — all heads\nBlue=L0, Red=L1  (bold=ref heads)",
             fontsize=9)
ax.grid(True, alpha=0.2)
ax.legend(fontsize=8)

plt.suptitle("Sloppy structure: G eigenvalues and B singular values across all heads\n"
             "Sloppy = log-spaced (steep slope); uniform = flat", fontsize=10)
plt.tight_layout()
plt.savefig("figs/exp8_spectra.png", dpi=300, bbox_inches="tight")
print("\nSaved figs/exp8_spectra.png")


# ── Fig 2: Sloppiness scatter — G_slop vs B_slop per head ────────────────────

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))

ax = axes2[0]
for l in range(n_layers):
    for h in range(n_heads):
        s = spectra[(l, h)]
        color = '#1f77b4' if l == 0 else '#d62728'
        ax.scatter(s['G_slop'], s['B_slop'], color=color, s=60, zorder=3)
        ax.annotate(f"L{l}H{h}", (s['G_slop'], s['B_slop']),
                    fontsize=7, xytext=(3, 2), textcoords='offset points')

ax.set_xlabel("G sloppiness: log10(|λ_1|/|λ_K|)", fontsize=9)
ax.set_ylabel("B sloppiness: log10(σ_1/σ_K)", fontsize=9)
ax.set_title("G vs B sloppiness per head\nBlue=L0, Red=L1", fontsize=9)
ax.grid(True, alpha=0.2)

ax = axes2[1]
for l in range(n_layers):
    for h in range(n_heads):
        s = spectra[(l, h)]
        color = '#1f77b4' if l == 0 else '#d62728'
        ax.scatter(s['G_pr'], s['B_pr'], color=color, s=60, zorder=3)
        ax.annotate(f"L{l}H{h}", (s['G_pr'], s['B_pr']),
                    fontsize=7, xytext=(3, 2), textcoords='offset points')

ax.set_xlabel("G participation ratio (effective # G modes)", fontsize=9)
ax.set_ylabel("B participation ratio (effective # B modes)", fontsize=9)
ax.set_title("G vs B participation ratio per head\nHigher = more spread across modes (less sloppy)",
             fontsize=9)
ax.grid(True, alpha=0.2)

plt.suptitle("Sloppiness metrics: do routing heads have non-sloppy B? Do content heads have sloppy G?",
             fontsize=10)
plt.tight_layout()
plt.savefig("figs/exp8_sloppiness_scatter.png", dpi=300, bbox_inches="tight")
print("Saved figs/exp8_sloppiness_scatter.png")

print("\nDone.")

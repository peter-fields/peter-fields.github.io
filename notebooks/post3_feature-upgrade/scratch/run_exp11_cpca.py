"""
Exp 11: Contrastive PCA via generalized eigenvalue decomposition.

Find directions v that maximize the ratio of IOI variance to baseline variance:

    λ = (v^T C_ioi v) / (v^T C_non v)

This is the generalized eigenvalue problem:   C_ioi v = λ C_non v

Eigenvectors with large λ (>> 1): amplified by IOI — circuit-specific structure.
Eigenvectors with small λ (<< 1): suppressed by IOI.
Eigenvectors with λ ≈ 1: same variance in both conditions — structural noise.

Algorithm: scipy.linalg.eigh(C_ioi, C_non_reg) where C_non_reg is
Ledoit-Wolf regularized to ensure positive definiteness.

Figures:
  exp11_eigenvalues.png   — eigenvalue spectrum (λ_k) for leading/trailing modes
  exp11_cpca_{k}.png      — loading profile for top contrastive eigenvectors

Run with: /opt/miniconda3/bin/python run_exp11_cpca.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import linalg
from sklearn.covariance import LedoitWolf

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figs", exist_ok=True)

# ── metadata ──────────────────────────────────────────────────────────────────
IOI_HEADS = {
    "Name Mover":          [(9, 6), (9, 9), (10, 0)],
    "Backup Name Mover":   [(9, 0), (9, 7), (10, 1), (10, 2), (10, 6), (10, 10), (11, 2), (11, 9)],
    "Negative Name Mover": [(10, 7), (11, 10)],
    "S-Inhibition":        [(7, 3), (7, 9), (8, 6), (8, 10)],
    "Induction":           [(5, 5), (6, 9)],
    "Duplicate Token":     [(0, 1), (3, 0)],
    "Previous Token":      [(2, 2), (4, 11)],
}
ROLE_ORDER = ["Name Mover", "Backup Name Mover", "Negative Name Mover",
              "S-Inhibition", "Induction", "Duplicate Token", "Previous Token", "Non-circuit"]
ROLE_COLORS = {
    "Name Mover":          "#d62728",
    "Backup Name Mover":   "#ff7f0e",
    "Negative Name Mover": "#9467bd",
    "S-Inhibition":        "#2ca02c",
    "Induction":           "#17becf",
    "Duplicate Token":     "#bcbd22",
    "Previous Token":      "#e377c2",
    "Non-circuit":         "#cccccc",
}
SHORT = {"Name Mover": "NM", "Backup Name Mover": "BNM", "Negative Name Mover": "NegNM",
         "S-Inhibition": "SI", "Induction": "Ind", "Duplicate Token": "DT",
         "Previous Token": "PT", "Non-circuit": "NC"}

ALL_CIRCUIT_HEADS = set(lh for hs in IOI_HEADS.values() for lh in hs)
HL_FLAT = [(l, h) for l in range(12) for h in range(12)]

def head_role(l, h):
    for role, heads in IOI_HEADS.items():
        if (l, h) in heads: return role
    return "Non-circuit"

HEAD_NAMES   = [f"L{l}H{h}" for l, h in HL_FLAT]
HEAD_CIRCUIT = [lh in ALL_CIRCUIT_HEADS for lh in HL_FLAT]
HEAD_ROLE    = [head_role(*lh) for lh in HL_FLAT]
HEAD_COLOR   = [ROLE_COLORS[r] for r in HEAD_ROLE]
N_HEAD = 144

# ── load cached arrays ────────────────────────────────────────────────────────
assert os.path.exists("outmag_ioi.npy"), "Run run_exp8_outmag.py first"
outmag_ioi = np.load("outmag_ioi.npy")   # (1000, 144)
outmag_non = np.load("outmag_non.npy")   # (1000, 144)
print(f"Loaded outmag_ioi {outmag_ioi.shape}, outmag_non {outmag_non.shape}")

# ── correlation matrices ───────────────────────────────────────────────────────
C_ioi = np.corrcoef(outmag_ioi.T)   # (144, 144)
C_non = np.corrcoef(outmag_non.T)   # (144, 144)

# Regularize C_non with Ledoit-Wolf so it's positive definite.
# We fit LW on the raw data (not the correlation matrix) — LW gives a
# regularized covariance; we convert to correlation form by dividing through
# by sqrt(diag) on both sides.
lw = LedoitWolf().fit(outmag_non)
C_non_cov_reg = lw.covariance_        # regularized covariance (144, 144)
sd = np.sqrt(np.diag(C_non_cov_reg))
C_non_reg = C_non_cov_reg / np.outer(sd, sd)   # regularized correlation matrix

print(f"Ledoit-Wolf shrinkage on C_non: {lw.shrinkage_:.4f}")
print(f"Condition number C_non (raw): {np.linalg.cond(C_non):.1f}")
print(f"Condition number C_non (reg): {np.linalg.cond(C_non_reg):.1f}")

# ── generalized eigenvalue decomposition ──────────────────────────────────────
# scipy.linalg.eigh(A, B) solves A v = λ B v, returns eigenvalues ascending.
# We want C_ioi v = λ C_non_reg v.
# Large λ = direction amplified by IOI (more variance in C_ioi than baseline).
# Small λ = direction suppressed by IOI.
vals, vecs = linalg.eigh(C_ioi, C_non_reg)

# Sort descending (largest ratio first).
idx = np.argsort(vals)[::-1]
vals = vals[idx]
vecs = vecs[:, idx]

print(f"\nTop-10 generalized eigenvalues (C_ioi/C_non ratio):")
for k in range(10):
    v = vecs[:, k]
    c8 = sum(1 for i in np.argsort(np.abs(v))[::-1][:8] if HEAD_CIRCUIT[i])
    top3 = [(HEAD_NAMES[i], SHORT[HEAD_ROLE[i]])
            for i in np.argsort(np.abs(v))[::-1][:3]]
    print(f"  k={k:2d}  λ={vals[k]:.3f}  circ/top8={c8}  top: {top3}")

print(f"\nBottom-10 generalized eigenvalues (suppressed by IOI):")
for k in range(N_HEAD - 10, N_HEAD):
    v = vecs[:, k]
    c8 = sum(1 for i in np.argsort(np.abs(v))[::-1][:8] if HEAD_CIRCUIT[i])
    top3 = [(HEAD_NAMES[i], SHORT[HEAD_ROLE[i]])
            for i in np.argsort(np.abs(v))[::-1][:3]]
    print(f"  k={k:3d}  λ={vals[k]:.4f}  circ/top8={c8}  top: {top3}")

# ── Fig 1: eigenvalue spectrum ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: full spectrum
ax = axes[0]
ks = np.arange(N_HEAD)
colors = ["#d62728" if vals[k] > 2.0 else ("#ff9900" if vals[k] > 1.2 else
          ("#aaaaaa" if vals[k] > 0.8 else "#4444cc"))
          for k in range(N_HEAD)]
ax.bar(ks, vals, color=colors, width=0.9, alpha=0.85)
ax.axhline(1.0, color="k", lw=1.0, ls="--", label="λ=1 (same variance in both conditions)")
ax.set_xlabel("Contrastive eigenvector index", fontsize=10)
ax.set_ylabel("λ = IOI variance / baseline variance in this direction", fontsize=10)
ax.set_title("Contrastive PCA eigenvalue spectrum\n"
             "λ >> 1: amplified by IOI  |  λ << 1: suppressed by IOI", fontsize=9)
ax.legend(fontsize=9)

# Right: zoom on top-20
ax = axes[1]
ax.bar(np.arange(20), vals[:20], color=colors[:20], width=0.9, alpha=0.85)
ax.axhline(1.0, color="k", lw=1.0, ls="--")
for k in range(20):
    c8 = sum(1 for i in np.argsort(np.abs(vecs[:, k]))[::-1][:8] if HEAD_CIRCUIT[i])
    ax.text(k, vals[k] + 0.05, str(c8), ha="center", va="bottom", fontsize=7,
            fontweight="bold" if c8 >= 4 else "normal",
            color="#d62728" if c8 >= 4 else "black")
ax.set_xlabel("Top-20 contrastive eigenvector index", fontsize=10)
ax.set_ylabel("λ", fontsize=10)
ax.set_title("Top-20 contrastive eigenvectors\n"
             "Number above bar = circuit heads in top-8 loaders", fontsize=9)

plt.tight_layout()
plt.savefig("figs/exp11_eigenvalues.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  → figs/exp11_eigenvalues.png")

# ── Fig 2: loading profiles for top contrastive eigenvectors ──────────────────
xs = np.arange(N_HEAD)

def plot_cpca_evec(k, ax=None):
    v = vecs[:, k]
    c8 = sum(1 for i in np.argsort(np.abs(v))[::-1][:8] if HEAD_CIRCUIT[i])
    n_circ = len(ALL_CIRCUIT_HEADS)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(20, 4))

    ax.bar(xs, v, color=HEAD_COLOR, alpha=0.85, width=0.9)
    ax.axhline(0, color="k", lw=0.6)
    for i, is_circ in enumerate(HEAD_CIRCUIT):
        if is_circ:
            ax.axvline(i, color="k", lw=0.5, alpha=0.3, zorder=0)
    ax.set_xticks([l * 12 + 6 for l in range(12)])
    ax.set_xticklabels([f"L{l}" for l in range(12)], fontsize=9)
    ax.set_xlim(-0.5, N_HEAD - 0.5)
    ax.set_ylabel("Loading", fontsize=9)
    ax.set_title(
        f"Contrastive eigenvector {k}  (λ = {vals[k]:.3f}, "
        f"IOI variance is {vals[k]:.1f}× baseline)\n"
        f"{c8}/{min(8, n_circ)} circuit heads in top-8 loaders  "
        f"(chance ≈ {n_circ/N_HEAD*8:.1f}/8)",
        fontsize=9
    )

    if standalone:
        legend_handles = [mpatches.Patch(color=ROLE_COLORS[r], label=SHORT[r])
                          for r in ROLE_ORDER]
        ax.legend(handles=legend_handles, fontsize=8, loc="upper right", ncol=4, framealpha=0.9)
        plt.tight_layout()
        fname = f"figs/exp11_cpca_{k:02d}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  → {fname}")

    return c8

# Individual figures for top contrastive modes
TOP_K = 10
for k in range(TOP_K):
    c8 = plot_cpca_evec(k)
    v = vecs[:, k]
    order = np.argsort(np.abs(v))[::-1]
    print(f"\n  cPCA evec {k} (λ={vals[k]:.3f}) top-12 loaders:")
    for rank, i in enumerate(order[:12], 1):
        tag = "CIRCUIT" if HEAD_CIRCUIT[i] else ""
        print(f"    {rank:2d}. {HEAD_NAMES[i]:8s} ({SHORT[HEAD_ROLE[i]]:5s})  v={v[i]:+.4f}  {tag}")

# Panel figure: all top-10 in one image
fig, axes = plt.subplots(TOP_K, 1, figsize=(20, 4 * TOP_K), sharex=True)
for k, ax in enumerate(axes):
    plot_cpca_evec(k, ax=ax)
legend_handles = [mpatches.Patch(color=ROLE_COLORS[r], label=SHORT[r]) for r in ROLE_ORDER]
axes[0].legend(handles=legend_handles, fontsize=8, loc="upper right", ncol=4, framealpha=0.9)
plt.tight_layout()
plt.savefig("figs/exp11_cpca_panel.png", dpi=120, bbox_inches="tight")
plt.close()
print("  → figs/exp11_cpca_panel.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n\nSummary of top-15 contrastive eigenvectors:")
print(f"  {'k':>3}  {'λ':>7}  {'circ/top8':>9}")
for k in range(15):
    v = vecs[:, k]
    c8 = sum(1 for i in np.argsort(np.abs(v))[::-1][:8] if HEAD_CIRCUIT[i])
    print(f"  {k:3d}  {vals[k]:7.3f}  {c8}/8")
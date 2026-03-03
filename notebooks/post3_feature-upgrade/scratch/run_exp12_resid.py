"""
Exp 12: Project C_non structure out of C_ioi eigenvectors.

For each eigenvector v_k of C_ioi, subtract the component that lies in
C_non's top-K subspace:

    v_resid = v_k - V_non_K V_non_K^T v_k

The residual shows only the part of v_k that is NOT shared structural noise.
Plot loading profiles of v_resid colored by circuit role.

This is the simplest version of the idea: work with C_ioi's eigenvectors,
strip the baseline structure out of each one, see what's left.

Figures:
  exp12_resid_{k}.png   — residual loading profile for C_ioi evec k
  exp12_resid_panel.png — all top-15 in one panel

Run with: /opt/miniconda3/bin/python run_exp12_resid.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
N_CIRC = len(ALL_CIRCUIT_HEADS)
CHANCE = N_CIRC / N_HEAD

# ── load and compute correlation matrices ─────────────────────────────────────
assert os.path.exists("outmag_ioi.npy"), "Run run_exp8_outmag.py first"
outmag_ioi = np.load("outmag_ioi.npy")
outmag_non = np.load("outmag_non.npy")

C_ioi = np.corrcoef(outmag_ioi.T)
C_non = np.corrcoef(outmag_non.T)

# Eigendecompose both, sort descending.
vals_ioi, vecs_ioi = np.linalg.eigh(C_ioi)
vals_non, vecs_non = np.linalg.eigh(C_non)
vals_ioi = vals_ioi[np.argsort(vals_ioi)[::-1]]
vecs_ioi = vecs_ioi[:, np.argsort(vals_ioi[::-1].argsort()[::-1])]  # re-sort columns
# cleaner sort:
order_ioi = np.argsort(np.linalg.eigh(C_ioi)[0])[::-1]
order_non = np.argsort(np.linalg.eigh(C_non)[0])[::-1]
vals_ioi = np.linalg.eigh(C_ioi)[0][order_ioi]
vecs_ioi = np.linalg.eigh(C_ioi)[1][:, order_ioi]
vals_non = np.linalg.eigh(C_non)[0][order_non]
vecs_non = np.linalg.eigh(C_non)[1][:, order_non]

# ── projection matrix for C_non top-K ────────────────────────────────────────
K = 6    # number of C_non modes to project out (covers ~80% of C_non variance)
V_non_K = vecs_non[:, :K]                          # (144, K)
P_non   = V_non_K @ V_non_K.T                      # (144, 144) — projects ONTO C_non subspace
P_perp  = np.eye(N_HEAD) - P_non                   # projects AWAY from C_non subspace

# fraction of C_non variance captured by top-K
var_frac = vals_non[:K].sum() / vals_non.sum()
print(f"C_non top-{K} eigenvectors capture {var_frac:.1%} of C_non variance")

# ── compute residuals and circuit enrichment ───────────────────────────────────
xs = np.arange(N_HEAD)

def circ_in_topk(v, k=8):
    return sum(1 for i in np.argsort(np.abs(v))[::-1][:k] if HEAD_CIRCUIT[i])

print(f"\n{'k':>3}  {'λ_ioi':>6}  {'raw c/8':>7}  {'resid c/8':>9}  {'overlap':>7}")
residuals = []
overlaps  = []
for k in range(N_HEAD):
    v      = vecs_ioi[:, k]
    proj   = P_non @ v
    v_resid = P_perp @ v           # component orthogonal to C_non top-K

    overlap = float(np.dot(proj, proj))   # fraction in C_non subspace
    overlaps.append(overlap)
    residuals.append(v_resid)

    if k < 20:
        c_raw   = circ_in_topk(v)
        c_resid = circ_in_topk(v_resid)
        print(f"{k:3d}  {vals_ioi[k]:6.2f}  {c_raw}/8      {c_resid}/8        {overlap:.2f}")

residuals = np.array(residuals)   # (N_HEAD, N_HEAD), row k = residual of evec k
overlaps  = np.array(overlaps)

# ── figures: raw vs residual side-by-side for top-15 evecs ────────────────────
legend_handles = [mpatches.Patch(color=ROLE_COLORS[r], label=SHORT[r]) for r in ROLE_ORDER]

TOP_K = 15
fig, axes = plt.subplots(TOP_K, 2, figsize=(26, 3.5 * TOP_K), sharex=True)

for k in range(TOP_K):
    v       = vecs_ioi[:, k]
    v_resid = residuals[k]
    overlap = overlaps[k]
    c_raw   = circ_in_topk(v)
    c_resid = circ_in_topk(v_resid)

    for ax, data, label, c in [
        (axes[k, 0], v,       "raw",    c_raw),
        (axes[k, 1], v_resid, "residual (C_non projected out)", c_resid),
    ]:
        ax.bar(xs, data, color=HEAD_COLOR, alpha=0.85, width=0.9)
        ax.axhline(0, color="k", lw=0.5)
        for i, is_circ in enumerate(HEAD_CIRCUIT):
            if is_circ:
                ax.axvline(i, color="k", lw=0.4, alpha=0.25, zorder=0)
        ax.set_xticks([l * 12 + 6 for l in range(12)])
        ax.set_xticklabels([f"L{l}" for l in range(12)], fontsize=8)
        ax.set_xlim(-0.5, N_HEAD - 0.5)
        ax.set_ylabel("loading", fontsize=8)
        ax.set_title(
            f"Evec {k} ({label})  λ={vals_ioi[k]:.2f}  overlap={overlap:.2f}  "
            f"circ/top8 = {c}/8  (chance {CHANCE*8:.1f})",
            fontsize=8
        )

axes[0, 0].legend(handles=legend_handles, fontsize=7, loc="upper right",
                  ncol=4, framealpha=0.9)
plt.suptitle(
    f"C_ioi eigenvectors: raw vs residual after projecting out C_non top-{K} modes\n"
    f"(C_non top-{K} captures {var_frac:.0%} of baseline variance — the NC layer-depth structure)",
    fontsize=10, y=1.001
)
plt.tight_layout()
plt.savefig("figs/exp12_resid_panel.png", dpi=100, bbox_inches="tight")
plt.close()
print(f"\n  → figs/exp12_resid_panel.png")

# ── individual residual figures for evecs with best circuit enrichment ─────────
best = sorted(range(20), key=lambda k: circ_in_topk(residuals[k]), reverse=True)[:6]
print(f"\nTop-6 evecs by residual circuit enrichment (top-8): {best}")

for k in best:
    v_resid = residuals[k]
    c_resid = circ_in_topk(v_resid)
    order   = np.argsort(np.abs(v_resid))[::-1]

    fig, ax = plt.subplots(figsize=(20, 4))
    ax.bar(xs, v_resid, color=HEAD_COLOR, alpha=0.85, width=0.9)
    ax.axhline(0, color="k", lw=0.6)
    for i, is_circ in enumerate(HEAD_CIRCUIT):
        if is_circ:
            ax.axvline(i, color="k", lw=0.5, alpha=0.3, zorder=0)
    ax.set_xticks([l * 12 + 6 for l in range(12)])
    ax.set_xticklabels([f"L{l}" for l in range(12)], fontsize=9)
    ax.set_xlim(-0.5, N_HEAD - 0.5)
    ax.set_ylabel("residual loading", fontsize=10)
    ax.set_title(
        f"C_ioi evec {k} — residual after projecting out C_non top-{K} modes\n"
        f"λ_ioi={vals_ioi[k]:.2f}  C_non overlap={overlaps[k]:.2f}  "
        f"{c_resid}/8 circuit heads in top-8  (chance {CHANCE*8:.1f}/8)",
        fontsize=9
    )
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right", ncol=4, framealpha=0.9)
    plt.tight_layout()
    fname = f"figs/exp12_resid_{k:02d}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  Evec {k} residual top-12 loaders:")
    for rank, i in enumerate(order[:12], 1):
        tag = "CIRCUIT" if HEAD_CIRCUIT[i] else ""
        print(f"    {rank:2d}. {HEAD_NAMES[i]:8s} ({SHORT[HEAD_ROLE[i]]:5s})  "
              f"v_resid={v_resid[i]:+.4f}  {tag}")
    print(f"  → {fname}")
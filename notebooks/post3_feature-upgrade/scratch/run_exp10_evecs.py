"""
Exp 10: IOI-specific eigenvectors of C_ioi.

Plan:
  1. Compute C_ioi = corrcoef(outmag_ioi) and C_non = corrcoef(outmag_non).
  2. Eigendecompose each: eigenvectors sorted by eigenvalue (descending).
  3. For each eigenvector of C_ioi, compute overlap with the top-10 subspace
     of C_non: overlap_k = ||V_non_10^T v_k||^2  (fraction of variance explained
     by the structural non-IOI modes).
     High overlap (~1) = the mode exists in both conditions = generic structural mode.
     Low overlap (~0) = IOI-specific mode.
  4. Plot bar chart loading profiles for the top IOI-specific eigenvectors,
     colored by circuit role, with the overlap score as subtitle.

Figures:
  exp10_overlap.png          -- overlap of each C_ioi evec with C_non top-10 subspace
  exp10_evec_{k}.png         -- loading profile for IOI-specific evec k

Run with: /opt/miniconda3/bin/python run_exp10_evecs.py
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

# ── load cached arrays ────────────────────────────────────────────────────────
assert os.path.exists("outmag_ioi.npy"), "Run run_exp8_outmag.py first"
outmag_ioi = np.load("outmag_ioi.npy")   # (1000, 144)
outmag_non = np.load("outmag_non.npy")   # (1000, 144)
print(f"Loaded outmag_ioi {outmag_ioi.shape}, outmag_non {outmag_non.shape}")

# ── compute correlation matrices and eigenvectors ─────────────────────────────
C_ioi = np.corrcoef(outmag_ioi.T)   # (144, 144)
C_non = np.corrcoef(outmag_non.T)   # (144, 144)

# Eigendecomposition — numpy returns in ascending order, so reverse.
vals_ioi, vecs_ioi = np.linalg.eigh(C_ioi)
vals_non, vecs_non = np.linalg.eigh(C_non)

# Sort descending (largest eigenvalue first).
idx_ioi = np.argsort(vals_ioi)[::-1]
idx_non = np.argsort(vals_non)[::-1]
vals_ioi = vals_ioi[idx_ioi]
vecs_ioi = vecs_ioi[:, idx_ioi]   # columns are eigenvectors
vals_non = vals_non[idx_non]
vecs_non = vecs_non[:, idx_non]

print(f"Top-5 eigenvalues C_ioi: {vals_ioi[:5].round(2)}")
print(f"Top-5 eigenvalues C_non: {vals_non[:5].round(2)}")

# ── overlap analysis ──────────────────────────────────────────────────────────
# For each eigenvector of C_ioi, measure how much of it lives in the top-10
# subspace of C_non.  V_non_10 is the (144, 10) matrix of the 10 leading
# eigenvectors of C_non.
# overlap_k = ||V_non_10^T v_k||^2 = sum of squared projections onto those 10 dirs
# = fraction of |v_k|^2 explained by the non-IOI structural modes.
N_NON_SUBSPACE = 10
V_non_10 = vecs_non[:, :N_NON_SUBSPACE]   # (144, 10)

overlaps = []
for k in range(N_HEAD):
    v = vecs_ioi[:, k]
    proj = V_non_10.T @ v          # (10,)
    overlap = float(np.dot(proj, proj))
    overlaps.append(overlap)
overlaps = np.array(overlaps)

# IOI-specific = low overlap
print("\nOverlap of C_ioi evecs with C_non top-10 subspace:")
for k in range(15):
    v = vecs_ioi[:, k]
    top_circ = sorted(
        [(abs(v[i]), HEAD_NAMES[i], HEAD_ROLE[i], HEAD_CIRCUIT[i])
         for i in range(N_HEAD)],
        reverse=True
    )[:5]
    circ_in_top8 = sum(1 for i in np.argsort(np.abs(v))[::-1][:8] if HEAD_CIRCUIT[i])
    print(f"  Evec {k:2d}  λ={vals_ioi[k]:.2f}  overlap={overlaps[k]:.2f}  "
          f"circ_in_top8={circ_in_top8}  "
          f"top loaders: {[(n, SHORT[r]) for _, n, r, _ in top_circ[:3]]}")

# ── Fig 1: overlap vs evec index ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ks = np.arange(N_HEAD)
colors = ["#d62728" if overlaps[k] < 0.5 else ("#ff9900" if overlaps[k] < 0.8 else "#aaaaaa")
          for k in range(N_HEAD)]
ax.bar(ks, overlaps, color=colors, width=0.9, alpha=0.85)
ax.axhline(0.5, color="#d62728", lw=1.2, ls="--", label="0.5 threshold (IOI-specific)")
ax.axhline(0.8, color="#ff9900", lw=1.2, ls="--", label="0.8 threshold (partially novel)")
ax.set_xlabel("Eigenvector index of C_ioi (sorted by eigenvalue, largest first)", fontsize=10)
ax.set_ylabel("Overlap with C_non top-10 subspace", fontsize=10)
ax.set_title("How much of each C_ioi eigenvector is explained by the non-IOI structural modes?\n"
             "Low overlap = genuinely IOI-specific mode; high overlap = exists in both conditions",
             fontsize=9)
ax.set_xlim(-0.5, 30.5)   # zoom in on leading evecs where the action is
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("figs/exp10_overlap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figs/exp10_overlap.png")

# ── Fig 2: loading profiles for IOI-specific eigenvectors ─────────────────────
# Show evecs with overlap < 0.8 among the first 15 (where eigenvalues are large).
IOI_SPECIFIC_THRESHOLD = 0.8
candidates = [k for k in range(15) if overlaps[k] < IOI_SPECIFIC_THRESHOLD]
print(f"\nIOI-specific evecs (overlap < {IOI_SPECIFIC_THRESHOLD}) in first 15: {candidates}")

xs = np.arange(N_HEAD)
for k in candidates:
    v = vecs_ioi[:, k]
    overlap = overlaps[k]
    circ_in_top8 = sum(1 for i in np.argsort(np.abs(v))[::-1][:8] if HEAD_CIRCUIT[i])
    n_circ = len(ALL_CIRCUIT_HEADS)

    fig, ax = plt.subplots(figsize=(20, 4))
    ax.bar(xs, v, color=HEAD_COLOR, alpha=0.85, width=0.9)
    ax.axhline(0, color="k", lw=0.6)

    # Vertical marks at circuit head positions.
    for i, is_circ in enumerate(HEAD_CIRCUIT):
        if is_circ:
            ax.axvline(i, color="k", lw=0.5, alpha=0.3, zorder=0)

    # Layer boundary ticks.
    ax.set_xticks([l * 12 + 6 for l in range(12)])
    ax.set_xticklabels([f"L{l}" for l in range(12)], fontsize=9)
    ax.set_xlim(-0.5, N_HEAD - 0.5)

    ax.set_ylabel("Loading (component of eigenvector)", fontsize=10)
    ax.set_title(
        f"C_ioi eigenvector {k}  (λ = {vals_ioi[k]:.2f})\n"
        f"Overlap with C_non top-10 subspace: {overlap:.2f}  "
        f"({(1-overlap)*100:.0f}% IOI-specific)  |  "
        f"{circ_in_top8}/{min(8, n_circ)} circuit heads in top-8 loaders  "
        f"(chance ≈ {n_circ/N_HEAD*8:.1f}/8)",
        fontsize=9
    )

    # Legend.
    legend_handles = [mpatches.Patch(color=ROLE_COLORS[r], label=SHORT[r])
                      for r in ROLE_ORDER]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right", ncol=4, framealpha=0.9)

    plt.tight_layout()
    fname = f"figs/exp10_evec_{k:02d}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {fname}")

    # Print top loaders.
    order = np.argsort(np.abs(v))[::-1]
    print(f"\n  Evec {k} top-12 loaders:")
    for rank, i in enumerate(order[:12], 1):
        circ_tag = "CIRCUIT" if HEAD_CIRCUIT[i] else ""
        print(f"    {rank:2d}. {HEAD_NAMES[i]:8s} ({SHORT[HEAD_ROLE[i]]:5s})  v={v[i]:+.4f}  {circ_tag}")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n\nSummary — all evecs in first 20 with overlap < 0.8:")
print(f"  {'k':>3}  {'λ':>6}  {'overlap':>7}  {'novel%':>6}  {'circ/top8':>9}")
for k in range(20):
    if overlaps[k] < 0.8:
        v = vecs_ioi[:, k]
        c8 = sum(1 for i in np.argsort(np.abs(v))[::-1][:8] if HEAD_CIRCUIT[i])
        print(f"  {k:3d}  {vals_ioi[k]:6.2f}  {overlaps[k]:7.2f}  {(1-overlaps[k])*100:5.0f}%  {c8}/8")
"""
Post 3 — Exp 6: Factor analysis at k=6, focused circuit graph visualization.
Loads cached X_ioi.npy / X_non.npy (run run_exp5_fa.py first to generate).

Figures:
  fa6_Jdiff_sorted.png   — J_diff heatmap with rows/cols sorted by role
  fa6_network.png        — network diagram of top J_diff edges
  fa6_loadings.png       — loading profiles at k=6

Run with: /opt/miniconda3/bin/python run_exp6_k6.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtrans
from scipy import stats
from sklearn.decomposition import FactorAnalysis

os.makedirs("figs", exist_ok=True)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── metadata (no model needed) ────────────────────────────────────────────────
IOI_HEADS = {
    "Name Mover":          [(9, 6), (9, 9), (10, 0)],
    "Backup Name Mover":   [(9, 0), (9, 7), (10, 1), (10, 2), (10, 6), (10, 10), (11, 2), (11, 9)],
    "Negative Name Mover": [(10, 7), (11, 10)],
    "S-Inhibition":        [(7, 3), (7, 9), (8, 6), (8, 10)],
    "Induction":           [(5, 5), (6, 9)],
    "Duplicate Token":     [(0, 1), (3, 0)],
    "Previous Token":      [(2, 2), (4, 11)],
}
ROLE_ORDER = ["Name Mover","Backup Name Mover","Negative Name Mover",
              "S-Inhibition","Induction","Duplicate Token","Previous Token",
              "Non-circuit","MLP"]
ROLE_COLORS = {
    "Name Mover":          "#d62728",
    "Backup Name Mover":   "#ff7f0e",
    "Negative Name Mover": "#9467bd",
    "S-Inhibition":        "#2ca02c",
    "Induction":           "#17becf",
    "Duplicate Token":     "#bcbd22",
    "Previous Token":      "#e377c2",
    "Non-circuit":         "#cccccc",
    "MLP":                 "#4dac26",
}
SHORT = {"Name Mover":"NM","Backup Name Mover":"BNM","Negative Name Mover":"NegNM",
         "S-Inhibition":"SI","Induction":"Ind","Duplicate Token":"DT","Previous Token":"PT",
         "Non-circuit":"NC","MLP":"MLP"}

ALL_CIRCUIT_HEADS = set(lh for hs in IOI_HEADS.values() for lh in hs)
HL_FLAT  = [(l, h) for l in range(12) for h in range(12)]

def head_role(l, h):
    for role, heads in IOI_HEADS.items():
        if (l, h) in heads: return role
    return "Non-circuit"

FEAT_NAMES   = [f"L{l}H{h}" for l, h in HL_FLAT] + [f"MLP_L{l}" for l in range(12)]
FEAT_CIRCUIT = [lh in ALL_CIRCUIT_HEADS for lh in HL_FLAT] + [False]*12
FEAT_ROLE    = [head_role(*lh) for lh in HL_FLAT] + ["MLP"]*12
FEAT_COLOR   = [ROLE_COLORS[r] for r in FEAT_ROLE]
N_FEAT = 156

# ── load cached features ──────────────────────────────────────────────────────
assert os.path.exists("X_ioi.npy"), "Run run_exp5_fa.py first to generate X_ioi.npy"
X_ioi = np.load("X_ioi.npy")
X_non = np.load("X_non.npy")
print(f"Loaded X_ioi {X_ioi.shape}, X_non {X_non.shape}")

# z-score within each class
mu_i, sd_i = X_ioi.mean(0), X_ioi.std(0) + 1e-10
mu_n, sd_n = X_non.mean(0), X_non.std(0) + 1e-10
Xz_ioi = (X_ioi - mu_i) / sd_i
Xz_non = (X_non - mu_n) / sd_n

# ── fit at k=6 ────────────────────────────────────────────────────────────────
K = 6
fa_ioi = FactorAnalysis(n_components=K, random_state=42, max_iter=2000)
fa_non = FactorAnalysis(n_components=K, random_state=42, max_iter=2000)
fa_ioi.fit(Xz_ioi); fa_non.fit(Xz_non)

W_ioi = fa_ioi.components_.T   # (156, 6)
W_non = fa_non.components_.T

J_ioi  = W_ioi @ W_ioi.T
J_non  = W_non @ W_non.T
J_diff = J_ioi - J_non

print(f"k={K}  log-lik IOI={fa_ioi.score(Xz_ioi):.3f}  non-IOI={fa_non.score(Xz_non):.3f}")
print(f"J_diff range: [{J_diff.min():.3f}, {J_diff.max():.3f}]")

# ── Fig 1: J_diff sorted by role ─────────────────────────────────────────────
# Reorder features: circuit heads grouped by role, then non-circuit, then MLP
role_sort_key = {r: i for i, r in enumerate(ROLE_ORDER)}
sort_order = sorted(range(N_FEAT), key=lambda i: (role_sort_key[FEAT_ROLE[i]],
                                                    i // 12,   # layer within role
                                                    i % 12))   # head within layer
Js = J_diff[np.ix_(sort_order, sort_order)]
sorted_roles  = [FEAT_ROLE[i]  for i in sort_order]
sorted_names  = [FEAT_NAMES[i] for i in sort_order]
sorted_colors = [FEAT_COLOR[i] for i in sort_order]
sorted_circuit= [FEAT_CIRCUIT[i] for i in sort_order]

# Block boundaries: where role changes
boundaries = [0]
for k_b in range(1, N_FEAT):
    if sorted_roles[k_b] != sorted_roles[k_b-1]:
        boundaries.append(k_b)
boundaries.append(N_FEAT)

fig, ax = plt.subplots(figsize=(13, 11))
vmax = np.abs(Js).max()
im = ax.imshow(Js, cmap="PRGn_r", vmin=-vmax, vmax=vmax, aspect="auto")

# Role block dividers
for b in boundaries[1:-1]:
    ax.axvline(b - 0.5, color="white", lw=1.2, alpha=0.8)
    ax.axhline(b - 0.5, color="white", lw=1.2, alpha=0.8)

# Role labels on axes
mid = [(boundaries[i] + boundaries[i+1]) // 2 for i in range(len(boundaries)-1)]
role_labels = [sorted_roles[boundaries[i]] for i in range(len(boundaries)-1)]
ax.set_xticks(mid)
ax.set_xticklabels([SHORT[r] for r in role_labels], fontsize=9, rotation=45, ha="right")
ax.set_yticks(mid)
ax.set_yticklabels([SHORT[r] for r in role_labels], fontsize=9)

fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
ax.set_title(f"J_diff sorted by circuit role  (k={K})\n"
             "Rows/cols ordered: NM → BNM → NegNM → SI → Ind → DT → PT → NC → MLP",
             fontsize=11)
plt.tight_layout()
plt.savefig("figs/fa6_Jdiff_sorted.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figs/fa6_Jdiff_sorted.png")

# ── Fig 2: Network diagram ────────────────────────────────────────────────────
# Nodes = any head/MLP that appears in a strong edge
# Edges = |J_diff[i,j]| > threshold (mean + 2.5σ, off-diagonal)
mask_off = ~np.eye(N_FEAT, dtype=bool)
Jd_off   = np.abs(J_diff[mask_off])
thresh   = Jd_off.mean() + 2.5 * Jd_off.std()

rows, cols = np.where((np.abs(J_diff) > thresh) & mask_off)
edges = [(r, c, J_diff[r, c]) for r, c in zip(rows, cols) if r < c]
edges.sort(key=lambda e: abs(e[2]), reverse=True)

print(f"\nNetwork: threshold={thresh:.4f}, {len(edges)} edges")

# Collect active nodes
active_nodes = sorted(set(r for r,c,v in edges) | set(c for r,c,v in edges))
print(f"Active nodes: {len(active_nodes)}")
for idx in active_nodes:
    r = FEAT_ROLE[idx]
    print(f"  {FEAT_NAMES[idx]:10s}  {SHORT[r]}")

# Layout: place nodes on a circle, grouped by role
# Group active nodes by role
from collections import defaultdict
role_groups = defaultdict(list)
for idx in active_nodes:
    role_groups[FEAT_ROLE[idx]].append(idx)

# Assign angular positions: role groups get arcs, gaps between groups
import math
node_angles = {}
total = len(active_nodes)
gap = 2 * math.pi / (len(role_groups) * 3 + total)  # small gap between groups
angle = 0
for role in ROLE_ORDER:
    if role not in role_groups: continue
    for idx in sorted(role_groups[role]):
        node_angles[idx] = angle
        angle += gap * 3
    angle += gap * 6  # larger gap between role groups

R = 10.0
pos = {idx: (R * math.cos(a), R * math.sin(a)) for idx, a in node_angles.items()}

fig, ax = plt.subplots(figsize=(14, 14))
ax.set_aspect("equal")
ax.axis("off")

# Draw edges first (behind nodes)
max_abs = max(abs(v) for r,c,v in edges) if edges else 1
for r, c, v in edges:
    if r not in pos or c not in pos: continue
    x0, y0 = pos[r]; x1, y1 = pos[c]
    alpha = 0.3 + 0.5 * abs(v) / max_abs
    lw    = 0.5 + 3.0 * abs(v) / max_abs
    color = "#8B0000" if v > 0 else "#003080"  # dark red=positive, dark blue=negative
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, alpha=alpha, zorder=1)

# Draw nodes
for idx in active_nodes:
    if idx not in pos: continue
    x, y = pos[idx]
    role  = FEAT_ROLE[idx]
    color = ROLE_COLORS[role]
    size  = 220 if FEAT_CIRCUIT[idx] else 110
    edge_c = "black" if FEAT_CIRCUIT[idx] else "gray"
    ax.scatter(x, y, s=size, c=color, edgecolors=edge_c, linewidths=1.5,
               zorder=3)
    # Label
    name = FEAT_NAMES[idx]
    # offset label outward
    norm = math.sqrt(x**2 + y**2)
    lx, ly = x * 1.18, y * 1.18
    ax.text(lx, ly, name, ha="center", va="center", fontsize=7.5,
            fontweight="bold" if FEAT_CIRCUIT[idx] else "normal")

# Legend: node colors
legend_patches = [mpatches.Patch(color=ROLE_COLORS[r], label=SHORT[r])
                  for r in ROLE_ORDER if any(FEAT_ROLE[i]==r for i in active_nodes)]
# Edge legend
legend_patches += [
    plt.Line2D([0],[0], color="#8B0000", lw=2, label="J_diff > 0 (co-activate on IOI)"),
    plt.Line2D([0],[0], color="#003080", lw=2, label="J_diff < 0 (anti-correlate on IOI)"),
]
ax.legend(handles=legend_patches, fontsize=9, loc="lower right", framealpha=0.9)
ax.set_title(f"J_diff circuit graph — k={K},  threshold = mean+2.5σ ({thresh:.3f})\n"
             f"{len(edges)} edges, {len(active_nodes)} nodes  |  large nodes = known circuit heads",
             fontsize=12)
plt.tight_layout()
plt.savefig("figs/fa6_network.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figs/fa6_network.png")

# ── Fig 3: Loading profiles at k=6 ───────────────────────────────────────────
fig, axes = plt.subplots(K, 1, figsize=(20, 2.5*K), sharex=True)
xs = np.arange(N_FEAT)
for k_i, ax in enumerate(axes):
    loadings = W_ioi[:, k_i]
    ax.bar(xs, loadings, color=FEAT_COLOR, alpha=0.85, width=0.9)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel(f"Factor {k_i+1}", fontsize=10)
    ax.axvline(143.5, color="k", lw=1.0, ls="--", alpha=0.5)
    # Mark circuit heads with thin vertical lines
    for i, c in enumerate(FEAT_CIRCUIT[:144]):
        if c: ax.axvline(i, color="k", lw=0.3, alpha=0.2, zorder=0)
    # Label top 5 loaders
    top = np.argsort(np.abs(loadings))[-5:]
    for i in top:
        ax.text(i, loadings[i] + (0.02 if loadings[i]>=0 else -0.04),
                SHORT[FEAT_ROLE[i]], ha="center", fontsize=6,
                color=ROLE_COLORS[FEAT_ROLE[i]], fontweight="bold")

axes[-1].set_xticks([l*12+6 for l in range(12)] + [150])
axes[-1].set_xticklabels([f"L{l}" for l in range(12)] + ["MLP"], fontsize=9)
legend_handles = [mpatches.Patch(color=c, label=r) for r, c in ROLE_COLORS.items()]
axes[0].legend(handles=legend_handles, fontsize=8, loc="upper right", ncol=5, framealpha=0.9)
plt.suptitle(f"W_ioi loading profiles — k={K}  (z-scored log Var)", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("figs/fa6_loadings.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figs/fa6_loadings.png")

# ── edge stats summary ────────────────────────────────────────────────────────
print(f"\nEdge breakdown (|J_diff| > {thresh:.4f}, mean+2.5σ):")
cc  = sum(1 for r,c,v in edges if FEAT_CIRCUIT[r] and FEAT_CIRCUIT[c])
cnc = sum(1 for r,c,v in edges if FEAT_CIRCUIT[r] != FEAT_CIRCUIT[c])
nn  = sum(1 for r,c,v in edges if not FEAT_CIRCUIT[r] and not FEAT_CIRCUIT[c])
tot = len(edges)
p_cc   = (len(ALL_CIRCUIT_HEADS)/N_FEAT)**2
p_cnc  = 2*(len(ALL_CIRCUIT_HEADS)/N_FEAT)*(1-len(ALL_CIRCUIT_HEADS)/N_FEAT)
print(f"  CC  {cc:3d}/{tot} = {cc/max(tot,1):.1%}   (chance {p_cc:.1%})")
print(f"  CNC {cnc:3d}/{tot} = {cnc/max(tot,1):.1%}   (chance {p_cnc:.1%})")
print(f"  NN  {nn:3d}/{tot} = {nn/max(tot,1):.1%}")
print(f"\nTop 15 edges:")
for r,c,v in edges[:15]:
    tag = "CC" if FEAT_CIRCUIT[r] and FEAT_CIRCUIT[c] else \
          "CNC" if FEAT_CIRCUIT[r] or FEAT_CIRCUIT[c] else "NN"
    print(f"  {FEAT_NAMES[r]:10s} ({SHORT[FEAT_ROLE[r]]:5s}) ↔ "
          f"{FEAT_NAMES[c]:10s} ({SHORT[FEAT_ROLE[c]]:5s})  "
          f"J={v:+.3f}  [{tag}]")

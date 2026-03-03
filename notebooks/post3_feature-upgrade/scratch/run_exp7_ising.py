"""
Exp 7: Full-rank Gaussian graphical model on 2000 pooled feature vectors.

Stack X_ioi (1000×156) and X_non (1000×156) → 2000×156.
Center the columns (subtract mean), compute the 156×156 covariance C,
invert it to get the precision matrix J = C⁻¹.

In a Gaussian graphical model, J[i,j] ≠ 0 means heads i and j are
conditionally dependent given all other heads — i.e., they are directly
coupled, not just correlated via a third head.

Figures:
  exp7_J_sorted.png   — J heatmap sorted by circuit role
  exp7_network.png    — network diagram of large |J_ij| edges

Run with: /opt/miniconda3/bin/python run_exp7_ising.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
from collections import defaultdict
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

# ── load and stack ─────────────────────────────────────────────────────────────
assert os.path.exists("X_ioi.npy"), "Run run_exp5_fa.py first to generate X_ioi.npy"
X_ioi = np.load("X_ioi.npy")   # (1000, 156): log(Var_v) for each of 1000 IOI prompts
X_non = np.load("X_non.npy")   # (1000, 156): same for 1000 non-IOI prompts
print(f"Loaded X_ioi {X_ioi.shape}, X_non {X_non.shape}")

# Standardize each condition separately (subtract its own mean, divide by its own std).
# Then fit Ledoit-Wolf precision matrices for each condition independently.
# J_diff = J_ioi - J_non asks: which pairs of heads become MORE conditionally
# dependent when the circuit is active vs. when it is not?
mu_i, sd_i = X_ioi.mean(0), X_ioi.std(0) + 1e-10
mu_n, sd_n = X_non.mean(0), X_non.std(0) + 1e-10
Xz_ioi = (X_ioi - mu_i) / sd_i
Xz_non = (X_non - mu_n) / sd_n

lw_ioi = LedoitWolf().fit(Xz_ioi)
lw_non = LedoitWolf().fit(Xz_non)

J_ioi = lw_ioi.precision_   # (156, 156)
J_non = lw_non.precision_   # (156, 156)
J     = J_ioi - J_non       # differential precision matrix

print(f"Ledoit-Wolf shrinkage — IOI: {lw_ioi.shrinkage_:.4f}, non-IOI: {lw_non.shrinkage_:.4f}")
print(f"Condition numbers — IOI: {np.linalg.cond(lw_ioi.covariance_):.1f}, "
      f"non-IOI: {np.linalg.cond(lw_non.covariance_):.1f}")
print(f"J_diff range: [{J.min():.4f}, {J.max():.4f}]")

# Remove diagonal for visualization
J_viz = J.copy()
np.fill_diagonal(J_viz, 0.0)

# ── Fig 1: J sorted by circuit role ───────────────────────────────────────────
role_sort_key = {r: i for i, r in enumerate(ROLE_ORDER)}
sort_order = sorted(range(N_FEAT),
                    key=lambda i: (role_sort_key[FEAT_ROLE[i]], i // 12, i % 12))

Js = J_viz[np.ix_(sort_order, sort_order)]
sorted_roles = [FEAT_ROLE[i] for i in sort_order]

boundaries = [0]
for k in range(1, N_FEAT):
    if sorted_roles[k] != sorted_roles[k-1]:
        boundaries.append(k)
boundaries.append(N_FEAT)

fig, ax = plt.subplots(figsize=(13, 11))
vmax = np.percentile(np.abs(Js), 99)   # 99th percentile to avoid outlier squashing
im = ax.imshow(Js, cmap="PRGn_r", vmin=-vmax, vmax=vmax, aspect="auto")

for b in boundaries[1:-1]:
    ax.axvline(b - 0.5, color="white", lw=1.2, alpha=0.8)
    ax.axhline(b - 0.5, color="white", lw=1.2, alpha=0.8)

mid = [(boundaries[i] + boundaries[i+1]) // 2 for i in range(len(boundaries)-1)]
role_labels = [sorted_roles[boundaries[i]] for i in range(len(boundaries)-1)]
ax.set_xticks(mid)
ax.set_xticklabels([SHORT[r] for r in role_labels], fontsize=9, rotation=45, ha="right")
ax.set_yticks(mid)
ax.set_yticklabels([SHORT[r] for r in role_labels], fontsize=9)

fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
ax.set_title("J_diff = J_ioi − J_non, sorted by circuit role\n"
             "J_diff[i,j] > 0: more conditionally coupled on IOI  |  < 0: less coupled on IOI\n"
             "(diagonal zeroed; colorscale clipped at 99th percentile)",
             fontsize=10)
plt.tight_layout()
plt.savefig("figs/exp7_J_sorted.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figs/exp7_J_sorted.png")

# ── Fig 2: Network diagram of strong conditional dependencies ─────────────────
mask_off = ~np.eye(N_FEAT, dtype=bool)
J_off    = np.abs(J_viz[mask_off])
thresh   = J_off.mean() + 2.5 * J_off.std()

rows, cols = np.where((np.abs(J_viz) > thresh) & mask_off)
edges = [(r, c, J_viz[r, c]) for r, c in zip(rows, cols) if r < c]
edges.sort(key=lambda e: abs(e[2]), reverse=True)

print(f"\nNetwork: threshold = mean+2.5σ = {thresh:.4f}")
print(f"  {len(edges)} edges")

active_nodes = sorted(set(r for r,c,v in edges) | set(c for r,c,v in edges))
print(f"  {len(active_nodes)} nodes")
for idx in active_nodes:
    print(f"    {FEAT_NAMES[idx]:10s}  {SHORT[FEAT_ROLE[idx]]}")

# Circular layout, grouped by role
role_groups = defaultdict(list)
for idx in active_nodes:
    role_groups[FEAT_ROLE[idx]].append(idx)

node_angles = {}
gap = 2 * math.pi / (len(role_groups) * 3 + len(active_nodes))
angle = 0.0
for role in ROLE_ORDER:
    if role not in role_groups:
        continue
    for idx in sorted(role_groups[role]):
        node_angles[idx] = angle
        angle += gap * 3
    angle += gap * 6   # larger gap between role groups

R = 10.0
pos = {idx: (R * math.cos(a), R * math.sin(a)) for idx, a in node_angles.items()}

fig, ax = plt.subplots(figsize=(14, 14))
ax.set_aspect("equal")
ax.axis("off")

max_abs = max(abs(v) for r,c,v in edges) if edges else 1.0
for r, c, v in edges:
    if r not in pos or c not in pos:
        continue
    x0, y0 = pos[r]; x1, y1 = pos[c]
    alpha = 0.3 + 0.5 * abs(v) / max_abs
    lw    = 0.5 + 3.0 * abs(v) / max_abs
    color = "#8B0000" if v > 0 else "#003080"
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, alpha=alpha, zorder=1)

for idx in active_nodes:
    if idx not in pos:
        continue
    x, y = pos[idx]
    color  = ROLE_COLORS[FEAT_ROLE[idx]]
    size   = 220 if FEAT_CIRCUIT[idx] else 110
    edge_c = "black" if FEAT_CIRCUIT[idx] else "gray"
    ax.scatter(x, y, s=size, c=color, edgecolors=edge_c, linewidths=1.5, zorder=3)
    lx, ly = x * 1.18, y * 1.18
    ax.text(lx, ly, FEAT_NAMES[idx], ha="center", va="center", fontsize=7.5,
            fontweight="bold" if FEAT_CIRCUIT[idx] else "normal")

legend_patches = [mpatches.Patch(color=ROLE_COLORS[r], label=SHORT[r])
                  for r in ROLE_ORDER if any(FEAT_ROLE[i]==r for i in active_nodes)]
legend_patches += [
    plt.Line2D([0],[0], color="#8B0000", lw=2, label="J > 0  (conditionally co-active)"),
    plt.Line2D([0],[0], color="#003080", lw=2, label="J < 0  (conditionally anti-correlated)"),
]
ax.legend(handles=legend_patches, fontsize=9, loc="lower right", framealpha=0.9)
ax.set_title(f"J_diff circuit graph — threshold = mean+2.5σ ({thresh:.3f})\n"
             f"{len(edges)} edges, {len(active_nodes)} nodes  |  large nodes = known circuit heads",
             fontsize=12)
plt.tight_layout()
plt.savefig("figs/exp7_network.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figs/exp7_network.png")

# ── edge stats ─────────────────────────────────────────────────────────────────
print(f"\nEdge breakdown (|J| > {thresh:.4f}):")
cc  = sum(1 for r,c,v in edges if FEAT_CIRCUIT[r] and FEAT_CIRCUIT[c])
cnc = sum(1 for r,c,v in edges if FEAT_CIRCUIT[r] != FEAT_CIRCUIT[c])
nn  = sum(1 for r,c,v in edges if not FEAT_CIRCUIT[r] and not FEAT_CIRCUIT[c])
tot = len(edges)
p_cc  = (len(ALL_CIRCUIT_HEADS) / N_FEAT) ** 2
p_cnc = 2 * (len(ALL_CIRCUIT_HEADS) / N_FEAT) * (1 - len(ALL_CIRCUIT_HEADS) / N_FEAT)
print(f"  CC  {cc:3d}/{tot} = {cc/max(tot,1):.1%}   (chance {p_cc:.1%})")
print(f"  CNC {cnc:3d}/{tot} = {cnc/max(tot,1):.1%}   (chance {p_cnc:.1%})")
print(f"  NN  {nn:3d}/{tot} = {nn/max(tot,1):.1%}")
print(f"\nTop 20 edges by |J|:")
for r, c, v in edges[:20]:
    tag = "CC"  if FEAT_CIRCUIT[r] and FEAT_CIRCUIT[c] else \
          "CNC" if FEAT_CIRCUIT[r] or  FEAT_CIRCUIT[c] else "NN"
    print(f"  {FEAT_NAMES[r]:10s} ({SHORT[FEAT_ROLE[r]]:5s}) ↔ "
          f"{FEAT_NAMES[c]:10s} ({SHORT[FEAT_ROLE[c]]:5s})  "
          f"J={v:+.4f}  [{tag}]")
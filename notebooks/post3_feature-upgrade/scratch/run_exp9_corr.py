"""
Exp 9: Correlation matrix of out_mag across prompts.

Two versions:
  (A) Pooled: correlation of out_mag across all 2000 prompts (IOI + non-IOI stacked,
      no z-scoring). Circuit heads are all high on the first 1000 rows and low on the
      second 1000 — so pairs of circuit heads will naturally correlate with each other.

  (B) Differential: C_ioi - C_non, where each is the correlation matrix computed
      separately on the 1000 IOI prompts and 1000 non-IOI prompts.
      Positive entries = pairs that co-vary MORE on IOI prompts specifically.

Figures:
  exp9_corr_pooled.png       — pooled correlation heatmap sorted by role
  exp9_corr_diff.png         — differential correlation heatmap sorted by role
  exp9_network_pooled.png    — network graph of strong pooled correlations
  exp9_network_diff.png      — network graph of strong differential correlations

Run with: /opt/miniconda3/bin/python run_exp9_corr.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
from collections import defaultdict

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
              "S-Inhibition","Induction","Duplicate Token","Previous Token","Non-circuit"]
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
SHORT = {"Name Mover":"NM","Backup Name Mover":"BNM","Negative Name Mover":"NegNM",
         "S-Inhibition":"SI","Induction":"Ind","Duplicate Token":"DT","Previous Token":"PT",
         "Non-circuit":"NC"}

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

# ── load cached out_mag arrays ────────────────────────────────────────────────
assert os.path.exists("outmag_ioi.npy"), "Run run_exp8_outmag.py first"
outmag_ioi = np.load("outmag_ioi.npy")   # (1000, 144)
outmag_non = np.load("outmag_non.npy")   # (1000, 144)
print(f"Loaded outmag_ioi {outmag_ioi.shape}, outmag_non {outmag_non.shape}")

# ── compute correlation matrices ──────────────────────────────────────────────
# (A) Pooled: stack all 2000, no z-scoring, compute correlation
X_pooled = np.vstack([outmag_ioi, outmag_non])   # (2000, 144)
C_pooled  = np.corrcoef(X_pooled.T)               # (144, 144)

# (B) Separate, then difference
C_ioi  = np.corrcoef(outmag_ioi.T)   # (144, 144)
C_non  = np.corrcoef(outmag_non.T)   # (144, 144)
C_diff = C_ioi - C_non               # positive = more correlated on IOI

print(f"C_pooled range: [{C_pooled.min():.3f}, {C_pooled.max():.3f}]")
print(f"C_diff range:   [{C_diff.min():.3f}, {C_diff.max():.3f}]")

# ── helpers ───────────────────────────────────────────────────────────────────
role_sort_key = {r: i for i, r in enumerate(ROLE_ORDER)}
sort_order = sorted(range(N_HEAD),
                    key=lambda i: (role_sort_key[HEAD_ROLE[i]], i // 12, i % 12))
sorted_roles = [HEAD_ROLE[i] for i in sort_order]

boundaries = [0]
for k in range(1, N_HEAD):
    if sorted_roles[k] != sorted_roles[k-1]:
        boundaries.append(k)
boundaries.append(N_HEAD)
mid = [(boundaries[i] + boundaries[i+1]) // 2 for i in range(len(boundaries)-1)]
role_labels = [sorted_roles[boundaries[i]] for i in range(len(boundaries)-1)]


def plot_heatmap(C, title, fname):
    Cs = C[np.ix_(sort_order, sort_order)]
    np.fill_diagonal(Cs, 0.0)
    vmax = np.percentile(np.abs(Cs), 99)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(Cs, cmap="PRGn_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for b in boundaries[1:-1]:
        ax.axvline(b - 0.5, color="white", lw=1.0, alpha=0.8)
        ax.axhline(b - 0.5, color="white", lw=1.0, alpha=0.8)
    ax.set_xticks(mid)
    ax.set_xticklabels([SHORT[r] for r in role_labels], fontsize=9, rotation=45, ha="right")
    ax.set_yticks(mid)
    ax.set_yticklabels([SHORT[r] for r in role_labels], fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(f"figs/{fname}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → figs/{fname}")


def plot_network(C, title, fname, thresh_sigma=2.5):
    mask_off = ~np.eye(N_HEAD, dtype=bool)
    np.fill_diagonal(C, 0.0)   # ignore self-correlations
    C_off  = np.abs(C[mask_off])
    thresh = C_off.mean() + thresh_sigma * C_off.std()

    rows, cols = np.where((np.abs(C) > thresh) & mask_off)
    edges = [(r, c, C[r, c]) for r, c in zip(rows, cols) if r < c]
    edges.sort(key=lambda e: abs(e[2]), reverse=True)

    cc  = sum(1 for r,c,v in edges if HEAD_CIRCUIT[r] and HEAD_CIRCUIT[c])
    cnc = sum(1 for r,c,v in edges if HEAD_CIRCUIT[r] != HEAD_CIRCUIT[c])
    nn  = sum(1 for r,c,v in edges if not HEAD_CIRCUIT[r] and not HEAD_CIRCUIT[c])
    tot = len(edges)
    p_cc  = (len(ALL_CIRCUIT_HEADS) / N_HEAD) ** 2
    p_cnc = 2 * (len(ALL_CIRCUIT_HEADS) / N_HEAD) * (1 - len(ALL_CIRCUIT_HEADS) / N_HEAD)
    print(f"\n{title}")
    print(f"  threshold={thresh:.4f} ({thresh_sigma}σ), {tot} edges")
    print(f"  CC  {cc:3d}/{tot} = {cc/max(tot,1):.1%}  (chance {p_cc:.1%})")
    print(f"  CNC {cnc:3d}/{tot} = {cnc/max(tot,1):.1%}  (chance {p_cnc:.1%})")
    print(f"  NN  {nn:3d}/{tot} = {nn/max(tot,1):.1%}")
    print(f"  Top 10 edges:")
    for r, c, v in edges[:10]:
        tag = "CC" if HEAD_CIRCUIT[r] and HEAD_CIRCUIT[c] else \
              "CNC" if HEAD_CIRCUIT[r] or HEAD_CIRCUIT[c] else "NN"
        print(f"    {HEAD_NAMES[r]:8s} ({SHORT[HEAD_ROLE[r]]:5s}) ↔ "
              f"{HEAD_NAMES[c]:8s} ({SHORT[HEAD_ROLE[c]]:5s})  r={v:+.3f}  [{tag}]")

    if not edges:
        print("  No edges above threshold — skipping network plot")
        return

    active_nodes = sorted(set(r for r,c,v in edges) | set(c for r,c,v in edges))
    role_groups = defaultdict(list)
    for idx in active_nodes:
        role_groups[HEAD_ROLE[idx]].append(idx)

    node_angles = {}
    gap = 2 * math.pi / (len(role_groups) * 3 + len(active_nodes))
    angle = 0.0
    for role in ROLE_ORDER:
        if role not in role_groups: continue
        for idx in sorted(role_groups[role]):
            node_angles[idx] = angle
            angle += gap * 3
        angle += gap * 6

    R = 10.0
    pos = {idx: (R * math.cos(a), R * math.sin(a)) for idx, a in node_angles.items()}

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_aspect("equal"); ax.axis("off")

    max_abs = max(abs(v) for r,c,v in edges)
    for r, c, v in edges:
        if r not in pos or c not in pos: continue
        x0, y0 = pos[r]; x1, y1 = pos[c]
        alpha = 0.3 + 0.5 * abs(v) / max_abs
        lw    = 0.5 + 3.0 * abs(v) / max_abs
        color = "#8B0000" if v > 0 else "#003080"
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, alpha=alpha, zorder=1)

    for idx in active_nodes:
        if idx not in pos: continue
        x, y = pos[idx]
        color  = ROLE_COLORS[HEAD_ROLE[idx]]
        size   = 220 if HEAD_CIRCUIT[idx] else 110
        edge_c = "black" if HEAD_CIRCUIT[idx] else "gray"
        ax.scatter(x, y, s=size, c=color, edgecolors=edge_c, linewidths=1.5, zorder=3)
        lx, ly = x * 1.18, y * 1.18
        ax.text(lx, ly, HEAD_NAMES[idx], ha="center", va="center", fontsize=7.5,
                fontweight="bold" if HEAD_CIRCUIT[idx] else "normal")

    legend_patches = [mpatches.Patch(color=ROLE_COLORS[r], label=SHORT[r])
                      for r in ROLE_ORDER if any(HEAD_ROLE[i]==r for i in active_nodes)]
    legend_patches += [
        plt.Line2D([0],[0], color="#8B0000", lw=2, label="r > 0 (co-activate)"),
        plt.Line2D([0],[0], color="#003080", lw=2, label="r < 0 (anti-correlate)"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="lower right", framealpha=0.9)
    ax.set_title(f"{title}  (threshold = mean+{thresh_sigma}σ = {thresh:.3f})\n"
                 f"{tot} edges, {len(active_nodes)} nodes  |  large nodes = known circuit heads",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(f"figs/{fname}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → figs/{fname}")


# ── heatmaps ──────────────────────────────────────────────────────────────────
plot_heatmap(C_pooled.copy(),
             "Correlation of out_mag — all 2000 prompts pooled (no z-scoring)\n"
             "Circuit heads all high on IOI rows, low on non-IOI rows → they naturally co-correlate",
             "exp9_corr_pooled.png")

plot_heatmap(C_diff.copy(),
             "Differential correlation: C_ioi − C_non\n"
             "Positive = pairs that co-vary MORE on IOI prompts than on non-IOI prompts",
             "exp9_corr_diff.png")

# ── networks at increasing thresholds ─────────────────────────────────────────
for sigma in [2.5, 3.5, 5.0]:
    plot_network(C_diff.copy(),
                 f"Differential correlation network ({sigma}σ threshold)",
                 f"exp9_network_diff_{sigma}s.png",
                 thresh_sigma=sigma)

# ── per-head score and precision curve ────────────────────────────────────────
# For each head, compute: mean positive C_diff with all other heads.
# This summarises "how much did this head's co-activation with others increase on IOI?"
# Heads that fire together with other circuit heads on IOI should score highest.
np.fill_diagonal(C_diff, 0.0)
head_score = np.clip(C_diff, 0, None).mean(axis=1)   # (144,) — mean positive diff-corr

order = np.argsort(head_score)[::-1]   # highest first
circuit_mask = np.array(HEAD_CIRCUIT)
chance = len(ALL_CIRCUIT_HEADS) / N_HEAD

# Precision curve
prec = []
n_circ = 0
for k, idx in enumerate(order, 1):
    if circuit_mask[idx]: n_circ += 1
    prec.append(n_circ / k)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(np.arange(1, N_HEAD + 1), prec, color="#d62728", lw=2)
ax.axhline(chance, color="gray", lw=1.2, ls="--", label=f"chance ({chance:.1%})")
ax.set_xlabel("Top-k heads by mean positive ΔCorr", fontsize=11)
ax.set_ylabel("Precision (fraction that are circuit heads)", fontsize=11)
ax.set_title("Precision curve: ranking heads by mean positive differential correlation\n"
             "(for each head: average increase in co-activation with other heads on IOI vs non-IOI)",
             fontsize=10)
ax.legend(fontsize=10)
ax.set_xlim(1, N_HEAD); ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("figs/exp9_precision_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figs/exp9_precision_curve.png")

# Ranked head list
print(f"\nHeads ranked by mean positive ΔCorr (top 30):")
print(f"  {'Head':8s}  {'Role':20s}  {'Score':7s}  Circuit?")
for k, idx in enumerate(order[:30], 1):
    circ = "YES" if circuit_mask[idx] else ""
    print(f"  {k:2d}. {HEAD_NAMES[idx]:8s}  {HEAD_ROLE[idx]:20s}  {head_score[idx]:.4f}  {circ}")

# Summary: how many of the top-20 are circuit heads?
top20_circ = sum(circuit_mask[order[:20]])
print(f"\nTop 20: {top20_circ}/20 are circuit heads  "
      f"(expected by chance: {int(chance*20+0.5)}/20 = {chance:.1%})")
"""
Exp 13: Two projection approaches compared.

K = 9 (Marchenko-Pastur threshold for C_non, N=1000, P=144, bulk max ≈ 1.903).
Those 9 modes capture 85.3% of C_non variance — the NC layer-depth structure.

Approach A (Exp 12 with correct K):
  - Eigendecompose C_ioi → eigenvectors v_k
  - For each v_k, subtract the component in C_non's top-K subspace
  - Plot the residual loading profile

Approach B (new — project matrix first):
  - P = I - V_non_K V_non_K^T   (project away C_non's top-K modes)
  - C_ioi_clean = P C_ioi P      (remove shared NC structure from C_ioi)
  - Eigendecompose C_ioi_clean
  - The eigenvectors already live in the IOI-specific subspace

Figures:
  exp13_spectrum.png         — eigenvalue spectra: C_non, C_ioi, C_ioi_clean
  exp13_panel_A.png          — Approach A residual loading profiles (top-15 evecs)
  exp13_panel_B.png          — Approach B loading profiles (top-15 evecs of C_ioi_clean)
  exp13_best_A_{k}.png       — individual figures for best Approach A evecs
  exp13_best_B_{k}.png       — individual figures for best Approach B evecs

Run with: /opt/miniconda3/bin/python run_exp13_proj.py
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
legend_handles = [mpatches.Patch(color=ROLE_COLORS[r], label=SHORT[r]) for r in ROLE_ORDER]

# ── load data ─────────────────────────────────────────────────────────────────
assert os.path.exists("outmag_ioi.npy"), "Run run_exp8_outmag.py first"
outmag_ioi = np.load("outmag_ioi.npy")
outmag_non = np.load("outmag_non.npy")
C_ioi = np.corrcoef(outmag_ioi.T)
C_non = np.corrcoef(outmag_non.T)

# ── Marchenko-Pastur threshold ────────────────────────────────────────────────
N, P  = 1000, 144
gamma = P / N
mp_max = (1 + gamma**0.5)**2
K = int((np.linalg.eigvalsh(C_non) > mp_max).sum())   # number of signal modes
print(f"MP bulk max = {mp_max:.3f}  →  K = {K} signal modes in C_non")

# Eigendecompose C_non, sort descending
vals_non, vecs_non = np.linalg.eigh(C_non)
order_non = np.argsort(vals_non)[::-1]
vals_non  = vals_non[order_non]
vecs_non  = vecs_non[:, order_non]
V_non_K   = vecs_non[:, :K]           # (144, K) — top-K C_non eigenvectors

# Projection operators
P_non  = V_non_K @ V_non_K.T          # onto C_non subspace
P_perp = np.eye(N_HEAD) - P_non       # orthogonal complement

print(f"C_non top-{K} modes capture "
      f"{vals_non[:K].sum() / vals_non.sum():.1%} of C_non variance")

# Eigendecompose C_ioi, sort descending
vals_ioi, vecs_ioi = np.linalg.eigh(C_ioi)
order_ioi = np.argsort(vals_ioi)[::-1]
vals_ioi  = vals_ioi[order_ioi]
vecs_ioi  = vecs_ioi[:, order_ioi]

# ── Approach B: project matrix first, then decompose ─────────────────────────
C_ioi_clean = P_perp @ C_ioi @ P_perp
vals_clean, vecs_clean = np.linalg.eigh(C_ioi_clean)
order_clean = np.argsort(vals_clean)[::-1]
vals_clean  = vals_clean[order_clean]
vecs_clean  = vecs_clean[:, order_clean]

# ── helpers ───────────────────────────────────────────────────────────────────
xs = np.arange(N_HEAD)

def circ_in_topk(v, k=8):
    return sum(1 for i in np.argsort(np.abs(v))[::-1][:k] if HEAD_CIRCUIT[i])

def bar_plot(ax, v, title):
    ax.bar(xs, v, color=HEAD_COLOR, alpha=0.85, width=0.9)
    ax.axhline(0, color="k", lw=0.5)
    for i, is_circ in enumerate(HEAD_CIRCUIT):
        if is_circ:
            ax.axvline(i, color="k", lw=0.4, alpha=0.25, zorder=0)
    ax.set_xticks([l * 12 + 6 for l in range(12)])
    ax.set_xticklabels([f"L{l}" for l in range(12)], fontsize=8)
    ax.set_xlim(-0.5, N_HEAD - 0.5)
    ax.set_ylabel("loading", fontsize=8)
    ax.set_title(title, fontsize=8)

# ── Fig 1: eigenvalue spectra ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
k_show = 30
ax.plot(range(k_show), vals_non[:k_show],  "o-", color="#aaaaaa", lw=1.5, ms=4, label="C_non")
ax.plot(range(k_show), vals_ioi[:k_show],  "s-", color="#1f77b4", lw=1.5, ms=4, label="C_ioi")
ax.plot(range(k_show), vals_clean[:k_show],"^-", color="#d62728", lw=1.5, ms=4,
        label="C_ioi_clean  (C_non top-9 projected out)")
ax.axhline(mp_max, color="k", lw=1, ls="--", label=f"MP bulk max ({mp_max:.2f})")
ax.axvline(K - 0.5, color="k", lw=0.8, ls=":", alpha=0.5, label=f"K={K} cutoff")
ax.set_xlabel("Eigenvector index", fontsize=10)
ax.set_ylabel("Eigenvalue", fontsize=10)
ax.set_title("Eigenvalue spectra: C_non, C_ioi, and C_ioi with C_non structure projected out\n"
             "Dashed line = Marchenko-Pastur bulk max (noise threshold for N=1000, P=144)",
             fontsize=9)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("figs/exp13_spectrum.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figs/exp13_spectrum.png")

# ── Approach A: residual of C_ioi eigenvectors ────────────────────────────────
print(f"\nApproach A — project C_non out of C_ioi eigenvectors (K={K}):")
print(f"  {'k':>3}  {'λ_ioi':>6}  {'overlap':>7}  {'raw c/8':>7}  {'resid c/8':>10}")
resid_A = []
for k in range(N_HEAD):
    v       = vecs_ioi[:, k]
    v_resid = P_perp @ v
    overlap = float(np.dot(P_non @ v, P_non @ v))
    resid_A.append(v_resid)
    if k < 20:
        print(f"  {k:3d}  {vals_ioi[k]:6.2f}  {overlap:7.2f}  "
              f"{circ_in_topk(v):4d}/8  {circ_in_topk(v_resid):6d}/8")
resid_A = np.array(resid_A)

# panel
TOP = 15
fig, axes = plt.subplots(TOP, 2, figsize=(26, 3.5 * TOP), sharex=True)
for k in range(TOP):
    v       = vecs_ioi[:, k]
    v_resid = resid_A[k]
    overlap = float(np.dot(P_non @ v, P_non @ v))
    bar_plot(axes[k, 0], v,
             f"Evec {k} raw  λ={vals_ioi[k]:.2f}  circ/8={circ_in_topk(v)}")
    bar_plot(axes[k, 1], v_resid,
             f"Evec {k} residual  overlap={overlap:.2f}  circ/8={circ_in_topk(v_resid)}")
axes[0, 0].legend(handles=legend_handles, fontsize=7, loc="upper right", ncol=4)
plt.suptitle(f"Approach A: residuals of C_ioi eigenvectors after projecting out C_non top-{K}",
             fontsize=10, y=1.001)
plt.tight_layout()
plt.savefig("figs/exp13_panel_A.png", dpi=80, bbox_inches="tight")
plt.close()
print(f"  → figs/exp13_panel_A.png")

# ── Approach B: eigenvectors of C_ioi_clean ───────────────────────────────────
print(f"\nApproach B — eigenvectors of C_ioi_clean (C_non top-{K} projected out of matrix):")
print(f"  {'k':>3}  {'λ_clean':>8}  {'circ/8':>7}")
for k in range(20):
    v = vecs_clean[:, k]
    print(f"  {k:3d}  {vals_clean[k]:8.4f}  {circ_in_topk(v)}/8")

fig, axes = plt.subplots(TOP, 1, figsize=(20, 3.5 * TOP), sharex=True)
for k in range(TOP):
    v = vecs_clean[:, k]
    c = circ_in_topk(v)
    bar_plot(axes[k], v,
             f"C_ioi_clean evec {k}  λ={vals_clean[k]:.4f}  "
             f"circ/top8={c}/8  (chance {CHANCE*8:.1f})")
axes[0].legend(handles=legend_handles, fontsize=7, loc="upper right", ncol=4)
plt.suptitle(
    f"Approach B: eigenvectors of C_ioi_clean\n"
    f"(C_non top-{K} modes projected out of C_ioi before decomposing)",
    fontsize=10, y=1.001
)
plt.tight_layout()
plt.savefig("figs/exp13_panel_B.png", dpi=80, bbox_inches="tight")
plt.close()
print(f"  → figs/exp13_panel_B.png")

# ── best individual figures ───────────────────────────────────────────────────
def save_individual(v, title, fname):
    c = circ_in_topk(v)
    order = np.argsort(np.abs(v))[::-1]
    fig, ax = plt.subplots(figsize=(20, 4))
    bar_plot(ax, v, title + f"\n{c}/8 circuit heads in top-8  (chance {CHANCE*8:.1f}/8)")
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right", ncol=4, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  {fname}  [{c}/8 circuit]")
    print(f"  Top-12 loaders:")
    for rank, i in enumerate(order[:12], 1):
        tag = "CIRCUIT" if HEAD_CIRCUIT[i] else ""
        print(f"    {rank:2d}. {HEAD_NAMES[i]:8s} ({SHORT[HEAD_ROLE[i]]:5s})  "
              f"v={v[i]:+.4f}  {tag}")

# Approach A: best by residual enrichment
best_A = sorted(range(20), key=lambda k: circ_in_topk(resid_A[k]), reverse=True)[:5]
print(f"\nApproach A best evecs (by residual circ/8): {best_A}")
for k in best_A:
    v_resid = resid_A[k]
    overlap = float(np.dot(P_non @ vecs_ioi[:, k], P_non @ vecs_ioi[:, k]))
    save_individual(
        v_resid,
        f"Approach A: C_ioi evec {k} residual  "
        f"(λ_ioi={vals_ioi[k]:.2f}, C_non overlap={overlap:.2f})",
        f"figs/exp13_A_{k:02d}.png"
    )

# Approach B: top by eigenvalue
print(f"\nApproach B top evecs (by eigenvalue):")
for k in range(8):
    v = vecs_clean[:, k]
    save_individual(
        v,
        f"Approach B: C_ioi_clean evec {k}  λ={vals_clean[k]:.4f}",
        f"figs/exp13_B_{k:02d}.png"
    )

# ── summary comparison ────────────────────────────────────────────────────────
print("\n\n── Summary ──────────────────────────────────────────────────────────────")
print("Approach A (residual of C_ioi evecs):")
for k in range(15):
    v       = vecs_ioi[:, k]
    v_resid = resid_A[k]
    overlap = float(np.dot(P_non @ v, P_non @ v))
    c_raw   = circ_in_topk(v)
    c_resid = circ_in_topk(v_resid)
    marker  = " ← interesting" if c_resid >= 4 and c_resid > c_raw else ""
    print(f"  k={k:2d}  λ={vals_ioi[k]:5.2f}  overlap={overlap:.2f}  "
          f"raw={c_raw}/8  resid={c_resid}/8{marker}")

print("\nApproach B (eigenvecs of C_ioi_clean):")
for k in range(15):
    v = vecs_clean[:, k]
    c = circ_in_topk(v)
    marker = " ← interesting" if c >= 4 else ""
    print(f"  k={k:2d}  λ={vals_clean[k]:7.4f}  circ/8={c}/8{marker}")
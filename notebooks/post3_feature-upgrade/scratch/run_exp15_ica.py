"""
Exp 15: Contrastive ICA.

Pipeline:
  1. ICA on outmag_non  → K baseline independent components (mixing matrix A_non).
     These are the independent activation patterns of non-circuit heads.
  2. Project outmag_ioi onto the orthogonal complement of A_non's column space.
     This removes baseline NC patterns from the IOI feature matrix.
  3. ICA on the residual outmag_ioi → IOI-specific independent components.
     These are sources that (a) don't appear in the baseline and (b) are
     mutually independent — a stronger claim than PCA orthogonality.

K_non = 9  (Marchenko-Pastur threshold for N=1000, P=144)
K_ioi = scan over 2..8  (we don't know how many IOI-specific sources there are)

Stability check: run FastICA 10 times with different seeds, keep components
that appear consistently (high mean pairwise absolute correlation with the
same component across runs).

Figures:
  exp15_mixing_non.png     — ICA baseline mixing matrix (heatmap, heads × components)
  exp15_ica_ioi_{k}.png   — IOI-specific ICA component loading profiles
  exp15_stability.png      — component stability across random seeds

Run with: /opt/miniconda3/bin/python run_exp15_ica.py
"""

import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
from sklearn.decomposition import FastICA
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import linear_sum_assignment

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figs", exist_ok=True)

# ── metadata ──────────────────────────────────────────────────────────────────
IOI_HEADS = {
    "Name Mover":          [(9,6),(9,9),(10,0)],
    "Backup Name Mover":   [(9,0),(9,7),(10,1),(10,2),(10,6),(10,10),(11,2),(11,9)],
    "Negative Name Mover": [(10,7),(11,10)],
    "S-Inhibition":        [(7,3),(7,9),(8,6),(8,10)],
    "Induction":           [(5,5),(6,9)],
    "Duplicate Token":     [(0,1),(3,0)],
    "Previous Token":      [(2,2),(4,11)],
}
ROLE_ORDER = ["Name Mover","Backup Name Mover","Negative Name Mover",
              "S-Inhibition","Induction","Duplicate Token","Previous Token","Non-circuit"]
ROLE_COLORS = {
    "Name Mover":"#d62728","Backup Name Mover":"#ff7f0e",
    "Negative Name Mover":"#9467bd","S-Inhibition":"#2ca02c",
    "Induction":"#17becf","Duplicate Token":"#bcbd22",
    "Previous Token":"#e377c2","Non-circuit":"#cccccc",
}
SHORT = {"Name Mover":"NM","Backup Name Mover":"BNM","Negative Name Mover":"NegNM",
         "S-Inhibition":"SI","Induction":"Ind","Duplicate Token":"DT",
         "Previous Token":"PT","Non-circuit":"NC"}
ALL_CIRCUIT = set(lh for hs in IOI_HEADS.values() for lh in hs)
HL_FLAT = [(l,h) for l in range(12) for h in range(12)]
def role(l,h):
    for r,hs in IOI_HEADS.items():
        if (l,h) in hs: return r
    return "Non-circuit"
HEAD_NAMES   = [f"L{l}H{h}" for l,h in HL_FLAT]
HEAD_CIRCUIT = np.array([lh in ALL_CIRCUIT for lh in HL_FLAT])
HEAD_ROLE    = [role(*lh) for lh in HL_FLAT]
HEAD_COLOR   = [ROLE_COLORS[r] for r in HEAD_ROLE]
N = 144; N_CIRC = HEAD_CIRCUIT.sum(); CHANCE = N_CIRC / N
legend_handles = [mpatches.Patch(color=ROLE_COLORS[r], label=SHORT[r]) for r in ROLE_ORDER]

# ── load data ─────────────────────────────────────────────────────────────────
outmag_ioi = np.load("outmag_ioi.npy")   # (1000, 144)
outmag_non = np.load("outmag_non.npy")   # (1000, 144)
print(f"Data: outmag_ioi {outmag_ioi.shape}, outmag_non {outmag_non.shape}")

# ── Step 1: Choose K_non from permutation test, then ICA on outmag_non ────────
N_SEEDS = 20
N_PERM_NON = 200
rng_non = np.random.default_rng(42)

print(f"\nStep 1: Permutation test for K_non ({N_PERM_NON} shuffles)...")
C_non_full = np.corrcoef(outmag_non.T)
vals_non_full = np.sort(np.linalg.eigvalsh(C_non_full))[::-1]
null_max_non = []
for _ in range(N_PERM_NON):
    X_shuf = outmag_non.copy()
    for j in range(X_shuf.shape[1]):
        rng_non.shuffle(X_shuf[:, j])
    null_max_non.append(np.linalg.eigvalsh(np.corrcoef(X_shuf.T)).max())
null_thresh_non = np.percentile(null_max_non, 95)
K_NON = int((vals_non_full > null_thresh_non).sum())
print(f"  Null 95th percentile: {null_thresh_non:.3f}  →  K_non = {K_NON}")

print(f"  Running FastICA K={K_NON}, {N_SEEDS} seeds for stability...")

def run_ica(X, K, seed):
    ica = FastICA(n_components=K, random_state=seed, max_iter=2000, tol=1e-4)
    ica.fit(X)
    # Normalize columns of mixing matrix to unit norm
    A = ica.mixing_                    # (n_features, K)
    A = A / np.linalg.norm(A, axis=0) # unit columns
    return A

# Run multiple times, match components across runs using Hungarian algorithm
A_non_runs = [run_ica(outmag_non, K_NON, seed) for seed in range(N_SEEDS)]

def match_to_reference(A_ref, A_new):
    """Match columns of A_new to A_ref by max absolute correlation, return reordered A_new."""
    corr = np.abs(A_ref.T @ A_new)    # (K, K)
    row_ind, col_ind = linear_sum_assignment(-corr)
    A_matched = A_new[:, col_ind]
    # Flip signs to match reference
    for k in range(A_ref.shape[1]):
        if A_ref[:, k] @ A_matched[:, k] < 0:
            A_matched[:, k] *= -1
    return A_matched, corr[row_ind, col_ind]

A_non_ref = A_non_runs[0]
matched = [A_non_ref]
match_corrs = []
for A in A_non_runs[1:]:
    A_m, corr_vals = match_to_reference(A_non_ref, A)
    matched.append(A_m)
    match_corrs.append(corr_vals)

A_non_stack = np.stack(matched, axis=0)   # (N_SEEDS, 144, K_NON)
A_non_mean  = A_non_stack.mean(axis=0)    # (144, K_NON) — mean mixing matrix
A_non_std   = A_non_stack.std(axis=0)     # (144, K_NON) — variability across seeds

stability_non = np.array(match_corrs).mean(axis=0)   # mean |corr| per component
print(f"  Baseline ICA component stability (mean |corr| across seeds):")
for k in range(K_NON):
    print(f"    Component {k}: {stability_non[k]:.3f}")

# Use mean mixing matrix as our estimate
A_non = A_non_mean   # (144, K_NON), unit-norm columns

# ── Step 2: Project outmag_ioi onto complement of A_non column space ──────────
# The column space of A_non spans the baseline independent source patterns.
# We project each prompt's feature vector onto the orthogonal complement.
# Note: A_non columns are already unit-norm but may not be orthogonal (ICA
# doesn't require orthogonality). Use QR to get an orthonormal basis.
Q_non, _ = np.linalg.qr(A_non)   # Q_non: (144, K_NON) orthonormal basis for col(A_non)
P_non  = Q_non @ Q_non.T          # (144, 144) projection onto A_non subspace
P_perp = np.eye(N) - P_non        # (144, 144) orthogonal complement

# Project: apply P_perp to the feature dimension of each prompt
outmag_ioi_resid = outmag_ioi @ P_perp   # (1000, 144), baseline patterns removed
print(f"\nStep 2: Projected outmag_ioi onto complement of baseline ICA space")
print(f"  Variance retained: {outmag_ioi_resid.var() / outmag_ioi.var():.1%}")

# ── Step 3: Choose K_ioi from permutation test on residual ───────────────────
# Shuffle each column independently to destroy correlations while preserving
# marginals; 95th percentile of null max eigenvalue is the noise floor.
N_PERM = 200
rng = np.random.default_rng(0)

C_ioi_resid = np.corrcoef(outmag_ioi_resid.T)   # (144, 144)
vals_resid   = np.sort(np.linalg.eigvalsh(C_ioi_resid))[::-1]

print(f"\nStep 3: Permutation test for K_ioi ({N_PERM} shuffles)...")
null_max = []
for _ in range(N_PERM):
    X_shuf = outmag_ioi_resid.copy()
    for j in range(X_shuf.shape[1]):
        rng.shuffle(X_shuf[:, j])
    null_max.append(np.linalg.eigvalsh(np.corrcoef(X_shuf.T)).max())

null_thresh = np.percentile(null_max, 95)
K_IOI = int((vals_resid > null_thresh).sum())
print(f"  Null 95th percentile: {null_thresh:.3f}")
print(f"  K_ioi = {K_IOI}  (MP alternative: "
      f"{(1 + ((N - K_NON) / outmag_ioi_resid.shape[0])**0.5)**2:.3f})")

# ICA on residual with principled K
print(f"  Running FastICA K={K_IOI}, {N_SEEDS} seeds for stability...")
A_ioi_runs  = [run_ica(outmag_ioi_resid, K_IOI, seed) for seed in range(N_SEEDS)]
A_ioi_ref   = A_ioi_runs[0]
matched_ioi = [A_ioi_ref]
corrs_ioi   = []
for A in A_ioi_runs[1:]:
    A_m, cv = match_to_reference(A_ioi_ref, A)
    matched_ioi.append(A_m)
    corrs_ioi.append(cv)
stability_ioi = np.array(corrs_ioi).mean(axis=0)
A_ioi = np.stack(matched_ioi).mean(axis=0)   # (144, K_IOI)

def auc(v): return roc_auc_score(HEAD_CIRCUIT, np.abs(v))
auc_vals = [auc(A_ioi[:, k]) for k in range(K_IOI)]
print(f"  Stability: {[f'{s:.3f}' for s in stability_ioi]}")
print(f"  AUC:       {[f'{a:.3f}' for a in auc_vals]}")

# ── Post-2 candidate heads: loadings across all IOI components ────────────────
candidates = [(8, 1), (8, 11)]
print(f"\nPost-2 candidate head loadings across IOI components:")
print(f"  {'Head':>6}  " + "  ".join(f"C{k:02d}" for k in range(K_IOI)))
for l, h in candidates:
    idx = l * 12 + h
    row = "  ".join(f"{A_ioi[idx, k]:+.3f}" for k in range(K_IOI))
    print(f"  L{l}H{h:>2}  {row}")

# ── Figures ───────────────────────────────────────────────────────────────────
print(f"\nK_ioi = {K_IOI} (permutation test, no label tuning)")

xs = np.arange(N)

def plot_component(v, title, fname):
    a = auc(v)
    order = np.argsort(np.abs(v))[::-1]
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.bar(xs, v, color=HEAD_COLOR, alpha=0.85, width=0.9)
    ax.axhline(0, color="k", lw=0.6)
    for i, ic in enumerate(HEAD_CIRCUIT):
        if ic: ax.axvline(i, color="k", lw=0.5, alpha=0.3, zorder=0)
    ax.set_xticks([l*12+6 for l in range(12)])
    ax.set_xticklabels([f"L{l}" for l in range(12)], fontsize=9)
    ax.set_xlim(-0.5, N-0.5)
    ax.set_ylabel("Mixing coefficient (head loading)", fontsize=10)
    ax.set_title(f"{title}\nAUC={a:.3f}  (chance=0.500)", fontsize=9)
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right", ncol=4, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  {fname}  [AUC={a:.3f}]")
    print(f"  Top-12 loaders:")
    for rank, i in enumerate(order[:12], 1):
        tag = "CIRCUIT" if HEAD_CIRCUIT[i] else ""
        print(f"    {rank:2d}. {HEAD_NAMES[i]:8s} ({SHORT[HEAD_ROLE[i]]:5s})"
              f"  a={v[i]:+.4f}  {tag}")

# IOI-specific components
for k in range(K_IOI):
    plot_component(
        A_ioi[:, k],
        f"Contrastive ICA: IOI-specific component {k}  "
        f"(stability={stability_ioi[k]:.3f})",
        f"figs/exp15_ica_ioi_{k:02d}.png"
    )

# Panel of all IOI-specific components
fig, axes = plt.subplots(K_IOI, 1, figsize=(20, 4*K_IOI), sharex=True)
if K_IOI == 1: axes = [axes]
for k, ax in enumerate(axes):
    v = A_ioi[:, k]; a = auc(v)
    ax.bar(xs, v, color=HEAD_COLOR, alpha=0.85, width=0.9)
    ax.axhline(0, color="k", lw=0.5)
    for i, ic in enumerate(HEAD_CIRCUIT):
        if ic: ax.axvline(i, color="k", lw=0.4, alpha=0.25, zorder=0)
    ax.set_xticks([l*12+6 for l in range(12)])
    ax.set_xticklabels([f"L{l}" for l in range(12)], fontsize=8)
    ax.set_xlim(-0.5, N-0.5)
    ax.set_ylabel("loading", fontsize=8)
    ax.set_title(f"ICA IOI component {k}  stab={stability_ioi[k]:.3f}  "
                 f"AUC={a:.3f}", fontsize=9)
axes[0].legend(handles=legend_handles, fontsize=7, loc="upper right", ncol=4)
plt.suptitle(
    f"Contrastive ICA: IOI-specific independent components (K_non={K_NON}, K_ioi={K_IOI})\n"
    f"Baseline ICA patterns projected out before decomposing IOI data",
    fontsize=10, y=1.001
)
plt.tight_layout()
plt.savefig("figs/exp15_ica_panel.png", dpi=100, bbox_inches="tight")
plt.close()
print(f"\n  → figs/exp15_ica_panel.png")

# Baseline mixing matrix heatmap
role_sort = sorted(range(N), key=lambda i: (ROLE_ORDER.index(HEAD_ROLE[i]), i))
fig, ax = plt.subplots(figsize=(K_NON * 1.2 + 2, 10))
im = ax.imshow(A_non[role_sort, :], aspect="auto", cmap="RdBu_r",
               vmin=-np.percentile(np.abs(A_non), 99),
               vmax= np.percentile(np.abs(A_non), 99))
ax.set_xticks(range(K_NON))
ax.set_xticklabels([f"NC-{k}" for k in range(K_NON)], fontsize=9)
# mark role boundaries
sorted_roles = [HEAD_ROLE[i] for i in role_sort]
boundaries = [0]
for j in range(1, N):
    if sorted_roles[j] != sorted_roles[j-1]: boundaries.append(j)
boundaries.append(N)
for b in boundaries[1:-1]:
    ax.axhline(b - 0.5, color="white", lw=1.0)
mid = [(boundaries[j]+boundaries[j+1])//2 for j in range(len(boundaries)-1)]
role_lbls = [sorted_roles[boundaries[j]] for j in range(len(boundaries)-1)]
ax.set_yticks(mid)
ax.set_yticklabels([SHORT[r] for r in role_lbls], fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.03)
ax.set_title(f"Baseline ICA mixing matrix (K={K_NON})\nRows=heads (sorted by role), "
             f"Cols=independent NC source patterns", fontsize=9)
plt.tight_layout()
plt.savefig("figs/exp15_mixing_non.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figs/exp15_mixing_non.png")

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n\n── Summary ─────────────────────────────────────────────────────────────────")
print(f"Contrastive ICA  K_non={K_NON}  K_ioi={K_IOI}")
print(f"  {'Comp':>5}  {'Stability':>9}  {'AUC':>7}")
for k in range(K_IOI):
    print(f"  {k:5d}  {stability_ioi[k]:9.3f}  {auc_vals[k]:.3f}")

# ── ROC curves panel ──────────────────────────────────────────────────────────
ncols = 4
nrows = (K_IOI + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
axes_flat = axes.flatten() if nrows > 1 else list(axes) if ncols > 1 else [axes]
for k in range(K_IOI):
    ax = axes_flat[k]
    fpr, tpr, _ = roc_curve(HEAD_CIRCUIT, np.abs(A_ioi[:, k]))
    ax.plot(fpr, tpr, color="#d62728", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title(f"Comp {k}  AUC={auc_vals[k]:.3f}  stab={stability_ioi[k]:.3f}",
                 fontsize=9)
    ax.set_xlabel("FPR", fontsize=8); ax.set_ylabel("TPR", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
for ax in axes_flat[K_IOI:]:
    ax.axis("off")
plt.suptitle(
    f"ROC curves: circuit vs non-circuit discrimination per ICA component\n"
    f"Score = |loading|, positives = known circuit heads (n={N_CIRC})",
    fontsize=10
)
plt.tight_layout()
plt.savefig("figs/exp15_roc.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figs/exp15_roc.png")
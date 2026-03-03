"""
Exp 16: Contrastive ICA on Var_v — comparison with out_mag (Exp 15).

Same pipeline as Exp 15, feature matrix swapped from out_mag to Var_v.
Var_v = E_pi[||v||^2] - ||mu_v||^2  (variance of value vectors under attention)
out_mag = ||mu_v||^2 / d            (squared magnitude of attention-weighted mean value)

Purpose: check whether out_mag's better single-head discrimination (Exp 8: 30x ratio
vs 12x for Var_v) also translates to better ICA separation. If AUC is similar,
the ICA is robust to feature choice; if out_mag wins, use it for the post.

Run with: /opt/miniconda3/bin/python run_exp16_varv.py
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
feat_ioi = np.load("varv_ioi.npy")   # (1000, 144)
feat_non = np.load("varv_non.npy")   # (1000, 144)
print(f"Data: varv_ioi {feat_ioi.shape}, varv_non {feat_non.shape}")

# ── helpers ───────────────────────────────────────────────────────────────────
N_SEEDS = 20

def run_ica(X, K, seed):
    ica = FastICA(n_components=K, random_state=seed, max_iter=2000, tol=1e-4)
    ica.fit(X)
    A = ica.mixing_
    A = A / np.linalg.norm(A, axis=0)
    return A

def match_to_reference(A_ref, A_new):
    corr = np.abs(A_ref.T @ A_new)
    row_ind, col_ind = linear_sum_assignment(-corr)
    A_matched = A_new[:, col_ind]
    for k in range(A_ref.shape[1]):
        if A_ref[:, k] @ A_matched[:, k] < 0:
            A_matched[:, k] *= -1
    return A_matched, corr[row_ind, col_ind]

def perm_K(X, n_perm=200, seed=0):
    """Permutation test: return number of eigenvalues above 95th pct null max."""
    rng = np.random.default_rng(seed)
    C = np.corrcoef(X.T)
    vals = np.sort(np.linalg.eigvalsh(C))[::-1]
    null_max = []
    for _ in range(n_perm):
        Xs = X.copy()
        for j in range(Xs.shape[1]): rng.shuffle(Xs[:, j])
        null_max.append(np.linalg.eigvalsh(np.corrcoef(Xs.T)).max())
    thresh = np.percentile(null_max, 95)
    return int((vals > thresh).sum()), thresh

def auc(v): return roc_auc_score(HEAD_CIRCUIT, np.abs(v))

# ── Step 1: K_non from permutation test, ICA on feat_non ─────────────────────
print(f"\nStep 1: Permutation test for K_non (200 shuffles)...")
K_NON, thresh_non = perm_K(feat_non, seed=42)
print(f"  Null 95th percentile: {thresh_non:.3f}  →  K_non = {K_NON}")

print(f"  Running FastICA K={K_NON}, {N_SEEDS} seeds...")
A_non_runs = [run_ica(feat_non, K_NON, seed) for seed in range(N_SEEDS)]
A_non_ref = A_non_runs[0]
matched = [A_non_ref]; match_corrs = []
for A in A_non_runs[1:]:
    A_m, cv = match_to_reference(A_non_ref, A)
    matched.append(A_m); match_corrs.append(cv)
stability_non = np.array(match_corrs).mean(axis=0)
A_non = np.stack(matched).mean(axis=0)
print(f"  Baseline stability: {[f'{s:.3f}' for s in stability_non]}")

# ── Step 2: project feat_ioi onto complement of A_non column space ────────────
Q_non, _ = np.linalg.qr(A_non)
P_perp = np.eye(N) - Q_non @ Q_non.T
feat_ioi_resid = feat_ioi @ P_perp
print(f"\nStep 2: Variance retained: {feat_ioi_resid.var() / feat_ioi.var():.1%}")

# ── Step 3: K_ioi from permutation test, ICA on residual ─────────────────────
print(f"\nStep 3: Permutation test for K_ioi (200 shuffles)...")
K_IOI, thresh_ioi = perm_K(feat_ioi_resid, seed=0)
print(f"  Null 95th percentile: {thresh_ioi:.3f}  →  K_ioi = {K_IOI}")

print(f"  Running FastICA K={K_IOI}, {N_SEEDS} seeds...")
A_ioi_runs = [run_ica(feat_ioi_resid, K_IOI, seed) for seed in range(N_SEEDS)]
A_ioi_ref = A_ioi_runs[0]
matched_ioi = [A_ioi_ref]; corrs_ioi = []
for A in A_ioi_runs[1:]:
    A_m, cv = match_to_reference(A_ioi_ref, A)
    matched_ioi.append(A_m); corrs_ioi.append(cv)
stability_ioi = np.array(corrs_ioi).mean(axis=0)
A_ioi = np.stack(matched_ioi).mean(axis=0)

auc_vals = [auc(A_ioi[:, k]) for k in range(K_IOI)]
print(f"  Stability: {[f'{s:.3f}' for s in stability_ioi]}")
print(f"  AUC:       {[f'{a:.3f}' for a in auc_vals]}")

# ── Panel figure ──────────────────────────────────────────────────────────────
xs = np.arange(N)
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
    ax.set_title(f"Comp {k}  stab={stability_ioi[k]:.3f}  AUC={a:.3f}", fontsize=9)
axes[0].legend(handles=legend_handles, fontsize=7, loc="upper right", ncol=4)
plt.suptitle(
    f"Contrastive ICA on Var_v (Exp 16)  K_non={K_NON}  K_ioi={K_IOI}\n"
    f"Compare Exp 15 (out_mag): AUC range 0.664–0.772",
    fontsize=10, y=1.001
)
plt.tight_layout()
plt.savefig("figs/exp16_varv_panel.png", dpi=100, bbox_inches="tight")
plt.close()
print(f"\n  → figs/exp16_varv_panel.png")

# ── ROC curves ────────────────────────────────────────────────────────────────
ncols = 4
nrows = (K_IOI + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
axes_flat = axes.flatten() if nrows > 1 else list(axes) if ncols > 1 else [axes]
for k in range(K_IOI):
    ax = axes_flat[k]
    fpr, tpr, _ = roc_curve(HEAD_CIRCUIT, np.abs(A_ioi[:, k]))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2)
    ax.plot([0,1],[0,1],"k--",lw=0.8)
    ax.set_title(f"Comp {k}  AUC={auc_vals[k]:.3f}  stab={stability_ioi[k]:.3f}", fontsize=9)
    ax.set_xlabel("FPR", fontsize=8); ax.set_ylabel("TPR", fontsize=8)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
for ax in axes_flat[K_IOI:]: ax.axis("off")
plt.suptitle("ROC curves: Var_v contrastive ICA (Exp 16)", fontsize=10)
plt.tight_layout()
plt.savefig("figs/exp16_varv_roc.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figs/exp16_varv_roc.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n\n── Summary ────────────────────────────────────────────────────────────────")
print(f"Var_v contrastive ICA  K_non={K_NON}  K_ioi={K_IOI}")
print(f"  {'Comp':>5}  {'Stability':>9}  {'AUC':>7}")
for k in range(K_IOI):
    print(f"  {k:5d}  {stability_ioi[k]:9.3f}  {auc_vals[k]:.3f}")
print(f"\n  AUC range: {min(auc_vals):.3f} – {max(auc_vals):.3f}")
print(f"  out_mag (Exp 15): 0.664 – 0.772  (K_ioi=12)")
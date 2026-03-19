"""
Post 4 — Experiment 15: Effective channel rank of the L0H3 → L1H6 induction circuit

The K-composition matrix for the induction circuit is:
  M = W_K[1,6].T @ |G| @ W_O[0,3].T    (d_head x d_head = 64 x 64)

where |G| = positive-definite version of the induction head's content metric.

SVD of M gives singular values sigma_1 >= sigma_2 >= ... >= sigma_64.
Effective rank = (sum sigma_i)^2 / sum sigma_i^2   (participation ratio)
  = 1 if rank-1, = 64 if uniform.

Questions:
1. How many effective modes does the L0H3 -> L1H6 channel communicate through?
2. How does this compare to random (d_head x d_head) baseline?
3. Does G-weighting sharpen or broaden the rank vs. identity metric?

Run with: /opt/miniconda3/bin/python run_exp15_channel_rank.py
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
d_model  = model.cfg.d_model
d_head   = model.cfg.d_head

W_Q = model.W_Q.cpu().numpy()   # (n_layers, n_heads, d_model, d_head)
W_K = model.W_K.cpu().numpy()
W_V = model.W_V.cpu().numpy()
W_O = model.W_O.cpu().numpy()   # (n_layers, n_heads, d_head, d_model)
W_E = model.W_E.cpu().numpy()

h_induction = 6
h_prev      = 3

# ── Build |G| from induction head ─────────────────────────────────────────────

WQK_ind = W_Q[1, h_induction] @ W_K[1, h_induction].T
G_raw   = (WQK_ind + WQK_ind.T) / 2.0
eigvals_G, eigvecs_G = np.linalg.eigh(G_raw)
G_abs   = eigvecs_G @ np.diag(np.abs(eigvals_G)) @ eigvecs_G.T   # |G|, positive semidefinite

# BOS removal
bos_id = model.tokenizer.bos_token_id
v_bos  = W_E[bos_id] / (np.linalg.norm(W_E[bos_id]) + 1e-10)
P_bos  = np.eye(d_model) - np.outer(v_bos, v_bos)
G_abs  = P_bos @ G_abs @ P_bos

print(f"|G| Frobenius norm: {np.linalg.norm(G_abs):.3f}")

# ── K-composition matrices ─────────────────────────────────────────────────────

# All L0 → L1 pairs
WK_1 = W_K[1]    # (n_heads, d_model, d_head)
WO_0 = W_O[0]    # (n_heads, d_head, d_model)

# Standard K-comp matrix (identity metric): W_K.T @ W_O.T = (d_head, d_model) @ (d_model, d_head)
# G-weighted K-comp matrix: W_K.T @ G @ W_O.T

def comp_matrices(WK, WO, G=None):
    """
    WK: (d_model, d_head), WO: (d_head, d_model)
    Returns (d_head, d_head) composition matrix.
    Standard: WK.T @ WO.T
    G-weighted: WK.T @ G @ WO.T
    """
    if G is None:
        return WK.T @ WO.T
    else:
        return WK.T @ G @ WO.T

def effective_rank(sv):
    """Participation ratio: (sum sv)^2 / sum sv^2"""
    sv = sv[sv > 1e-10]
    return (sv.sum() ** 2) / (sv ** 2).sum()

def stable_rank(sv):
    """Stable rank: sum sv_i^2 / sv_1^2 (less sensitive to noise floor)"""
    return (sv ** 2).sum() / (sv[0] ** 2)

# ── Analyze the primary induction circuit: L0H3 → L1H6 ───────────────────────

M_std = comp_matrices(WK_1[h_induction], WO_0[h_prev])
M_G   = comp_matrices(WK_1[h_induction], WO_0[h_prev], G_abs)

sv_std = np.linalg.svd(M_std, compute_uv=False)
sv_G   = np.linalg.svd(M_G,   compute_uv=False)

print(f"\n=== L0H{h_prev} → L1H{h_induction} (induction circuit) ===")
print(f"Standard (identity metric):")
print(f"  Top-5 SVs: {sv_std[:5].round(3)}")
print(f"  Effective rank: {effective_rank(sv_std):.2f}")
print(f"  Stable rank:    {stable_rank(sv_std):.2f}")

print(f"\nG-weighted:")
print(f"  Top-5 SVs: {sv_G[:5].round(3)}")
print(f"  Effective rank: {effective_rank(sv_G):.2f}")
print(f"  Stable rank:    {stable_rank(sv_G):.2f}")

# ── Compare all L0 → L1 pairs ─────────────────────────────────────────────────

print(f"\n=== All L0 → L1 pairs: effective rank (G-weighted) ===")
print(f"{'Pair':>12s}  {'std_rank':>9s}  {'G_rank':>8s}  {'kcomp_G':>9s}")

def kcomp_G_score(WO, WK, G):
    M = WK.T @ G @ WO.T
    numer = np.linalg.norm(M, 'fro')
    norm_O = np.sqrt(np.trace(WO @ G @ WO.T) + 1e-10)
    norm_K = np.sqrt(np.trace(WK.T @ G @ WK) + 1e-10)
    return numer / (norm_O * norm_K + 1e-10)

for l0h in range(n_heads):
    for l1h in range(n_heads):
        M_s = comp_matrices(WK_1[l1h], WO_0[l0h])
        M_g = comp_matrices(WK_1[l1h], WO_0[l0h], G_abs)
        sv_s = np.linalg.svd(M_s, compute_uv=False)
        sv_g = np.linalg.svd(M_g, compute_uv=False)
        kc   = kcomp_G_score(WO_0[l0h], WK_1[l1h], G_abs)
        tag = " ← induction circuit" if l0h == h_prev and l1h == h_induction else ""
        if kc > 0.08 or (l0h == h_prev and l1h == h_induction):
            print(f"  L0H{l0h}→L1H{l1h}:  {effective_rank(sv_s):8.2f}  {effective_rank(sv_g):8.2f}  {kc:9.4f}{tag}")

# ── Figure: SV spectrum for key circuits ──────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1: SV spectrum, L0H3 → L1H6, standard vs G-weighted
ax = axes[0]
k = np.arange(1, d_head + 1)
ax.semilogy(k, sv_std / sv_std[0], 'o-', lw=1.5, ms=4, label='Standard (identity)', color='#1f77b4')
ax.semilogy(k, sv_G   / sv_G[0],   's-', lw=1.5, ms=4, label='G-weighted',          color='#d62728')
ax.axhline(1 / d_head, color='gray', lw=1, linestyle='--', label='uniform (rank=64)')
ax.set_xlabel("SV rank", fontsize=9)
ax.set_ylabel("Normalized SV (log scale)", fontsize=9)
ax.set_title(f"L0H{h_prev} → L1H{h_induction} induction circuit\nSV spectrum: standard vs G-weighted",
             fontsize=8)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# Panel 2: Effective rank heatmap, G-weighted, all L0→L1 pairs
eff_rank_G = np.zeros((n_heads, n_heads))
for l0h in range(n_heads):
    for l1h in range(n_heads):
        M_g = comp_matrices(WK_1[l1h], WO_0[l0h], G_abs)
        sv  = np.linalg.svd(M_g, compute_uv=False)
        eff_rank_G[l0h, l1h] = effective_rank(sv)

im = axes[1].imshow(eff_rank_G, cmap='viridis', vmin=1, vmax=d_head,
                    aspect='auto', interpolation='nearest')
plt.colorbar(im, ax=axes[1], fraction=0.04, label='Effective rank')
axes[1].set_xticks(range(n_heads))
axes[1].set_xticklabels([f"L1H{h}" for h in range(n_heads)], rotation=45, ha='right', fontsize=7)
axes[1].set_yticks(range(n_heads))
axes[1].set_yticklabels([f"L0H{h}" for h in range(n_heads)], fontsize=7)
axes[1].set_title("G-weighted effective rank: all L0→L1 pairs\n"
                  f"Low = concentrated channel. Max possible = {d_head}", fontsize=8)
axes[1].plot(h_induction, h_prev, 'r*', ms=12)   # mark induction circuit

# Panel 3: K-comp score vs effective rank (scatter), G-weighted
kcomp_scores = np.zeros((n_heads, n_heads))
for l0h in range(n_heads):
    for l1h in range(n_heads):
        kcomp_scores[l0h, l1h] = kcomp_G_score(WO_0[l0h], WK_1[l1h], G_abs)

ax3 = axes[2]
x = eff_rank_G.flatten()
y = kcomp_scores.flatten()
ax3.scatter(x, y, alpha=0.5, s=20, color='#1f77b4')
# Highlight induction circuit
ax3.scatter([eff_rank_G[h_prev, h_induction]], [kcomp_scores[h_prev, h_induction]],
            color='red', s=80, zorder=5,
            label=f"L0H{h_prev}→L1H{h_induction} (induction)")
ax3.set_xlabel("G-weighted effective rank", fontsize=9)
ax3.set_ylabel("G-weighted K-comp score", fontsize=9)
ax3.set_title("K-comp score vs effective channel rank\n"
              "High comp + low rank = concentrated circuit", fontsize=8)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.2)

plt.suptitle(
    "Experiment 15: Effective channel rank of L0→L1 K-composition\n"
    "G-weighted metric: how many modes does the induction circuit communicate through?",
    fontsize=9)
plt.tight_layout()
plt.savefig("figs/exp15_channel_rank.png", dpi=250, bbox_inches="tight")
print("\nSaved figs/exp15_channel_rank.png")
print("\nDone.")

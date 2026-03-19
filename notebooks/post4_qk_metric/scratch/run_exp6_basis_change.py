"""
Post 4 — Experiment 6: G+B basis change using known head types

The G+B decomposition of W_QK is exact (always true by definition):
    W_QK^h = G^h + B^h   where G^h = sym(W_QK^h), B^h = anti(W_QK^h)

This is a basis-choice experiment. We pick:
  - G basis: eigenvectors of G^{induction} (L1H6) — the head whose job IS content matching
  - B basis: singular vector pairs of B^{prev-token} (identified by max ||B^h||_F in L0)
             — the head whose job IS directed routing

Then every other head's G^h, B^h, W_V^h, W_O^h is expressed exactly in those bases.
The computation is unchanged. We're just choosing coordinates.

Coefficients:
  a^h_i  = v_i^T G^h v_i           (G content engagement on direction i)
  b^h_j  = u_j^T B^h w_j           (B routing engagement on pair j)
  c^h_i  = ||W_V^h^T v_i||^2       (W_V reads direction v_i)
  d^h_i  = ||W_O^h v_i||^2         (W_O writes direction v_i)

Expected pattern:
  - Prev-token heads: large b, small a
  - Induction heads:  large a, small b
  - W_V/W_O of induction head: large c, d on same G directions as large a

Model: attn-only-2l (2L x 8H, d_model=512, d_head=64)
Run with: /opt/miniconda3/bin/python run_exp6_basis_change.py
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
print(f"Loaded: {model.cfg.n_layers}L x {model.cfg.n_heads}H, "
      f"d_model={model.cfg.d_model}, d_head={model.cfg.d_head}")

n_layers = model.cfg.n_layers
n_heads  = model.cfg.n_heads
d_model  = model.cfg.d_model

W_Q = model.W_Q.cpu().numpy()   # (2, 8, 512, 64)
W_K = model.W_K.cpu().numpy()
W_V = model.W_V.cpu().numpy()   # (2, 8, 512, 64)
W_O = model.W_O.cpu().numpy()   # (2, 8,  64, 512)


# ── Identify reference heads ──────────────────────────────────────────────────

# G^h and B^h for all heads
G_heads = {}
B_heads = {}
for l in range(n_layers):
    for h in range(n_heads):
        WQK = W_Q[l, h] @ W_K[l, h].T
        G_heads[(l, h)] = (WQK + WQK.T) / 2.0
        B_heads[(l, h)] = (WQK - WQK.T) / 2.0

# Reference heads identified by activation patterns on repeated-token sequence (exp10):
#   Induction head: L1H6  (induction score=0.604)
#   Prev-token head: L0H3 (prev-token score=0.488)
# Note: weight-based heuristics (max ||G||_F, max ||B||_F) gave wrong answers (L1H5, L0H5).
h_induction = 6
h_prev      = 3
print(f"Induction head (activation-based): L1H{h_induction}")
print(f"Prev-token head (activation-based): L0H{h_prev}")


# ── Extract G basis from induction head ───────────────────────────────────────

K_G = 16   # number of G basis vectors to keep

G_ind = G_heads[(1, h_induction)]
eigvals_G, eigvecs_G = np.linalg.eigh(G_ind)   # sorted ascending
# sort by |eigenvalue| descending
idx = np.argsort(np.abs(eigvals_G))[::-1]
eigvals_G = eigvals_G[idx]
eigvecs_G = eigvecs_G[:, idx]                   # (d_model, d_model), columns = eigenvectors

V_G = eigvecs_G[:, :K_G]                        # (d_model, K_G) — G basis

print(f"\nG basis (L1H{h_induction}) top-{K_G} eigenvalues: {eigvals_G[:K_G].round(3)}")


# ── Extract B basis from prev-token head ──────────────────────────────────────

K_B = 8    # number of B singular vector pairs to keep

B_prev = B_heads[(0, h_prev)]
U_B, sv_B, Vt_B = np.linalg.svd(B_prev)        # skew-sym: sv come in pairs

# B basis: pairs (U_B[:,j], Vt_B[j,:]) for j = 0, ..., K_B-1
# Coefficients b^h_j = U_B[:,j].T @ B^h @ Vt_B[j,:]

print(f"\nB basis (L0H{h_prev}) top-{K_B} singular values: {sv_B[:K_B].round(3)}")


# ── Compute coefficients for all heads ────────────────────────────────────────

# a^h_i = v_i^T G^h v_i                     scalar, G engagement on content direction i
# b^h_j = u_j^T B^h w_j                     scalar, B engagement on routing pair j
# c^h_i = ||W_V^h^T v_i||^2                 scalar, W_V reads direction v_i
# d^h_i = ||W_O^h v_i||^2                   scalar, W_O writes direction v_i

a_coeff = np.zeros((n_layers, n_heads, K_G))   # G content coefficients
b_coeff = np.zeros((n_layers, n_heads, K_B))   # B routing coefficients
c_coeff = np.zeros((n_layers, n_heads, K_G))   # W_V read coefficients
d_coeff = np.zeros((n_layers, n_heads, K_G))   # W_O write coefficients

for l in range(n_layers):
    for h in range(n_heads):
        G_h = G_heads[(l, h)]
        B_h = B_heads[(l, h)]
        WV_h = W_V[l, h]   # (d_model, d_head)
        WO_h = W_O[l, h]   # (d_head,  d_model)

        for i in range(K_G):
            v = V_G[:, i]
            a_coeff[l, h, i] = v @ G_h @ v
            c_coeff[l, h, i] = np.linalg.norm(WV_h.T @ v) ** 2
            d_coeff[l, h, i] = np.linalg.norm(WO_h @ v) ** 2

        for j in range(K_B):
            u = U_B[:, j]
            w = Vt_B[j, :]
            b_coeff[l, h, j] = u @ B_h @ w


# ── Summary scalars per head ──────────────────────────────────────────────────

# Total G engagement (sum |a^h_i|), B engagement (sum |b^h_j|)
G_total = np.abs(a_coeff).sum(axis=-1)    # (n_layers, n_heads)
B_total = np.abs(b_coeff).sum(axis=-1)
WV_total = c_coeff.sum(axis=-1)
WO_total = d_coeff.sum(axis=-1)

print("\n=== Summary per head ===")
print(f"{'Head':8s}  {'G_total':>8s}  {'B_total':>8s}  {'WV_G':>8s}  {'WO_G':>8s}")
for l in range(n_layers):
    for h in range(n_heads):
        tag = ""
        if l == 1 and h == h_induction: tag = " ← induction"
        if l == 0 and h == h_prev:      tag = " ← prev-token"
        print(f"  L{l}H{h}:   "
              f"  {G_total[l,h]:7.3f}   {B_total[l,h]:7.3f}"
              f"   {WV_total[l,h]:7.3f}   {WO_total[l,h]:7.3f}{tag}")


# ── Figures ───────────────────────────────────────────────────────────────────

colors = ['#1f77b4', '#d62728']   # L0 blue, L1 red
markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']

# Fig 1: G_total vs B_total scatter — head type separation
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for l in range(n_layers):
    for h in range(n_heads):
        ax.scatter(G_total[l, h], B_total[l, h],
                   color=colors[l], marker=markers[h], s=80, zorder=3)
        ax.annotate(f"L{l}H{h}", (G_total[l, h], B_total[l, h]),
                    fontsize=7, xytext=(4, 2), textcoords='offset points')

ax.set_xlabel(f"G engagement (sum |a^h_i|, top-{K_G} G basis dirs)", fontsize=9)
ax.set_ylabel(f"B engagement (sum |b^h_j|, top-{K_B} B basis pairs)", fontsize=9)
ax.set_title(
    f"G vs B engagement per head\n"
    f"G basis from L1H{h_induction} (induction), B basis from L0H{h_prev} (prev-token)",
    fontsize=9)
ax.grid(True, alpha=0.2)

# Annotate reference heads
ax.annotate(f"← induction (G ref)", (G_total[1, h_induction], B_total[1, h_induction]),
            fontsize=7, color='red', xytext=(6, -10), textcoords='offset points')
ax.annotate(f"← prev-token (B ref)", (G_total[0, h_prev], B_total[0, h_prev]),
            fontsize=7, color='blue', xytext=(6, 4), textcoords='offset points')

# Fig 2: W_V read vs W_O write scatter on G basis
ax = axes[1]
for l in range(n_layers):
    for h in range(n_heads):
        ax.scatter(WV_total[l, h], WO_total[l, h],
                   color=colors[l], marker=markers[h], s=80, zorder=3)
        ax.annotate(f"L{l}H{h}", (WV_total[l, h], WO_total[l, h]),
                    fontsize=7, xytext=(4, 2), textcoords='offset points')

ax.set_xlabel(f"W_V read mass on G basis (sum ||W_V^T v_i||^2)", fontsize=9)
ax.set_ylabel(f"W_O write mass on G basis (sum ||W_O v_i||^2)", fontsize=9)
ax.set_title(
    f"W_V read vs W_O write on G basis\n"
    f"G basis from L1H{h_induction} (induction)",
    fontsize=9)
ax.grid(True, alpha=0.2)

plt.suptitle(
    "Experiment 6: G+B basis change — expressing all heads in induction/prev-token coordinate system\n"
    "Blue = L0, Red = L1",
    fontsize=10)
plt.tight_layout()
plt.savefig("figs/exp6_basis_change.png", dpi=300, bbox_inches="tight")
print("\nSaved figs/exp6_basis_change.png")


# ── Fig 3: Per-direction coefficient heatmap for top G dirs ──────────────────

K_show = 8   # show top-K_show G directions

fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))

head_labels = [f"L{l}H{h}" for l in range(n_layers) for h in range(n_heads)]
x = np.arange(n_layers * n_heads)

def coeff_heatmap(ax, data, title, ylabel, cmap='RdBu_r'):
    """data: (n_layers*n_heads, K) coefficient matrix"""
    im = ax.imshow(data.T, aspect='auto', cmap=cmap,
                   vmin=-np.abs(data).max(), vmax=np.abs(data).max())
    plt.colorbar(im, ax=ax)
    ax.set_xticks(x)
    ax.set_xticklabels(head_labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9)

# G content coefficients a^h_i
a_flat = a_coeff.reshape(-1, K_G)[:, :K_show]
coeff_heatmap(axes3[0, 0], a_flat,
              f"G content coefficients a^h_i = v_i^T G^h v_i\n(top-{K_show} G basis dirs)",
              "G basis direction i")

# B routing coefficients b^h_j
b_flat = b_coeff.reshape(-1, K_B)
coeff_heatmap(axes3[0, 1], b_flat,
              f"B routing coefficients b^h_j = u_j^T B^h w_j\n(top-{K_B} B basis pairs)",
              "B basis pair j",
              cmap='PuOr')

# W_V read coefficients c^h_i
c_flat = c_coeff.reshape(-1, K_G)[:, :K_show]
coeff_heatmap(axes3[1, 0], c_flat,
              f"W_V read mass c^h_i = ||W_V^T v_i||^2\n(top-{K_show} G basis dirs)",
              "G basis direction i",
              cmap='Greens')

# W_O write coefficients d^h_i
d_flat = d_coeff.reshape(-1, K_G)[:, :K_show]
coeff_heatmap(axes3[1, 1], d_flat,
              f"W_O write mass d^h_i = ||W_O v_i||^2\n(top-{K_show} G basis dirs)",
              "G basis direction i",
              cmap='Oranges')

plt.suptitle(
    f"Per-direction coefficient heatmaps — all heads expressed in induction/prev-token basis\n"
    f"G basis: L1H{h_induction} eigenvectors  |  B basis: L0H{h_prev} singular vector pairs",
    fontsize=10)
plt.tight_layout()
plt.savefig("figs/exp6_coeff_heatmap.png", dpi=300, bbox_inches="tight")
print("Saved figs/exp6_coeff_heatmap.png")

print("\nDone.")

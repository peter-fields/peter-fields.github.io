"""
Post 4 — Experiment 13: Joint Approximate Diagonalization (JAD) basis for G and B

Find basis V such that each head's G^h = sum_i a^h_i v_i v_i^T has maximally
concentrated (sloppy) coefficient profiles — not the mean, but each head individually.

Algorithm: Jacobi JAD — find orthogonal V maximizing sum_h ||diag(V^T G^h V)||_F^2,
equivalently minimizing sum_h ||off-diag(V^T G^h V)||_F^2.

For each pair (i,j), the optimal Givens rotation angle θ satisfies:
  tan(4θ) = -2Q/P   where
    h_h = C^h[i,j]  (off-diagonal of current projection of G^h)
    d_h = (C^h[i,i] - C^h[j,j]) / 2
    Q = sum_h h_h * d_h
    P = sum_h h_h^2 - sum_h d_h^2

Starting point: BOS-cleaned G_crude eigenvectors (unsupervised).

Key test: does the JAD basis match the induction head's eigenvectors (supervised)?
If yes — induction head already found the natural jointly-diagonal basis.
If no — heads have diverse geometries that don't all concentrate on the same directions.

Model: attn-only-2l. Reference: L1H6 (induction), L0H3 (prev-token) from exp10.
Run with: /opt/miniconda3/bin/python run_exp13_jad_basis.py
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

W_Q = model.W_Q.cpu().numpy()
W_K = model.W_K.cpu().numpy()
W_E = model.W_E.cpu().numpy()

h_induction = 6; h_prev = 3
K = 16   # subspace dimension

# ── Compute G^h and B^h for all heads ────────────────────────────────────────

G_heads = {}
B_heads = {}
for l in range(n_layers):
    for h in range(n_heads):
        WQK = W_Q[l, h] @ W_K[l, h].T
        G_heads[(l, h)] = (WQK + WQK.T) / 2.0
        B_heads[(l, h)] = (WQK - WQK.T) / 2.0

G_crude = np.mean(list(G_heads.values()), axis=0)
B_crude = np.mean(list(B_heads.values()), axis=0)

# ── BOS removal helper ────────────────────────────────────────────────────────

bos_id = model.tokenizer.bos_token_id
v_bos  = W_E[bos_id] / (np.linalg.norm(W_E[bos_id]) + 1e-10)

def debos_and_orthogonalize(V):
    """Project out BOS direction from columns of V, re-orthogonalize via QR."""
    V2 = V - np.outer(v_bos, v_bos @ V)
    Q, _ = np.linalg.qr(V2)
    return Q[:, :V.shape[1]]

# ── Starting basis: G_crude top-K eigenvectors (BOS-cleaned) ─────────────────

eigvals_crude, eigvecs_crude = np.linalg.eigh(G_crude)
idx = np.argsort(np.abs(eigvals_crude))[::-1]
V0 = debos_and_orthogonalize(eigvecs_crude[:, idx[:K]])   # (d_model, K) — starting point

# ── Jacobi JAD ───────────────────────────────────────────────────────────────

def jacobi_jad(matrices_3d, V_init, n_sweeps=50, tol=1e-10):
    """
    Joint approximate diagonalization of symmetric matrices via Jacobi sweeps.
    matrices_3d: (n_matrices, K, K) — already projected to K-dim subspace
    V_init: (K, K) identity or other starting rotation
    Returns: V (K, K) orthogonal, C (n_matrices, K, K) approximately diagonal
    """
    n_mat, K, _ = matrices_3d.shape
    V = V_init.copy()
    C = matrices_3d.copy()

    prev_off = np.inf
    for sweep in range(n_sweeps):
        off = sum(np.sum(C[m]**2) - np.sum(np.diag(C[m])**2) for m in range(n_mat))
        if abs(prev_off - off) < tol:
            print(f"  Converged at sweep {sweep}, off-diag sum = {off:.6f}")
            break
        prev_off = off

        for i in range(K - 1):
            for j in range(i + 1, K):
                # h_m = C^m[i,j], d_m = (C^m[i,i] - C^m[j,j]) / 2
                h = C[:, i, j]           # (n_mat,)
                d = (C[:, i, i] - C[:, j, j]) / 2.0

                Q_val = np.dot(h, d)
                P_val = np.dot(h, h) - np.dot(d, d)

                if abs(Q_val) < 1e-15 and abs(P_val) < 1e-15:
                    continue

                theta = np.arctan2(-2 * Q_val, P_val) / 4.0

                c, s = np.cos(theta), np.sin(theta)

                # Givens rotation matrix (K x K)
                G_rot = np.eye(K)
                G_rot[i, i] =  c;  G_rot[i, j] = s
                G_rot[j, i] = -s;  G_rot[j, j] = c

                V = V @ G_rot
                C = np.einsum('ki,mkl,lj->mij', G_rot, C, G_rot)

    off_final = sum(np.sum(C[m]**2) - np.sum(np.diag(C[m])**2) for m in range(n_mat))
    print(f"  Final off-diag sum = {off_final:.6f}")
    return V, C

# Project all G^h onto starting subspace V0
all_G_keys = [(l, h) for l in range(n_layers) for h in range(n_heads)]
G_proj = np.array([V0.T @ G_heads[k] @ V0 for k in all_G_keys])  # (n_heads, K, K)
B_proj = np.array([V0.T @ B_heads[k] @ V0 for k in all_G_keys])

print("Running JAD for G...")
R_G, C_G = jacobi_jad(G_proj, np.eye(K), n_sweeps=100)

print("Running JAD for B...")
R_B, C_B = jacobi_jad(B_proj, np.eye(K), n_sweeps=100)

# Final bases in d_model space
V_jad_G = V0 @ R_G   # (d_model, K) — JAD G basis
V_jad_B = V0 @ R_B   # (d_model, K) — JAD B basis


# ── Coefficient profiles in JAD basis ────────────────────────────────────────

def coeff_diagonal(C_all, head_keys):
    """Extract diagonal coefficients from approximately-diagonal projected matrices."""
    return np.array([[C_all[m][i, i] for i in range(C_all.shape[1])]
                     for m in range(len(head_keys))])

A_jad_G = coeff_diagonal(C_G, all_G_keys)   # (n_heads, K) — G coefficients in JAD basis
A_jad_B = coeff_diagonal(C_B, all_G_keys)   # (n_heads, K) — B coefficients in JAD basis

head_labels = [f"L{l}H{h}" for l, h in all_G_keys]

print("\nPer-head G coefficient sloppiness in JAD basis (log10 range):")
for idx_h, (l, h) in enumerate(all_G_keys):
    a = np.abs(A_jad_G[idx_h])
    a = a[a > 1e-10]
    sloppy = np.log10(a.max() / (a.min() + 1e-10)) if len(a) > 1 else 0
    tag = ""
    if l == 1 and h == h_induction: tag = " ← induction"
    if l == 0 and h == h_prev:      tag = " ← prev-token"
    print(f"  L{l}H{h}: {sloppy:.3f}{tag}")


# ── Compare JAD G basis to induction head G basis (principal angles) ──────────

def principal_angles(A, B):
    """Mean cosine of principal angles between column spaces of A and B."""
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    sv = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.clip(sv, 0, 1)

# Induction head G basis (BOS-cleaned)
G_ind = G_heads[(1, h_induction)]
eigvals_ind, eigvecs_ind = np.linalg.eigh(G_ind)
idx_ind = np.argsort(np.abs(eigvals_ind))[::-1]
V_ind = debos_and_orthogonalize(eigvecs_ind[:, idx_ind[:K]])

# Prev-token head B basis (BOS-cleaned)
U_prev, sv_prev, Vt_prev = np.linalg.svd(B_heads[(0, h_prev)])
V_prev_B = debos_and_orthogonalize(U_prev[:, :K])

cos_jad_G_vs_ind = principal_angles(V_jad_G, V_ind)
cos_jad_B_vs_prev = principal_angles(V_jad_B, V_prev_B)
cos_jad_G_vs_crude = principal_angles(V_jad_G, V0)

print(f"\nPrincipal angle cosines (mean over top-{K} directions):")
print(f"  JAD-G  vs induction head G:   {cos_jad_G_vs_ind.mean():.3f}  "
      f"top-4: {cos_jad_G_vs_ind[:4].round(3)}")
print(f"  JAD-B  vs prev-token head B:  {cos_jad_B_vs_prev.mean():.3f}  "
      f"top-4: {cos_jad_B_vs_prev[:4].round(3)}")
print(f"  JAD-G  vs G_crude start:      {cos_jad_G_vs_crude.mean():.3f}  "
      f"(sanity: should be high, same subspace)")
print(f"  Random baseline: ~{np.sqrt(K / d_model):.2f}")


# ── Figures ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# Heatmap: G coefficients in JAD basis
vmax = np.abs(A_jad_G).max()
im = axes[0].imshow(A_jad_G.T, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
plt.colorbar(im, ax=axes[0], fraction=0.03)
axes[0].set_xticks(range(len(head_labels)))
axes[0].set_xticklabels(head_labels, rotation=45, ha='right', fontsize=6)
axes[0].set_ylabel("JAD G direction i", fontsize=9)
axes[0].set_title("G coefficients in JAD basis\na^h_i = v_i^T G^h v_i\n"
                  "(JAD → each head maximally concentrated)", fontsize=8)

# Heatmap: B coefficients in JAD basis
vmax_b = np.abs(A_jad_B).max()
im2 = axes[1].imshow(A_jad_B.T, aspect='auto', cmap='PuOr', vmin=-vmax_b, vmax=vmax_b)
plt.colorbar(im2, ax=axes[1], fraction=0.03)
axes[1].set_xticks(range(len(head_labels)))
axes[1].set_xticklabels(head_labels, rotation=45, ha='right', fontsize=6)
axes[1].set_ylabel("JAD B direction i", fontsize=9)
axes[1].set_title("B coefficients in JAD basis\nb^h_i = v_i^T B^h v_i\n"
                  "(same V, antisym part)", fontsize=8)

# Principal angle comparison
ax = axes[2]
x = np.arange(1, K + 1)
ax.plot(x, cos_jad_G_vs_ind,    'o-', lw=1.5, ms=5, label='JAD-G vs induction head G')
ax.plot(x, cos_jad_B_vs_prev,   's-', lw=1.5, ms=5, label='JAD-B vs prev-token head B')
ax.plot(x, cos_jad_G_vs_crude,  '^--', lw=1, ms=4, label='JAD-G vs G_crude (start)')
ax.axhline(np.sqrt(K / d_model), color='red', lw=1, linestyle='--',
           label=f'random baseline ≈{np.sqrt(K/d_model):.2f}')
ax.set_xlabel("Principal angle rank", fontsize=9)
ax.set_ylabel("cos(principal angle)", fontsize=9)
ax.set_title("Do JAD bases match known head bases?\n1.0 = identical, random ≈ 0.18", fontsize=8)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.2)
ax.set_ylim(0, 1.05)

plt.suptitle(
    "Experiment 13: JAD basis — find V where each head's G and B profiles are maximally concentrated\n"
    f"Starting from BOS-cleaned G_crude subspace. Reference: L1H{h_induction} induction, L0H{h_prev} prev-token",
    fontsize=9)
plt.tight_layout()
plt.savefig("figs/exp13_jad_basis.png", dpi=300, bbox_inches="tight")
print("\nSaved figs/exp13_jad_basis.png")
print("\nDone.")

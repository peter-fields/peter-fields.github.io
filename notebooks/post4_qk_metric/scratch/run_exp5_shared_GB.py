"""
Post 4 — Experiment 5: Does B from W_QK predict K-composition in W_OV?
Run with: /opt/miniconda3/bin/python run_exp5_shared_GB.py

Claim: G and B are shared geometric objects, not just W_QK decomposition artifacts.
Test: B_crude and G_crude are extracted from W_QK (routing geometry). Do they also
predict the K-composition structure in W_OV (W_O writer / W_K reader)?

If B-filtered K-composition alignment recovers the same sparse pattern as raw
alignment, then B is the K-composition channel — shared across W_QK and W_OV.

K-composition: W_O[L0,h] writes into residual stream.
               W_K[L1,h'] reads from residual stream to form keys.
Both W_O write directions and W_K read directions should project onto B's
RIGHT singular vectors (key-side directions of B) if B is the shared channel.

For comparison, also filter through G's top eigenvectors (content directions).

Model: attn-only-2l (2L x 8H, d_model=512, d_head=64).
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
W_O = model.W_O.cpu().numpy()   # (2, 8,  64, 512)


# ── G_crude and B_crude from W_QK ────────────────────────────────────────────

G_all, B_all = [], []
for l in range(n_layers):
    for h in range(n_heads):
        WQK = W_Q[l, h] @ W_K[l, h].T
        G_all.append((WQK + WQK.T) / 2.0)
        B_all.append((WQK - WQK.T) / 2.0)

G_crude = np.array(G_all).mean(axis=0)
B_crude = np.array(B_all).mean(axis=0)

# G: eigendecomposition (symmetric)
eigvals_G, eigvecs_G = np.linalg.eigh(G_crude)
idx = np.argsort(np.abs(eigvals_G))[::-1]
eigvecs_G = eigvecs_G[:, idx]

# B: SVD (skew-symmetric → singular values in pairs)
U_B, sv_B, Vt_B = np.linalg.svd(B_crude)

print(f"G_crude Frobenius norm: {np.linalg.norm(G_crude):.4f}")
print(f"B_crude Frobenius norm: {np.linalg.norm(B_crude):.4f}")
print(f"B singular values (pairs): {sv_B[:8].round(4)}")


# ── K-composition alignment: raw, B-filtered, G-filtered ─────────────────────

def kcomp_align(WO_dirs, WK_dirs):
    """Normalised K-comp alignment between write directions and read directions.
    WO_dirs: (d_model, d_head) — what L0h writes
    WK_dirs: (d_model, d_head) — what L1h' reads
    Returns scalar alignment score."""
    raw = np.linalg.norm(WK_dirs.T @ WO_dirs, 'fro')
    denom = (np.linalg.norm(WO_dirs, 'fro') * np.linalg.norm(WK_dirs, 'fro') + 1e-10)
    return raw / denom


K_B = 8   # top B singular modes
K_G = 8   # top G eigenmodes

# Projectors onto B right singular vectors (key-side directions of B)
P_B = Vt_B[:K_B, :].T @ Vt_B[:K_B, :]   # (d_model, d_model)

# Projector onto G top eigenvectors (content directions)
P_G = eigvecs_G[:, :K_G] @ eigvecs_G[:, :K_G].T

raw_align = np.zeros((n_heads, n_heads))
B_align   = np.zeros((n_heads, n_heads))
G_align   = np.zeros((n_heads, n_heads))

for h0 in range(n_heads):
    WO = W_O[0, h0].T        # (d_model, d_head) — write directions (cols of W_O^T)
    for h1 in range(n_heads):
        WK = W_K[1, h1]      # (d_model, d_head) — read directions (cols of W_K)

        raw_align[h0, h1] = kcomp_align(WO, WK)
        B_align[h0, h1]   = kcomp_align(P_B @ WO, P_B @ WK)
        G_align[h0, h1]   = kcomp_align(P_G @ WO, P_G @ WK)

head_labels = [f"L0H{h}" for h in range(n_heads)]

print(f"\nRaw K-comp alignment (L0h→L1h'):")
print(f"{'':8s}", end='')
for h1 in range(n_heads): print(f"  L1H{h1}", end='')
print()
for h0 in range(n_heads):
    print(f"  L0H{h0}:  ", end='')
    for h1 in range(n_heads): print(f"  {raw_align[h0,h1]:.3f}", end='')
    print()

print(f"\nB-filtered (right sv, key directions):")
print(f"{'':8s}", end='')
for h1 in range(n_heads): print(f"  L1H{h1}", end='')
print()
for h0 in range(n_heads):
    print(f"  L0H{h0}:  ", end='')
    for h1 in range(n_heads): print(f"  {B_align[h0,h1]:.3f}", end='')
    print()

print(f"\nG-filtered (top eigenvectors, content directions):")
print(f"{'':8s}", end='')
for h1 in range(n_heads): print(f"  L1H{h1}", end='')
print()
for h0 in range(n_heads):
    print(f"  L0H{h0}:  ", end='')
    for h1 in range(n_heads): print(f"  {G_align[h0,h1]:.3f}", end='')
    print()

corr_B = np.corrcoef(raw_align.ravel(), B_align.ravel())[0, 1]
corr_G = np.corrcoef(raw_align.ravel(), G_align.ravel())[0, 1]
print(f"\nCorrelation with raw alignment:")
print(f"  B-filtered: {corr_B:.4f}")
print(f"  G-filtered: {corr_G:.4f}")
print(f"  (higher → that filter recovers K-composition structure)")


# ════════════════════════════════════════════════════════════════════════════
# TEST 2b: Stacked SVD per matrix type
# Stack all heads of each matrix type, find shared directions.
# Rotation-invariant across heads — finds union of directions, not average.
# Compare subspace alignment: W_Q ↔ W_K (G-like), W_V ↔ W_Q/W_K, W_O separate.
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("TEST 2b: Stacked SVD per matrix type")
print("="*60)

W_V = model.W_V.cpu().numpy()   # (2, 8, 512, 64)
W_E = model.W_E.cpu().numpy()

def stacked_left_sv(matrices, k):
    """Stack list of (d_model, d_head) matrices, return top-k left singular vectors."""
    M = np.hstack(matrices)   # (d_model, N)
    U, sv, _ = np.linalg.svd(M, full_matrices=False)
    print(f"  stacked shape {M.shape}, top-5 sv: {sv[:5].round(3)}")
    return U[:, :k], sv

def principal_angle_cosines(A, B):
    """Mean cosine of principal angles between column spaces of A and B."""
    from numpy.linalg import qr, svd
    Qa, _ = qr(A); Qb, _ = qr(B)
    sv = svd(Qa.T @ Qb, compute_uv=False)
    return np.clip(sv, 0, 1)

K_stack = 32

print("\nW_Q stacked:")
U_Q, sv_Q = stacked_left_sv([W_Q[l, h] for l in range(n_layers) for h in range(n_heads)], K_stack)
print("\nW_K stacked:")
U_K, sv_K = stacked_left_sv([W_K[l, h] for l in range(n_layers) for h in range(n_heads)], K_stack)
print("\nW_V stacked:")
U_V, sv_V = stacked_left_sv([W_V[l, h] for l in range(n_layers) for h in range(n_heads)], K_stack)
print("\nW_O write directions stacked (W_O.T cols = rows of W_O):")
U_O, sv_O = stacked_left_sv([W_O[l, h].T for l in range(n_layers) for h in range(n_heads)], K_stack)

# W_E projection mass for each stacked subspace
_, sv_we, Vt_we = np.linalg.svd(W_E, full_matrices=False)
cumvar = np.cumsum(sv_we**2) / (sv_we**2).sum()
k_we = int(np.searchsorted(cumvar, 0.90)) + 1
V_we = Vt_we[:k_we].T

def we_mass(U):
    proj = U.T @ V_we
    return (proj**2).sum(axis=1)

mass_Q = we_mass(U_Q)
mass_K = we_mass(U_K)
mass_V = we_mass(U_V)
mass_O = we_mass(U_O)

print(f"\nW_E projection mass (top-{K_stack} shared directions):")
print(f"  W_Q: mean={mass_Q.mean():.3f}  top-5: {mass_Q[:5].round(3)}")
print(f"  W_K: mean={mass_K.mean():.3f}  top-5: {mass_K[:5].round(3)}")
print(f"  W_V: mean={mass_V.mean():.3f}  top-5: {mass_V[:5].round(3)}")
print(f"  W_O: mean={mass_O.mean():.3f}  top-5: {mass_O[:5].round(3)}")

# Principal angles between subspaces
cos_QK = principal_angle_cosines(U_Q, U_K)
cos_QV = principal_angle_cosines(U_Q, U_V)
cos_KV = principal_angle_cosines(U_K, U_V)
cos_QO = principal_angle_cosines(U_Q, U_O)
cos_KO = principal_angle_cosines(U_K, U_O)
cos_VO = principal_angle_cosines(U_V, U_O)

print(f"\nPrincipal angle cosines between stacked subspaces (top-{K_stack}):")
print(f"  W_Q ↔ W_K: mean={cos_QK.mean():.3f}  (both routing → should share G directions)")
print(f"  W_Q ↔ W_V: mean={cos_QV.mean():.3f}  (both read from resid stream)")
print(f"  W_K ↔ W_V: mean={cos_KV.mean():.3f}")
print(f"  W_Q ↔ W_O: mean={cos_QO.mean():.3f}")
print(f"  W_K ↔ W_O: mean={cos_KO.mean():.3f}")
print(f"  W_V ↔ W_O: mean={cos_VO.mean():.3f}  (read vs write — G content channel?)")


# ── FIGURES ───────────────────────────────────────────────────────────────────

vmax = raw_align.max()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

titles = [
    f"Raw K-comp alignment\n||W_K[L1]^T W_O[L0]^T||_F (normalised)",
    f"B-filtered (top-{K_B} right sv of B_crude)\ncorr with raw = {corr_B:.3f}",
    f"G-filtered (top-{K_G} eigvecs of G_crude)\ncorr with raw = {corr_G:.3f}",
]
data   = [raw_align, B_align, G_align]
cmaps  = ['Blues', 'Oranges', 'Greens']

for ax, d, title, cmap in zip(axes, data, titles, cmaps):
    im = ax.imshow(d, cmap=cmap, vmin=0, vmax=d.max())
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f'L1H{h}' for h in range(n_heads)], fontsize=7, rotation=45)
    ax.set_yticks(range(n_heads))
    ax.set_yticklabels([f'L0H{h}' for h in range(n_heads)], fontsize=7)
    ax.set_xlabel("L1 head (W_K reader)", fontsize=9)
    ax.set_ylabel("L0 head (W_O writer)", fontsize=9)
    ax.set_title(title, fontsize=9)

plt.suptitle(
    "attn-only-2l: Does B from W_QK routing geometry predict K-composition in W_OV?\n"
    "B-filtered = project W_O write directions and W_K read directions onto B right singular vectors",
    fontsize=10
)
plt.tight_layout()
plt.savefig("figs/exp5_shared_GB.png", dpi=300, bbox_inches="tight")
print("\nSaved figs/exp5_shared_GB.png")


# ── Figure 2: stacked SVD W_E masses + principal angles ──────────────────────

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))

ax = axes2[0]
x = np.arange(K_stack)
ax.plot(x, mass_Q, 'o-', ms=4, lw=1.2, label='W_Q', color='#1f77b4')
ax.plot(x, mass_K, 's-', ms=4, lw=1.2, label='W_K', color='#ff7f0e')
ax.plot(x, mass_V, '^-', ms=4, lw=1.2, label='W_V', color='#2ca02c')
ax.plot(x, mass_O, 'x-', ms=4, lw=1.2, label='W_O (write)', color='#d62728')
ax.axhline(0.5, color='gray', lw=1, linestyle='--', alpha=0.5, label='threshold')
ax.set_xlabel(f"Shared direction rank (top-{K_stack})", fontsize=9)
ax.set_ylabel("W_E projection mass", fontsize=9)
ax.set_title("Stacked SVD: W_E mass of shared directions\nper matrix type", fontsize=9)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

ax = axes2[1]
pairs   = ['W_Q↔W_K', 'W_Q↔W_V', 'W_K↔W_V', 'W_Q↔W_O', 'W_K↔W_O', 'W_V↔W_O']
means   = [cos_QK.mean(), cos_QV.mean(), cos_KV.mean(),
           cos_QO.mean(), cos_KO.mean(), cos_VO.mean()]
colors  = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2']
bars = ax.bar(pairs, means, color=colors, alpha=0.8)
ax.set_ylabel("Mean cos(principal angle)", fontsize=9)
ax.set_title(f"Subspace alignment between stacked matrix types\n(top-{K_stack} shared directions)", fontsize=9)
ax.set_xticklabels(pairs, rotation=30, ha='right', fontsize=8)
ax.set_ylim(0, 1.05)
ax.axhline(np.sqrt(K_stack / d_model), color='red', lw=1, linestyle='--',
           label=f'random baseline ≈{np.sqrt(K_stack/d_model):.2f}')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2, axis='y')

plt.suptitle("Stacked SVD per matrix type: shared direction structure\n"
             "W_Q ↔ W_K alignment tests whether G directions are shared across routing matrices",
             fontsize=10)
plt.tight_layout()
plt.savefig("figs/exp5_stacked_svd.png", dpi=300, bbox_inches="tight")
print("Saved figs/exp5_stacked_svd.png")
print("\nDone.")

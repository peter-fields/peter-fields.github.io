"""
Post 4 — QK Metric, Experiment 5: G and B as geometric foundation of attn-only-2l
Run with: /opt/miniconda3/bin/python run_exp5_geometric_foundation.py

Two claims:
  1. G is a real shared geometry: G_QK (from W_QK sym), G_V (from W_V outer products),
     G_O (from W_O outer products) share eigenvectors — checked via principal angles.
  2. Content vs compute falls out of G and B: G eigenvectors have high W_E projection
     mass, B singular vectors have low W_E projection mass. This is a per-head property,
     not just a global average.

Eigenvector comparisons only — scales are incomparable across sources.
Model: attn-only-2l (2L x 8H, d_model=512).
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
print(f"Loaded: {model.cfg.n_layers}L x {model.cfg.n_heads}H, d_model={model.cfg.d_model}")

n_layers = model.cfg.n_layers   # 2
n_heads  = model.cfg.n_heads    # 8
d_model  = model.cfg.d_model    # 512

W_Q = model.W_Q.cpu().numpy()   # (2, 8, 512, 64)
W_K = model.W_K.cpu().numpy()
W_V = model.W_V.cpu().numpy()   # (2, 8, 512, 64)
W_O = model.W_O.cpu().numpy()   # (2, 8,  64, 512)
W_E = model.W_E.cpu().numpy()   # (vocab, 512)


# ── Per-head G^h, B^h, WVV^h, WOO^h ─────────────────────────────────────────

G_heads, B_heads, WVV_heads, WOO_heads = [], [], [], []

for l in range(n_layers):
    for h in range(n_heads):
        WQK = W_Q[l, h] @ W_K[l, h].T          # (512, 512)
        G_heads.append((WQK + WQK.T) / 2.0)
        B_heads.append((WQK - WQK.T) / 2.0)
        WVV_heads.append(W_V[l, h] @ W_V[l, h].T)     # (512, 512) outer product of W_V cols
        WOO_heads.append(W_O[l, h].T @ W_O[l, h])     # (512, 512) outer product of W_O rows

G_heads  = np.array(G_heads)    # (16, 512, 512)
B_heads  = np.array(B_heads)
WVV_heads = np.array(WVV_heads)
WOO_heads = np.array(WOO_heads)

n_heads_total = G_heads.shape[0]  # 16

# ── Shared G from three sources, shared B ────────────────────────────────────

G_QK = G_heads.mean(axis=0)     # (512, 512) symmetric
G_V  = WVV_heads.mean(axis=0)   # (512, 512) symmetric PSD
G_O  = WOO_heads.mean(axis=0)   # (512, 512) symmetric PSD
B_QK = B_heads.mean(axis=0)     # (512, 512) antisymmetric

print(f"\nG_QK Frobenius norm: {np.linalg.norm(G_QK):.4f}")
print(f"G_V  Frobenius norm: {np.linalg.norm(G_V):.4f}")
print(f"G_O  Frobenius norm: {np.linalg.norm(G_O):.4f}")
print(f"B_QK Frobenius norm: {np.linalg.norm(B_QK):.4f}")


# ── Eigendecompose G sources ──────────────────────────────────────────────────

def top_eigvecs(M, k):
    """Top-k eigenvectors of symmetric M by absolute eigenvalue."""
    ev, evec = np.linalg.eigh(M)
    idx = np.argsort(np.abs(ev))[::-1]
    return evec[:, idx[:k]], ev[idx[:k]]

def orth(V):
    """Orthonormalize columns of V via QR."""
    Q, _ = np.linalg.qr(V)
    return Q

K = 32   # subspace rank to compare

evecs_GQK, evals_GQK = top_eigvecs(G_QK, K)
evecs_GV,  evals_GV  = top_eigvecs(G_V,  K)
evecs_GO,  evals_GO  = top_eigvecs(G_O,  K)

U_B, sv_B, _ = np.linalg.svd(B_QK)
evecs_B = U_B[:, :K]

print(f"\nTop-5 |eigenvalues|:")
print(f"  G_QK: {np.abs(evals_GQK[:5]).round(4)}")
print(f"  G_V:  {np.abs(evals_GV[:5]).round(4)}")
print(f"  G_O:  {np.abs(evals_GO[:5]).round(4)}")
print(f"  B_QK sv: {sv_B[:5].round(4)}")


# ── Principal angles between top-k subspaces ─────────────────────────────────

def principal_angle_cosines(A, B):
    """Cosines of principal angles between column spaces of A and B.
    A, B: (d, k). Returns k cosine values in descending order."""
    Qa = orth(A)
    Qb = orth(B)
    M = Qa.T @ Qb
    sv = np.linalg.svd(M, compute_uv=False)
    return np.clip(sv, 0, 1)

cos_QK_V  = principal_angle_cosines(evecs_GQK, evecs_GV)
cos_QK_O  = principal_angle_cosines(evecs_GQK, evecs_GO)
cos_V_O   = principal_angle_cosines(evecs_GV,  evecs_GO)
cos_G_B   = principal_angle_cosines(evecs_GQK, evecs_B)   # G vs B: should be small

print(f"\nPrincipal angle cosines — global means (top-{K} subspaces):")
print(f"  G_QK vs G_V  — mean: {cos_QK_V.mean():.4f}, min: {cos_QK_V.min():.4f}")
print(f"  G_QK vs G_O  — mean: {cos_QK_O.mean():.4f}, min: {cos_QK_O.min():.4f}")
print(f"  G_V  vs G_O  — mean: {cos_V_O.mean():.4f},  min: {cos_V_O.min():.4f}")
print(f"  G_QK vs B_QK — mean: {cos_G_B.mean():.4f}, min: {cos_G_B.min():.4f}  (should be ~0)")

# ── Per-head G consistency: do G_QK^h, G_V^h, G_O^h agree within each head? ─
# Use top-k_h eigenvectors per head (smaller k since heads are lower rank)
K_h = 8   # d_head = 64 but useful structure in top few modes
head_labels = [f"L{l}H{h}" for l in range(n_layers) for h in range(n_heads)]
colors_layer = ['#1f77b4'] * n_heads + ['#ff7f0e'] * n_heads

print(f"\n── Per-head G consistency (top-{K_h} subspace per head) ──")
print(f"{'Head':6s}  {'QK↔V':8s}  {'QK↔O':8s}  {'V↔O':8s}")

per_head_cos_QKV = []
per_head_cos_QKO = []
per_head_cos_VO  = []

for i, lbl in enumerate(head_labels):
    Gh  = G_heads[i]
    VVh = WVV_heads[i]
    OOh = WOO_heads[i]

    evQK, _ = top_eigvecs(Gh,  K_h)
    evV,  _ = top_eigvecs(VVh, K_h)
    evO,  _ = top_eigvecs(OOh, K_h)

    c_qkv = principal_angle_cosines(evQK, evV).mean()
    c_qko = principal_angle_cosines(evQK, evO).mean()
    c_vo  = principal_angle_cosines(evV,  evO).mean()

    per_head_cos_QKV.append(c_qkv)
    per_head_cos_QKO.append(c_qko)
    per_head_cos_VO.append(c_vo)

    print(f"  {lbl:6s}  {c_qkv:.4f}    {c_qko:.4f}    {c_vo:.4f}")

per_head_cos_QKV = np.array(per_head_cos_QKV)
per_head_cos_QKO = np.array(per_head_cos_QKO)
per_head_cos_VO  = np.array(per_head_cos_VO)

print(f"\n  Mean across heads:")
print(f"  G_QK^h vs G_V^h: {per_head_cos_QKV.mean():.4f}  (global mean was {cos_QK_V.mean():.4f})")
print(f"  G_QK^h vs G_O^h: {per_head_cos_QKO.mean():.4f}  (global mean was {cos_QK_O.mean():.4f})")
print(f"  G_V^h  vs G_O^h: {per_head_cos_VO.mean():.4f}   (global mean was {cos_V_O.mean():.4f})")


# ── W_E projection mass ───────────────────────────────────────────────────────

def we_subspace(W_E, var_threshold=0.90):
    _, sv, Vt = np.linalg.svd(W_E, full_matrices=False)
    cumvar = np.cumsum(sv**2) / (sv**2).sum()
    k = int(np.searchsorted(cumvar, var_threshold)) + 1
    print(f"  W_E: top-{k} right singular vectors capture {var_threshold*100:.0f}% variance")
    return Vt[:k].T

def we_mass(vecs, V_we):
    """vecs: (d_model, n). Returns W_E projection mass for each column."""
    proj = vecs.T @ V_we
    return (proj**2).sum(axis=1)

V_we = we_subspace(W_E)

mass_G = we_mass(evecs_GQK, V_we)
mass_B = we_mass(evecs_B,   V_we)

print(f"\nW_E projection mass (top-{K} modes):")
print(f"  G_QK eigenvectors — mean: {mass_G.mean():.3f}")
print(f"  B_QK singular vecs — mean: {mass_B.mean():.3f}")


# ── Per-head content vs compute profile ──────────────────────────────────────
# For each head: how much of sym(W_QK^h) lives in G subspace?
#                how much of anti(W_QK^h) lives in B subspace?
# Also: W_E mass of per-head G^h top eigvec and B^h top singvec.

P_G = evecs_GQK @ evecs_GQK.T   # projection onto shared G subspace (512, 512)
P_B = evecs_B   @ evecs_B.T     # projection onto shared B subspace

g_in_G, b_in_B = [], []
we_mass_g_per_head, we_mass_b_per_head = [], []

for i in range(n_heads_total):
    Gh = G_heads[i]
    Bh = B_heads[i]

    # fraction of Frobenius norm captured by shared subspace
    g_in_G.append(np.linalg.norm(P_G @ Gh @ P_G) / (np.linalg.norm(Gh) + 1e-10))
    b_in_B.append(np.linalg.norm(P_B @ Bh @ P_B) / (np.linalg.norm(Bh) + 1e-10))

    # W_E mass of this head's top G eigvec and top B singvec
    ev_h, evec_h = np.linalg.eigh(Gh)
    top_g = evec_h[:, np.argmax(np.abs(ev_h))].reshape(-1, 1)
    U_bh, _, _ = np.linalg.svd(Bh)
    top_b = U_bh[:, :1]

    we_mass_g_per_head.append(float(we_mass(top_g, V_we)[0]))
    we_mass_b_per_head.append(float(we_mass(top_b, V_we)[0]))

g_in_G = np.array(g_in_G)
b_in_B = np.array(b_in_B)
we_mass_g_per_head = np.array(we_mass_g_per_head)
we_mass_b_per_head = np.array(we_mass_b_per_head)

print(f"\nPer-head G fraction in shared G subspace:")
for i, lbl in enumerate(head_labels):
    print(f"  {lbl}: G_in_G={g_in_G[i]:.3f}  B_in_B={b_in_B[i]:.3f}  "
          f"WE_mass_g={we_mass_g_per_head[i]:.3f}  WE_mass_b={we_mass_b_per_head[i]:.3f}")


# ── FIGURES ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# ── Top-left: principal angle cosines ────────────────────────────────────────
ax = axes[0, 0]
x = np.arange(1, K + 1)
ax.plot(x, cos_QK_V, 'o-', ms=4, lw=1.2, label='G_QK vs G_V', color='#2ca02c')
ax.plot(x, cos_QK_O, 's-', ms=4, lw=1.2, label='G_QK vs G_O', color='#d62728')
ax.plot(x, cos_V_O,  '^-', ms=4, lw=1.2, label='G_V  vs G_O',  color='#9467bd')
ax.plot(x, cos_G_B,  'x-', ms=4, lw=1.0, label='G_QK vs B_QK (should→0)', color='gray', alpha=0.7)
ax.axhline(1.0, color='black', lw=0.5, linestyle='--', alpha=0.3)
ax.axhline(0.0, color='black', lw=0.5, linestyle='--', alpha=0.3)
ax.set_xlabel("Principal angle index", fontsize=9)
ax.set_ylabel("cos(angle)", fontsize=9)
ax.set_title("G subspace consistency: principal angle cosines\n"
             "between top-32 eigenspaces of G_QK, G_V, G_O", fontsize=9)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)
ax.set_ylim(-0.05, 1.1)

# ── Top-right: eigenspectra comparison (normalized) ──────────────────────────
ax = axes[0, 1]
k_plot = 64
ev_qk = np.sort(np.abs(np.linalg.eigh(G_QK)[0]))[::-1][:k_plot]
ev_v  = np.sort(np.abs(np.linalg.eigh(G_V)[0]))[::-1][:k_plot]
ev_o  = np.sort(np.abs(np.linalg.eigh(G_O)[0]))[::-1][:k_plot]
# normalize by max eigenvalue so scales are comparable
ax.semilogy(np.arange(1, k_plot+1), ev_qk / ev_qk[0], 'o-', ms=3, lw=1, label='G_QK', color='#1f77b4')
ax.semilogy(np.arange(1, k_plot+1), ev_v  / ev_v[0],  's-', ms=3, lw=1, label='G_V',  color='#2ca02c')
ax.semilogy(np.arange(1, k_plot+1), ev_o  / ev_o[0],  '^-', ms=3, lw=1, label='G_O',  color='#d62728')
ax.set_xlabel("Eigenvalue rank", fontsize=9)
ax.set_ylabel("|eigenvalue| / max (log scale)", fontsize=9)
ax.set_title("Normalized eigenspectra of G_QK, G_V, G_O\n(sloppy structure)", fontsize=9)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# ── Bottom-left: W_E projection mass — G vs B global ─────────────────────────
ax = axes[1, 0]
bins = np.linspace(0, 1, 25)
ax.hist(mass_G, bins=bins, alpha=0.6, color='#2ca02c', label=f'G eigenvectors (mean={mass_G.mean():.2f})', density=True)
ax.hist(mass_B, bins=bins, alpha=0.6, color='#1f77b4', label=f'B singular vecs (mean={mass_B.mean():.2f})', density=True)
ax.axvline(mass_G.mean(), color='#2ca02c', lw=1.5, linestyle='--')
ax.axvline(mass_B.mean(), color='#1f77b4', lw=1.5, linestyle='--')
ax.set_xlabel("W_E projection mass (90% var subspace)", fontsize=9)
ax.set_ylabel("Density", fontsize=9)
ax.set_title("Content vs compute: G lives in W_E space, B does not\n(attn-only-2l)", fontsize=9)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# ── Bottom-right: per-head scatter — W_E mass of top G mode vs top B mode ────
ax = axes[1, 2]
for i, lbl in enumerate(head_labels):
    ax.scatter(we_mass_g_per_head[i], we_mass_b_per_head[i],
               color=colors_layer[i], s=60, zorder=3)
    ax.annotate(lbl, (we_mass_g_per_head[i], we_mass_b_per_head[i]),
                fontsize=7, xytext=(4, 2), textcoords='offset points')
ax.set_xlabel("W_E mass of top G^h eigenvector (content)", fontsize=9)
ax.set_ylabel("W_E mass of top B^h singular vector (compute)", fontsize=9)
ax.set_title("Per-head content vs compute geometry\nblue=L0, orange=L1", fontsize=9)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.axhline(0.5, color='gray', lw=0.5, linestyle='--', alpha=0.5)
ax.axvline(0.5, color='gray', lw=0.5, linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.2)

# legend for layers
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#1f77b4', label='L0'), Patch(color='#ff7f0e', label='L1')],
          fontsize=8, loc='upper right')

# ── Bottom-middle: per-head G consistency ────────────────────────────────────
ax = axes[1, 1]
x = np.arange(n_heads_total)
w = 0.25
ax.bar(x - w, per_head_cos_QKV, width=w, label='G_QK^h vs G_V^h', color='#2ca02c', alpha=0.8)
ax.bar(x,     per_head_cos_QKO, width=w, label='G_QK^h vs G_O^h', color='#d62728', alpha=0.8)
ax.bar(x + w, per_head_cos_VO,  width=w, label='G_V^h  vs G_O^h',  color='#9467bd', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(head_labels, rotation=45, fontsize=7)
ax.set_ylabel("Mean cos(principal angle)", fontsize=9)
ax.set_title(f"Per-head G consistency (top-{K_h} subspace)\n"
             "Do G_QK^h, G_V^h, G_O^h agree within each head?", fontsize=9)
ax.legend(fontsize=7)
ax.axhline(cos_QK_V.mean(), color='#2ca02c', lw=1, linestyle='--', alpha=0.5, label='global mean QK↔V')
ax.axhline(cos_QK_O.mean(), color='#d62728', lw=1, linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.2)
ax.set_ylim(0, 1.05)

# ── Bottom-right: per-head scatter ────────────────────────────────────────────
ax = axes[1, 2]

plt.suptitle(
    "attn-only-2l: G and B as geometric foundation\n"
    "G_QK (W_QK sym), G_V (W_V outer products), G_O (W_O outer products) share eigenvectors\n"
    "Content vs compute falls out of G vs B subspace structure",
    fontsize=10
)
plt.tight_layout()
plt.savefig("figs/exp5_geometric_foundation.png", dpi=300, bbox_inches="tight")
print("\nSaved figs/exp5_geometric_foundation.png")
print("\nDone.")

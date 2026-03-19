"""
Post 4 — QK Metric, Experiment 2: G eigenvectors split into token vs compute subspace
Run with: /opt/miniconda3/bin/python run_exp2_compute_subspace.py

Key idea:
  W_E spans only ~50-100 effective dims of the 768-dim residual stream.
  G_crude eigenvectors that project strongly onto W_E = token-identity directions.
  G_crude eigenvectors that project weakly onto W_E = compute directions.

  For a 2-layer attention-only model (no MLPs), the compute subspace is
  entirely defined by what L0 heads write (W_O) and L1 heads read (W_K).
  Prediction: G's stiff compute eigenvectors align with W_O of L0 heads
  AND W_K of L1 heads — specifically the K-composition channel.

Models:
  - gpt2-small (loaded from exp1 .npy cache if available)
  - attn-only-2l (Elhage-style, the cleanest test)
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


# ── Helper: compute G_crude from a model ─────────────────────────────────────

def compute_G_crude(model):
    """Mean of W_QK^h_sym = (W_Q^h W_K^{h,T} + W_K^h W_Q^{h,T}) / 2 across all heads."""
    W_Q = model.W_Q.cpu().numpy()   # (n_layers, n_heads, d_model, d_head)
    W_K = model.W_K.cpu().numpy()
    n_layers, n_heads, d_model, _ = W_Q.shape
    sym_sum = np.zeros((d_model, d_model))
    count = 0
    for l in range(n_layers):
        for h in range(n_heads):
            WQK = W_Q[l, h] @ W_K[l, h].T   # (d_model, d_model)
            sym_sum += (WQK + WQK.T) / 2.0
            count += 1
    return sym_sum / count, n_layers, n_heads, d_model


def eigen_pd(G):
    """Eigendecompose G (symmetric), return descending eigenvalues + eigenvectors.
    Also return PD version (zero out negative eigenvalues)."""
    eigvals, eigvecs = np.linalg.eigh(G)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx].copy()
    eigvecs = eigvecs[:, idx].copy()
    G_pd = eigvecs @ np.diag(np.maximum(eigvals, 0)) @ eigvecs.T
    return eigvals, eigvecs, G_pd


def we_projection_mass(eigvecs, W_E, k=None):
    """For each eigenvector, compute fraction of its energy in W_E's row space.
    Uses top-k right singular vectors of W_E (default: all with sv > 1% of max)."""
    _, S, Vt = np.linalg.svd(W_E, full_matrices=False)  # Vt: (min, d_model)
    if k is None:
        k = int((S > 0.01 * S[0]).sum())
    V_top = Vt[:k].T   # (d_model, k) — W_E's row space basis
    # projection mass for each eigvec: ||V_top^T v||^2 / ||v||^2
    proj = eigvecs.T @ V_top   # (d_model, k)
    mass = (proj ** 2).sum(axis=1)   # (d_model,) — already normalized since ||v||=1
    print(f"  W_E effective rank (sv > 1% max): {k}")
    return mass, k


def subspace_alignment(vecs_A, vecs_B):
    """Principal angle alignment between two sets of vectors.
    Returns mean squared cosine of principal angles (0=orthogonal, 1=identical)."""
    # QR both sets
    Q_A, _ = np.linalg.qr(vecs_A)
    Q_B, _ = np.linalg.qr(vecs_B)
    # singular values of Q_A^T Q_B = cosines of principal angles
    sv = np.linalg.svd(Q_A.T @ Q_B, compute_uv=False)
    return (sv ** 2).mean()


# ── GPT-2 small ───────────────────────────────────────────────────────────────

print("=" * 60)
print("GPT-2 small")
print("=" * 60)

# Load cached G_crude if available from exp1, else recompute
if os.path.exists("G_crude.npy") and os.path.exists("G_crude_eigvals.npy"):
    print("Loading cached G_crude from exp1...")
    G_gpt2 = np.load("G_crude.npy")
    eigvals_gpt2 = np.load("G_crude_eigvals.npy")
    eigvecs_gpt2 = np.load("G_crude_eigvecs.npy")
else:
    print("Recomputing G_crude for GPT-2 small...")
    model_gpt2 = HookedTransformer.from_pretrained("gpt2-small")
    G_gpt2, _, _, _ = compute_G_crude(model_gpt2)
    eigvals_gpt2, eigvecs_gpt2, _ = eigen_pd(G_gpt2)
    model_gpt2 = None

# Load GPT-2 just for W_E
model_gpt2 = HookedTransformer.from_pretrained("gpt2-small")
W_E_gpt2 = model_gpt2.W_E.cpu().numpy()   # (50257, 768)
print(f"GPT-2 W_E shape: {W_E_gpt2.shape}")

mass_gpt2, k_gpt2 = we_projection_mass(eigvecs_gpt2, W_E_gpt2)
print(f"GPT-2: top-5 eigenvalues: {eigvals_gpt2[:5].round(4)}")
print(f"GPT-2: W_E projection mass of top-5 eigvecs: {mass_gpt2[:5].round(3)}")
model_gpt2 = None


# ── attn-only-2l ──────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("attn-only-2l")
print("=" * 60)

model_2l = HookedTransformer.from_pretrained("attn-only-2l")
print(f"Loaded: {model_2l.cfg.n_layers}L x {model_2l.cfg.n_heads}H, "
      f"d_model={model_2l.cfg.d_model}, d_head={model_2l.cfg.d_head}")

G_2l, n_layers_2l, n_heads_2l, d_model_2l = compute_G_crude(model_2l)
eigvals_2l, eigvecs_2l, G_pd_2l = eigen_pd(G_2l)

W_E_2l = model_2l.W_E.cpu().numpy()
print(f"attn-only-2l W_E shape: {W_E_2l.shape}")

mass_2l, k_2l = we_projection_mass(eigvecs_2l, W_E_2l)
print(f"attn-only-2l: top-10 eigenvalues: {eigvals_2l[:10].round(4)}")
print(f"attn-only-2l: W_E projection mass of top-10 eigvecs: {mass_2l[:10].round(3)}")

# Number of positive eigenvalues
n_pos_2l = (eigvals_2l > 0).sum()
n_neg_2l = (eigvals_2l < 0).sum()
print(f"attn-only-2l: {n_pos_2l} positive, {n_neg_2l} negative eigenvalues")

# ── Compute subspace analysis for attn-only-2l ───────────────────────────────
# Identify compute eigenvectors: stiff AND low W_E projection mass

# W_O shape: (n_layers, n_heads, d_head, d_model) — rows live in d_model
# W_K shape: (n_layers, n_heads, d_model, d_head) — columns live in d_model
W_O_2l = model_2l.W_O.cpu().numpy()   # (2, n_heads, d_head, d_model)
W_K_2l = model_2l.W_K.cpu().numpy()   # (2, n_heads, d_model, d_head)

print(f"\nW_O shape: {W_O_2l.shape}")
print(f"W_K shape: {W_K_2l.shape}")

# For each head: compute alignment of its W_O (layer 0) or W_K (layer 1) output
# subspace with the top compute eigenvectors of G

# Define compute eigenvectors: eigvecs with W_E mass < 0.3 AND eigval > 0
compute_mask = (mass_2l < 0.3) & (eigvals_2l > 0)
n_compute = compute_mask.sum()
print(f"\nCompute eigenvectors (W_E mass < 0.3, eigval > 0): {n_compute}")

compute_evecs = eigvecs_2l[:, compute_mask]   # (d_model, n_compute)

# Token eigenvectors: W_E mass > 0.3 AND eigval > 0
token_mask = (mass_2l > 0.3) & (eigvals_2l > 0)
n_token = token_mask.sum()
print(f"Token eigenvectors (W_E mass > 0.3, eigval > 0): {n_token}")

# Alignment of each head's W_O (L0) with compute vs token subspace
print("\nL0 head W_O alignment with compute subspace vs token subspace:")
for h in range(n_heads_2l):
    Wo = W_O_2l[0, h]   # (d_head, d_model) — rows = output directions in d_model
    Wo_T = Wo.T          # (d_model, d_head)
    align_compute = subspace_alignment(Wo_T, compute_evecs) if n_compute > 0 else 0
    align_token   = subspace_alignment(Wo_T, eigvecs_2l[:, token_mask]) if n_token > 0 else 0
    print(f"  L0H{h}: W_O → compute={align_compute:.3f}  token={align_token:.3f}")

print("\nL1 head W_K alignment with compute subspace vs token subspace:")
for h in range(n_heads_2l):
    Wk = W_K_2l[1, h]   # (d_model, d_head) — columns = input directions in d_model
    align_compute = subspace_alignment(Wk, compute_evecs) if n_compute > 0 else 0
    align_token   = subspace_alignment(Wk, eigvecs_2l[:, token_mask]) if n_token > 0 else 0
    print(f"  L1H{h}: W_K → compute={align_compute:.3f}  token={align_token:.3f}")

# Cross-layer K-composition: for each (L0 head, L1 head) pair,
# what is the alignment of W_O[L0] with W_K[L1]?
print("\nK-composition alignment W_O[L0,h] <-> W_K[L1,h'] (mean sq cosine of principal angles):")
kcomp = np.zeros((n_heads_2l, n_heads_2l))
for h0 in range(n_heads_2l):
    Wo = W_O_2l[0, h0].T   # (d_model, d_head)
    for h1 in range(n_heads_2l):
        Wk = W_K_2l[1, h1]   # (d_model, d_head)
        kcomp[h0, h1] = subspace_alignment(Wo, Wk)
print(kcomp.round(3))

model_2l = None


# ── FIGURE 1: Eigenvalue vs W_E projection mass — both models ────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, eigvals, mass, eigvecs, label, d_model in [
    (axes[0], eigvals_gpt2, mass_gpt2, eigvecs_gpt2, "GPT-2 small (12L×12H, d=768)", 768),
    (axes[1], eigvals_2l,   mass_2l,   eigvecs_2l,   "attn-only-2l (2L×8H, d=256)",  d_model_2l),
]:
    pos = eigvals > 0
    neg = eigvals <= 0

    sc_pos = ax.scatter(mass[pos], eigvals[pos], s=6, c='#2ca02c', alpha=0.5,
                         label='positive eigval')
    sc_neg = ax.scatter(mass[neg], np.abs(eigvals[neg]), s=6, c='#d62728', alpha=0.3,
                         label='|negative eigval|')

    ax.axvline(0.3, color='gray', lw=0.8, linestyle='--', label='W_E mass = 0.3')
    ax.set_xlabel("W_E projection mass  (fraction of eigvec energy in W_E row space)",
                  fontsize=9)
    ax.set_ylabel("|Eigenvalue|", fontsize=9)
    ax.set_title(label, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Annotate: label top-3 stiff compute directions
    compute_stiff = np.where((mass < 0.3) & (eigvals > 0))[0]
    if len(compute_stiff) > 0:
        top_compute = compute_stiff[np.argsort(eigvals[compute_stiff])[::-1][:3]]
        for idx in top_compute:
            ax.annotate(f"EV{idx+1}", (mass[idx], eigvals[idx]),
                        fontsize=7, color='#2ca02c',
                        xytext=(mass[idx]+0.02, eigvals[idx]),
                        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

plt.suptitle(
    "G_crude eigenvectors: token-identity (right) vs compute (left) directions\n"
    "Stiff + low W_E mass = shared compute channels; Stiff + high W_E mass = shared token geometry",
    fontsize=11
)
plt.tight_layout()
plt.savefig("figs/exp2_token_vs_compute.png", dpi=300, bbox_inches="tight")
print("\nSaved figs/exp2_token_vs_compute.png")


# ── FIGURE 2 (attn-only-2l): K-composition heatmap ───────────────────────────

fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))

# Left: K-composition W_O[L0] <-> W_K[L1]
im = axes2[0].imshow(kcomp, cmap='hot', vmin=0, vmax=kcomp.max())
axes2[0].set_xlabel("L1 head (W_K)", fontsize=9)
axes2[0].set_ylabel("L0 head (W_O)", fontsize=9)
axes2[0].set_title("K-composition: W_O[L0,h] ↔ W_K[L1,h']\n(mean sq cos of principal angles)",
                    fontsize=9)
plt.colorbar(im, ax=axes2[0])

# Middle: L0 W_O alignment with G compute eigenvectors
n_heads_2l_val = W_O_2l.shape[1]
model_2l_reload = HookedTransformer.from_pretrained("attn-only-2l")
W_O_2l_r = model_2l_reload.W_O.cpu().numpy()
W_K_2l_r = model_2l_reload.W_K.cpu().numpy()
model_2l_reload = None

wo_compute = np.array([
    subspace_alignment(W_O_2l_r[0, h].T, compute_evecs) if n_compute > 0 else 0
    for h in range(n_heads_2l_val)
])
wk_compute = np.array([
    subspace_alignment(W_K_2l_r[1, h], compute_evecs) if n_compute > 0 else 0
    for h in range(n_heads_2l_val)
])

x = np.arange(n_heads_2l_val)
w = 0.35
axes2[1].bar(x - w/2, wo_compute, w, label='L0 W_O → compute', color='#1f77b4')
axes2[1].bar(x + w/2, wk_compute, w, label='L1 W_K → compute', color='#ff7f0e')
axes2[1].set_xticks(x)
axes2[1].set_xticklabels([f"H{h}" for h in x], fontsize=8)
axes2[1].set_ylabel("Alignment with G compute eigvecs", fontsize=9)
axes2[1].set_title("attn-only-2l: which heads use the\nshared compute subspace?", fontsize=9)
axes2[1].legend(fontsize=8)
axes2[1].grid(True, alpha=0.2, axis='y')

# Right: eigenvalue spectrum for attn-only-2l colored by W_E mass
sc = axes2[2].scatter(
    range(1, d_model_2l + 1), eigvals_2l,
    c=mass_2l, cmap='RdYlGn_r', s=8, vmin=0, vmax=1
)
axes2[2].axhline(0, color='gray', lw=0.5, linestyle='--')
axes2[2].set_xlabel("Eigenvalue rank", fontsize=9)
axes2[2].set_ylabel("Eigenvalue", fontsize=9)
axes2[2].set_title("attn-only-2l G eigenspectrum\n(color = W_E projection mass)", fontsize=9)
plt.colorbar(sc, ax=axes2[2], label='W_E mass')

plt.suptitle("attn-only-2l: G compute subspace = K-composition channels?", fontsize=11)
plt.tight_layout()
plt.savefig("figs/exp2_2l_kcomposition.png", dpi=300, bbox_inches="tight")
print("Saved figs/exp2_2l_kcomposition.png")

print("\nDone.")

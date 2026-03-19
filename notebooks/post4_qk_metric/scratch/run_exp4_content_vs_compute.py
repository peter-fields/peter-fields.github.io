"""
Post 4 — QK Metric, Experiment 4: Content vs compute via W_E projection
Run with: /opt/miniconda3/bin/python run_exp4_content_vs_compute.py

Key question: do G eigenvectors and B singular vectors live in different
parts of the residual stream?

  G eigenvectors: should be in W_E's content subspace (token identity)
  B singular vectors: might be more in compute subspace (K-composition channels)

W_E projection defined properly: top-k right singular vectors of W_E
capturing 90% of its Frobenius variance — the genuine token-identity subspace.

Also: filtered vocab projections for B (remove outlier tokens by W_E norm).
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


def compute_GB(model):
    W_Q = model.W_Q.cpu().numpy()
    W_K = model.W_K.cpu().numpy()
    n_layers, n_heads, d_model, _ = W_Q.shape
    G_all, B_all = [], []
    for l in range(n_layers):
        for h in range(n_heads):
            WQK = W_Q[l, h] @ W_K[l, h].T
            G_all.append((WQK + WQK.T) / 2.0)
            B_all.append((WQK - WQK.T) / 2.0)
    G_crude = np.array(G_all).mean(axis=0)
    B_crude = np.array(B_all).mean(axis=0)
    return G_crude, B_crude, d_model


def we_subspace(W_E, var_threshold=0.90):
    """Right singular vectors of W_E capturing var_threshold of Frobenius variance."""
    _, sv, Vt = np.linalg.svd(W_E, full_matrices=False)
    cumvar = np.cumsum(sv**2) / (sv**2).sum()
    k = int(np.searchsorted(cumvar, var_threshold)) + 1
    print(f"  W_E: top-{k} singular vectors capture {var_threshold*100:.0f}% variance "
          f"(out of {Vt.shape[0]} total, d_model={Vt.shape[1]})")
    return Vt[:k].T, sv, k   # (d_model, k)


def we_projection_mass(vecs, V_we):
    """vecs: (d_model, n) — each column is a unit vector.
    Returns projection mass onto W_E subspace for each column."""
    proj = vecs.T @ V_we   # (n, k)
    return (proj**2).sum(axis=1)   # (n,)


def filter_vocab(W_E, percentile=99):
    """Return mask of 'normal' tokens: W_E row norm below given percentile."""
    norms = np.linalg.norm(W_E, axis=1)
    threshold = np.percentile(norms, percentile)
    return norms <= threshold


def vocab_proj(vec, W_E, model, mask, top_n=10):
    """Top tokens for a direction vec, filtered by mask."""
    scores = W_E @ vec
    scores_filtered = scores.copy()
    scores_filtered[~mask] = -np.inf
    top_pos = np.argsort(scores_filtered)[-top_n:][::-1]
    scores_filtered2 = scores.copy()
    scores_filtered2[~mask] = np.inf
    top_neg = np.argsort(scores_filtered2)[:top_n]
    pos_tokens = [model.to_string([i]).strip() for i in top_pos]
    neg_tokens = [model.to_string([i]).strip() for i in top_neg]
    return pos_tokens, neg_tokens


# ── Run on both models ────────────────────────────────────────────────────────

results = {}

for model_name, label in [("gpt2-small", "GPT-2 small"), ("attn-only-2l", "attn-only-2l")]:
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")

    model = HookedTransformer.from_pretrained(model_name)
    W_E = model.W_E.cpu().numpy()
    G_crude, B_crude, d_model = compute_GB(model)

    # W_E subspace (90% variance)
    V_we, sv_we, k_we = we_subspace(W_E, var_threshold=0.90)

    # Vocab filter
    mask = filter_vocab(W_E, percentile=99)
    print(f"  Vocab: {mask.sum()} / {len(mask)} tokens after filtering top-1% by norm")

    # G eigenvectors
    eigvals_G, eigvecs_G = np.linalg.eigh(G_crude)
    idx = np.argsort(eigvals_G)[::-1]
    eigvals_G = eigvals_G[idx].copy()
    eigvecs_G = eigvecs_G[:, idx].copy()

    # B singular vectors
    U_B, sv_B, Vt_B = np.linalg.svd(B_crude)

    # W_E projection mass for all G eigenvectors and B left singular vectors
    mass_G = we_projection_mass(eigvecs_G, V_we)
    mass_B_u = we_projection_mass(U_B, V_we)       # query directions
    mass_B_v = we_projection_mass(Vt_B.T, V_we)    # key directions

    print(f"\n  G eigenvectors W_E projection mass:")
    print(f"    Top-10 positive eigvecs: {mass_G[:10].round(3)}")
    print(f"    Mean (all positive eigvecs): {mass_G[eigvals_G>0].mean():.3f}")
    print(f"    Mean (all negative eigvecs): {mass_G[eigvals_G<0].mean():.3f}")

    print(f"\n  B left singular vectors W_E projection mass:")
    print(f"    Top-10 modes: {mass_B_u[:10].round(3)}")
    print(f"    Mean (all): {mass_B_u.mean():.3f}")

    print(f"\n  B right singular vectors W_E projection mass:")
    print(f"    Top-10 modes: {mass_B_v[:10].round(3)}")
    print(f"    Mean (all): {mass_B_v.mean():.3f}")

    # Is B systematically lower W_E projection than G?
    mean_G_pos = mass_G[eigvals_G > 0].mean()
    mean_B = (mass_B_u.mean() + mass_B_v.mean()) / 2
    print(f"\n  G (positive eigvecs) mean W_E mass: {mean_G_pos:.3f}")
    print(f"  B (avg u/v) mean W_E mass:           {mean_B:.3f}")
    print(f"  Difference (G - B): {mean_G_pos - mean_B:.3f}")

    # Filtered vocab projections for top B modes
    print(f"\n  B_crude top singular modes (filtered vocab):")
    for k in range(min(6, len(sv_B))):
        u = U_B[:, k]
        v = Vt_B[k, :]
        pos_u, neg_u = vocab_proj(u, W_E, model, mask)
        pos_v, neg_v = vocab_proj(v, W_E, model, mask)
        print(f"    SV{k+1} (σ={sv_B[k]:.4f})")
        print(f"      query+ : {pos_u[:8]}")
        print(f"      query- : {neg_u[:8]}")
        print(f"      key+   : {pos_v[:8]}")
        print(f"      key-   : {neg_v[:8]}")

    # Filtered vocab projections for top G eigenvectors
    print(f"\n  G_crude top eigenvectors (filtered vocab):")
    for k in range(min(6, d_model)):
        v = eigvecs_G[:, k]
        pos_v, neg_v = vocab_proj(v, W_E, model, mask)
        print(f"    EV{k+1} (λ={eigvals_G[k]:.4f})")
        print(f"      + : {pos_v[:8]}")
        print(f"      - : {neg_v[:8]}")

    results[model_name] = dict(
        eigvals_G=eigvals_G, eigvecs_G=eigvecs_G,
        sv_B=sv_B, U_B=U_B, Vt_B=Vt_B,
        mass_G=mass_G, mass_B_u=mass_B_u, mass_B_v=mass_B_v,
        d_model=d_model, label=label
    )
    model = None


# ── FIGURE: W_E projection mass distribution — G vs B ────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

for col, mname in enumerate(["gpt2-small", "attn-only-2l"]):
    r = results[mname]
    eigvals_G = r['eigvals_G']
    mass_G    = r['mass_G']
    mass_B_u  = r['mass_B_u']
    mass_B_v  = r['mass_B_v']
    sv_B      = r['sv_B']
    lbl       = r['label']

    # Top row: scatter eigenvalue vs W_E mass, colored G pos/neg and B
    ax = axes[0, col]
    n_G = len(eigvals_G)
    pos_G = eigvals_G > 0
    ax.scatter(mass_G[pos_G],  eigvals_G[pos_G],  s=5, alpha=0.5,
               color='#2ca02c', label='G positive eigvec')
    ax.scatter(mass_G[~pos_G], np.abs(eigvals_G[~pos_G]), s=5, alpha=0.3,
               color='#d62728', label='|G negative eigvec|')
    ax.scatter(mass_B_u[:20], sv_B[:20], s=30, alpha=0.9,
               color='#1f77b4', marker='^', label='B left sv (query dir)')
    ax.scatter(mass_B_v[:20], sv_B[:20], s=30, alpha=0.9,
               color='#ff7f0e', marker='v', label='B right sv (key dir)')
    ax.set_xlabel("W_E projection mass (90% var subspace)", fontsize=9)
    ax.set_ylabel("|Eigenvalue| / Singular value", fontsize=9)
    ax.set_title(f"{lbl}\nG eigvecs vs B singular vectors", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # Bottom row: histogram comparison
    ax = axes[1, col]
    bins = np.linspace(0, 1, 30)
    ax.hist(mass_G[pos_G],  bins=bins, alpha=0.5, color='#2ca02c',
            label='G positive eigvecs', density=True)
    ax.hist(mass_B_u,        bins=bins, alpha=0.5, color='#1f77b4',
            label='B singular vecs (query)', density=True)
    ax.axvline(mass_G[pos_G].mean(),  color='#2ca02c', lw=1.5, linestyle='--',
               label=f'G mean={mass_G[pos_G].mean():.2f}')
    ax.axvline(mass_B_u.mean(), color='#1f77b4', lw=1.5, linestyle='--',
               label=f'B mean={mass_B_u.mean():.2f}')
    ax.set_xlabel("W_E projection mass", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title(f"{lbl}\nW_E projection mass distribution", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

plt.suptitle(
    "Content vs compute: do G eigenvectors and B singular vectors\n"
    "live in different parts of the residual stream?",
    fontsize=11
)
plt.tight_layout()
plt.savefig("figs/exp4_content_vs_compute.png", dpi=300, bbox_inches="tight")
print("\nSaved figs/exp4_content_vs_compute.png")
print("\nDone.")

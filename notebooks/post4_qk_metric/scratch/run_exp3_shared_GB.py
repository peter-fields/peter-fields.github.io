"""
Post 4 — QK Metric, Experiment 3: Shared G (symmetric) and B (2-form) across heads
Run with: /opt/miniconda3/bin/python run_exp3_shared_GB.py

W_QK^h = G^h + B^h
  G^h = (W_QK^h + W_QK^h^T) / 2   symmetric bilinear form (proto-metric)
  B^h = (W_QK^h - W_QK^h^T) / 2   antisymmetric 2-form (directed routing)

Shared structure:
  G_crude = mean_h G^h   -- shared content-matching geometry
  B_crude = mean_h B^h   -- shared directed-routing geometry

B_crude is real skew-symmetric. Analyzed via SVD: singular vectors (u, v) mean
"query direction u prefers key direction v" (directed, not mutual).

Key question: are G_crude and B_crude low-rank (genuine shared geometry)
or flat (each head is idiosyncratic)?
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
    """Compute W_QK^h, G^h, B^h for all heads. Return stacked arrays."""
    W_Q = model.W_Q.cpu().numpy()   # (n_layers, n_heads, d_model, d_head)
    W_K = model.W_K.cpu().numpy()
    n_layers, n_heads, d_model, _ = W_Q.shape
    G_all = []
    B_all = []
    for l in range(n_layers):
        for h in range(n_heads):
            WQK = W_Q[l, h] @ W_K[l, h].T
            G_all.append((WQK + WQK.T) / 2.0)
            B_all.append((WQK - WQK.T) / 2.0)
    return (np.array(G_all), np.array(B_all),
            n_layers, n_heads, d_model)


def analyze_model(name, model, W_E, label):
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")

    G_all, B_all, n_layers, n_heads, d_model = compute_GB(model)
    n_heads_total = G_all.shape[0]

    G_crude = G_all.mean(axis=0)   # (d_model, d_model)
    B_crude = B_all.mean(axis=0)   # (d_model, d_model) skew-symmetric

    print(f"G_crude Frobenius norm: {np.linalg.norm(G_crude):.4f}")
    print(f"B_crude Frobenius norm: {np.linalg.norm(B_crude):.4f}")
    print(f"Symmetry check B_crude (should be ~0): "
          f"{np.linalg.norm(B_crude + B_crude.T):.6f}")

    # ── G: eigendecomposition ─────────────────────────────────────────────────
    eigvals_G, eigvecs_G = np.linalg.eigh(G_crude)
    idx = np.argsort(eigvals_G)[::-1]
    eigvals_G = eigvals_G[idx].copy()
    eigvecs_G = eigvecs_G[:, idx].copy()
    n_pos_G = (eigvals_G > 0).sum()
    n_neg_G = (eigvals_G < 0).sum()
    print(f"\nG_crude: {n_pos_G} positive, {n_neg_G} negative eigenvalues")
    print(f"Top-5 G eigenvalues: {eigvals_G[:5].round(4)}")
    print(f"Bottom-5 G eigenvalues: {eigvals_G[-5:].round(4)}")

    # ── B: SVD (skew-symmetric, so singular values come in pairs) ────────────
    U_B, sv_B, Vt_B = np.linalg.svd(B_crude)
    print(f"\nB_crude top-10 singular values: {sv_B[:10].round(4)}")
    print(f"B_crude bottom-5 singular values: {sv_B[-5:].round(6)}")

    # ── Vocab projections ─────────────────────────────────────────────────────
    N = 5   # top N modes to inspect

    print(f"\nG_crude top eigenvectors — vocab projection (W_E @ v):")
    for k in range(N):
        v = eigvecs_G[:, k]
        scores = W_E @ v
        top = [model.to_string([i]) for i in np.argsort(scores)[-8:][::-1]]
        bot = [model.to_string([i]) for i in np.argsort(scores)[:8]]
        print(f"  EV{k+1} (λ={eigvals_G[k]:.4f})  + {top}  - {bot}")

    print(f"\nB_crude top singular modes — directed routing (query→key):")
    for k in range(N):
        u = U_B[:, k]       # query direction
        v = Vt_B[k, :]     # key direction
        scores_u = W_E @ u
        scores_v = W_E @ v
        top_u = [model.to_string([i]) for i in np.argsort(scores_u)[-8:][::-1]]
        top_v = [model.to_string([i]) for i in np.argsort(scores_v)[-8:][::-1]]
        print(f"  SV{k+1} (σ={sv_B[k]:.4f})")
        print(f"    query (u): {top_u}")
        print(f"    key   (v): {top_v}")

    # ── Per-head variance around shared G and B ───────────────────────────────
    # How much does each head deviate from the shared structure?
    G_resid_norms = np.array([np.linalg.norm(G_all[h] - G_crude)
                               for h in range(n_heads_total)])
    B_resid_norms = np.array([np.linalg.norm(B_all[h] - B_crude)
                               for h in range(n_heads_total)])
    print(f"\nPer-head residual from shared G — mean: {G_resid_norms.mean():.4f}, "
          f"std: {G_resid_norms.std():.4f}")
    print(f"Per-head residual from shared B — mean: {B_resid_norms.mean():.4f}, "
          f"std: {B_resid_norms.std():.4f}")
    print(f"G_crude norm / mean head G norm: "
          f"{np.linalg.norm(G_crude) / np.array([np.linalg.norm(G_all[h]) for h in range(n_heads_total)]).mean():.4f}")
    print(f"B_crude norm / mean head B norm: "
          f"{np.linalg.norm(B_crude) / np.array([np.linalg.norm(B_all[h]) for h in range(n_heads_total)]).mean():.4f}")

    return (G_crude, B_crude, eigvals_G, eigvecs_G, sv_B, U_B, Vt_B,
            G_all, B_all, d_model)


# ── Run on both models ────────────────────────────────────────────────────────

model_gpt2 = HookedTransformer.from_pretrained("gpt2-small")
W_E_gpt2 = model_gpt2.W_E.cpu().numpy()
res_gpt2 = analyze_model("gpt2", model_gpt2, W_E_gpt2, "GPT-2 small (12L×12H, d=768)")
model_gpt2 = None

model_2l = HookedTransformer.from_pretrained("attn-only-2l")
W_E_2l = model_2l.W_E.cpu().numpy()
res_2l = analyze_model("attn-only-2l", model_2l, W_E_2l, "attn-only-2l (2L×8H, d=512)")
model_2l = None


# ── FIGURE: G and B singular value spectra — both models ─────────────────────

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

for col, (res, lbl) in enumerate([
    (res_gpt2, "GPT-2 small"),
    (res_2l,   "attn-only-2l"),
]):
    (G_crude, B_crude, eigvals_G, eigvecs_G, sv_B, U_B, Vt_B,
     G_all, B_all, d_model) = res

    # Top row: G eigenvalue spectrum
    ax = axes[0, col]
    pos = eigvals_G > 0
    neg = eigvals_G <= 0
    ax.plot(np.where(pos)[0] + 1, eigvals_G[pos], '.', ms=3,
            color='#2ca02c', label='positive', alpha=0.7)
    ax.plot(np.where(neg)[0] + 1, np.abs(eigvals_G[neg]), '.', ms=3,
            color='#d62728', label='|negative|', alpha=0.5)
    ax.axhline(0, color='gray', lw=0.5, linestyle='--')
    ax.set_xlabel("Eigenvalue rank", fontsize=9)
    ax.set_ylabel("Eigenvalue", fontsize=9)
    ax.set_title(f"{lbl} — G_crude eigenspectrum", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Bottom row: B singular value spectrum
    ax = axes[1, col]
    ax.plot(range(1, len(sv_B) + 1), sv_B, 'k-', lw=0.8, alpha=0.7)
    ax.set_xlabel("Singular value rank", fontsize=9)
    ax.set_ylabel("Singular value", fontsize=9)
    ax.set_title(f"{lbl} — B_crude singular values\n"
                 f"(skew-symmetric 2-form, directed routing)", fontsize=10)
    ax.grid(True, alpha=0.2)

    # Annotate ratio: how much of total sv is in top-10?
    frac = sv_B[:10].sum() / sv_B.sum()
    ax.annotate(f"Top-10 capture {frac:.1%} of total",
                xy=(10, sv_B[9]), xytext=(d_model * 0.3, sv_B[0] * 0.7),
                fontsize=8, color='#1f77b4',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

plt.suptitle(
    "Shared geometry across all heads: G_crude (metric) + B_crude (2-form)\n"
    "Low-rank structure in either = genuine shared geometry, not per-head noise",
    fontsize=11
)
plt.tight_layout()
plt.savefig("figs/exp3_GB_spectra.png", dpi=300, bbox_inches="tight")
print("\nSaved figs/exp3_GB_spectra.png")

# ── Save ─────────────────────────────────────────────────────────────────────
np.save("B_crude_gpt2.npy",  res_gpt2[1])
np.save("B_crude_2l.npy",    res_2l[1])
np.save("sv_B_gpt2.npy",     res_gpt2[4])
np.save("sv_B_2l.npy",       res_2l[4])
print("Saved B_crude and sv_B arrays.")
print("\nDone.")

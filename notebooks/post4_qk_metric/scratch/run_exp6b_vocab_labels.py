"""
Post 4 — Experiment 6b: Vocab labels for G basis directions (attn-only-2l)

The G basis from exp6 (induction head L1H5 eigenvectors) defines content directions.
Here we project those directions onto the token embedding space to ask:
"What tokens does each G direction distinguish?"

Method: project each G basis vector v_i onto W_E rows → get a score per token.
Top positive tokens = tokens this direction attends TO (keys).
Top negative tokens = tokens this direction attends AWAY FROM.

Also: show the sloppy (log-spaced) eigenvalue spectrum of G^{induction}.

Run with: /opt/miniconda3/bin/python run_exp6b_vocab_labels.py
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
W_E = model.W_E.cpu().numpy()   # (vocab, d_model)

# Tokenizer for decoding
tokenizer = model.tokenizer

# ── Identify reference heads (same logic as exp6) ─────────────────────────────

G_heads = {}
B_heads = {}
for l in range(n_layers):
    for h in range(n_heads):
        WQK = W_Q[l, h] @ W_K[l, h].T
        G_heads[(l, h)] = (WQK + WQK.T) / 2.0
        B_heads[(l, h)] = (WQK - WQK.T) / 2.0

G_norms_L1 = {h: np.linalg.norm(G_heads[(1, h)], 'fro') for h in range(n_heads)}
h_induction = max(G_norms_L1, key=G_norms_L1.get)
B_norms_L0  = {h: np.linalg.norm(B_heads[(0, h)], 'fro') for h in range(n_heads)}
h_prev      = max(B_norms_L0, key=B_norms_L0.get)
print(f"Induction head: L1H{h_induction},  Prev-token head: L0H{h_prev}")

# ── G basis from induction head ────────────────────────────────────────────────

K_G = 12

G_ind = G_heads[(1, h_induction)]
eigvals_G, eigvecs_G = np.linalg.eigh(G_ind)
idx = np.argsort(np.abs(eigvals_G))[::-1]
eigvals_G = eigvals_G[idx]
eigvecs_G = eigvecs_G[:, idx]
V_G = eigvecs_G[:, :K_G]   # (d_model, K_G)

# ── Sloppy spectrum: eigenvalue magnitudes on log scale ───────────────────────

fig_spec, ax_spec = plt.subplots(figsize=(8, 4))
ax_spec.semilogy(np.arange(1, K_G + 1), np.abs(eigvals_G[:K_G]),
                 'o-', color='#1f77b4', ms=6, lw=1.5)
ax_spec.set_xlabel("Eigenvalue rank", fontsize=10)
ax_spec.set_ylabel("|eigenvalue|  (log scale)", fontsize=10)
ax_spec.set_title(
    f"G^{{induction}} (L1H{h_induction}) eigenvalue spectrum — 'sloppy' structure\n"
    f"Log-spaced magnitudes → multi-scale content geometry",
    fontsize=9)
ax_spec.grid(True, alpha=0.3)
ax_spec.set_xticks(np.arange(1, K_G + 1))
plt.tight_layout()
plt.savefig("figs/exp6b_G_spectrum.png", dpi=300, bbox_inches="tight")
print("Saved figs/exp6b_G_spectrum.png")


# ── Vocab projections: filter outlier-norm tokens ─────────────────────────────

# W_E row norms — filter top 1% to remove outlier tokens
row_norms = np.linalg.norm(W_E, axis=1)
thresh = np.percentile(row_norms, 99)
mask = row_norms < thresh
W_E_filt = W_E[mask]
token_ids_filt = np.where(mask)[0]

N_top = 8   # tokens to show per direction

print(f"\nVocab projections (top {N_top} per G direction, outliers filtered):\n")
vocab_results = []

for i in range(K_G):
    v = V_G[:, i]
    scores = W_E_filt @ v   # (vocab_filt,) — projection score per token
    top_pos_idx = np.argsort(scores)[::-1][:N_top]
    top_neg_idx = np.argsort(scores)[:N_top]

    top_pos_tokens = [tokenizer.decode([token_ids_filt[j]]) for j in top_pos_idx]
    top_neg_tokens = [tokenizer.decode([token_ids_filt[j]]) for j in top_neg_idx]

    sign = "+" if eigvals_G[i] > 0 else "-"
    print(f"G dir {i+1:2d}  (λ={eigvals_G[i]:+.3f})")
    print(f"  + side: {top_pos_tokens}")
    print(f"  - side: {top_neg_tokens}")

    vocab_results.append({
        'i': i,
        'eigval': eigvals_G[i],
        'pos_tokens': top_pos_tokens,
        'neg_tokens': top_neg_tokens,
    })


# ── Figure: vocab label grid ──────────────────────────────────────────────────

K_show = 8
fig_vocab, axes_v = plt.subplots(K_show, 1, figsize=(10, K_show * 1.2))

for i, ax in enumerate(axes_v):
    r = vocab_results[i]
    pos_str = "  |  ".join([f'"{t.strip()}"' for t in r['pos_tokens'][:6]])
    neg_str = "  |  ".join([f'"{t.strip()}"' for t in r['neg_tokens'][:6]])
    ax.text(0.01, 0.65, f"+ {pos_str}", transform=ax.transAxes,
            fontsize=7.5, color='#2ca02c', va='center')
    ax.text(0.01, 0.25, f"- {neg_str}", transform=ax.transAxes,
            fontsize=7.5, color='#d62728', va='center')
    ax.set_ylabel(f"G dir {i+1}\nλ={r['eigval']:+.2f}", fontsize=7, rotation=0,
                  labelpad=55, va='center')
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

plt.suptitle(
    f"Vocab labels for G basis directions — L1H{h_induction} (induction head)\n"
    f"Each direction: tokens that score high (+) vs low (-) when projected onto W_E",
    fontsize=10, y=1.01)
plt.tight_layout()
plt.savefig("figs/exp6b_vocab_labels.png", dpi=300, bbox_inches="tight")
print("\nSaved figs/exp6b_vocab_labels.png")


# ── B basis vocab projections (prev-token head) ───────────────────────────────

K_B = 6
B_prev = B_heads[(0, h_prev)]
U_B, sv_B, Vt_B = np.linalg.svd(B_prev)

print(f"\n\nB basis (L0H{h_prev}) vocab projections (query side = U_B, key side = Vt_B):\n")
for j in range(K_B):
    u = U_B[:, j]    # query-side direction
    w = Vt_B[j, :]   # key-side direction

    q_scores = W_E_filt @ u
    k_scores = W_E_filt @ w

    top_q = [tokenizer.decode([token_ids_filt[idx]]) for idx in np.argsort(q_scores)[::-1][:N_top]]
    top_k = [tokenizer.decode([token_ids_filt[idx]]) for idx in np.argsort(k_scores)[::-1][:N_top]]

    print(f"B pair {j+1}  (σ={sv_B[j]:.3f})")
    print(f"  query side (attends FROM): {top_q}")
    print(f"  key   side (attends TO  ): {top_k}")

print("\nDone.")

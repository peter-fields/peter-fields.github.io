"""
Post 4 — Experiment 14: Token-token similarity under G and B

Once we have G and B, we can compute inner products between token embeddings:
  S^G[i,j] = W_E[i]^T G W_E[j]   — symmetric content similarity in attention geometry
  S^B[i,j] = W_E[i]^T B W_E[j]   — antisymmetric directed routing preference

S^G: "which tokens does the attention mechanism treat as similar for content matching?"
S^B: "which token i preferentially attends TO j, asymmetrically?"

No forward pass. No activations. Pure weight geometry.

G: from induction head L1H6 (BOS-cleaned, absolute eigenvalues — positive definite)
B: from prev-token head L0H3 (BOS-cleaned)

Run with: /opt/miniconda3/bin/python run_exp14_token_similarity.py
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
W_Q = model.W_Q.cpu().numpy()
W_K = model.W_K.cpu().numpy()
W_E = model.W_E.cpu().numpy()   # (vocab, d_model)
tokenizer = model.tokenizer

h_induction = 6; h_prev = 3

# ── Build G and B from reference heads ───────────────────────────────────────

WQK_ind  = W_Q[1, h_induction] @ W_K[1, h_induction].T
WQK_prev = W_Q[0, h_prev]      @ W_K[0, h_prev].T

# G: |G|^{induction} — positive definite, BOS-cleaned
G_raw = (WQK_ind + WQK_ind.T) / 2.0
eigvals_G, eigvecs_G = np.linalg.eigh(G_raw)
G = eigvecs_G @ np.diag(np.abs(eigvals_G)) @ eigvecs_G.T   # (d_model, d_model)

# B: anti(W_QK^{prev-token})
B = (WQK_prev - WQK_prev.T) / 2.0

# BOS removal from metric matrices (project out BOS row and column)
bos_id = model.tokenizer.bos_token_id
v_bos  = W_E[bos_id] / (np.linalg.norm(W_E[bos_id]) + 1e-10)
P_bos  = np.eye(G.shape[0]) - np.outer(v_bos, v_bos)   # projection onto BOS-complement
G = P_bos @ G @ P_bos
B = P_bos @ B @ P_bos

print(f"G Frobenius norm: {np.linalg.norm(G):.3f}")
print(f"B Frobenius norm: {np.linalg.norm(B):.3f}")


# ── Select interesting tokens ─────────────────────────────────────────────────

# Filter: remove BOS/EOS/pad, remove outlier-norm tokens (top 1%)
row_norms = np.linalg.norm(W_E, axis=1)
thresh    = np.percentile(row_norms, 99)

# Take low-index tokens (common in GPT-2 tokenizer) + some ranges
candidate_ids = list(range(1, 3000))   # skip 0=BOS

# Filter by norm
candidate_ids = [i for i in candidate_ids
                 if row_norms[i] < thresh and i != bos_id]

# Decode and keep tokens that are readable single words / punctuation / short code
interesting = []
for tok_id in candidate_ids:
    tok = tokenizer.decode([tok_id])
    # Keep: single common words (with leading space), punctuation, short code keywords
    tok_stripped = tok.strip()
    if len(tok_stripped) == 0:
        continue
    if len(tok_stripped) <= 12 and tok_stripped.replace(' ', '').replace('.', '').replace(
            ',', '').replace('!', '').replace('?', '').replace(':', '').replace(
            ';', '').replace('(', '').replace(')', '').replace('[', '').replace(
            ']', '').replace('{', '').replace('}', '').replace("'", '').replace(
            '"', '').replace('-', '').replace('_', '').replace('#', '').replace(
            '/', '').replace('*', '').replace('=', '').replace('+', '').replace(
            '<', '').replace('>', '').replace('@', '').replace('\\', '').isalnum() or \
       tok_stripped in '.,!?;:()[]{}"\'-_#/\\*=+<>@':
        interesting.append(tok_id)
    if len(interesting) >= 300:
        break

print(f"Selected {len(interesting)} tokens")

# Extract embeddings
E_sel = W_E[interesting]                      # (N, d_model)
tok_strs = [tokenizer.decode([i]).strip() or repr(tokenizer.decode([i]))
            for i in interesting]


# ── Compute similarity matrices ───────────────────────────────────────────────

# G-weighted similarity
S_G = E_sel @ G @ E_sel.T                    # (N, N) symmetric

# B-weighted similarity (antisymmetric)
S_B = E_sel @ B @ E_sel.T                    # (N, N) antisymmetric, S_B[i,j] = -S_B[j,i]

# Standard cosine similarity (identity metric baseline)
E_norm = E_sel / (np.linalg.norm(E_sel, axis=1, keepdims=True) + 1e-10)
S_cos  = E_norm @ E_norm.T

print(f"\nS_G range: [{S_G.min():.3f}, {S_G.max():.3f}]")
print(f"S_B range: [{S_B.min():.3f}, {S_B.max():.3f}]  (should be antisymmetric)")
print(f"S_cos range: [{S_cos.min():.3f}, {S_cos.max():.3f}]")
print(f"Antisymmetry check: ||S_B + S_B^T||_F = {np.linalg.norm(S_B + S_B.T):.6f}")


# ── Top directed pairs from B ─────────────────────────────────────────────────

N = len(interesting)
N_top = 20

# Flatten upper triangle of S_B (i < j), these have S_B[i,j] = -S_B[j,i]
triu_i, triu_j = np.triu_indices(N, k=1)
b_vals = S_B[triu_i, triu_j]

# Most positive: i strongly attends TOWARD j (not vice versa)
top_pos = np.argsort(b_vals)[::-1][:N_top]
# Most negative: j strongly attends TOWARD i (not vice versa)
top_neg = np.argsort(b_vals)[:N_top]

print(f"\nTop {N_top} directed pairs by B (i → j means i-query prefers j-key):")
print(f"{'Score':>8s}  {'Query (i)':>15s} → {'Key (j)':<15s}")
for k in top_pos:
    i, j = triu_i[k], triu_j[k]
    print(f"  {b_vals[k]:7.3f}  {repr(tok_strs[i]):>15s} → {repr(tok_strs[j]):<15s}")

print(f"\nBottom {N_top} (reversed direction: j → i):")
for k in top_neg:
    i, j = triu_i[k], triu_j[k]
    print(f"  {b_vals[k]:7.3f}  {repr(tok_strs[j]):>15s} → {repr(tok_strs[i]):<15s}")

# Top G pairs (highest content similarity, excluding diagonal)
np.fill_diagonal(S_G, 0)
triu_g = S_G[triu_i, triu_j]
top_g  = np.argsort(triu_g)[::-1][:N_top]

print(f"\nTop {N_top} G-similar token pairs (content similarity):")
print(f"{'G-sim':>8s}  {'cos-sim':>8s}  Pair")
for k in top_g:
    i, j = triu_i[k], triu_j[k]
    print(f"  {triu_g[k]:7.3f}   {S_cos[triu_i[k], triu_j[k]]:7.3f}   "
          f"{repr(tok_strs[i])} ↔ {repr(tok_strs[j])}")


# ── Figures ───────────────────────────────────────────────────────────────────

# Show a focused subset: top-60 tokens by G row-sum (most "active" in content geometry)
g_activity = np.abs(S_G).sum(axis=1) + np.abs(S_B).sum(axis=1)
top60_idx   = np.argsort(g_activity)[::-1][:60]
top60_strs  = [tok_strs[i] for i in top60_idx]

S_G_sub = S_G[np.ix_(top60_idx, top60_idx)]
S_B_sub = S_B[np.ix_(top60_idx, top60_idx)]
S_cos_sub = S_cos[np.ix_(top60_idx, top60_idx)]
np.fill_diagonal(S_G_sub, 0)
np.fill_diagonal(S_cos_sub, 0)

fig, axes = plt.subplots(1, 3, figsize=(18, 7))

def token_heatmap(ax, M, labels, title, cmap='RdBu_r', sym=True):
    vmax = np.abs(M).max()
    im = ax.imshow(M, cmap=cmap, vmin=-vmax if sym else 0, vmax=vmax,
                   aspect='auto', interpolation='nearest')
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_title(title, fontsize=8)

token_heatmap(axes[0], S_cos_sub, top60_strs,
              "Standard cosine similarity\nW_E[i]^T W_E[j] (identity metric)",
              cmap='RdBu_r')

token_heatmap(axes[1], S_G_sub, top60_strs,
              f"G-weighted similarity\nW_E[i]^T G W_E[j] (induction head L1H{h_induction} metric)",
              cmap='RdBu_r')

token_heatmap(axes[2], S_B_sub, top60_strs,
              f"B-directed routing\nW_E[i]^T B W_E[j] (prev-token head L0H{h_prev})\n"
              f"Antisymmetric: [i,j] = -[j,i], no weights needed",
              cmap='PuOr')

plt.suptitle(
    "Token-token similarity under G (content geometry) and B (directed routing)\n"
    "Pure weight geometry — no forward pass, no activations",
    fontsize=10)
plt.tight_layout()
plt.savefig("figs/exp14_token_similarity.png", dpi=250, bbox_inches="tight")
print("\nSaved figs/exp14_token_similarity.png")
print("\nDone.")

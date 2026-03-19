"""
Post 4 — Experiment 10: Identify induction and prev-token heads by activation patterns

Run a repeated-token sequence through attn-only-2l and look at attention patterns.
- Induction head: attends to token at position (current - period), i.e. one below diagonal
  on the repeated half of the sequence. Pattern: [A B C ... A B C] → induction head at
  position of second A attends back to position of first A+1 (= B in first copy).
- Prev-token head: attends to position i-1 (one below main diagonal), uniformly.

Also check: which heads show the classic induction stripe?
Cross-reference with our weight-based heuristics (max ||G||_F, max ||B||_F).

Run with: /opt/miniconda3/bin/python run_exp10_identify_heads.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

os.makedirs("figs", exist_ok=True)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from transformer_lens import HookedTransformer, utils
torch.set_grad_enabled(False)

model = HookedTransformer.from_pretrained("attn-only-2l")
n_layers = model.cfg.n_layers
n_heads  = model.cfg.n_heads

# ── Build repeated-token sequence ─────────────────────────────────────────────
# Classic induction test: [BOS A B C D E F G H A B C D E F G H]
# Induction head at position of 2nd A (pos 9) should attend to pos 1 (= first A+1 = B)

seq_len = 20   # half-length of repeated block
torch.manual_seed(42)
rand_tokens = torch.randint(1000, 10000, (1, seq_len))   # avoid special tokens
repeated = torch.cat([rand_tokens, rand_tokens], dim=1)  # (1, 2*seq_len)

# Prepend BOS
bos = torch.tensor([[model.tokenizer.bos_token_id]])
tokens = torch.cat([bos, repeated], dim=1)   # (1, 2*seq_len + 1)
T = tokens.shape[1]

print(f"Sequence length: {T}")
print(f"Tokens: {tokens[0, :10].tolist()} ...")

# ── Run forward pass, cache attention patterns ────────────────────────────────

logits, cache = model.run_with_cache(tokens)

# attn patterns: (batch, n_heads, dest, src)
attn = {}
for l in range(n_layers):
    attn[l] = cache[f"blocks.{l}.attn.hook_pattern"][0].cpu().numpy()  # (n_heads, T, T)


# ── Score each head for induction and prev-token behavior ─────────────────────

# Induction score: mean attention on the "induction diagonal"
# For the repeated block (positions seq_len+1 to 2*seq_len), position i should attend
# to position i - seq_len (the matching token in the first copy).
# We measure mean attn[dest, dest - seq_len] for dest in second half.

# Prev-token score: mean attention on the subdiagonal (pos i attends to pos i-1)

induction_scores = np.zeros((n_layers, n_heads))
prev_token_scores = np.zeros((n_layers, n_heads))

second_half = range(seq_len + 1, T)   # positions in the second copy

for l in range(n_layers):
    for h in range(n_heads):
        pat = attn[l][h]   # (T, T)

        # Induction: attn[dest, dest - seq_len + 1] for dest in second half
        # Induction head at position of 2nd A attends to B1 (= first A pos + 1)
        ind_vals = [pat[dest, dest - seq_len + 1] for dest in second_half if dest - seq_len + 1 >= 0]
        induction_scores[l, h] = np.mean(ind_vals)

        # Prev-token: attn[dest, dest-1] for dest > 0
        prev_vals = [pat[dest, dest - 1] for dest in range(1, T)]
        prev_token_scores[l, h] = np.mean(prev_vals)


print("\n=== Induction scores (attn to matching token in first copy) ===")
print(f"{'Head':8s}", end='')
for h in range(n_heads): print(f"  H{h}", end='')
print()
for l in range(n_layers):
    print(f"  L{l}:    ", end='')
    for h in range(n_heads): print(f"  {induction_scores[l,h]:.3f}", end='')
    print()

print("\n=== Prev-token scores (attn to position i-1) ===")
print(f"{'Head':8s}", end='')
for h in range(n_heads): print(f"  H{h}", end='')
print()
for l in range(n_layers):
    print(f"  L{l}:    ", end='')
    for h in range(n_heads): print(f"  {prev_token_scores[l,h]:.3f}", end='')
    print()

# Identify top heads
ind_flat = np.argmax(induction_scores)
prev_flat = np.argmax(prev_token_scores[0])   # prev-token should be L0
l_ind, h_ind = divmod(ind_flat, n_heads)
h_prev = prev_flat

print(f"\nTop induction head (by activation): L{l_ind}H{h_ind}  "
      f"(score={induction_scores[l_ind,h_ind]:.3f})")
print(f"Top prev-token head L0 (by activation): L0H{h_prev}  "
      f"(score={prev_token_scores[0,h_prev]:.3f})")

# Compare to weight-based heuristics from exp6
G_heads = {}
B_heads = {}
W_Q = model.W_Q.cpu().numpy()
W_K = model.W_K.cpu().numpy()
for l in range(n_layers):
    for h in range(n_heads):
        WQK = W_Q[l, h] @ W_K[l, h].T
        G_heads[(l, h)] = (WQK + WQK.T) / 2.0
        B_heads[(l, h)] = (WQK - WQK.T) / 2.0

G_norms_L1 = {h: np.linalg.norm(G_heads[(1, h)], 'fro') for h in range(n_heads)}
B_norms_L0 = {h: np.linalg.norm(B_heads[(0, h)], 'fro') for h in range(n_heads)}
h_ind_weight  = max(G_norms_L1, key=G_norms_L1.get)
h_prev_weight = max(B_norms_L0, key=B_norms_L0.get)

print(f"\nWeight-based heuristic (exp6):")
print(f"  Induction (max ||G||_F L1): L1H{h_ind_weight}")
print(f"  Prev-token (max ||B||_F L0): L0H{h_prev_weight}")
print(f"\nELHAGE 2021 reported: L0H7 (prev-token), L1H4 (induction)")


# ── Figures ───────────────────────────────────────────────────────────────────

# Fig 1: Attention patterns for all L1 heads on repeated-token sequence
fig, axes = plt.subplots(2, n_heads, figsize=(20, 6))

for l in range(n_layers):
    for h in range(n_heads):
        ax = axes[l, h]
        im = ax.imshow(attn[l][h], cmap='Blues', vmin=0, vmax=1,
                       aspect='auto', interpolation='nearest')
        ax.set_title(f"L{l}H{h}\nind={induction_scores[l,h]:.2f} "
                     f"prev={prev_token_scores[l,h]:.2f}", fontsize=6)
        ax.set_xticks([]); ax.set_yticks([])
        # Mark the induction diagonal for L1 heads
        if l == 1:
            for dest in second_half:
                src = dest - seq_len + 1
                if 0 <= src < T:
                    ax.plot(src, dest, 'r.', ms=2)

plt.suptitle(
    f"Attention patterns on repeated-token sequence (length {T})\n"
    f"Red dots = expected induction positions (dest, dest-{seq_len})",
    fontsize=10)
plt.tight_layout()
plt.savefig("figs/exp10_attn_patterns.png", dpi=200, bbox_inches="tight")
print("\nSaved figs/exp10_attn_patterns.png")


# Fig 2: Induction and prev-token scores bar chart
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4))

x = np.arange(n_heads)
colors_L0 = '#1f77b4'; colors_L1 = '#d62728'

ax = axes2[0]
ax.bar(x - 0.2, induction_scores[0], 0.4, label='L0', color=colors_L0, alpha=0.8)
ax.bar(x + 0.2, induction_scores[1], 0.4, label='L1', color=colors_L1, alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels([f'H{h}' for h in range(n_heads)])
ax.set_ylabel("Induction score", fontsize=9)
ax.set_title("Induction score per head\n(attn to matching token in first copy)", fontsize=9)
ax.legend(); ax.grid(True, alpha=0.2, axis='y')
ax.axvline(h_ind - 0.5 + (0.2 if l_ind==1 else -0.2),
           color='red', lw=1, linestyle='--', alpha=0.5)

ax = axes2[1]
ax.bar(x - 0.2, prev_token_scores[0], 0.4, label='L0', color=colors_L0, alpha=0.8)
ax.bar(x + 0.2, prev_token_scores[1], 0.4, label='L1', color=colors_L1, alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels([f'H{h}' for h in range(n_heads)])
ax.set_ylabel("Prev-token score", fontsize=9)
ax.set_title("Prev-token score per head\n(attn to position i-1)", fontsize=9)
ax.legend(); ax.grid(True, alpha=0.2, axis='y')

plt.suptitle(
    f"Head identification by activation pattern\n"
    f"Activation: ind=L{l_ind}H{h_ind}, prev=L0H{h_prev}  |  "
    f"Weight heuristic: ind=L1H{h_ind_weight}, prev=L0H{h_prev_weight}  |  "
    f"Elhage: ind=L1H4, prev=L0H7",
    fontsize=9)
plt.tight_layout()
plt.savefig("figs/exp10_head_scores.png", dpi=200, bbox_inches="tight")
print("Saved figs/exp10_head_scores.png")

print("\nDone.")

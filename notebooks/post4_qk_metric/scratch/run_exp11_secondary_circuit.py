"""
Post 4 — Experiment 11: Secondary circuit investigation — L0H4 → L1H7/L1H3

From exp9 G-corrected K-comp:
  L0H4 → L1H7: G-corrected=0.134 (highest in table)
  L0H4 → L1H3: G-corrected=0.130 (second)

From exp10: L1H7 induction score=0.060 (second after L1H6=0.604)
From exp8: L0H4 is anomalous — G_slop=1.020, G_top1=23.5%, concentrated single mode

Questions:
1. What does L0H4's attention pattern look like on repeated-token sequence?
2. What does L1H7's attention pattern look like — does it show an induction stripe?
3. Is L0H4→L1H7 a secondary induction circuit, or something else?
4. What is L0H4's attention pattern on non-repeated text?

Run with: /opt/miniconda3/bin/python run_exp11_secondary_circuit.py
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

# ── Test 1: repeated-token sequence ──────────────────────────────────────────

seq_len = 20
torch.manual_seed(42)
rand_tokens = torch.randint(1000, 10000, (1, seq_len))
repeated    = torch.cat([rand_tokens, rand_tokens], dim=1)
bos         = torch.tensor([[model.tokenizer.bos_token_id]])
tokens_rep  = torch.cat([bos, repeated], dim=1)   # (1, 41)
T = tokens_rep.shape[1]

_, cache_rep = model.run_with_cache(tokens_rep)
attn_rep = {l: cache_rep[f"blocks.{l}.attn.hook_pattern"][0].cpu().numpy()
            for l in range(n_layers)}

# ── Test 2: longer random sequence (no repetition) ───────────────────────────

torch.manual_seed(99)
rand_long  = torch.randint(1000, 10000, (1, 40))
tokens_rnd = torch.cat([bos, rand_long], dim=1)   # (1, 41)

_, cache_rnd = model.run_with_cache(tokens_rnd)
attn_rnd = {l: cache_rnd[f"blocks.{l}.attn.hook_pattern"][0].cpu().numpy()
            for l in range(n_layers)}

# ── Compute induction scores ──────────────────────────────────────────────────

second_half = range(seq_len + 1, T)

def induction_score(pat, seq_len, T):
    vals = [pat[dest, dest - seq_len + 1]
            for dest in range(seq_len + 1, T)
            if dest - seq_len + 1 >= 0]
    return np.mean(vals)

def prev_token_score(pat, T):
    return np.mean([pat[dest, dest - 1] for dest in range(1, T)])

print("=== Induction scores on repeated sequence ===")
for l in range(n_layers):
    for h in range(n_heads):
        s = induction_score(attn_rep[l][h], seq_len, T)
        if s > 0.02:
            print(f"  L{l}H{h}: {s:.3f}")

print("\n=== Prev-token scores on repeated sequence ===")
for l in range(n_layers):
    for h in range(n_heads):
        s = prev_token_score(attn_rep[l][h], T)
        if s > 0.1:
            print(f"  L{l}H{h}: {s:.3f}")

# ── Attention entropy per head (how diffuse is the pattern?) ─────────────────

def attn_entropy(pat):
    """Mean entropy of attention distribution per destination position."""
    pat_safe = np.clip(pat, 1e-10, 1)
    return np.mean(-np.sum(pat_safe * np.log(pat_safe), axis=-1))

print("\n=== Attention entropy (higher = more diffuse) ===")
print("Repeated sequence:")
for l in range(n_layers):
    for h in [3, 4, 6, 7]:   # focus heads
        e_rep = attn_entropy(attn_rep[l][h])
        e_rnd = attn_entropy(attn_rnd[l][h])
        print(f"  L{l}H{h}: repeated={e_rep:.2f}  random={e_rnd:.2f}  "
              f"delta={e_rep - e_rnd:+.2f}")

# ── What does L0H4 attend to? Top attended positions ─────────────────────────

print("\n=== L0H4 attention on repeated sequence (top src per dest) ===")
print("Format: dest → top-3 sources (position, weight)")
pat_L0H4_rep = attn_rep[0][4]
for dest in range(1, min(T, 15)):
    top3 = np.argsort(pat_L0H4_rep[dest])[::-1][:3]
    print(f"  pos {dest:2d}: " +
          "  ".join(f"src={s} ({pat_L0H4_rep[dest,s]:.2f})" for s in top3))

print("\n=== L1H7 attention on repeated sequence (positions 21-35) ===")
pat_L1H7_rep = attn_rep[1][7]
for dest in range(seq_len + 1, min(T, seq_len + 16)):
    top3 = np.argsort(pat_L1H7_rep[dest])[::-1][:3]
    expected_ind = dest - seq_len + 1
    print(f"  pos {dest:2d} (expected ind src={expected_ind}): " +
          "  ".join(f"src={s} ({pat_L1H7_rep[dest,s]:.2f})" for s in top3))


# ── Figures ───────────────────────────────────────────────────────────────────

focus_heads = [(0, 3), (0, 4), (1, 6), (1, 7), (1, 3)]
n_focus = len(focus_heads)

fig, axes = plt.subplots(2, n_focus, figsize=(4 * n_focus, 8))

for col, (l, h) in enumerate(focus_heads):
    for row, (attn, label) in enumerate([(attn_rep, "repeated"), (attn_rnd, "random")]):
        ax = axes[row, col]
        im = ax.imshow(attn[l][h], cmap='Blues', vmin=0, vmax=1,
                       aspect='auto', interpolation='nearest')
        plt.colorbar(im, ax=ax, fraction=0.04)

        # Mark induction diagonal for L1 heads
        if l == 1 and label == "repeated":
            for dest in second_half:
                src = dest - seq_len + 1
                if 0 <= src < T:
                    ax.plot(src, dest, 'r.', ms=3)

        ind_s = induction_score(attn[l][h], seq_len, T)
        prev_s = prev_token_score(attn[l][h], T)
        ax.set_title(f"L{l}H{h} — {label}\nind={ind_s:.3f}  prev={prev_s:.3f}",
                     fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

plt.suptitle(
    "Secondary circuit investigation: L0H4, L1H7, L1H3 vs primary circuit L0H3, L1H6\n"
    "Top row: repeated-token sequence  |  Bottom row: random sequence\n"
    "Red dots = expected induction positions",
    fontsize=9)
plt.tight_layout()
plt.savefig("figs/exp11_secondary_circuit.png", dpi=200, bbox_inches="tight")
print("\nSaved figs/exp11_secondary_circuit.png")
print("\nDone.")

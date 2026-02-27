import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
import random
import os
from itertools import combinations

os.chdir('/Users/pfields/Git/peter-fields.github.io/notebooks/post2_attention-diagnostics/scratch')

torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained("gpt2-small")

IOI_HEADS = {
    "Name Mover": [(9, 6), (9, 9), (10, 0)],
    "Backup Name Mover": [(9, 0), (9, 7), (10, 1), (10, 2), (10, 6), (10, 10), (11, 2), (11, 9)],
    "Negative Name Mover": [(10, 7), (11, 10)],
    "S-Inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "Induction": [(5, 5), (6, 9)],
    "Duplicate Token": [(0, 1), (3, 0)],
    "Previous Token": [(2, 2), (4, 11)],
}
ALL_CIRCUIT_HEADS = set()
for heads in IOI_HEADS.values():
    ALL_CIRCUIT_HEADS.update(heads)

TEMPLATES_ABBA = [
    "When {A} and {B} went to the store, {B} gave a drink to",
    "When {A} and {B} went to the park, {B} handed a ball to",
    "When {A} and {B} arrived at the office, {B} passed a note to",
    "When {A} and {B} got to the restaurant, {B} offered a menu to",
    "When {A} and {B} walked into the room, {B} showed a book to",
    "After {A} and {B} met at the cafe, {B} sent a message to",
    "After {A} and {B} sat down for dinner, {B} gave a gift to",
]
TEMPLATES_BABA = [t.replace("{A} and {B}", "{B} and {A}") for t in TEMPLATES_ABBA]
TEMPLATES_NON_IOI = [
    "When {A} and {B} went to the store, {C} gave a drink to",
    "When {A} and {B} went to the park, {C} handed a ball to",
    "When {A} and {B} arrived at the office, {C} passed a note to",
    "When {A} and {B} got to the restaurant, {C} offered a menu to",
    "When {A} and {B} walked into the room, {C} showed a book to",
    "After {A} and {B} met at the cafe, {C} sent a message to",
    "After {A} and {B} sat down for dinner, {C} gave a gift to",
]
NAMES = ["Mary", "John", "Alice", "Bob", "Sarah", "Tom",
         "Emma", "James", "Lisa", "David", "Kate", "Mark"]

def generate_ioi(n, seed=42):
    random.seed(seed)
    templates = TEMPLATES_ABBA + TEMPLATES_BABA
    seen, prompts = set(), []
    while len(prompts) < n:
        t = random.choice(templates)
        a, b = random.sample(NAMES, 2)
        p = t.format(A=a, B=b)
        if p not in seen:
            seen.add(p); prompts.append(p)
    return prompts

def generate_non_ioi(n, seed=43):
    random.seed(seed)
    seen, prompts = set(), []
    while len(prompts) < n:
        t = random.choice(TEMPLATES_NON_IOI)
        a, b, c = random.sample(NAMES, 3)
        p = t.format(A=a, B=b, C=c)
        if p not in seen:
            seen.add(p); prompts.append(p)
    return prompts

ioi_prompts = generate_ioi(50)
non_ioi_prompts = generate_non_ioi(50)

def compute_kl(model, prompts):
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    kl_all = np.zeros((len(prompts), n_layers, n_heads))
    for p_idx, prompt in enumerate(prompts):
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens)
        seq_len = tokens.shape[1]
        log_n = np.log(seq_len)
        for layer in range(n_layers):
            attn = cache["pattern", layer]
            pi = attn[0, :, -1, :]
            log_pi = torch.log(pi + 1e-12)
            entropy = -(pi * log_pi).sum(dim=-1)
            kl_all[p_idx, layer, :] = (log_n - entropy.cpu().numpy()) / log_n
    return kl_all

print("Computing KL on IOI prompts...")
kl_ioi = compute_kl(model, ioi_prompts)
print("Computing KL on non-IOI prompts...")
kl_non = compute_kl(model, non_ioi_prompts)

head_labels = [(l, h) for l in range(12) for h in range(12)]
C_ioi  = np.corrcoef(kl_ioi.reshape(50, -1).T)
C_non  = np.corrcoef(kl_non.reshape(50, -1).T)
C_diff = C_ioi - C_non

pairs = list(combinations(range(144), 2))
results = []
for i, j in pairs:
    li, hi = head_labels[i]
    lj, hj = head_labels[j]
    c = C_diff[i, j]
    i_c = (li, hi) in ALL_CIRCUIT_HEADS
    j_c = (lj, hj) in ALL_CIRCUIT_HEADS
    if i_c and j_c:   cat = "circuit-circuit"
    elif i_c or j_c:  cat = "circuit-non"
    else:             cat = "non-non"
    results.append((abs(c), c, li, hi, lj, hj, cat))

results.sort(key=lambda x: x[0], reverse=True)

cats     = [r[6] for r in results]
abs_corr = [r[0] for r in results]

for k in [10, 20, 50, 100, 200]:
    top_k = results[:k]
    cc = sum(1 for r in top_k if r[6] == "circuit-circuit")
    cn = sum(1 for r in top_k if r[6] == "circuit-non")
    nn = sum(1 for r in top_k if r[6] == "non-non")
    print(f"Top {k:>4}: circuit-circuit={cc}, circuit-non={cn}, non-non={nn}")

COLORS  = {"circuit-circuit": "#d62728", "circuit-non": "#ff7f0e", "non-non": "#aaaaaa"}
ZORDERS = {"circuit-circuit": 3, "circuit-non": 2, "non-non": 1}
ALPHAS  = {"circuit-circuit": 1.0, "circuit-non": 0.8, "non-non": 0.3}

def draw_barcode(ax, cats, abs_corr):
    for cat in ["non-non", "circuit-non", "circuit-circuit"]:
        idx = [i for i, c in enumerate(cats) if c == cat]
        vals = [abs_corr[i] for i in idx]
        ax.vlines(idx, 0, vals,
                  color=COLORS[cat], alpha=ALPHAS[cat],
                  linewidth=0.4 if cat == "non-non" else 0.8,
                  zorder=ZORDERS[cat])

fig, ax = plt.subplots(figsize=(13, 5))
draw_barcode(ax, cats, abs_corr)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=COLORS["circuit-circuit"], lw=2, label="circuit-circuit"),
    Line2D([0], [0], color=COLORS["circuit-non"],     lw=2, label="circuit-non-circuit"),
    Line2D([0], [0], color=COLORS["non-non"],         lw=1, alpha=0.5, label="non-non"),
]
ax.legend(handles=legend_elements, fontsize=10, loc="upper right")
ax.set_xlabel("Rank (by |C_diff,ij|)", fontsize=12)
ax.set_ylabel("|C_IOI - C_nonIOI|", fontsize=12)
ax.set_title("C_diff = C_IOI - C_nonIOI — all 10,296 pairs ranked\n"
             "Do circuit-circuit pairs concentrate at top?", fontsize=12)
ax.set_xlim(-50, len(results) + 50)
ax.set_ylim(0, max(abs_corr) * 1.05)
ax.grid(True, alpha=0.15, axis='y')

# Inset: top 100
ax_inset = ax.inset_axes([0.35, 0.35, 0.45, 0.55])
draw_barcode(ax_inset, cats[:100], abs_corr[:100])
ax_inset.set_xlim(-1, 100)
ax_inset.set_ylim(min(abs_corr[:100]) * 0.95, max(abs_corr[:100]) * 1.02)
ax_inset.set_title("Top 100", fontsize=9)
ax_inset.tick_params(labelsize=7)
ax_inset.grid(True, alpha=0.2, axis='y')
ax_inset.set_xlabel("Rank", fontsize=8)
ax_inset.set_ylabel("|C_diff|", fontsize=8)
ax.indicate_inset_zoom(ax_inset, edgecolor="black", alpha=0.4)

plt.tight_layout()
plt.savefig("fig5_cdiff_ranking.png", dpi=300, bbox_inches="tight")
print("Saved fig5_cdiff_ranking.png")
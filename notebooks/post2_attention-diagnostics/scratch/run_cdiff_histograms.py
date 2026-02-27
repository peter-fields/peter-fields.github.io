"""
For each of the 144 heads, plot two overlaid histograms:
  - C_diff values vs the 23 circuit heads (orange)
  - C_diff values vs the 121 non-circuit heads (blue)
Fixed axes across all 144 plots. Saved to cdiff_histograms/.
"""
import os, json, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)

OUT_DIR = '/Users/pfields/Git/peter-fields.github.io/notebooks/post2_attention-diagnostics/scratch/cdiff_histograms'
os.makedirs(OUT_DIR, exist_ok=True)

IOI_HEADS = {
    "Name Mover":          [(9, 6), (9, 9), (10, 0)],
    "Backup Name Mover":   [(9, 0), (9, 7), (10, 1), (10, 2), (10, 6), (10, 10), (11, 2), (11, 9)],
    "Negative Name Mover": [(10, 7), (11, 10)],
    "S-Inhibition":        [(7, 3), (7, 9), (8, 6), (8, 10)],
    "Induction":           [(5, 5), (6, 9)],
    "Duplicate Token":     [(0, 1), (3, 0)],
    "Previous Token":      [(2, 2), (4, 11)],
}
ALL_CIRCUIT_HEADS = set(h for hs in IOI_HEADS.values() for h in hs)

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

def gen_ioi(n, seed=42):
    random.seed(seed); t = TEMPLATES_ABBA + TEMPLATES_BABA; seen, out = set(), []
    while len(out) < n:
        tmpl = random.choice(t); a, b = random.sample(NAMES, 2); p = tmpl.format(A=a, B=b)
        if p not in seen: seen.add(p); out.append(p)
    return out

def gen_non(n, seed=43):
    random.seed(seed); seen, out = set(), []
    while len(out) < n:
        tmpl = random.choice(TEMPLATES_NON_IOI); a, b, c = random.sample(NAMES, 3)
        p = tmpl.format(A=a, B=b, C=c)
        if p not in seen: seen.add(p); out.append(p)
    return out

def compute_kl(prompts):
    kl = np.zeros((len(prompts), 12, 12))
    for i, p in enumerate(prompts):
        tok = model.to_tokens(p)
        _, cache = model.run_with_cache(tok)
        log_n = np.log(tok.shape[1])
        for l in range(12):
            pi = cache["pattern", l][0, :, -1, :]
            ent = -(pi * torch.log(pi + 1e-12)).sum(-1)
            kl[i, l, :] = (log_n - ent.cpu().numpy()) / log_n
    return kl

print("Loading model...")
model = HookedTransformer.from_pretrained("gpt2-small")
print("Computing KL...")
kl_ioi = compute_kl(gen_ioi(50))
kl_non = compute_kl(gen_non(50))
print("Done.")

hl = [(l, h) for l in range(12) for h in range(12)]
circuit_idx     = [i for i, (l, h) in enumerate(hl) if (l, h) in ALL_CIRCUIT_HEADS]
noncircuit_idx  = [i for i, (l, h) in enumerate(hl) if (l, h) not in ALL_CIRCUIT_HEADS]

C_IOI   = np.corrcoef(kl_ioi.reshape(50, -1).T)   # (144, 144)
C_nonIOI = np.corrcoef(kl_non.reshape(50, -1).T)
C_diff  = C_IOI - C_nonIOI                         # (144, 144)

def head_label(idx):
    l, h = hl[idx]
    for role, heads in IOI_HEADS.items():
        if (l, h) in heads:
            return f"L{l}H{h} [{role}]"
    return f"L{l}H{h} [non-circuit]"

# Global axis limits — use true min/max so no outlier is clipped
mask_offdiag = ~np.eye(144, dtype=bool)
all_vals = C_diff[mask_offdiag]
xlim = max(abs(all_vals.min()), abs(all_vals.max()))
BINS = np.linspace(-xlim, xlim, 50)

# Y-axis: fix after a first pass to find max count across all plots
print("First pass to find y-axis limit...")
max_count = 0
for i in range(144):
    row = C_diff[i]
    c_vals = [row[j] for j in circuit_idx    if j != i]
    n_vals = [row[j] for j in noncircuit_idx if j != i]
    for vals in [c_vals, n_vals]:
        counts, _ = np.histogram(vals, bins=BINS)
        max_count = max(max_count, counts.max())
YLIM = max_count * 1.15

print(f"Global x range: [{-xlim:.3f}, {xlim:.3f}]  y max: {max_count}")
print("Generating 144 histograms...")

for i in range(144):
    l, h = hl[i]
    row = C_diff[i]

    c_vals = [row[j] for j in circuit_idx    if j != i]
    n_vals = [row[j] for j in noncircuit_idx if j != i]

    is_circuit = (l, h) in ALL_CIRCUIT_HEADS
    fig, ax = plt.subplots(figsize=(5, 3.5))

    ax.hist(n_vals, bins=BINS, color='#1f77b4', alpha=0.55, label=f'vs non-circuit (n={len(n_vals)})')
    ax.hist(c_vals, bins=BINS, color='#ff7f0e', alpha=0.75, label=f'vs circuit (n={len(c_vals)})')

    ax.axvline(0, color='black', lw=0.8, ls='--')
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(0, YLIM)
    ax.set_xlabel('C_diff (C_IOI − C_nonIOI)', fontsize=9)
    ax.set_ylabel('Count', fontsize=9)

    border = '★ ' if is_circuit else ''
    ax.set_title(f'{border}{head_label(i)}', fontsize=10,
                 color='#d62728' if is_circuit else 'black')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.2, axis='y')

    fname = f"L{l:02d}H{h:02d}.png"
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=120)
    plt.close(fig)

print(f"Saved 144 plots to {OUT_DIR}/")
print("Files: L00H00.png … L11H11.png")
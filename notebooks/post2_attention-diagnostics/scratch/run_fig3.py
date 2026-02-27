import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
import random
import os

os.chdir('/Users/pfields/Git/peter-fields.github.io/notebooks')

torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained("gpt2-small")
print(f"Loaded: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads/layer")

# --- Head labels ---
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

# --- Prompts ---
TEMPLATES_ABBA = [
    "When {A} and {B} went to the store, {B} gave a drink to",
    "When {A} and {B} went to the park, {B} handed a ball to",
    "When {A} and {B} arrived at the office, {B} passed a note to",
    "When {A} and {B} got to the restaurant, {B} offered a menu to",
    "When {A} and {B} walked into the room, {B} showed a book to",
    "After {A} and {B} met at the cafe, {B} sent a message to",
    "After {A} and {B} sat down for dinner, {B} gave a gift to",
]
TEMPLATES_BABA = [
    "When {B} and {A} went to the store, {B} gave a drink to",
    "When {B} and {A} went to the park, {B} handed a ball to",
    "When {B} and {A} arrived at the office, {B} passed a note to",
    "When {B} and {A} got to the restaurant, {B} offered a menu to",
    "When {B} and {A} walked into the room, {B} showed a book to",
    "After {B} and {A} met at the cafe, {B} sent a message to",
    "After {B} and {A} sat down for dinner, {B} gave a gift to",
]
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

def generate_unique_prompts_ioi(n, seed=42):
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

def generate_unique_prompts_non_ioi(n, seed=43):
    random.seed(seed)
    seen, prompts = set(), []
    while len(prompts) < n:
        t = random.choice(TEMPLATES_NON_IOI)
        a, b, c = random.sample(NAMES, 3)
        p = t.format(A=a, B=b, C=c)
        if p not in seen:
            seen.add(p); prompts.append(p)
    return prompts

ioi_prompts = generate_unique_prompts_ioi(50)
non_ioi_prompts = generate_unique_prompts_non_ioi(50)

# --- Diagnostics ---
def compute_diagnostics(model, prompts):
    n_layers, n_heads, n_prompts = model.cfg.n_layers, model.cfg.n_heads, len(prompts)
    kl_all = np.zeros((n_prompts, n_layers, n_heads))
    var_all = np.zeros((n_prompts, n_layers, n_heads))
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
            kl_norm = (log_n - entropy.cpu().numpy()) / log_n
            mean_log_pi = (pi * log_pi).sum(dim=-1, keepdim=True)
            var_log_pi = (pi * (log_pi - mean_log_pi) ** 2).sum(dim=-1)
            var_norm = var_log_pi.cpu().numpy() / (log_n ** 2)
            kl_all[p_idx, layer, :] = kl_norm
            var_all[p_idx, layer, :] = var_norm
    return kl_all, var_all

print("Computing diagnostics on IOI prompts...")
kl_ioi, var_ioi = compute_diagnostics(model, ioi_prompts)
print("Computing diagnostics on non-IOI prompts...")
kl_non_ioi, var_non_ioi = compute_diagnostics(model, non_ioi_prompts)
print("Done!")

# --- Figure 3 ---
_delta_kl_nc = kl_ioi.mean(axis=0) - kl_non_ioi.mean(axis=0)
_delta_var_nc = var_ioi.mean(axis=0) - var_non_ioi.mean(axis=0)

non_circuit_list = [(l, h) for l in range(12) for h in range(12)
                    if (l, h) not in ALL_CIRCUIT_HEADS]
non_circuit_sorted = sorted(non_circuit_list,
                            key=lambda lh: abs(_delta_kl_nc[lh[0], lh[1]]),
                            reverse=True)

interesting = non_circuit_sorted[:4]
boring = non_circuit_sorted[-12:][::-1]
fig3_heads = [(lh, True) for lh in interesting] + [(lh, False) for lh in boring]

all_kl_vals, all_var_vals = [], []
for (l, h), _ in fig3_heads:
    all_kl_vals.extend(kl_ioi[:, l, h].tolist())
    all_kl_vals.extend(kl_non_ioi[:, l, h].tolist())
    all_var_vals.extend(var_ioi[:, l, h].tolist())
    all_var_vals.extend(var_non_ioi[:, l, h].tolist())
kl_lo = max(0, min(all_kl_vals) - 0.03)
kl_hi = min(1, max(all_kl_vals) + 0.03)
var_lo = max(0, min(all_var_vals) - 0.01)
var_hi = max(all_var_vals) + 0.01

ncols, nrows = 4, 4
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.2))
axes_flat = axes.flatten()

COLOR_INTERESTING = "#ff7f0e"
COLOR_BORING      = "#aec7e8"

for idx, ((l, h), is_interesting) in enumerate(fig3_heads):
    ax = axes_flat[idx]
    color = COLOR_INTERESTING if is_interesting else COLOR_BORING

    kl_act    = kl_ioi[:, l, h]
    var_act   = var_ioi[:, l, h]
    kl_inact  = kl_non_ioi[:, l, h]
    var_inact = var_non_ioi[:, l, h]

    ax.scatter(kl_inact, var_inact, alpha=0.3, s=18, c="#cccccc", label="Non-IOI", zorder=1)
    ax.scatter(kl_act,   var_act,   alpha=0.4, s=18, c=color,     label="IOI",     zorder=2)

    mean_active   = (kl_act.mean(),   var_act.mean())
    mean_inactive = (kl_inact.mean(), var_inact.mean())

    ax.scatter(*mean_inactive, s=120, c="#555555", marker="*",
               edgecolors="black", linewidths=0.6, zorder=4)
    ax.scatter(*mean_active,   s=120, c=color,     marker="*",
               edgecolors="black", linewidths=0.6, zorder=4)

    ax.annotate("", xy=mean_active, xytext=mean_inactive,
                arrowprops=dict(arrowstyle="->", color="black",
                               lw=1.5, connectionstyle="arc3,rad=0.1"),
                zorder=3)

    dkl = _delta_kl_nc[l, h]
    tag = " \u2605" if is_interesting else ""
    ax.set_title(f"L{l}H{h}{tag}  \u0394KL={dkl:+.3f}", fontsize=9)
    ax.set_xlim(kl_lo, kl_hi)
    ax.set_ylim(var_lo, var_hi)

    if idx % ncols == 0:
        ax.set_ylabel(r"Norm. $\chi$", fontsize=9)
    if idx >= (nrows - 1) * ncols:
        ax.set_xlabel(r"Norm. KL", fontsize=9)

    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)

for idx in range(len(fig3_heads), nrows * ncols):
    axes_flat[idx].set_visible(False)

fig.suptitle(
    "Figure 3: Non-Circuit Heads \u2014 IOI (colored) vs Non-IOI (gray)\n"
    "\u2605 = largest |\u0394KL|.  Arrow: non-IOI mean \u2192 IOI mean  |  Axes matched to Fig 2d",
    fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("fig3_non_circuit_grid.png", dpi=300, bbox_inches="tight")
print("Saved fig3_non_circuit_grid.png")

print("\nInteresting heads (top |delta KL|):")
for (l, h) in interesting:
    print(f"  L{l}H{h}  dKL={_delta_kl_nc[l,h]:+.4f}  dvar={_delta_var_nc[l,h]:+.4f}")
print("Boring heads (smallest |delta KL|):")
for (l, h) in boring:
    print(f"  L{l}H{h}  dKL={_delta_kl_nc[l,h]:+.4f}  dvar={_delta_var_nc[l,h]:+.4f}")
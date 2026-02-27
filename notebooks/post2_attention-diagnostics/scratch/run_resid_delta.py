import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from transformer_lens import HookedTransformer
import random
import os

os.chdir('/Users/pfields/Git/peter-fields.github.io/notebooks/post2_attention-diagnostics/scratch')

torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained("gpt2-small")

IOI_HEADS = {
    "Name Mover":          [(9, 6), (9, 9), (10, 0)],
    "Backup Name Mover":   [(9, 0), (9, 7), (10, 1), (10, 2), (10, 6), (10, 10), (11, 2), (11, 9)],
    "Negative Name Mover": [(10, 7), (11, 10)],
    "S-Inhibition":        [(7, 3), (7, 9), (8, 6), (8, 10)],
    "Induction":           [(5, 5), (6, 9)],
    "Duplicate Token":     [(0, 1), (3, 0)],
    "Previous Token":      [(2, 2), (4, 11)],
}
ROLE_COLORS = {
    "Name Mover":          "#d62728",
    "Backup Name Mover":   "#ff9896",
    "Negative Name Mover": "#9467bd",
    "S-Inhibition":        "#2ca02c",
    "Induction":           "#1f77b4",
    "Duplicate Token":     "#8c564b",
    "Previous Token":      "#e377c2",
    "Non-circuit":         "#cccccc",
    "MLP":                 "#17becf",
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

def generate_ioi(n, seed=42):
    random.seed(seed); templates = TEMPLATES_ABBA + TEMPLATES_BABA
    seen, prompts = set(), []
    while len(prompts) < n:
        t = random.choice(templates); a, b = random.sample(NAMES, 2)
        p = t.format(A=a, B=b)
        if p not in seen: seen.add(p); prompts.append(p)
    return prompts

def generate_non_ioi(n, seed=43):
    random.seed(seed); seen, prompts = set(), []
    while len(prompts) < n:
        t = random.choice(TEMPLATES_NON_IOI); a, b, c = random.sample(NAMES, 3)
        p = t.format(A=a, B=b, C=c)
        if p not in seen: seen.add(p); prompts.append(p)
    return prompts

# Component index layout:
#   0..143  : attention heads (l*12 + h), type "H"
#   144..155: MLPs (layer 0..11),         type "M"
N_HEADS = 144
N_MLPS  = 12
N_COMP  = N_HEADS + N_MLPS

comp_labels = [(l, h, "H") for l in range(12) for h in range(12)] + \
              [(l, None, "M") for l in range(12)]

circuit_mask = np.array([
    (ctype == "H" and (l, h) in ALL_CIRCUIT_HEADS)
    for l, h, ctype in comp_labels
])

def comp_role(l, h, ctype):
    if ctype == "M": return "MLP"
    for role, heads in IOI_HEADS.items():
        if (l, h) in heads: return role
    return "Non-circuit"

def comp_color(l, h, ctype):
    return ROLE_COLORS[comp_role(l, h, ctype)]

def comp_name(l, h, ctype):
    return f"L{l}M" if ctype == "M" else f"L{l}H{h}"

def compute_resid_delta(prompts):
    """
    For each prompt, compute ||Δresidual||_2 at the final token position
    for every attention head and MLP.

    Per-head residual contribution = z_h @ W_O[h]  (d_model vector)
    where z = cache["z", layer], W_O = model.blocks[layer].attn.W_O
    MLP: cache["mlp_out", layer]

    Returns X: (n_prompts, N_COMP)
    """
    X = np.zeros((len(prompts), N_COMP))
    for i, prompt in enumerate(prompts):
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens)
        for layer in range(12):
            # Attention heads: z has shape (1, seq, n_heads, d_head)
            z   = cache["z", layer]                    # (1, seq, n_heads, d_head)
            W_O = model.blocks[layer].attn.W_O         # (n_heads, d_head, d_model)
            for head in range(12):
                # per-head residual contribution at final token
                delta = (z[0, -1, head, :] @ W_O[head]).cpu().numpy()  # (d_model,)
                X[i, layer * 12 + head] = np.linalg.norm(delta)
            # MLP
            mlp_out = cache["mlp_out", layer]          # (1, seq, d_model)
            delta_mlp = mlp_out[0, -1, :].cpu().numpy()
            X[i, N_HEADS + layer] = np.linalg.norm(delta_mlp)
    return X

print("Computing residual deltas for IOI prompts...")
X_ioi = compute_resid_delta(generate_ioi(50))
print("Computing residual deltas for non-IOI prompts...")
X_non = compute_resid_delta(generate_non_ioi(50))

print(f"X_ioi shape: {X_ioi.shape}  X_non shape: {X_non.shape}")

C_ioi  = np.corrcoef(X_ioi.T)   # (156, 156)
C_non  = np.corrcoef(X_non.T)
C_diff = C_ioi - C_non

# ── Signed pair ranking ────────────────────────────────────────────────────────
pairs = []
for i in range(N_COMP):
    for j in range(i + 1, N_COMP):
        c = C_diff[i, j]
        li, hi, ti = comp_labels[i]
        lj, hj, tj = comp_labels[j]
        ic = circuit_mask[i]; jc = circuit_mask[j]
        ct = "CC" if ic and jc else ("CN" if ic or jc else "NN")
        # extra flag if either component is an MLP
        has_mlp = (ti == "M") or (tj == "M")
        pairs.append((c, i, j, ct, has_mlp))

pairs.sort(reverse=True)
cats    = [p[3] for p in pairs]
vals    = [p[0] for p in pairs]
n_pairs = len(pairs)

# ── Coverage stats ─────────────────────────────────────────────────────────────
print(f"\nTop-k C_diff pair coverage of circuit heads:")
print(f"{'k':>5}  {'CC pairs':>8}  {'heads seen':>10}  {'heads seen /23':>14}")
print("-" * 48)
for k in [10, 20, 50, 100, 200]:
    top = pairs[:k]
    heads_seen = set()
    for c, i, j, ct, hm in top:
        if circuit_mask[i]: heads_seen.add(comp_labels[i][:2])
        if circuit_mask[j]: heads_seen.add(comp_labels[j][:2])
    cc_count = sum(1 for p in top if p[3] == "CC")
    print(f"{k:>5}  {cc_count:>8}  {len(heads_seen):>10}  {len(heads_seen)}/{len(ALL_CIRCUIT_HEADS)}")

# Bottom-k
print(f"\nBottom-k C_diff pair coverage of circuit heads:")
for k in [10, 20, 50, 100]:
    bot = pairs[n_pairs - k:]
    cc_count = sum(1 for p in bot if p[3] == "CC")
    heads_seen = set()
    for c, i, j, ct, hm in bot:
        if circuit_mask[i]: heads_seen.add(comp_labels[i][:2])
        if circuit_mask[j]: heads_seen.add(comp_labels[j][:2])
    print(f"  bottom-{k}: {cc_count} CC  |  heads_seen = {len(heads_seen)}")

# Top/bottom table
def print_table(subset, label):
    print(f"\n{label}")
    print(f"{'Rk':<5} {'Comp i':<9} {'Role i':<24} {'Comp j':<9} {'Role j':<24} {'C_diff':>7}  cat")
    print("-" * 90)
    for k, (c, i, j, ct, hm) in enumerate(subset, 1):
        li, hi, ti = comp_labels[i]
        lj, hj, tj = comp_labels[j]
        ni = comp_name(li, hi, ti); nj = comp_name(lj, hj, tj)
        ri = comp_role(li, hi, ti); rj = comp_role(lj, hj, tj)
        print(f"{k:<5} {ni:<9} {ri:<24} {nj:<9} {rj:<24} {c:+.3f}  {ct}")

print_table(pairs[:20], "Top 20 positive C_diff pairs:")
print_table(pairs[n_pairs-20:], "Top 20 negative C_diff pairs:")

# Top MLP pairs
mlp_pairs = [(c, i, j, ct, hm) for c, i, j, ct, hm in pairs if hm]
print(f"\nTop 10 pairs involving an MLP (by signed C_diff):")
print_table(mlp_pairs[:10], "MLP pairs (positive):")
print_table(mlp_pairs[-10:], "MLP pairs (negative):")

# ── Figure ─────────────────────────────────────────────────────────────────────
COLORS  = {"CC": "#d62728", "CN": "#ff7f0e", "NN": "#bbbbbb"}
ALPHAS  = {"CC": 1.0,       "CN": 0.7,       "NN": 0.2}
WIDTHS  = {"CC": 1.0,       "CN": 0.6,       "NN": 0.35}
ZORDERS = {"CC": 3,         "CN": 2,         "NN": 1}

fig, axes = plt.subplots(1, 2, figsize=(15, 5),
                         gridspec_kw={"width_ratios": [3, 1]})

# Left: full signed ranking (color MLP pairs with a different marker)
ax = axes[0]
for cat in ["NN", "CN", "CC"]:
    idx = [i for i, c in enumerate(cats) if c == cat]
    ax.vlines(idx, 0, [vals[i] for i in idx],
              color=COLORS[cat], alpha=ALPHAS[cat],
              linewidth=WIDTHS[cat], zorder=ZORDERS[cat])

# Overlay MLP-involved pairs as dots
mlp_idx_all = [k for k, (c, i, j, ct, hm) in enumerate(pairs) if hm]
ax.scatter(mlp_idx_all, [vals[k] for k in mlp_idx_all],
           s=4, color="#17becf", zorder=4, alpha=0.6, linewidths=0)

ax.axhline(0, color="black", lw=0.8, zorder=5)
ax.set_xlabel("Rank (sorted by C_diff,ij)", fontsize=11)
ax.set_ylabel("C_diff,ij", fontsize=11)
ax.set_title(f"C_diff = C_IOI − C_nonIOI — residual stream ||Δresid|| per component\n"
             f"156 components: 144 attention heads + 12 MLPs  |  {n_pairs:,} pairs ranked by signed value",
             fontsize=10)
ax.set_xlim(-100, n_pairs + 100)
ax.grid(True, alpha=0.15, axis='y')

# Annotate top positive CC pairs
top_cc = [(k, p) for k, p in enumerate(pairs) if p[3] == "CC"][:3]
for rank, (k, p) in enumerate(top_cc):
    c, i, j, ct, hm = p
    li, hi, ti = comp_labels[i]; lj, hj, tj = comp_labels[j]
    ax.annotate(f"{comp_name(li,hi,ti)}–{comp_name(lj,hj,tj)}",
                xy=(k, c), xytext=(k + 200, c + 0.04),
                fontsize=7, color="#d62728",
                arrowprops=dict(arrowstyle="-", color="#d62728", lw=0.6))

# Right: zoom top/bottom 100
ax2 = axes[1]
zoom_n = 100
top_idx = list(range(zoom_n))
bot_idx = list(range(n_pairs - zoom_n, n_pairs))
x_top = list(range(zoom_n))
x_bot = list(range(zoom_n + 15, zoom_n + 15 + zoom_n))

for idx_list, x_list in [(top_idx, x_top), (bot_idx, x_bot)]:
    for cat in ["NN", "CN", "CC"]:
        sel = [(x, vals[i]) for x, i in zip(x_list, idx_list) if cats[i] == cat]
        if sel:
            xs, ys = zip(*sel)
            ax2.vlines(xs, 0, ys, color=COLORS[cat], alpha=ALPHAS[cat],
                       linewidth=max(WIDTHS[cat], 0.8), zorder=ZORDERS[cat])

ax2.axhline(0, color="black", lw=0.8)
ax2.axvspan(zoom_n + 2, zoom_n + 13, color="white", zorder=5)
ax2.text(zoom_n + 7.5, 0, "···", ha="center", va="center",
         fontsize=12, color="gray", zorder=6)
ax2.set_xlabel("Rank (top/bottom 100)", fontsize=10)
ax2.set_ylabel("C_diff,ij", fontsize=10)
ax2.set_title("Top & bottom 100", fontsize=10)
ax2.set_xlim(-3, zoom_n + 15 + zoom_n + 3)
ax2.grid(True, alpha=0.2, axis='y')

# Stats box
stats_lines = []
for k in [10, 20, 50]:
    top = pairs[:k]; bot = pairs[n_pairs - k:]
    cc_t = sum(1 for p in top if p[3] == "CC")
    cc_b = sum(1 for p in bot if p[3] == "CC")
    stats_lines.append(f"top-{k:>2}: {cc_t} CC  |  bot-{k:>2}: {cc_b} CC")
ax2.text(0.02, 0.02, "\n".join(stats_lines), transform=ax2.transAxes,
         fontsize=8, va='bottom', family='monospace',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

legend_elements = [
    Line2D([0], [0], color=COLORS["CC"], lw=2, label="circuit–circuit (CC)"),
    Line2D([0], [0], color=COLORS["CN"], lw=2, alpha=0.8, label="circuit–non-circuit (CN)"),
    Line2D([0], [0], color=COLORS["NN"], lw=1, alpha=0.5, label="non–non (NN)"),
    Line2D([0], [0], color="#17becf", lw=0, marker='o', ms=5, label="MLP involved"),
]
ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

plt.suptitle("Signed C_diff ranking — residual stream ||Δresid|| per component\n"
             "n=50 IOI / 50 non-IOI prompts, GPT-2 small  |  attention heads + MLPs",
             fontsize=10, y=1.01)
plt.tight_layout()
plt.savefig("fig_resid_delta_cdiff.png", dpi=300, bbox_inches="tight")
print("\nSaved fig_resid_delta_cdiff.png")

# ── Compare coverage: resid-delta vs KL ───────────────────────────────────────
print("\n--- Coverage comparison ---")
for k in [10, 20, 50, 100]:
    top = pairs[:k]
    heads_seen = set()
    for c, i, j, ct, hm in top:
        if circuit_mask[i]: heads_seen.add(comp_labels[i][:2])
        if circuit_mask[j]: heads_seen.add(comp_labels[j][:2])
    print(f"  resid-delta top-{k:>3}: {len(heads_seen)}/{len(ALL_CIRCUIT_HEADS)} circuit heads  ({sum(1 for p in top if p[3]=='CC')} CC pairs)")
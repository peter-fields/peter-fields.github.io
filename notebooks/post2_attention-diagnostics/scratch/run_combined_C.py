import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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
kl_ioi = compute_kl(model, generate_ioi(50))
print("Computing KL on non-IOI prompts...")
kl_non = compute_kl(model, generate_non_ioi(50))

head_labels = [(l, h) for l in range(12) for h in range(12)]
circuit_mask = np.array([(l, h) in ALL_CIRCUIT_HEADS for l, h in head_labels])

def head_role(l, h):
    for role, heads in IOI_HEADS.items():
        if (l, h) in heads:
            return role
    return "Non-circuit"

# Stack all 100 prompts together, compute single C
kl_combined = np.vstack([kl_ioi.reshape(50, -1), kl_non.reshape(50, -1)])  # (100, 144)
C_combined = np.corrcoef(kl_combined.T)  # (144, 144)

# Also compute C_IOI and C_nonIOI for comparison
C_ioi = np.corrcoef(kl_ioi.reshape(50, -1).T)
C_non = np.corrcoef(kl_non.reshape(50, -1).T)

# Eigendecompose all three
def eig_desc(C):
    vals, vecs = np.linalg.eigh(C)
    return vals[::-1], vecs[:, ::-1]

evals_comb, evecs_comb = eig_desc(C_combined)
evals_ioi,  evecs_ioi  = eig_desc(C_ioi)

print(f"\nC_combined top 10 eigenvalues: {evals_comb[:10].round(3)}")
print(f"C_IOI      top 10 eigenvalues: {evals_ioi[:10].round(3)}")

# Enrichment score for top eigenvectors of C_combined
print(f"\n{'EV':>4}  {'λ_combined':>12}  {'Enrichment':>12}")
print("-" * 35)
for k in range(20):
    vec = evecs_comb[:, k]
    abs_vec = np.abs(vec)
    mc = abs_vec[circuit_mask].mean()
    mn = abs_vec[~circuit_mask].mean()
    enrich = mc / mn
    marker = " <---" if enrich > 1.5 else ""
    print(f"{k+1:>4}  {evals_comb[k]:>12.3f}  {enrich:>12.3f}{marker}")

# Print top loadings for EV1 of C_combined
print("\nEV1 of C_combined — top loadings:")
vec1 = evecs_comb[:, 0]
top10 = np.argsort(np.abs(vec1))[::-1][:15]
for i in top10:
    l, h = head_labels[i]
    role = head_role(l, h)
    tag = " ***" if (l, h) in ALL_CIRCUIT_HEADS else ""
    print(f"  L{l}H{h} ({role:.22s}): {vec1[i]:+.4f}{tag}")

# --- Figure: compare EV1 loadings for C_IOI vs C_combined ---
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

def plot_eigenvec(ax, vec, eigenval, title, circuit_mask, head_labels):
    colors = [ROLE_COLORS[head_role(l, h)] for l, h in head_labels]
    order = np.argsort(vec)
    ax.bar(range(144), vec[order],
           color=[colors[i] for i in order],
           width=1.0, linewidth=0)
    circ_pos  = [rank for rank, i in enumerate(order) if circuit_mask[i]]
    circ_vals = [vec[i] for i in order if circuit_mask[i]]
    ax.scatter(circ_pos, circ_vals, s=25, c='black', zorder=3, marker='|')
    ax.set_title(f"{title}  (λ={eigenval:.2f})", fontsize=10)
    ax.set_xlabel("Head (sorted by loading)", fontsize=9)
    ax.set_ylabel("Loading", fontsize=9)
    ax.axhline(0, color='gray', lw=0.5)
    ax.tick_params(labelsize=7)
    ax.set_xlim(-1, 145)

plot_eigenvec(axes[0,0], evecs_ioi[:,0],  evals_ioi[0],
              "C_IOI  EV1", circuit_mask, head_labels)
plot_eigenvec(axes[0,1], evecs_comb[:,0], evals_comb[0],
              "C_combined  EV1", circuit_mask, head_labels)
plot_eigenvec(axes[1,0], evecs_ioi[:,1],  evals_ioi[1],
              "C_IOI  EV2", circuit_mask, head_labels)
plot_eigenvec(axes[1,1], evecs_comb[:,1], evals_comb[1],
              "C_combined  EV2", circuit_mask, head_labels)

legend_elements = [Patch(facecolor=ROLE_COLORS[r], label=r) for r in ROLE_COLORS]
fig.legend(handles=legend_elements, loc='lower center', fontsize=8, ncol=4,
           bbox_to_anchor=(0.5, -0.02))

plt.suptitle("C_IOI vs C_combined (IOI+nonIOI pooled) — top eigenvectors\n"
             "Black ticks = circuit heads", fontsize=12)
plt.tight_layout()
plt.savefig("combined_C_eigenvectors.png", dpi=300, bbox_inches="tight")
print("\nSaved combined_C_eigenvectors.png")
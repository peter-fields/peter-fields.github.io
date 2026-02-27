import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
kl_mat = kl_ioi.reshape(50, -1)  # (50, 144)

head_labels = [(l, h) for l in range(12) for h in range(12)]
circuit_mask = np.array([(l, h) in ALL_CIRCUIT_HEADS for l, h in head_labels])  # (144,)
n_circuit = circuit_mask.sum()      # 23
n_noncircuit = (~circuit_mask).sum()  # 121

def head_role(l, h):
    for role, heads in IOI_HEADS.items():
        if (l, h) in heads:
            return role
    return "Non-circuit"

C_ioi = np.corrcoef(kl_mat.T)  # (144, 144)
eigenvalues, eigenvectors = np.linalg.eigh(C_ioi)
eigenvalues = eigenvalues[::-1]
eigenvectors = eigenvectors[:, ::-1]

# Only look at the first 50 (meaningful, non-zero eigenmodes)
n_meaningful = 50

# For each eigenvector, compute enrichment:
# mean |loading| of circuit heads vs mean |loading| of non-circuit heads
enrichment = np.zeros(n_meaningful)
for k in range(n_meaningful):
    vec = eigenvectors[:, k]
    abs_vec = np.abs(vec)
    mean_circ    = abs_vec[circuit_mask].mean()
    mean_noncirc = abs_vec[~circuit_mask].mean()
    enrichment[k] = mean_circ / mean_noncirc  # >1 means circuit heads load more strongly

print(f"\nCircuit enrichment per eigenvector (circuit |loading| / non-circuit |loading|):")
print(f"{'EV':>4}  {'λ':>8}  {'Enrichment':>12}  {'Circuit mean':>14}  {'Non-circ mean':>14}")
print("-" * 60)
for k in range(n_meaningful):
    vec = eigenvectors[:, k]
    abs_vec = np.abs(vec)
    mc = abs_vec[circuit_mask].mean()
    mn = abs_vec[~circuit_mask].mean()
    marker = " <---" if enrichment[k] > 1.5 else ""
    print(f"{k+1:>4}  {eigenvalues[k]:>8.3f}  {enrichment[k]:>12.3f}  {mc:>14.4f}  {mn:>14.4f}{marker}")

# Find the most enriched eigenvector
best_k = np.argmax(enrichment)
print(f"\nMost enriched eigenvector: EV{best_k+1} (λ={eigenvalues[best_k]:.3f}, enrichment={enrichment[best_k]:.3f})")

# Print top loadings for top-3 most enriched eigenvectors
top_enriched = np.argsort(enrichment)[::-1][:5]
for k in top_enriched:
    vec = eigenvectors[:, k]
    print(f"\nEV{k+1} (λ={eigenvalues[k]:.3f}, enrichment={enrichment[k]:.3f}):")
    top5 = np.argsort(np.abs(vec))[::-1][:10]
    for i in top5:
        l, h = head_labels[i]
        role = head_role(l, h)
        tag = " ***" if (l, h) in ALL_CIRCUIT_HEADS else ""
        print(f"  L{l}H{h} ({role:.22s}): {vec[i]:+.4f}{tag}")

# --- Figure: enrichment profile + top-3 enriched eigenvectors ---
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# Top-left: enrichment vs eigenvalue rank
ax = axes[0, 0]
ax.bar(range(1, n_meaningful+1), enrichment, color=['#d62728' if e > 1.3 else '#aaaaaa' for e in enrichment])
ax.axhline(1.0, color='black', lw=0.8, linestyle='--', label='no enrichment')
ax.set_xlabel("Eigenvector rank", fontsize=10)
ax.set_ylabel("Circuit enrichment (ratio of mean |loading|)", fontsize=9)
ax.set_title("Circuit head enrichment across all 50 meaningful eigenvectors", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# Top-middle: eigenvalue scree (first 50 only)
ax = axes[0, 1]
ax.plot(range(1, n_meaningful+1), eigenvalues[:n_meaningful], 'k-', lw=1.5)
for k in top_enriched[:3]:
    ax.axvline(k+1, color='#d62728', alpha=0.5, lw=1.5, linestyle='--')
ax.set_xlabel("Eigenvector rank", fontsize=10)
ax.set_ylabel("Eigenvalue", fontsize=10)
ax.set_title("Scree (top 50 only) — red = most enriched EVs", fontsize=10)
ax.grid(True, alpha=0.2)

# Top-right: scatter enrichment vs eigenvalue
ax = axes[0, 2]
ax.scatter(eigenvalues[:n_meaningful], enrichment, c=range(n_meaningful),
           cmap='viridis_r', s=40, zorder=3)
for k in top_enriched[:3]:
    ax.annotate(f"EV{k+1}", (eigenvalues[k], enrichment[k]), fontsize=7,
                xytext=(5, 3), textcoords='offset points')
ax.axhline(1.0, color='gray', lw=0.8, linestyle='--')
ax.set_xlabel("Eigenvalue (λ)", fontsize=10)
ax.set_ylabel("Circuit enrichment", fontsize=10)
ax.set_title("Enrichment vs eigenvalue magnitude", fontsize=10)
ax.grid(True, alpha=0.2)

# Bottom row: top-3 most enriched eigenvectors
from matplotlib.patches import Patch
for plot_idx, k in enumerate(top_enriched[:3]):
    ax = axes[1, plot_idx]
    vec = eigenvectors[:, k]
    colors = [ROLE_COLORS[head_role(l, h)] for l, h in head_labels]
    order = np.argsort(vec)
    ax.bar(range(144), vec[order],
           color=[colors[i] for i in order],
           width=1.0, linewidth=0)
    # Mark circuit heads
    circuit_pos = [rank for rank, i in enumerate(order) if circuit_mask[i]]
    circuit_vals = [vec[i] for i in order if circuit_mask[i]]
    ax.scatter(circuit_pos, circuit_vals, s=25, c='black', zorder=3, marker='|')
    ax.set_title(f"EV{k+1}  (λ={eigenvalues[k]:.2f}, enrich={enrichment[k]:.2f})", fontsize=10)
    ax.set_xlabel("Head (sorted by loading)", fontsize=9)
    ax.set_ylabel("Loading", fontsize=9)
    ax.axhline(0, color='gray', lw=0.5)
    ax.tick_params(labelsize=7)
    ax.set_xlim(-1, 145)

legend_elements = [Patch(facecolor=ROLE_COLORS[r], label=r) for r in ROLE_COLORS]
fig.legend(handles=legend_elements, loc='lower right', fontsize=7, ncol=2)

plt.suptitle("Eigenvector scan: which modes have disproportionate circuit head loadings?", fontsize=12)
plt.tight_layout()
plt.savefig("eigenvector_scan.png", dpi=300, bbox_inches="tight")
print("\nSaved eigenvector_scan.png")
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

def head_role(l, h):
    for role, heads in IOI_HEADS.items():
        if (l, h) in heads:
            return role
    return "Non-circuit"

# Compute C_IOI and its eigendecomposition
C_ioi = np.corrcoef(kl_mat.T)  # (144, 144)
eigenvalues, eigenvectors = np.linalg.eigh(C_ioi)
# eigh returns ascending order — flip to descending
eigenvalues = eigenvalues[::-1]
eigenvectors = eigenvectors[:, ::-1]

print(f"\nTop 10 eigenvalues: {eigenvalues[:10].round(3)}")
print(f"Bottom 10 eigenvalues: {eigenvalues[-10:].round(3)}")

# --- Figure: scree plot + top/bottom eigenvectors colored by role ---
fig = plt.figure(figsize=(16, 10))

# 1. Scree plot
ax_scree = fig.add_subplot(2, 3, (1, 2))
ax_scree.plot(range(1, len(eigenvalues)+1), eigenvalues, 'k-', lw=1, alpha=0.6)
ax_scree.axhline(0, color='gray', lw=0.5, linestyle='--')
ax_scree.set_xlabel("Eigenvalue rank", fontsize=11)
ax_scree.set_ylabel("Eigenvalue", fontsize=11)
ax_scree.set_title("Eigenspectrum of C_IOI (144×144)", fontsize=11)
ax_scree.set_xlim(0, 145)
ax_scree.grid(True, alpha=0.2)

# Mark top and bottom
for k in [1, 2, 3]:
    ax_scree.axvline(k, color='#d62728', alpha=0.4, lw=1)
for k in range(135, 145):
    ax_scree.axvline(k, color='#1f77b4', alpha=0.2, lw=1)

# 2-4: Top 3 eigenvectors (loadings per head, colored by role)
for ev_idx in range(3):
    ax = fig.add_subplot(2, 3, ev_idx + 4) if ev_idx < 3 else None
    vec = eigenvectors[:, ev_idx]
    colors = [ROLE_COLORS[head_role(l, h)] for l, h in head_labels]
    is_circuit = [(l, h) in ALL_CIRCUIT_HEADS for l, h in head_labels]

    # Sort by loading value for bar chart
    order = np.argsort(vec)
    ax.bar(range(144), vec[order],
           color=[colors[i] for i in order],
           width=1.0, linewidth=0)

    # Mark circuit heads
    circuit_positions = [rank for rank, i in enumerate(order) if is_circuit[i]]
    circuit_vals = [vec[i] for i in order if is_circuit[i]]
    ax.scatter(circuit_positions, circuit_vals, s=20, c='black', zorder=3, marker='|')

    ax.set_title(f"Eigenvector {ev_idx+1}  (λ={eigenvalues[ev_idx]:.2f})", fontsize=10)
    ax.set_xlabel("Head (sorted by loading)", fontsize=9)
    ax.set_ylabel("Loading", fontsize=9)
    ax.axhline(0, color='gray', lw=0.5)
    ax.tick_params(labelsize=7)
    ax.set_xlim(-1, 145)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=ROLE_COLORS[r], label=r) for r in ROLE_COLORS]
fig.legend(handles=legend_elements, loc='upper right', fontsize=8,
           bbox_to_anchor=(1.0, 1.0), ncol=1)

plt.suptitle("C_IOI eigenspectrum — do small eigenmodes reveal circuit structure?", fontsize=12)
plt.tight_layout()
plt.savefig("eigenspectrum_C_IOI.png", dpi=300, bbox_inches="tight")
print("Saved eigenspectrum_C_IOI.png")

# Print which heads load most strongly on each eigenvector
for ev_idx in range(5):
    vec = eigenvectors[:, ev_idx]
    top_pos = np.argsort(vec)[-5:][::-1]
    top_neg = np.argsort(vec)[:5]
    print(f"\nEigenvector {ev_idx+1} (lambda={eigenvalues[ev_idx]:.3f}):")
    print("  Top positive loadings:")
    for i in top_pos:
        l, h = head_labels[i]
        print(f"    L{l}H{h} ({head_role(l,h):.20s}): {vec[i]:+.3f}")
    print("  Top negative loadings:")
    for i in top_neg:
        l, h = head_labels[i]
        print(f"    L{l}H{h} ({head_role(l,h):.20s}): {vec[i]:+.3f}")
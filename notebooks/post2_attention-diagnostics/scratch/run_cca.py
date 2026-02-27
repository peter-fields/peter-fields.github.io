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

def compute_diagnostics(model, prompts):
    kl  = np.zeros((len(prompts), model.cfg.n_layers, model.cfg.n_heads))
    chi = np.zeros((len(prompts), model.cfg.n_layers, model.cfg.n_heads))
    for i, prompt in enumerate(prompts):
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens)
        log_n = np.log(tokens.shape[1])
        for layer in range(model.cfg.n_layers):
            pi     = cache["pattern", layer][0, :, -1, :]
            log_pi = torch.log(pi + 1e-12)
            entropy = -(pi * log_pi).sum(-1)
            kl[i, layer, :] = (log_n - entropy.cpu().numpy()) / log_n
            mean_log = (pi * log_pi).sum(-1, keepdim=True)
            var_log  = (pi * (log_pi - mean_log)**2).sum(-1)
            chi[i, layer, :] = var_log.cpu().numpy() / log_n**2
    return kl, chi

print("Computing diagnostics...")
kl_ioi,  chi_ioi  = compute_diagnostics(model, generate_ioi(50))
kl_non,  chi_non  = compute_diagnostics(model, generate_non_ioi(50))

head_labels  = [(l, h) for l in range(12) for h in range(12)]
circuit_mask = np.array([(l, h) in ALL_CIRCUIT_HEADS for l, h in head_labels])

def head_role(l, h):
    for role, heads in IOI_HEADS.items():
        if (l, h) in heads: return role
    return "Non-circuit"

# Flatten to (50, 144)
KL_ioi  = kl_ioi.reshape(50, -1)
KL_non  = kl_non.reshape(50, -1)
CHI_ioi = chi_ioi.reshape(50, -1)
CHI_non = chi_non.reshape(50, -1)

def run_cca(X, Y, k=20):
    """
    PCA-reduced CCA.
    X, Y: (n, p) — both centered inside.
    Returns: canonical correlations r, KL directions U (p, k), chi directions V (p, k).
    """
    n = X.shape[0]
    X = X - X.mean(0)
    Y = Y - Y.mean(0)

    # PCA-reduce each to k components
    Ux, Sx, _ = np.linalg.svd(X, full_matrices=False)
    Uy, Sy, _ = np.linalg.svd(Y, full_matrices=False)
    X_r = Ux[:, :k] * Sx[:k]   # (n, k) — scores
    Y_r = Uy[:, :k] * Sy[:k]

    # Whiten
    X_w = X_r / (Sx[:k] / np.sqrt(n - 1))
    Y_w = Y_r / (Sy[:k] / np.sqrt(n - 1))

    # SVD of cross-covariance
    cross = X_w.T @ Y_w / (n - 1)   # (k, k)
    Uc, S_cc, Vc = np.linalg.svd(cross)

    # Canonical directions in PCA space
    A_r = Uc    # (k, k) — each column is a CCA direction in PCA-X space
    B_r = Vc.T  # (k, k)

    # Map back to original 144-dim space
    # X_r = Ux[:,:k] * Sx[:k], so X = Ux[:,:k] @ diag(Sx[:k]) @ Vx[:k,:]
    # direction in original space: Vx[:k,:].T @ diag(1/Sx[:k]) * sqrt(n-1) @ A_r
    _, _, Vx = np.linalg.svd(X, full_matrices=False)
    _, _, Vy = np.linalg.svd(Y, full_matrices=False)
    scale_x = np.sqrt(n - 1) / Sx[:k]
    scale_y = np.sqrt(n - 1) / Sy[:k]
    U_orig = Vx[:k, :].T @ (scale_x[:, None] * A_r)  # (144, k)
    V_orig = Vy[:k, :].T @ (scale_y[:, None] * B_r)  # (144, k)

    return S_cc, U_orig, V_orig

print("\nRunning CCA (IOI prompts: KL vs chi)...")
r_ioi, U_ioi, V_ioi = run_cca(KL_ioi, CHI_ioi, k=20)

print("\nRunning CCA (non-IOI prompts: KL vs chi)...")
r_non, U_non, V_non = run_cca(KL_non, CHI_non, k=20)

print(f"\nCanonical correlations — IOI vs non-IOI:")
print(f"{'CC':>4}  {'r_IOI':>8}  {'r_nonIOI':>10}  {'diff':>8}")
print("-" * 38)
for i in range(10):
    print(f"{i+1:>4}  {r_ioi[i]:>8.4f}  {r_non[i]:>10.4f}  {r_ioi[i]-r_non[i]:>+8.4f}")

# Enrichment for top CCA directions (IOI)
print(f"\nCircuit enrichment in CCA directions (IOI):")
print(f"{'CC':>4}  {'KL enrich':>12}  {'chi enrich':>12}  {'top KL head':<30}  {'top chi head'}")
print("-" * 90)
for i in range(10):
    u = np.abs(U_ioi[:, i]); v = np.abs(V_ioi[:, i])
    eu = u[circuit_mask].mean() / u[~circuit_mask].mean()
    ev = v[circuit_mask].mean() / v[~circuit_mask].mean()
    top_u = np.argmax(u); tu_l, tu_h = head_labels[top_u]
    top_v = np.argmax(v); tv_l, tv_h = head_labels[top_v]
    print(f"{i+1:>4}  {eu:>12.3f}  {ev:>12.3f}  "
          f"L{tu_l}H{tu_h} {head_role(tu_l,tu_h)[:22]:<24}  "
          f"L{tv_l}H{tv_h} {head_role(tv_l,tv_h)}")

# --- Figure ---
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# Top-left: canonical correlations IOI vs nonIOI
ax = axes[0, 0]
x = np.arange(1, 21)
ax.plot(x, r_ioi, 'o-', color='#d62728', label='IOI', lw=1.5, ms=5)
ax.plot(x, r_non, 's--', color='#aaaaaa', label='non-IOI', lw=1.5, ms=5)
ax.set_xlabel("Canonical component", fontsize=10)
ax.set_ylabel("Canonical correlation r", fontsize=10)
ax.set_title("CCA: KL vs χ\nIOI shows higher cross-modal coupling?", fontsize=10)
ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

# Top-middle: enrichment per CC
ax = axes[0, 1]
enrich_u = [np.abs(U_ioi[:,i])[circuit_mask].mean() / np.abs(U_ioi[:,i])[~circuit_mask].mean() for i in range(20)]
enrich_v = [np.abs(V_ioi[:,i])[circuit_mask].mean() / np.abs(V_ioi[:,i])[~circuit_mask].mean() for i in range(20)]
ax.plot(x, enrich_u, 'o-', color='#1f77b4', label='KL direction', lw=1.5, ms=5)
ax.plot(x, enrich_v, 's-', color='#ff7f0e', label='χ direction', lw=1.5, ms=5)
ax.axhline(1.0, color='gray', lw=0.8, ls='--')
ax.set_xlabel("Canonical component", fontsize=10)
ax.set_ylabel("Circuit enrichment", fontsize=10)
ax.set_title("Circuit head enrichment\nin CCA directions (IOI)", fontsize=10)
ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

# Top-right: IOI - nonIOI canonical correlations
ax = axes[0, 2]
diff_r = r_ioi - r_non
colors = ['#d62728' if d > 0 else '#1f77b4' for d in diff_r]
ax.bar(x, diff_r, color=colors)
ax.axhline(0, color='black', lw=0.8)
ax.set_xlabel("Canonical component", fontsize=10)
ax.set_ylabel("r_IOI - r_nonIOI", fontsize=10)
ax.set_title("Task-specific cross-modal coupling\n(positive = more coupled on IOI)", fontsize=10)
ax.grid(True, alpha=0.2)

# Bottom: top 3 CCA KL directions (bar plots colored by role)
def plot_cca_dir(ax, vec, title, circuit_mask, head_labels):
    colors = [ROLE_COLORS[head_role(l, h)] for l, h in head_labels]
    order = np.argsort(vec)
    ax.bar(range(144), vec[order], color=[colors[i] for i in order], width=1.0, linewidth=0)
    circ_pos  = [r for r, i in enumerate(order) if circuit_mask[i]]
    circ_vals = [vec[i] for i in order if circuit_mask[i]]
    ax.scatter(circ_pos, circ_vals, s=25, c='black', zorder=3, marker='|')
    ax.set_title(title, fontsize=9); ax.axhline(0, color='gray', lw=0.5)
    ax.tick_params(labelsize=7); ax.set_xlim(-1, 145)

for i, ax in enumerate(axes[1]):
    eu = np.abs(U_ioi[:,i])[circuit_mask].mean() / np.abs(U_ioi[:,i])[~circuit_mask].mean()
    plot_cca_dir(ax, U_ioi[:, i],
                 f"CC{i+1} KL direction  r={r_ioi[i]:.3f}  enrich={eu:.2f}",
                 circuit_mask, head_labels)
    ax.set_xlabel("Head (sorted by loading)", fontsize=9)
    ax.set_ylabel("Loading", fontsize=9)

legend_elements = [Patch(facecolor=ROLE_COLORS[r], label=r) for r in ROLE_COLORS]
fig.legend(handles=legend_elements, loc='lower center', fontsize=7, ncol=4,
           bbox_to_anchor=(0.5, -0.02))

plt.suptitle("CCA between KL and χ — does cross-modal coupling reveal circuit structure?", fontsize=12)
plt.tight_layout()
plt.savefig("cca_kl_chi.png", dpi=300, bbox_inches="tight")
print("\nSaved cca_kl_chi.png")
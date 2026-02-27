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
            kl[i, layer, :]  = (log_n - entropy.cpu().numpy()) / log_n
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

KL_ioi  = kl_ioi.reshape(50, -1)   # (50, 144)
KL_non  = kl_non.reshape(50, -1)
CHI_ioi = chi_ioi.reshape(50, -1)
CHI_non = chi_non.reshape(50, -1)

# --- Three C_diff matrices ---
# 1. KL only (144×144) — baseline
C_diff_kl = np.corrcoef(KL_ioi.T) - np.corrcoef(KL_non.T)

# 2. χ only (144×144)
C_diff_chi = np.corrcoef(CHI_ioi.T) - np.corrcoef(CHI_non.T)

# 3. Augmented [KL; χ] (288×288)
# Stack features: columns 0-143 = KL, 144-287 = χ
X_ioi = np.hstack([KL_ioi, CHI_ioi])   # (50, 288)
X_non = np.hstack([KL_non, CHI_non])
C_diff_aug = np.corrcoef(X_ioi.T) - np.corrcoef(X_non.T)   # (288, 288)

def eig_desc(C):
    vals, vecs = np.linalg.eigh(C)
    return vals[::-1].copy(), vecs[:, ::-1].copy()

vals_kl,  vecs_kl  = eig_desc(C_diff_kl)
vals_chi, vecs_chi = eig_desc(C_diff_chi)
vals_aug, vecs_aug = eig_desc(C_diff_aug)

# Augmented circuit mask: first 144 = KL block, next 144 = χ block
circuit_mask_aug = np.concatenate([circuit_mask, circuit_mask])

def enrichment(vec, cmask):
    a = np.abs(vec)
    return a[cmask].mean() / a[~cmask].mean()

print("\nTop eigenvalues comparison:")
print(f"{'k':>4}  {'C_diff_KL':>12}  {'C_diff_chi':>12}  {'C_diff_aug':>12}")
print("-" * 46)
for k in range(10):
    print(f"{k+1:>4}  {vals_kl[k]:>12.3f}  {vals_chi[k]:>12.3f}  {vals_aug[k]:>12.3f}")

print("\nCircuit enrichment in top eigenvectors:")
print(f"{'EV':>4}  {'KL-only':>10}  {'chi-only':>10}  {'aug(KL)':>10}  {'aug(chi)':>10}  {'aug(full)':>10}")
print("-" * 60)
for k in range(10):
    ek  = enrichment(vecs_kl[:, k],  circuit_mask)
    ec  = enrichment(vecs_chi[:, k], circuit_mask)
    # augmented vector: split into KL and χ halves
    v_aug = vecs_aug[:, k]
    ea_kl = enrichment(v_aug[:144],  circuit_mask)
    ea_chi= enrichment(v_aug[144:],  circuit_mask)
    ea_all= enrichment(v_aug, circuit_mask_aug)
    print(f"{k+1:>4}  {ek:>10.3f}  {ec:>10.3f}  {ea_kl:>10.3f}  {ea_chi:>10.3f}  {ea_all:>10.3f}")

# Cumulative variance explained (positive eigenvalues only)
def cum_var(vals, ks=[1,2,3,5,10]):
    pos = vals[vals > 0]
    cv = np.cumsum(pos) / pos.sum()
    return {k: cv[k-1] for k in ks}

print("\nCumulative variance (positive eigenvalues):")
print(f"{'k':>4}  {'KL-only':>10}  {'chi-only':>10}  {'augmented':>10}")
for k in [1, 2, 3, 5, 10]:
    ck  = cum_var(vals_kl)[k]
    cc  = cum_var(vals_chi)[k]
    ca  = cum_var(vals_aug)[k]
    print(f"{k:>4}  {ck:>10.1%}  {cc:>10.1%}  {ca:>10.1%}")

# Print top loadings for EV1 of each
for label, vals, vecs, cmask, is_aug in [
    ("C_diff KL",  vals_kl,  vecs_kl,  circuit_mask,     False),
    ("C_diff χ",   vals_chi, vecs_chi, circuit_mask,     False),
    ("C_diff aug", vals_aug, vecs_aug, circuit_mask_aug, True),
]:
    print(f"\n{label}  EV1 (λ={vals[0]:.3f})  top 10 loadings:")
    vec = vecs[:, 0]
    top = np.argsort(np.abs(vec))[::-1][:10]
    for i in top:
        if is_aug:
            block = "KL " if i < 144 else "χ  "
            j = i % 144
        else:
            block = ""; j = i
        l, h = head_labels[j]
        role = head_role(l, h)
        tag = " ***" if (l, h) in ALL_CIRCUIT_HEADS else ""
        print(f"  {block}L{l}H{h} ({role:.22s}): {vec[i]:+.4f}{tag}")

# --- Figure ---
fig, axes = plt.subplots(2, 3, figsize=(17, 10))

# Row 0: scree plots
for ax, vals, title, color in [
    (axes[0,0], vals_kl,  "C_diff (KL only)",       "#1f77b4"),
    (axes[0,1], vals_chi, "C_diff (χ only)",         "#ff7f0e"),
    (axes[0,2], vals_aug, "C_diff (augmented KL+χ)", "#2ca02c"),
]:
    n = len(vals)
    ax.plot(range(1, n+1), vals, '-', color=color, lw=1.2, alpha=0.8)
    ax.axhline(0, color='gray', lw=0.6, ls='--')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Eigenvalue rank", fontsize=9)
    ax.set_ylabel("Eigenvalue", fontsize=9)
    ax.grid(True, alpha=0.2)
    # shade positive
    pos_x = [i+1 for i, v in enumerate(vals) if v > 0]
    if pos_x:
        ax.axvspan(1, max(pos_x)+0.5, alpha=0.05, color=color)

# Row 1: EV1 bar plots
def plot_ev(ax, vec, title, cmask_144):
    """vec may be 144 or 288; for 288 we show KL half only."""
    v = vec[:144]
    colors = [ROLE_COLORS[head_role(l, h)] for l, h in head_labels]
    order = np.argsort(v)
    ax.bar(range(144), v[order], color=[colors[i] for i in order], width=1.0, linewidth=0)
    cp = [r for r, i in enumerate(order) if cmask_144[i]]
    cv = [v[i] for i in order if cmask_144[i]]
    ax.scatter(cp, cv, s=25, c='black', zorder=3, marker='|')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Head (sorted by KL loading)", fontsize=8)
    ax.set_ylabel("Loading", fontsize=8)
    ax.tick_params(labelsize=7); ax.set_xlim(-1, 145)

e1_kl  = enrichment(vecs_kl[:,0],       circuit_mask)
e1_chi = enrichment(vecs_chi[:,0],       circuit_mask)
e1_aug = enrichment(vecs_aug[:144, 0],   circuit_mask)

plot_ev(axes[1,0], vecs_kl[:,0],       f"EV1 KL-only   λ={vals_kl[0]:.2f}  enrich={e1_kl:.2f}",  circuit_mask)
plot_ev(axes[1,1], vecs_chi[:,0],      f"EV1 χ-only    λ={vals_chi[0]:.2f}  enrich={e1_chi:.2f}", circuit_mask)
plot_ev(axes[1,2], vecs_aug[:,0],      f"EV1 aug (KL half)  λ={vals_aug[0]:.2f}  enrich={e1_aug:.2f}", circuit_mask)

legend_elements = [Patch(facecolor=ROLE_COLORS[r], label=r) for r in ROLE_COLORS]
fig.legend(handles=legend_elements, loc='lower center', fontsize=7, ncol=4,
           bbox_to_anchor=(0.5, -0.02))

plt.suptitle("cPCA (C_diff) — KL only vs χ only vs augmented [KL, χ]\nDoes adding χ sharpen circuit signal?", fontsize=12)
plt.tight_layout()
plt.savefig("cdiff_augmented.png", dpi=300, bbox_inches="tight")
print("\nSaved cdiff_augmented.png")
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

def compute_kl_and_varv(prompts):
    """
    For each prompt, compute per head at final token position:
      KL  = log(n) - H(π)                         = attention selectivity
      Var_v = E_π[||v - E_π[v]||²]                = variance of value vectors under π
    Returns kl (n, 12, 12), varv (n, 12, 12)
    """
    kl   = np.zeros((len(prompts), 12, 12))
    varv = np.zeros((len(prompts), 12, 12))
    for i, prompt in enumerate(prompts):
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens)
        log_n = np.log(tokens.shape[1])
        for layer in range(12):
            pi_all = cache["pattern", layer][0, :, -1, :]  # (n_heads, seq)
            v_all  = cache["v",       layer][0, :, :, :]   # (seq, n_heads, d_head)
            for head in range(12):
                pi = pi_all[head].cpu().numpy()             # (seq,)
                v  = v_all[:, head, :].cpu().numpy()        # (seq, d_head)

                # KL
                ent = -(pi * np.log(pi + 1e-12)).sum()
                kl[i, layer, head] = (log_n - ent) / log_n

                # Var_v = E_π[||v - E_π[v]||²]
                ev = (pi[:, None] * v).sum(0)               # (d_head,) = E_π[v]
                diff_sq = ((v - ev[None, :]) ** 2).sum(-1)  # (seq,)
                varv[i, layer, head] = (pi * diff_sq).sum()
    return kl, varv

print("Computing KL + Var_v for IOI prompts...")
kl_ioi, varv_ioi = compute_kl_and_varv(generate_ioi(50))
print("Computing KL + Var_v for non-IOI prompts...")
kl_non, varv_non = compute_kl_and_varv(generate_non_ioi(50))

hl = [(l, h) for l in range(12) for h in range(12)]
circuit_mask = np.array([(l, h) in ALL_CIRCUIT_HEADS for l, h in hl])

def head_role(l, h):
    for role, heads in IOI_HEADS.items():
        if (l, h) in heads: return role
    return "Non-circuit"

KL_ioi   = kl_ioi.reshape(50, -1)
KL_non   = kl_non.reshape(50, -1)
VV_ioi   = varv_ioi.reshape(50, -1)
VV_non   = varv_non.reshape(50, -1)

C_diff_kl  = np.corrcoef(KL_ioi.T) - np.corrcoef(KL_non.T)
C_diff_vv  = np.corrcoef(VV_ioi.T) - np.corrcoef(VV_non.T)

# ── Coverage comparison ────────────────────────────────────────────────────────
def ranked_pairs(C_diff):
    pairs = []
    for i in range(144):
        for j in range(i + 1, 144):
            ic = circuit_mask[i]; jc = circuit_mask[j]
            ct = "CC" if ic and jc else ("CN" if ic or jc else "NN")
            pairs.append((C_diff[i, j], i, j, ct))
    pairs.sort(reverse=True)
    return pairs

pairs_kl = ranked_pairs(C_diff_kl)
pairs_vv = ranked_pairs(C_diff_vv)

print(f"\n{'k':>5}  {'KL CC pairs':>11}  {'KL heads':>9}  {'Vv CC pairs':>12}  {'Vv heads':>9}")
print("-" * 58)
for k in [10, 20, 50, 100, 200]:
    def stats(pairs):
        top = pairs[:k]
        seen = set()
        for c, i, j, ct in top:
            if circuit_mask[i]: seen.add(hl[i])
            if circuit_mask[j]: seen.add(hl[j])
        cc = sum(1 for p in top if p[3] == "CC")
        return cc, len(seen)
    cc_kl, h_kl = stats(pairs_kl)
    cc_vv, h_vv = stats(pairs_vv)
    print(f"{k:>5}  {cc_kl:>11}  {h_kl:>5}/{len(ALL_CIRCUIT_HEADS)}  {cc_vv:>12}  {h_vv:>5}/{len(ALL_CIRCUIT_HEADS)}")

# Top-20 table for each
def print_table(pairs, label, k=20):
    print(f"\n{label} — top {k}:")
    print(f"{'Rk':<4} {'Head i':<8} {'Role i':<24} {'Head j':<8} {'Role j':<24} {'C_diff':>7}  cat")
    print("-" * 85)
    for rk, (c, i, j, ct) in enumerate(pairs[:k], 1):
        li, hi = hl[i]; lj, hj = hl[j]
        print(f"{rk:<4} L{li}H{hi:<5} {head_role(li,hi):<24} L{lj}H{hj:<5} {head_role(lj,hj):<24} {c:+.3f}  {ct}")

print_table(pairs_kl, "KL C_diff")
print_table(pairs_vv, "Var_v C_diff")

# ── Figure: side-by-side signed ranking ───────────────────────────────────────
COLORS  = {"CC": "#d62728", "CN": "#ff7f0e", "NN": "#bbbbbb"}
ALPHAS  = {"CC": 1.0,       "CN": 0.7,       "NN": 0.2}
WIDTHS  = {"CC": 1.0,       "CN": 0.6,       "NN": 0.35}
ZORDERS = {"CC": 3,         "CN": 2,         "NN": 1}

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

for ax, pairs, title in [
    (axes[0], pairs_kl, "KL C_diff (attention selectivity)"),
    (axes[1], pairs_vv, "Var_v C_diff (value variance)"),
]:
    cats = [p[3] for p in pairs]
    vals = [p[0] for p in pairs]
    n = len(pairs)
    for cat in ["NN", "CN", "CC"]:
        idx = [i for i, c in enumerate(cats) if c == cat]
        ax.vlines(idx, 0, [vals[i] for i in idx],
                  color=COLORS[cat], alpha=ALPHAS[cat],
                  linewidth=WIDTHS[cat], zorder=ZORDERS[cat])
    ax.axhline(0, color="black", lw=0.8, zorder=4)
    ax.set_xlabel("Rank (sorted by C_diff,ij)", fontsize=10)
    ax.set_ylabel("C_diff,ij", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-100, n + 100)
    ax.grid(True, alpha=0.15, axis='y')

    # Stats box
    lines = []
    for k in [10, 20, 50]:
        top = pairs[:k]
        cc = sum(1 for p in top if p[3] == "CC")
        lines.append(f"top-{k:>2}: {cc} CC")
    ax.text(0.98, 0.98, "\n".join(lines), transform=ax.transAxes,
            fontsize=8, va='top', ha='right', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

legend_elements = [
    Line2D([0], [0], color=COLORS["CC"], lw=2, label="circuit–circuit (CC)"),
    Line2D([0], [0], color=COLORS["CN"], lw=2, alpha=0.8, label="circuit–non-circuit (CN)"),
    Line2D([0], [0], color=COLORS["NN"], lw=1, alpha=0.5, label="non–non (NN)"),
]
axes[0].legend(handles=legend_elements, fontsize=9, loc="upper right")

plt.suptitle("KL vs Var_v C_diff — which diagnostic better recovers the IOI circuit?\n"
             "n=50 IOI / 50 non-IOI, GPT-2 small, signed ranking",
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig("fig_varv_vs_kl_cdiff.png", dpi=300, bbox_inches="tight")
print("\nSaved fig_varv_vs_kl_cdiff.png")
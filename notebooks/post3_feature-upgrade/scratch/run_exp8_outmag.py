"""
Exp 8: Compare Var_v vs output magnitude as circuit-head discriminators.

For each head at the last token:
  Var_v    = (1/d) * [E_π[||v||²] - ||µ_v||²]   (already cached)
  out_mag  = ||µ_v||² / d                         (new — how much signal the head writes)

where µ_v = Σ_i π_i v_i is the attention-weighted mean of the value vectors.

The hypothesis: some circuit heads change *what they write* without necessarily
changing how diverse the values are (Var_v). Out_mag catches those heads.

Figures:
  exp8_delta_comparison.png  — side-by-side bar charts of ΔVar_v and Δout_mag
  exp8_precision_curve.png   — precision curves: how well each statistic ranks circuit heads

Run with: /opt/miniconda3/bin/python run_exp8_outmag.py
"""

import os, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figs", exist_ok=True)

# ── metadata ──────────────────────────────────────────────────────────────────
IOI_HEADS = {
    "Name Mover":          [(9, 6), (9, 9), (10, 0)],
    "Backup Name Mover":   [(9, 0), (9, 7), (10, 1), (10, 2), (10, 6), (10, 10), (11, 2), (11, 9)],
    "Negative Name Mover": [(10, 7), (11, 10)],
    "S-Inhibition":        [(7, 3), (7, 9), (8, 6), (8, 10)],
    "Induction":           [(5, 5), (6, 9)],
    "Duplicate Token":     [(0, 1), (3, 0)],
    "Previous Token":      [(2, 2), (4, 11)],
}
ROLE_ORDER = ["Name Mover","Backup Name Mover","Negative Name Mover",
              "S-Inhibition","Induction","Duplicate Token","Previous Token",
              "Non-circuit"]
ROLE_COLORS = {
    "Name Mover":          "#d62728",
    "Backup Name Mover":   "#ff7f0e",
    "Negative Name Mover": "#9467bd",
    "S-Inhibition":        "#2ca02c",
    "Induction":           "#17becf",
    "Duplicate Token":     "#bcbd22",
    "Previous Token":      "#e377c2",
    "Non-circuit":         "#cccccc",
}
SHORT = {"Name Mover":"NM","Backup Name Mover":"BNM","Negative Name Mover":"NegNM",
         "S-Inhibition":"SI","Induction":"Ind","Duplicate Token":"DT","Previous Token":"PT",
         "Non-circuit":"NC"}

ALL_CIRCUIT_HEADS = set(lh for hs in IOI_HEADS.values() for lh in hs)
HL_FLAT = [(l, h) for l in range(12) for h in range(12)]

def head_role(l, h):
    for role, heads in IOI_HEADS.items():
        if (l, h) in heads: return role
    return "Non-circuit"

HEAD_NAMES   = [f"L{l}H{h}" for l, h in HL_FLAT]
HEAD_CIRCUIT = [lh in ALL_CIRCUIT_HEADS for lh in HL_FLAT]
HEAD_ROLE    = [head_role(*lh) for lh in HL_FLAT]
HEAD_COLOR   = [ROLE_COLORS[r] for r in HEAD_ROLE]
N_HEAD = 144

# ── prompts (same seeds as exp5 so we use same 1000 prompts) ──────────────────
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
NAMES = ["Mary","John","Alice","Bob","Sarah","Tom","Emma","James","Lisa","David","Kate","Mark"]

def gen_ioi(n, seed=42):
    random.seed(seed); templates = TEMPLATES_ABBA + TEMPLATES_BABA
    seen, out = set(), []
    while len(out) < n:
        t = random.choice(templates); a, b = random.sample(NAMES, 2)
        p = t.format(A=a, B=b)
        if p not in seen: seen.add(p); out.append(p)
    return out

def gen_non_ioi(n, seed=43):
    random.seed(seed); seen, out = set(), []
    while len(out) < n:
        t = random.choice(TEMPLATES_NON_IOI); a, b, c = random.sample(NAMES, 3)
        p = t.format(A=a, B=b, C=c)
        if p not in seen: seen.add(p); out.append(p)
    return out

# ── feature extraction ────────────────────────────────────────────────────────
def compute_both(model, prompts):
    """
    Returns two arrays, each (n_prompts, 144):
      varv    — Var_v / d_head for each attention head at the last token
      out_mag — ||µ_v||² / d_head  (squared output magnitude, normalized)
    """
    nL, nH, dH, EPS = model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_head, 1e-8
    n = len(prompts)
    varv_arr    = np.zeros((n, nL * nH))
    outmag_arr  = np.zeros((n, nL * nH))

    for p_idx, prompt in enumerate(prompts):
        if p_idx % 100 == 0: print(f"  {p_idx}/{n}", flush=True)
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens)

        for layer in range(nL):
            pi   = cache["pattern", layer][0, :, -1, :]    # (nH, seq)
            v    = cache["v", layer][0].permute(1, 0, 2)   # (nH, seq, dH)
            mu_v = (pi.unsqueeze(-1) * v).sum(dim=1)        # (nH, dH)  — weighted mean value
            E_v2 = (pi * (v**2).sum(-1)).sum(-1)            # (nH,)

            mu_sq = (mu_v**2).sum(-1)        # ||µ_v||²  (nH,)
            vv    = (E_v2 - mu_sq) / dH      # Var_v
            om    = mu_sq / dH               # output magnitude (normalized)

            idx = layer * nH
            varv_arr[p_idx, idx:idx+nH]   = vv.cpu().numpy()
            outmag_arr[p_idx, idx:idx+nH] = om.cpu().numpy()

    return varv_arr, outmag_arr

# cache to disk so we don't rerun the model
CACHE_FILES = ["varv_ioi.npy","varv_non.npy","outmag_ioi.npy","outmag_non.npy"]
if all(os.path.exists(f) for f in CACHE_FILES):
    print("Loading cached arrays...")
    varv_ioi   = np.load("varv_ioi.npy")
    varv_non   = np.load("varv_non.npy")
    outmag_ioi = np.load("outmag_ioi.npy")
    outmag_non = np.load("outmag_non.npy")
else:
    from transformer_lens import HookedTransformer
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained("gpt2-small")
    print(f"Loaded model")

    N = 1000
    ioi_prompts     = gen_ioi(N)
    non_ioi_prompts = gen_non_ioi(N)

    print("Computing IOI features...")
    varv_ioi, outmag_ioi = compute_both(model, ioi_prompts)
    print("Computing non-IOI features...")
    varv_non, outmag_non = compute_both(model, non_ioi_prompts)

    np.save("varv_ioi.npy",   varv_ioi)
    np.save("varv_non.npy",   varv_non)
    np.save("outmag_ioi.npy", outmag_ioi)
    np.save("outmag_non.npy", outmag_non)
    print("Saved.")

print(f"varv_ioi {varv_ioi.shape}, outmag_ioi {outmag_ioi.shape}")

# ── deltas: mean(IOI) - mean(non-IOI) ─────────────────────────────────────────
delta_varv   = varv_ioi.mean(0)   - varv_non.mean(0)    # (144,)
delta_outmag = outmag_ioi.mean(0) - outmag_non.mean(0)  # (144,)

# ── Fig 1: side-by-side bar charts ────────────────────────────────────────────
xs = np.arange(N_HEAD)
fig, axes = plt.subplots(2, 1, figsize=(20, 7), sharex=True)

for ax, delta, label in zip(axes,
                             [delta_varv, delta_outmag],
                             ["Δ Var_v  (IOI − non-IOI)",
                              "Δ ||µ_v||²/d  (IOI − non-IOI)"]):
    ax.bar(xs, delta, color=HEAD_COLOR, alpha=0.85, width=0.9)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel(label, fontsize=10)
    # mark circuit heads
    for i, c in enumerate(HEAD_CIRCUIT):
        if c: ax.axvline(i, color="k", lw=0.4, alpha=0.25, zorder=0)

axes[-1].set_xticks([l*12+6 for l in range(12)])
axes[-1].set_xticklabels([f"L{l}" for l in range(12)], fontsize=9)

legend_handles = [mpatches.Patch(color=ROLE_COLORS[r], label=SHORT[r])
                  for r in ROLE_ORDER]
axes[0].legend(handles=legend_handles, fontsize=8, loc="upper left", ncol=4, framealpha=0.9)
plt.suptitle("Var_v vs output magnitude as circuit-head discriminators\n"
             "(thin black lines = known circuit head positions)", fontsize=11)
plt.tight_layout()
plt.savefig("figs/exp8_delta_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figs/exp8_delta_comparison.png")

# ── Mann-Whitney discrimination stats ─────────────────────────────────────────
circuit_mask = np.array(HEAD_CIRCUIT)

def discrimination_stats(delta, label):
    cc  = np.abs(delta[circuit_mask])
    nc  = np.abs(delta[~circuit_mask])
    u, p = stats.mannwhitneyu(cc, nc, alternative="greater")
    ratio = cc.mean() / nc.mean()
    print(f"  {label}: circuit mean={cc.mean():.4f}, NC mean={nc.mean():.4f}, "
          f"ratio={ratio:.2f}x, p={p:.4e}")
    return ratio, p

print("\nDiscrimination (Mann-Whitney |Δ| circuit > NC):")
discrimination_stats(delta_varv,   "|ΔVar_v|")
discrimination_stats(delta_outmag, "|Δout_mag|")

# ── Fig 2: precision curves ───────────────────────────────────────────────────
# Rank heads by |Δ|, walk down the list, track cumulative precision
# (fraction of top-k that are known circuit heads).
# Chance line = 20/144 ≈ 13.9%

def precision_curve(delta):
    order = np.argsort(np.abs(delta))[::-1]   # highest |Δ| first
    prec = []
    n_circ = 0
    for k, idx in enumerate(order, 1):
        if circuit_mask[idx]: n_circ += 1
        prec.append(n_circ / k)
    return prec

prec_varv   = precision_curve(delta_varv)
prec_outmag = precision_curve(delta_outmag)
chance = len(ALL_CIRCUIT_HEADS) / N_HEAD  # 20/144

fig, ax = plt.subplots(figsize=(8, 5))
ks = np.arange(1, N_HEAD + 1)
ax.plot(ks, prec_varv,   color="#1f77b4", lw=2, label="Var_v")
ax.plot(ks, prec_outmag, color="#d62728", lw=2, label="||µ_v||²/d  (output magnitude)")
ax.axhline(chance, color="gray", lw=1.2, ls="--", label=f"chance ({chance:.1%})")
ax.set_xlabel("Top-k heads by |Δ|", fontsize=11)
ax.set_ylabel("Precision (fraction that are circuit heads)", fontsize=11)
ax.set_title("Precision curve: how well does each statistic rank circuit heads?", fontsize=11)
ax.legend(fontsize=10)
ax.set_xlim(1, N_HEAD)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("figs/exp8_precision_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figs/exp8_precision_curve.png")

# ── which circuit heads does each miss? ───────────────────────────────────────
# "missed" = |Δ| below the median of all heads
varv_thresh   = np.median(np.abs(delta_varv))
outmag_thresh = np.median(np.abs(delta_outmag))

varv_caught   = set(i for i in range(N_HEAD) if HEAD_CIRCUIT[i] and abs(delta_varv[i])   > varv_thresh)
outmag_caught = set(i for i in range(N_HEAD) if HEAD_CIRCUIT[i] and abs(delta_outmag[i]) > outmag_thresh)

print(f"\nCircuit heads above median |Δ|:")
print(f"  Var_v catches:    {sorted(HEAD_NAMES[i] for i in varv_caught)}")
print(f"  out_mag catches:  {sorted(HEAD_NAMES[i] for i in outmag_caught)}")
print(f"  out_mag only:     {sorted(HEAD_NAMES[i] for i in outmag_caught - varv_caught)}")
print(f"  Var_v only:       {sorted(HEAD_NAMES[i] for i in varv_caught - outmag_caught)}")
print(f"  both miss:        {sorted(HEAD_NAMES[i] for i in range(N_HEAD) if HEAD_CIRCUIT[i] and i not in varv_caught and i not in outmag_caught)}")
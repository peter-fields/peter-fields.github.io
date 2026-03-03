"""
Post 3 — Experiment 4
  (a) ||mu_v||^2 (output norm) vs KL independence test
  (b) MLP Var(post-activations) at last token
  (c) Unified 90-feature picture: Var_v (78 heads) + Var_act (12 MLPs)

Run with: /opt/miniconda3/bin/python run_exp4.py
"""

import os, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import torch

os.makedirs("figs", exist_ok=True)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from transformer_lens import HookedTransformer
torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained("gpt2-small")
print(f"Loaded: {model.cfg.n_layers}L x {model.cfg.n_heads}H, d_head={model.cfg.d_head}, d_mlp={model.cfg.d_mlp}")

# ── head labels ───────────────────────────────────────────────────────────────
IOI_HEADS = {
    "Name Mover":          [(9, 6), (9, 9), (10, 0)],
    "Backup Name Mover":   [(9, 0), (9, 7), (10, 1), (10, 2), (10, 6), (10, 10), (11, 2), (11, 9)],
    "Negative Name Mover": [(10, 7), (11, 10)],
    "S-Inhibition":        [(7, 3), (7, 9), (8, 6), (8, 10)],
    "Induction":           [(5, 5), (6, 9)],
    "Duplicate Token":     [(0, 1), (3, 0)],
    "Previous Token":      [(2, 2), (4, 11)],
}
ALL_CIRCUIT_HEADS = set(lh for hs in IOI_HEADS.values() for lh in hs)
HL_FLAT = [(l, h) for l in range(12) for h in range(12)]

def head_role(l, h):
    for role, heads in IOI_HEADS.items():
        if (l, h) in heads: return role
    return "Non-circuit"

# ── prompts (same seeds as before) ───────────────────────────────────────────
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
    random.seed(seed)
    templates = TEMPLATES_ABBA + TEMPLATES_BABA
    seen, prompts = set(), []
    while len(prompts) < n:
        t = random.choice(templates); a, b = random.sample(NAMES, 2)
        p = t.format(A=a, B=b)
        if p not in seen: seen.add(p); prompts.append(p)
    return prompts

def gen_non_ioi(n, seed=43):
    random.seed(seed)
    seen, prompts = set(), []
    while len(prompts) < n:
        t = random.choice(TEMPLATES_NON_IOI); a, b, c = random.sample(NAMES, 3)
        p = t.format(A=a, B=b, C=c)
        if p not in seen: seen.add(p); prompts.append(p)
    return prompts

N = 1000
ioi_prompts     = gen_ioi(N)
non_ioi_prompts = gen_non_ioi(N)
print(f"n={N} per class.  IOI[0]: {ioi_prompts[0]}")

# ── compute everything in one pass ────────────────────────────────────────────
def compute_exp4(model, prompts):
    """
    Returns per-prompt, at last-token position:
      varv   : (n, n_layers, n_heads)  — Var_v (value-weighted variance)
      mu_sq  : (n, n_layers, n_heads)  — ||mu_v||^2 (output norm^2)
      kl     : (n, n_layers, n_heads)  — normalized KL
      var_act: (n, n_layers)           — Var(MLP post-activations)
    """
    nL, nH, dH  = model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_head
    n = len(prompts)

    varv    = np.zeros((n, nL, nH))
    mu_sq   = np.zeros((n, nL, nH))
    kl      = np.zeros((n, nL, nH))
    var_act = np.zeros((n, nL))

    for p_idx, prompt in enumerate(prompts):
        if p_idx % 100 == 0: print(f"  {p_idx}/{n}", flush=True)
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens)
        seq_len = tokens.shape[1]
        log_n   = np.log(seq_len)

        for layer in range(nL):
            # attention
            attn  = cache["pattern", layer]        # (1, nH, seq, seq)
            pi    = attn[0, :, -1, :]              # (nH, seq)
            log_pi = torch.log(pi + 1e-12)

            # KL
            H = -(pi * log_pi).sum(dim=-1)
            kl[p_idx, layer] = (log_n - H.cpu().numpy()) / log_n

            # value vectors
            v    = cache["v", layer][0, :, :, :]  # (seq, nH, dH)
            v    = v.permute(1, 0, 2)             # (nH, seq, dH)
            pi_e = pi.unsqueeze(-1)               # (nH, seq, 1)

            # mu_v = attended value  (nH, dH)
            mu_v = (pi_e * v).sum(dim=1)

            # E_pi[||v||^2]  (nH,)
            E_v2 = (pi * (v**2).sum(dim=-1)).sum(dim=-1)

            # ||mu_v||^2  (nH,)
            mu_sq[p_idx, layer]  = (mu_v**2).sum(dim=-1).cpu().numpy()

            # Var_v = E[||v||^2] - ||mu||^2, normalised by dH
            varv[p_idx, layer]   = ((E_v2 - (mu_v**2).sum(dim=-1)) / dH).cpu().numpy()

            # MLP post-activations at last token  (d_mlp,)
            post = cache["post", layer][0, -1, :]   # (d_mlp,)
            var_act[p_idx, layer] = post.var().cpu().item()

    return varv, mu_sq, kl, var_act

print("\nComputing IOI...")
varv_ioi, musq_ioi, kl_ioi, vact_ioi = compute_exp4(model, ioi_prompts)
print("Computing non-IOI...")
varv_non, musq_non, kl_non, vact_non = compute_exp4(model, non_ioi_prompts)
print("Done.\n")

# ── (a) ||mu_v||^2 vs KL independence ────────────────────────────────────────
print("=" * 60)
print("EXP 4a — corr(ΔKL, Δ||mu_v||²) vs corr(ΔKL, ΔVar_v)")
print("=" * 60)

delta_kl   = kl_ioi.mean(0)   - kl_non.mean(0)     # (12,12)
delta_varv = varv_ioi.mean(0) - varv_non.mean(0)
delta_musq = musq_ioi.mean(0) - musq_non.mean(0)

dkl_f    = delta_kl.flatten()
dvarv_f  = delta_varv.flatten()
dmusq_f  = delta_musq.flatten()

r_kl_varv, _ = stats.pearsonr(dkl_f, dvarv_f)
r_kl_musq, _ = stats.pearsonr(dkl_f, dmusq_f)
r_varv_musq, _ = stats.pearsonr(dvarv_f, dmusq_f)

print(f"corr(ΔKL,   ΔVar_v):    r = {r_kl_varv:.3f}  (Exp 2 reference)")
print(f"corr(ΔKL,   Δ||mu||²):  r = {r_kl_musq:.3f}")
print(f"corr(ΔVar_v, Δ||mu||²): r = {r_varv_musq:.3f}")

# discrimination: |Δ||mu||^2|
circ_musq = np.array([abs(delta_musq[l,h]) for l,h in ALL_CIRCUIT_HEADS])
nonc_musq = np.array([abs(delta_musq[l,h]) for l in range(12) for h in range(12)
                       if (l,h) not in ALL_CIRCUIT_HEADS])
_, p_musq = stats.mannwhitneyu(circ_musq, nonc_musq, alternative="greater")
print(f"\n|Δ||mu||²| discrimination: circ={circ_musq.mean():.4f}  nonc={nonc_musq.mean():.4f}  "
      f"ratio={circ_musq.mean()/nonc_musq.mean():.2f}x  MW p={p_musq:.2e}")

colors = ["#d62728" if lh in ALL_CIRCUIT_HEADS else "#aaaaaa" for lh in HL_FLAT]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, x, y, xl, yl, r in [
    (axes[0], dkl_f,   dvarv_f, "ΔKL",     "ΔVar_v",    r_kl_varv),
    (axes[1], dkl_f,   dmusq_f, "ΔKL",     "Δ||mu||²",  r_kl_musq),
    (axes[2], dvarv_f, dmusq_f, "ΔVar_v",  "Δ||mu||²",  r_varv_musq),
]:
    ax.scatter(x, y, c=colors, s=20, alpha=0.7)
    ax.set_xlabel(xl, fontsize=12); ax.set_ylabel(yl, fontsize=12)
    ax.set_title(f"r = {r:.3f}", fontsize=13)
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.axvline(0, color="k", lw=0.5, alpha=0.3)
plt.suptitle(f"Exp 4a: Var_v vs ||mu||² vs KL  (n={N}, red=circuit)", y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig("figs/exp4a_musq_vs_varv.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figs/exp4a_musq_vs_varv.png")

# ── (b) MLP Var(post-activations) ────────────────────────────────────────────
print()
print("=" * 60)
print("EXP 4b — MLP Var(post-activations) at last token")
print("=" * 60)

delta_vact = vact_ioi.mean(0) - vact_non.mean(0)   # (12,) one per layer

print("ΔVar_act per layer (IOI - non-IOI):")
for layer in range(12):
    print(f"  L{layer:2d}:  {delta_vact[layer]:+.5f}")

fig, ax = plt.subplots(figsize=(9, 4))
colors_mlp = ["#d62728" if delta_vact[l] > 0 else "#4878cf" for l in range(12)]
ax.bar(range(12), delta_vact, color=colors_mlp, alpha=0.8)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(12))
ax.set_xticklabels([f"L{l}" for l in range(12)], fontsize=11)
ax.set_ylabel("ΔVar(post-act)  [IOI − non-IOI]", fontsize=11)
ax.set_title(f"MLP activation variance shift by layer  (n={N})", fontsize=12)
plt.tight_layout()
plt.savefig("figs/exp4b_mlp_var_act.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figs/exp4b_mlp_var_act.png")

# ── (c) Unified 90-feature view ───────────────────────────────────────────────
print()
print("=" * 60)
print("EXP 4c — Unified 90-feature view: log(Var_v) [78 heads] + log(Var_act) [12 MLPs]")
print("=" * 60)

EPS = 1e-8

# Attention: log(Var_v) per head, mean over prompts, shape (12, 12)
log_varv_ioi_mean = np.log(varv_ioi.mean(0) + EPS)
log_varv_non_mean = np.log(varv_non.mean(0) + EPS)
delta_log_varv    = log_varv_ioi_mean - log_varv_non_mean   # (12, 12)

# MLP: log(Var_act) per layer, mean over prompts, shape (12,)
log_vact_ioi_mean = np.log(vact_ioi.mean(0) + EPS)
log_vact_non_mean = np.log(vact_non.mean(0) + EPS)
delta_log_vact    = log_vact_ioi_mean - log_vact_non_mean   # (12,)

# Combine into 90-d feature vector delta
feat_names = [f"L{l}H{h}" for l in range(12) for h in range(12)] + [f"MLP_L{l}" for l in range(12)]
feat_delta  = np.concatenate([delta_log_varv.flatten(), delta_log_vact])   # (156,)
feat_is_circuit = [lh in ALL_CIRCUIT_HEADS for lh in HL_FLAT] + [False]*12
N_FEAT = len(feat_delta)  # 156

print(f"Feature vector: {N_FEAT} components")
print(f"  Attention heads: {12*12}")
print(f"  MLP layers:      {12}")

# Sort by |delta| for a ranked-importance bar chart
order = np.argsort(np.abs(feat_delta))[::-1]
top_n = 30
print(f"\nTop {top_n} features by |Δlog(Var)|:")
for rank, idx in enumerate(order[:top_n]):
    tag = "(circuit)" if feat_is_circuit[idx] else ""
    print(f"  {rank+1:2d}. {feat_names[idx]:10s}  Δlog(Var)={feat_delta[idx]:+.3f}  {tag}")

fig, ax = plt.subplots(figsize=(20, 5))
bar_colors = []
for i, name in enumerate(feat_names):
    if name.startswith("MLP"):
        bar_colors.append("#2ca02c")
    elif feat_is_circuit[i]:
        bar_colors.append("#d62728")
    else:
        bar_colors.append("#aaaaaa")

xs = np.arange(N_FEAT)
ax.bar(xs, feat_delta, color=bar_colors, alpha=0.85, width=0.9)
ax.axhline(0, color="k", lw=0.7)

# x-axis: layer labels for attention block + MLP label
ax.set_xticks([l*12+6 for l in range(12)] + [144+6])
ax.set_xticklabels([f"L{l}" for l in range(12)] + ["MLPs"], fontsize=10)
ax.axvline(143.5, color="k", lw=1.5, linestyle="--", alpha=0.5)

# Annotate circuit heads with role initials
for i in range(144):
    if feat_is_circuit[i] and abs(feat_delta[i]) > 0.1:
        l, h = HL_FLAT[i]
        role = head_role(l, h)
        short = {"Name Mover":"NM","Backup Name Mover":"BNM","Negative Name Mover":"NegNM",
                 "S-Inhibition":"SI","Induction":"Ind","Duplicate Token":"DT","Previous Token":"PT"}[role]
        ax.text(i, feat_delta[i] + (0.05 if feat_delta[i]>0 else -0.12),
                short, ha="center", va="bottom", fontsize=6, color="#d62728")

import matplotlib.patches as mpatches
legend_handles = [
    mpatches.Patch(color="#d62728", label="Circuit attention head"),
    mpatches.Patch(color="#aaaaaa", label="Non-circuit attention head"),
    mpatches.Patch(color="#2ca02c", label="MLP layer"),
]
ax.legend(handles=legend_handles, fontsize=10, loc="lower left")
ax.set_ylabel("Δlog(Var)  [IOI − non-IOI]", fontsize=11)
ax.set_title(f"Unified 90-feature Δlog(Var) profile  (n={N} prompts each)", fontsize=13)
plt.tight_layout()
plt.savefig("figs/exp4c_unified_feature_delta.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figs/exp4c_unified_feature_delta.png")

# summary stats
circ_feat  = np.abs(feat_delta[np.array([i for i, c in enumerate(feat_is_circuit) if c and i < 144])])
nonc_feat  = np.abs(feat_delta[np.array([i for i, c in enumerate(feat_is_circuit) if not c and i < 144])])
mlp_feat   = np.abs(feat_delta[144:])
_, p_feat  = stats.mannwhitneyu(circ_feat, nonc_feat, alternative="greater")
print(f"\n|Δlog(Var_v)| circuit attn: {circ_feat.mean():.3f} ± {circ_feat.std():.3f}")
print(f"|Δlog(Var_v)| non-circuit:  {nonc_feat.mean():.3f} ± {nonc_feat.std():.3f}  MW p={p_feat:.2e}")
print(f"|Δlog(Var_act)| MLPs:        {mlp_feat.mean():.3f} ± {mlp_feat.std():.3f}")
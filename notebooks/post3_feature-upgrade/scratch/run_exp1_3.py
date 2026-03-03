"""
Post 3 — Experiments 1–3
Run with: /opt/miniconda3/bin/python run_exp1_3.py
"""

import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import torch

os.makedirs("figs", exist_ok=True)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Model ────────────────────────────────────────────────────────────────────
from transformer_lens import HookedTransformer
torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained("gpt2-small")
print(f"Loaded: {model.cfg.n_layers}L × {model.cfg.n_heads}H, d_head={model.cfg.d_head}")

# ── IOI head labels ──────────────────────────────────────────────────────────
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
        if (l, h) in heads:
            return role
    return "Non-circuit"

# ── Prompt generation ────────────────────────────────────────────────────────
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

def gen_ioi(n, seed=42):
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

def gen_non_ioi(n, seed=43):
    random.seed(seed)
    seen, prompts = set(), []
    while len(prompts) < n:
        t = random.choice(TEMPLATES_NON_IOI)
        a, b, c = random.sample(NAMES, 3)
        p = t.format(A=a, B=b, C=c)
        if p not in seen:
            seen.add(p); prompts.append(p)
    return prompts

N = 1000
ioi_prompts     = gen_ioi(N)
non_ioi_prompts = gen_non_ioi(N)

# Sanity check: first 50 must match post2
post2_ioi_0 = "When Mary and John got to the restaurant, Mary offered a menu to"
assert ioi_prompts[0] == post2_ioi_0, f"MISMATCH: {ioi_prompts[0]!r}"
print(f"Prompt check OK. IOI[0]: {ioi_prompts[0]}")
print(f"Prompt lengths: IOI {set(model.to_tokens(p).shape[1] for p in ioi_prompts[:5])}, "
      f"non-IOI {set(model.to_tokens(p).shape[1] for p in non_ioi_prompts[:5])}")

# ── Diagnostics ──────────────────────────────────────────────────────────────
def compute_all_diagnostics(model, prompts):
    n_layers  = model.cfg.n_layers
    n_heads   = model.cfg.n_heads
    d_head    = model.cfg.d_head
    n_prompts = len(prompts)

    kl_all   = np.zeros((n_prompts, n_layers, n_heads))
    chi_all  = np.zeros((n_prompts, n_layers, n_heads))
    varv_all = np.zeros((n_prompts, n_layers, n_heads))

    for p_idx, prompt in enumerate(prompts):
        if p_idx % 20 == 0:
            print(f"  {p_idx}/{n_prompts}", flush=True)
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens)
        seq_len = tokens.shape[1]
        log_n   = np.log(seq_len)

        for layer in range(n_layers):
            attn = cache["pattern", layer]      # (1, n_heads, seq_len, seq_len)
            pi   = attn[0, :, -1, :]            # (n_heads, seq_len)
            log_pi = torch.log(pi + 1e-12)

            # KL
            H = -(pi * log_pi).sum(dim=-1)
            kl_all[p_idx, layer, :] = (log_n - H.cpu().numpy()) / log_n

            # chi
            mean_log_pi = (pi * log_pi).sum(dim=-1, keepdim=True)
            var_log_pi  = (pi * (log_pi - mean_log_pi) ** 2).sum(dim=-1)
            chi_all[p_idx, layer, :] = var_log_pi.cpu().numpy() / (log_n ** 2)

            # Var_v
            v = cache["v", layer][0, :, :, :]  # (seq_len, n_heads, d_head)
            v = v.permute(1, 0, 2)             # (n_heads, seq_len, d_head)
            pi_e  = pi.unsqueeze(-1)            # (n_heads, seq_len, 1)
            mu_v  = (pi_e * v).sum(dim=1)       # (n_heads, d_head)
            E_v2  = (pi * (v ** 2).sum(dim=-1)).sum(dim=-1)  # (n_heads,)
            mu_sq = (mu_v ** 2).sum(dim=-1)
            varv_all[p_idx, layer, :] = ((E_v2 - mu_sq) / d_head).cpu().numpy()

    return kl_all, chi_all, varv_all

print("\nComputing IOI diagnostics...")
kl_ioi, chi_ioi, varv_ioi = compute_all_diagnostics(model, ioi_prompts)
print("Computing non-IOI diagnostics...")
kl_non, chi_non, varv_non = compute_all_diagnostics(model, non_ioi_prompts)
print("Done.\n")

# ── Experiment 1: |Δ| discrimination ─────────────────────────────────────────
print("=" * 60)
print("EXPERIMENT 1 — |Δ| circuit vs non-circuit (Mann-Whitney)")
print("=" * 60)

delta_kl   = kl_ioi.mean(axis=0)   - kl_non.mean(axis=0)
delta_chi  = chi_ioi.mean(axis=0)  - chi_non.mean(axis=0)
delta_varv = varv_ioi.mean(axis=0) - varv_non.mean(axis=0)

results_exp1 = {}
for key, delta in [("KL", delta_kl), ("chi", delta_chi), ("Varv", delta_varv)]:
    circ = np.array([abs(delta[l, h]) for l, h in ALL_CIRCUIT_HEADS])
    nonc = np.array([abs(delta[l, h]) for l in range(12) for h in range(12)
                     if (l, h) not in ALL_CIRCUIT_HEADS])
    u, p = stats.mannwhitneyu(circ, nonc, alternative="greater")
    results_exp1[key] = dict(circ_mean=circ.mean(), circ_std=circ.std(),
                              nonc_mean=nonc.mean(), nonc_std=nonc.std(), p=p)
    print(f"|Δ{key}|  circuit: {circ.mean():.4f} ± {circ.std():.4f}  "
          f"non-circuit: {nonc.mean():.4f} ± {nonc.std():.4f}  MW p={p:.2e}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, key, delta, label in [
    (axes[0], "KL",   delta_kl,   "KL"),
    (axes[1], "chi",  delta_chi,  "χ"),
    (axes[2], "Varv", delta_varv, "Var_v"),
]:
    circ = np.array([abs(delta[l, h]) for l, h in ALL_CIRCUIT_HEADS])
    nonc = np.array([abs(delta[l, h]) for l in range(12) for h in range(12)
                     if (l, h) not in ALL_CIRCUIT_HEADS])
    p = results_exp1[key]["p"]
    ax.hist(nonc, bins=20, alpha=0.5, color="#cccccc", label="Non-circuit", density=True)
    ax.hist(circ, bins=12, alpha=0.7, color="#d62728", label="Circuit", density=True)
    ax.set_xlabel(f"|Δ{label}|", fontsize=12)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"|Δ{label}|  p = {p:.2e}", fontsize=12)
    ax.legend()
plt.suptitle(f"Exp 1: circuit vs non-circuit discrimination (n={N} prompts each)", y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig("figs/exp1_delta_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figs/exp1_delta_distributions.png")

# ── Experiment 2: corr(ΔKL, ΔVar_v) vs corr(ΔKL, Δχ) ─────────────────────
print()
print("=" * 60)
print("EXPERIMENT 2 — corr(ΔKL, ΔVar_v) vs corr(ΔKL, Δχ)")
print("=" * 60)

dkl_f   = delta_kl.flatten()
dchi_f  = delta_chi.flatten()
dvarv_f = delta_varv.flatten()

r_kl_chi,  p_kl_chi  = stats.pearsonr(dkl_f, dchi_f)
r_kl_varv, p_kl_varv = stats.pearsonr(dkl_f, dvarv_f)
r_chi_varv, p_chi_varv = stats.pearsonr(dchi_f, dvarv_f)

print(f"corr(ΔKL,   Δχ):    r = {r_kl_chi:.3f}  (p = {p_kl_chi:.2e})")
print(f"corr(ΔKL,   ΔVar_v): r = {r_kl_varv:.3f}  (p = {p_kl_varv:.2e})")
print(f"corr(Δχ,    ΔVar_v): r = {r_chi_varv:.3f}  (p = {p_chi_varv:.2e})")

# Also compute on circuit heads only
dkl_circ   = np.array([delta_kl[l, h]   for l, h in ALL_CIRCUIT_HEADS])
dchi_circ  = np.array([delta_chi[l, h]  for l, h in ALL_CIRCUIT_HEADS])
dvarv_circ = np.array([delta_varv[l, h] for l, h in ALL_CIRCUIT_HEADS])
r_kl_varv_c, _ = stats.pearsonr(dkl_circ, dvarv_circ)
r_kl_chi_c,  _ = stats.pearsonr(dkl_circ, dchi_circ)
print(f"\nCircuit heads only (n=23):")
print(f"  corr(ΔKL, Δχ):    r = {r_kl_chi_c:.3f}")
print(f"  corr(ΔKL, ΔVar_v): r = {r_kl_varv_c:.3f}")

colors = ["#d62728" if lh in ALL_CIRCUIT_HEADS else "#aaaaaa" for lh in HL_FLAT]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, x, y, xl, yl, r in [
    (axes[0], dkl_f,   dchi_f,   "ΔKL", "Δχ",     r_kl_chi),
    (axes[1], dkl_f,   dvarv_f,  "ΔKL", "ΔVar_v", r_kl_varv),
    (axes[2], dchi_f,  dvarv_f,  "Δχ",  "ΔVar_v", r_chi_varv),
]:
    ax.scatter(x, y, c=colors, s=20, alpha=0.7)
    ax.set_xlabel(xl, fontsize=12); ax.set_ylabel(yl, fontsize=12)
    ax.set_title(f"r = {r:.3f}", fontsize=13)
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.axvline(0, color="k", lw=0.5, alpha=0.3)
plt.suptitle(f"Exp 2: pairwise correlations of deltas (n={N}, red=circuit)",
             y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig("figs/exp2_delta_correlations.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figs/exp2_delta_correlations.png")

# ── Experiment 3: log(Var_v) normality ───────────────────────────────────────
print()
print("=" * 60)
print("EXPERIMENT 3 — log(Var_v) normality")
print("=" * 60)

EPS = 1e-8
log_varv_ioi = np.log(varv_ioi + EPS)
log_varv_non = np.log(varv_non + EPS)

print(f"log(Var_v) range (IOI):     [{log_varv_ioi.min():.2f}, {log_varv_ioi.max():.2f}]")
print(f"log(Var_v) range (non-IOI): [{log_varv_non.min():.2f}, {log_varv_non.max():.2f}]")

# Shapiro-Wilk per head on pooled data
sw_w  = np.zeros((12, 12))
sw_p  = np.zeros((12, 12))
for l in range(12):
    for h in range(12):
        vals = np.concatenate([log_varv_ioi[:, l, h], log_varv_non[:, l, h]])
        w, p_sw = stats.shapiro(vals)
        sw_w[l, h] = w
        sw_p[l, h] = p_sw

frac_not_rejected = (sw_p > 0.05).mean()
print(f"Shapiro-Wilk (pooled IOI+non-IOI, n={2*N} per head):")
print(f"  Mean W: {sw_w.mean():.3f}  (1.0 = perfectly normal)")
print(f"  Fraction with p > 0.05 (Gaussian not rejected): {frac_not_rejected:.2%}")
print(f"  Min W: {sw_w.min():.3f} at L{np.unravel_index(sw_w.argmin(), (12,12))[0]}H{np.unravel_index(sw_w.argmin(), (12,12))[1]}")

# Histograms for key circuit heads
fig, axes = plt.subplots(3, 4, figsize=(16, 11))
axes_flat = axes.flatten()

SHOW = [(9,9),(9,6),(10,0),(9,0),(7,3),(7,9),(8,6),(5,5),(0,1),(2,2),(0,0),(6,6)]
for idx, (l, h) in enumerate(SHOW):
    ax = axes_flat[idx]
    vi = log_varv_ioi[:, l, h]
    vn = log_varv_non[:, l, h]
    color = "#d62728" if (l, h) in ALL_CIRCUIT_HEADS else "#aaaaaa"
    ax.hist(vn, bins=20, alpha=0.4, color="#cccccc", density=True, label="non-IOI")
    ax.hist(vi, bins=20, alpha=0.6, color=color,    density=True, label="IOI")
    all_v = np.concatenate([vi, vn])
    xs = np.linspace(all_v.min(), all_v.max(), 200)
    ax.plot(xs, stats.norm.pdf(xs, all_v.mean(), all_v.std()), "k--", lw=1.2)
    role = head_role(l, h)
    short = role.replace("Backup Name Mover","BkNM").replace("Name Mover","NM").replace("Negative Name Mover","NegNM").replace("S-Inhibition","S-Inhib").replace("Duplicate Token","DupTok").replace("Previous Token","PrevTok").replace("Non-circuit","NC")
    ax.set_title(f"L{l}H{h} ({short})  W={sw_w[l,h]:.3f}", fontsize=10)
    ax.set_xlabel("log(Var_v)", fontsize=9)
    if idx == 0:
        ax.legend(fontsize=8)

plt.suptitle(f"Exp 3: log(Var_v) histograms (n={N} each) + Gaussian fit (dashed)\nW = Shapiro-Wilk statistic",
             y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig("figs/exp3_log_varv_histograms.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figs/exp3_log_varv_histograms.png")

# ── Summary ──────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"n = {N} prompts per class")
print()
print("Exp 1 — discrimination (circuit > non-circuit, MW one-sided):")
for key, res in results_exp1.items():
    print(f"  |Δ{key}|  circ={res['circ_mean']:.4f}  nonc={res['nonc_mean']:.4f}  ratio={res['circ_mean']/res['nonc_mean']:.2f}x  p={res['p']:.2e}")
print()
print("Exp 2 — correlation of deltas (all 144 heads):")
print(f"  corr(ΔKL, Δχ):    {r_kl_chi:.3f}   (Post 2 reference ≈ 0.70)")
print(f"  corr(ΔKL, ΔVar_v): {r_kl_varv:.3f}")
print(f"  corr(Δχ,  ΔVar_v): {r_chi_varv:.3f}")
print()
print("Exp 3 — log(Var_v) normality:")
print(f"  Mean Shapiro-Wilk W: {sw_w.mean():.3f}")
print(f"  Fraction not rejected at 0.05: {frac_not_rejected:.2%}")
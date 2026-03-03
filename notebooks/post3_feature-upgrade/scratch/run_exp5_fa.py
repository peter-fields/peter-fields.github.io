"""
Post 3 — Experiment 5: Factor Analysis on log(Var_v) + log(Var_act)
  Feature vector: 156-d  [144 attention heads + 12 MLP layers], last token
  1. Elbow plot (log-likelihood vs k)
  2. Fit FactorAnalysis at chosen k on IOI and non-IOI
  3. J_act, J_null, J_diff = W_act W_act^T - W_null W_null^T
  4. J_diff matrix heatmap with circuit markers
  5. Thresholded circuit graph: edge fraction analysis
  6. Per-factor loading profiles

Run with: /opt/miniconda3/bin/python run_exp5_fa.py
"""

import os, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtrans
from scipy import stats
from sklearn.decomposition import FactorAnalysis
import torch

os.makedirs("figs", exist_ok=True)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from transformer_lens import HookedTransformer
torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained("gpt2-small")
print(f"Loaded: {model.cfg.n_layers}L x {model.cfg.n_heads}H  d_head={model.cfg.d_head}  d_mlp={model.cfg.d_mlp}")

# ── circuit labels ────────────────────────────────────────────────────────────
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
    "Backup Name Mover":   "#ff7f0e",
    "Negative Name Mover": "#9467bd",
    "S-Inhibition":        "#2ca02c",
    "Induction":           "#17becf",
    "Duplicate Token":     "#bcbd22",
    "Previous Token":      "#e377c2",
    "Non-circuit":         "#cccccc",
    "MLP":                 "#4dac26",
}
ALL_CIRCUIT_HEADS = set(lh for hs in IOI_HEADS.values() for lh in hs)
HL_FLAT = [(l, h) for l in range(12) for h in range(12)]

def head_role(l, h):
    for role, heads in IOI_HEADS.items():
        if (l, h) in heads: return role
    return "Non-circuit"

SHORT = {"Name Mover":"NM","Backup Name Mover":"BNM","Negative Name Mover":"NegNM",
         "S-Inhibition":"SI","Induction":"Ind","Duplicate Token":"DT","Previous Token":"PT",
         "Non-circuit":"NC","MLP":"MLP"}

# ── prompts ───────────────────────────────────────────────────────────────────
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

N = 1000
ioi_prompts = gen_ioi(N); non_ioi_prompts = gen_non_ioi(N)
print(f"n={N} per class")

# ── compute feature matrices ──────────────────────────────────────────────────
def compute_features(model, prompts):
    """
    Returns X: (n_prompts, 156)
      cols 0..143  : log(Var_v) per attention head (12x12), last token
      cols 144..155: log(Var_act) per MLP layer, last token
    """
    nL, nH, dH, EPS = model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_head, 1e-8
    n = len(prompts)
    varv    = np.zeros((n, nL, nH))
    var_act = np.zeros((n, nL))

    for p_idx, prompt in enumerate(prompts):
        if p_idx % 100 == 0: print(f"  {p_idx}/{n}", flush=True)
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens)

        for layer in range(nL):
            # attention Var_v
            pi   = cache["pattern", layer][0, :, -1, :]    # (nH, seq)
            v    = cache["v", layer][0].permute(1, 0, 2)   # (nH, seq, dH)
            mu_v = (pi.unsqueeze(-1) * v).sum(dim=1)        # (nH, dH)
            E_v2 = (pi * (v**2).sum(-1)).sum(-1)            # (nH,)
            varv[p_idx, layer] = ((E_v2 - (mu_v**2).sum(-1)) / dH).cpu().numpy()

            # MLP Var(post-act) at last token
            post = cache["post", layer][0, -1, :]           # (d_mlp,)
            var_act[p_idx, layer] = post.var().cpu().item()

    log_varv    = np.log(varv    + EPS)   # (n, 12, 12)
    log_var_act = np.log(var_act + EPS)   # (n, 12)

    X = np.concatenate([log_varv.reshape(n, -1), log_var_act], axis=1)  # (n, 156)
    return X

if os.path.exists("X_ioi.npy") and os.path.exists("X_non.npy"):
    print("Loading cached feature arrays...")
    X_ioi = np.load("X_ioi.npy")
    X_non = np.load("X_non.npy")
else:
    print("\nComputing IOI features...")
    X_ioi = compute_features(model, ioi_prompts)
    print("Computing non-IOI features...")
    X_non = compute_features(model, non_ioi_prompts)
print(f"Done. X shape: {X_ioi.shape}")
np.save("X_ioi.npy", X_ioi)
np.save("X_non.npy", X_non)
print("Saved X_ioi.npy, X_non.npy")

# ── feature names and metadata ────────────────────────────────────────────────
FEAT_NAMES   = [f"L{l}H{h}" for l,h in HL_FLAT] + [f"MLP_L{l}" for l in range(12)]
FEAT_CIRCUIT = [lh in ALL_CIRCUIT_HEADS for lh in HL_FLAT] + [False]*12
FEAT_ROLE    = [head_role(*lh) for lh in HL_FLAT] + ["MLP"]*12
FEAT_COLOR   = [ROLE_COLORS[r] for r in FEAT_ROLE]
N_FEAT = len(FEAT_NAMES)   # 156

# ── z-score within each class, per feature ────────────────────────────────────
# Center and scale each feature independently within each condition.
# This puts attention heads and MLP features on the same footing
# despite their very different raw variances.
mu_ioi  = X_ioi.mean(0);  sd_ioi  = X_ioi.std(0) + 1e-10
mu_non  = X_non.mean(0);  sd_non  = X_non.std(0) + 1e-10

Xz_ioi = (X_ioi - mu_ioi) / sd_ioi
Xz_non = (X_non - mu_non) / sd_non

print(f"\nZ-scored feature ranges:  IOI [{Xz_ioi.min():.2f}, {Xz_ioi.max():.2f}]  "
      f"non-IOI [{Xz_non.min():.2f}, {Xz_non.max():.2f}]")

# ── 1. Elbow plot ─────────────────────────────────────────────────────────────
print("\n--- Elbow plot ---")
K_RANGE = list(range(1, 21))
ll_ioi, ll_non = [], []
for k in K_RANGE:
    fa_i = FactorAnalysis(n_components=k, random_state=42, max_iter=2000)
    fa_n = FactorAnalysis(n_components=k, random_state=42, max_iter=2000)
    fa_i.fit(Xz_ioi); fa_n.fit(Xz_non)
    ll_ioi.append(fa_i.score(Xz_ioi))
    ll_non.append(fa_n.score(Xz_non))
    print(f"  k={k:2d}  IOI={ll_ioi[-1]:.3f}  non-IOI={ll_non[-1]:.3f}")

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(K_RANGE, ll_ioi, "o-", label="IOI (activating)")
ax.plot(K_RANGE, ll_non, "s-", label="non-IOI (null)")
ax.set_xlabel("n_components (k)", fontsize=12)
ax.set_ylabel("Mean log-likelihood", fontsize=12)
ax.set_title("Factor Analysis elbow — z-scored log(Var_v) + log(Var_act)", fontsize=12)
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figs/fa_elbow.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figs/fa_elbow.png")

# ── 2. Fit at chosen k ────────────────────────────────────────────────────────
# Visual elbow: pick where the gain starts flattening.
# We'll try k=5 as the default; update K_SELECTED after inspecting the elbow plot.
# Gains from elbow:
gains = [ll_ioi[i] - ll_ioi[i-1] for i in range(1, len(ll_ioi))]
# Find first k where gain drops below 10% of initial gain
init_gain = gains[0]
K_AUTO = next((K_RANGE[i+1] for i, g in enumerate(gains) if g < 0.10 * init_gain), 5)
K_SELECTED = K_AUTO
print(f"\nAuto-selected k={K_SELECTED} (first k where marginal gain < 10% of initial)")

fa_ioi = FactorAnalysis(n_components=K_SELECTED, random_state=42, max_iter=2000)
fa_non = FactorAnalysis(n_components=K_SELECTED, random_state=42, max_iter=2000)
fa_ioi.fit(Xz_ioi); fa_non.fit(Xz_non)

# W: (N_FEAT, k) — loading matrix
W_ioi = fa_ioi.components_.T   # (156, k)
W_non = fa_non.components_.T

print(f"W_ioi shape: {W_ioi.shape}")
print(f"log-lik IOI:     {fa_ioi.score(Xz_ioi):.3f}")
print(f"log-lik non-IOI: {fa_non.score(Xz_non):.3f}")

# ── 3. J matrices ─────────────────────────────────────────────────────────────
J_ioi  = W_ioi @ W_ioi.T    # (156, 156)
J_non  = W_non @ W_non.T
J_diff = J_ioi - J_non

print(f"\nJ_diff range: [{J_diff.min():.4f}, {J_diff.max():.4f}]")
print(f"J_ioi  range: [{J_ioi.min():.4f},  {J_ioi.max():.4f}]")

# ── 4. J_diff heatmap ─────────────────────────────────────────────────────────
circuit_idx = [i for i, c in enumerate(FEAT_CIRCUIT) if c]

def add_circuit_ticks(ax, idx_list, color="black", lw=1.5):
    for idx in idx_list:
        t = mtrans.blended_transform_factory(ax.transAxes, ax.transData)
        ax.add_artist(plt.Line2D([-0.02, 0], [idx, idx], transform=t,
                                 color=color, lw=lw, clip_on=False))
        t = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
        ax.add_artist(plt.Line2D([idx, idx], [-0.02, 0], transform=t,
                                 color=color, lw=lw, clip_on=False))

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
vmax = max(abs(J_ioi).max(), abs(J_non).max(), abs(J_diff).max())

for ax, mat, title in [
    (axes[0], J_ioi,  f"J_ioi  (k={K_SELECTED})"),
    (axes[1], J_non,  f"J_null (k={K_SELECTED})"),
    (axes[2], J_diff, "J_diff = J_ioi − J_null"),
]:
    im = ax.imshow(mat, cmap="PRGn_r", vmin=-vmax, vmax=vmax, aspect="auto")
    add_circuit_ticks(ax, circuit_idx)

    # Layer boundary lines (every 12 heads)
    for i in range(1, 12):
        ax.axvline(i*12 - 0.5, color="white", lw=0.3, alpha=0.5)
        ax.axhline(i*12 - 0.5, color="white", lw=0.3, alpha=0.5)
    # MLP separator
    ax.axvline(143.5, color="white", lw=1.2, alpha=0.9)
    ax.axhline(143.5, color="white", lw=1.2, alpha=0.9)

    ax.set_xticks([l*12+6 for l in range(12)] + [150])
    ax.set_xticklabels([f"L{l}" for l in range(12)] + ["MLP"], fontsize=9)
    ax.set_yticks([l*12+6 for l in range(12)] + [150])
    ax.set_yticklabels([f"L{l}" for l in range(12)] + ["MLP"], fontsize=9)
    ax.set_title(title, fontsize=12)

cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.suptitle(f"Factor analysis coupling matrices — z-scored log(Var), k={K_SELECTED}\n"
             "Black margin ticks = IOI circuit heads | dashed line = MLP block",
             fontsize=11, y=1.01)
plt.tight_layout(rect=[0, 0, 0.91, 1])
plt.savefig("figs/fa_J_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figs/fa_J_matrices.png")

# ── 5. Circuit graph: threshold J_diff ────────────────────────────────────────
print("\n--- Circuit graph analysis ---")

mask_off = ~np.eye(N_FEAT, dtype=bool)
Jd_off   = np.abs(J_diff[mask_off])
mu_off, sd_off = Jd_off.mean(), Jd_off.std()

for n_sigma in [2, 3]:
    thresh  = mu_off + n_sigma * sd_off
    rows, cols = np.where((np.abs(J_diff) > thresh) & mask_off)
    edges   = [(J_diff[r,c], FEAT_NAMES[r], FEAT_NAMES[c],
                FEAT_CIRCUIT[r], FEAT_CIRCUIT[c]) for r,c in zip(rows,cols) if r < c]
    edges.sort(key=lambda e: abs(e[0]), reverse=True)

    cc  = sum(1 for e in edges if e[3] and e[4])
    cnc = sum(1 for e in edges if e[3] != e[4])
    nn  = sum(1 for e in edges if not e[3] and not e[4])
    tot = len(edges)

    p_cc  = (len(ALL_CIRCUIT_HEADS)/N_FEAT)**2
    p_cnc = 2*(len(ALL_CIRCUIT_HEADS)/N_FEAT)*(1 - len(ALL_CIRCUIT_HEADS)/N_FEAT)

    print(f"\n{n_sigma}σ threshold ({thresh:.4f}) — {tot} edges:")
    print(f"  CC  = {cc}/{tot} = {cc/max(tot,1):.1%}  (chance: {p_cc:.1%})")
    print(f"  CNC = {cnc}/{tot} = {cnc/max(tot,1):.1%}  (chance: {p_cnc:.1%})")
    print(f"  NN  = {nn}/{tot} = {nn/max(tot,1):.1%}")
    if tot > 0:
        print(f"  Top edges:")
        for val, n1, n2, c1, c2 in edges[:10]:
            tag = "CC" if c1 and c2 else ("CNC" if c1 or c2 else "NN")
            r1  = FEAT_ROLE[FEAT_NAMES.index(n1)]
            r2  = FEAT_ROLE[FEAT_NAMES.index(n2)]
            print(f"    {n1} ({SHORT[r1]}) ↔ {n2} ({SHORT[r2]})  J={val:+.4f}  [{tag}]")

# ── 6. Loading profiles ───────────────────────────────────────────────────────
print("\n--- Loading profiles ---")

fig, axes = plt.subplots(K_SELECTED, 1, figsize=(20, 2.8*K_SELECTED), sharex=True)
if K_SELECTED == 1: axes = [axes]

xs = np.arange(N_FEAT)
for k, ax in enumerate(axes):
    loadings = W_ioi[:, k]
    ax.bar(xs, loadings, color=FEAT_COLOR, alpha=0.85, width=0.9)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel(f"Factor {k+1}", fontsize=11)

    # Mark circuit heads with tick lines
    for i in circuit_idx:
        ax.axvline(i, color="k", lw=0.4, alpha=0.25, zorder=0)

    # MLP separator
    ax.axvline(143.5, color="k", lw=1.2, linestyle="--", alpha=0.5)

    # Annotate top loaders
    top_idx = np.argsort(np.abs(loadings))[-5:]
    for i in top_idx:
        role = FEAT_ROLE[i]
        ax.text(i, loadings[i] + (0.02 if loadings[i]>=0 else -0.04),
                SHORT[role], ha="center", fontsize=6,
                color=ROLE_COLORS[role], fontweight="bold")

axes[-1].set_xticks([l*12+6 for l in range(12)] + [150])
axes[-1].set_xticklabels([f"L{l}" for l in range(12)] + ["MLP"], fontsize=9)

legend_handles = [mpatches.Patch(color=c, label=r) for r, c in ROLE_COLORS.items()]
axes[0].legend(handles=legend_handles, fontsize=8, loc="upper right",
               ncol=5, framealpha=0.9)

plt.suptitle(f"W_ioi loading profiles — k={K_SELECTED} factors  (z-scored log(Var))\n"
             "Color = circuit role | Dashed line = MLP block",
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("figs/fa_loading_profiles.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figs/fa_loading_profiles.png")

# ── 7. J_diff: zoom on attention-only (144x144) and annotate roles ─────────────
print("\n--- J_diff attention zoom ---")

Jd_attn = J_diff[:144, :144]
vmax_a  = np.abs(Jd_attn).max()
circ_idx_attn = [i for i, c in enumerate(FEAT_CIRCUIT[:144]) if c]

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(Jd_attn, cmap="PRGn_r", vmin=-vmax_a, vmax=vmax_a, aspect="auto")
add_circuit_ticks(ax, circ_idx_attn, color="black")

for i in range(1, 12):
    ax.axvline(i*12 - 0.5, color="white", lw=0.4, alpha=0.6)
    ax.axhline(i*12 - 0.5, color="white", lw=0.4, alpha=0.6)

ax.set_xticks([l*12+6 for l in range(12)])
ax.set_xticklabels([f"L{l}" for l in range(12)], fontsize=10)
ax.set_yticks([l*12+6 for l in range(12)])
ax.set_yticklabels([f"L{l}" for l in range(12)], fontsize=10)
fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
ax.set_title(f"J_diff — attention heads only (144×144), k={K_SELECTED}\n"
             "Black margin ticks = IOI circuit heads", fontsize=12)
plt.tight_layout()
plt.savefig("figs/fa_Jdiff_attn_zoom.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figs/fa_Jdiff_attn_zoom.png")

print("\nAll done.")
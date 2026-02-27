import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
import random, os

os.chdir('/Users/pfields/Git/peter-fields.github.io/notebooks/post2_attention-diagnostics/scratch')

torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained("gpt2-small")

IOI_HEADS = {
    "Name Mover":          [(9,6),(9,9),(10,0)],
    "Backup Name Mover":   [(9,0),(9,7),(10,1),(10,2),(10,6),(10,10),(11,2),(11,9)],
    "Negative Name Mover": [(10,7),(11,10)],
    "S-Inhibition":        [(7,3),(7,9),(8,6),(8,10)],
    "Induction":           [(5,5),(6,9)],
    "Duplicate Token":     [(0,1),(3,0)],
    "Previous Token":      [(2,2),(4,11)],
}
ROLE_COLORS = {
    "Name Mover":          "#d62728",
    "Backup Name Mover":   "#ff9896",
    "Negative Name Mover": "#9467bd",
    "S-Inhibition":        "#2ca02c",
    "Induction":           "#1f77b4",
    "Duplicate Token":     "#8c564b",
    "Previous Token":      "#e377c2",
}
ALL = set(h for hs in IOI_HEADS.values() for h in hs)

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
NAMES = ["Mary","John","Alice","Bob","Sarah","Tom","Emma","James","Lisa","David","Kate","Mark"]

def gen_ioi(n, seed=42):
    random.seed(seed); t=TEMPLATES_ABBA+TEMPLATES_BABA; seen,out=set(),[]
    while len(out)<n:
        tmpl=random.choice(t); a,b=random.sample(NAMES,2); p=tmpl.format(A=a,B=b)
        if p not in seen: seen.add(p); out.append(p)
    return out

def gen_non(n, seed=43):
    random.seed(seed); seen,out=set(),[]
    while len(out)<n:
        tmpl=random.choice(TEMPLATES_NON_IOI); a,b,c=random.sample(NAMES,3); p=tmpl.format(A=a,B=b,C=c)
        if p not in seen: seen.add(p); out.append(p)
    return out

def compute_kl(prompts):
    kl=np.zeros((len(prompts),12,12))
    for i,p in enumerate(prompts):
        tok=model.to_tokens(p); _,cache=model.run_with_cache(tok); log_n=np.log(tok.shape[1])
        for l in range(12):
            pi=cache["pattern",l][0,:,-1,:]; ent=-(pi*torch.log(pi+1e-12)).sum(-1)
            kl[i,l,:]=(log_n-ent.cpu().numpy())/log_n
    return kl

print("Computing KL...")
kl_ioi = compute_kl(gen_ioi(50))
kl_non = compute_kl(gen_non(50))

hl = [(l,h) for l in range(12) for h in range(12)]
cmask = np.array([(l,h) in ALL for l,h in hl])
circuit_idx = [i for i,(l,h) in enumerate(hl) if (l,h) in ALL]

C_ioi  = np.corrcoef(kl_ioi.reshape(50,-1).T)
C_non  = np.corrcoef(kl_non.reshape(50,-1).T)
C_diff = C_ioi - C_non

# ── Fig 1: three heatmaps ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

def draw_heatmap(ax, C, title, vmax=None, cmap='RdBu_r', symmetric=True):
    vm = vmax or (np.abs(C).max() * 0.8)
    vmin = -vm if symmetric else 0
    im = ax.imshow(C, cmap=cmap, vmin=vmin, vmax=vm, aspect='auto', interpolation='nearest')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Layer boundary lines
    for l in range(1, 12):
        ax.axhline(l*12 - 0.5, color='white', lw=0.4, alpha=0.5)
        ax.axvline(l*12 - 0.5, color='white', lw=0.4, alpha=0.5)

    # Mark circuit heads with colored ticks on both axes
    for i in circuit_idx:
        l, h = hl[i]
        role = next(r for r,hs in IOI_HEADS.items() if (l,h) in hs)
        c = ROLE_COLORS[role]
        ax.plot([-3, -1], [i, i], color=c, lw=2, clip_on=False)
        ax.plot([i, i], [144, 146], color=c, lw=2, clip_on=False)

    # Layer labels on x axis
    ax.set_xticks([l*12 + 5.5 for l in range(12)])
    ax.set_xticklabels([f"L{l}" for l in range(12)], fontsize=7)
    ax.set_yticks([l*12 + 5.5 for l in range(12)])
    ax.set_yticklabels([f"L{l}" for l in range(12)], fontsize=7)
    ax.set_title(title, fontsize=11)

draw_heatmap(axes[0], C_ioi,  "C_IOI",    vmax=1.0, symmetric=False, cmap='viridis')
draw_heatmap(axes[1], C_non,  "C_nonIOI", vmax=1.0, symmetric=False, cmap='viridis')
draw_heatmap(axes[2], C_diff, "C_diff = C_IOI − C_nonIOI", vmax=1.5)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=ROLE_COLORS[r], label=r) for r in ROLE_COLORS]
fig.legend(handles=legend_elements, loc='lower center', fontsize=7,
           ncol=7, bbox_to_anchor=(0.5, -0.06))

plt.suptitle("Cross-head KL correlation matrices  (144 heads × 144 heads, n=50 prompts each)\n"
             "Colored ticks = known IOI circuit heads", fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig("fig_C_heatmaps.png", dpi=200, bbox_inches="tight")
print("Saved fig_C_heatmaps.png")

# ── Fig 2: top-50 C_diff pairs and circuit coverage ──────────────────────────
def role(l,h):
    for r,hs in IOI_HEADS.items():
        if (l,h) in hs: return r
    return "Non-circuit"

pairs = sorted(
    [(abs(C_diff[i,j]), C_diff[i,j], i, j) for i in range(144) for j in range(i+1,144)],
    reverse=True
)

top50 = pairs[:50]
heads_seen = set()
for _,_,i,j in top50:
    heads_seen.add(hl[i]); heads_seen.add(hl[j])

circuit_seen    = heads_seen & ALL
circuit_missing = ALL - heads_seen

print(f"\nTop-50 |C_diff| pairs cover {len(circuit_seen)}/{len(ALL)} circuit heads")
print(f"Seen:    {sorted(circuit_seen)}")
print(f"Missing: {sorted(circuit_missing)}")

# Bar chart: top 50 pairs colored by cat
fig2, ax = plt.subplots(figsize=(13, 4))
COLORS = {"CC":"#d62728","CN":"#ff7f0e","NN":"#bbbbbb"}

xs, ys, cs = [], [], []
for k,(av,v,i,j) in enumerate(top50):
    ic=cmask[i]; jc=cmask[j]
    ct="CC" if ic and jc else ("CN" if ic or jc else "NN")
    xs.append(k); ys.append(v); cs.append(COLORS[ct])

bars = ax.bar(xs, ys, color=cs, width=0.8, linewidth=0)
ax.axhline(0, color='black', lw=0.8)

# Annotate circuit-circuit bars
for k,(av,v,i,j) in enumerate(top50):
    ic=cmask[i]; jc=cmask[j]
    if ic and jc:
        li,hi=hl[i]; lj,hj=hl[j]
        ax.text(k, av+0.02, f"L{li}H{hi}\nL{lj}H{hj}",
                ha='center', va='bottom', fontsize=5.5, color='#d62728', rotation=90)

ax.set_xlabel("Rank (by |C_diff,ij|)", fontsize=11)
ax.set_ylabel("C_diff,ij  (signed)", fontsize=11)
ax.set_title(f"Top 50 pairs by |C_diff|  —  covers {len(circuit_seen)}/{len(ALL)} known circuit heads\n"
             f"Missing: {', '.join(f'L{l}H{h}' for l,h in sorted(circuit_missing)) or 'none'}",
             fontsize=10)
ax.set_xlim(-1, 50)

from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=COLORS[c], label=c) for c in ["CC","CN","NN"]],
          fontsize=9, loc='upper right')
ax.grid(True, alpha=0.2, axis='y')
plt.tight_layout()
plt.savefig("fig_cdiff_top50.png", dpi=300, bbox_inches="tight")
print("Saved fig_cdiff_top50.png")

# Also print the top-50 table
print(f"\n{'Rk':<4} {'Head i':<8} {'Role i':<22} {'Head j':<8} {'Role j':<22} {'C_diff':>7}  cat")
print("-"*80)
for k,(av,v,i,j) in enumerate(top50,1):
    li,hi=hl[i]; lj,hj=hl[j]
    ic=cmask[i]; jc=cmask[j]
    ct="CC" if ic and jc else ("CN" if ic or jc else "NN")
    print(f"{k:<4} L{li}H{hi:<5} {role(li,hi):<22} L{lj}H{hj:<5} {role(lj,hj):<22} {v:+.3f}  {ct}")
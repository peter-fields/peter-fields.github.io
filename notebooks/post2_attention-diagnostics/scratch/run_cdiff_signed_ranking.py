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
    "Name Mover":          [(9,6),(9,9),(10,0)],
    "Backup Name Mover":   [(9,0),(9,7),(10,1),(10,2),(10,6),(10,10),(11,2),(11,9)],
    "Negative Name Mover": [(10,7),(11,10)],
    "S-Inhibition":        [(7,3),(7,9),(8,6),(8,10)],
    "Induction":           [(5,5),(6,9)],
    "Duplicate Token":     [(0,1),(3,0)],
    "Previous Token":      [(2,2),(4,11)],
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

print("Computing...")
kl_ioi = compute_kl(gen_ioi(50))
kl_non = compute_kl(gen_non(50))

hl = [(l,h) for l in range(12) for h in range(12)]
cmask = np.array([(l,h) in ALL for l,h in hl])

def role(l,h):
    for r,hs in IOI_HEADS.items():
        if (l,h) in hs: return r
    return "Non-circuit"

C_diff = np.corrcoef(kl_ioi.reshape(50,-1).T) - np.corrcoef(kl_non.reshape(50,-1).T)

# Build sorted pair list (signed, descending)
pairs = []
for i in range(144):
    for j in range(i+1, 144):
        c = C_diff[i,j]
        ic=cmask[i]; jc=cmask[j]
        ct = "CC" if ic and jc else ("CN" if ic or jc else "NN")
        pairs.append((c, i, j, ct))

pairs.sort(reverse=True)  # descending by signed value

cats   = [p[3] for p in pairs]
vals   = [p[0] for p in pairs]
n      = len(pairs)

# --- Figure ---
COLORS  = {"CC": "#d62728", "CN": "#ff7f0e", "NN": "#bbbbbb"}
ALPHAS  = {"CC": 1.0,       "CN": 0.7,       "NN": 0.25}
WIDTHS  = {"CC": 1.0,       "CN": 0.6,       "NN": 0.4}
ZORDERS = {"CC": 3,         "CN": 2,         "NN": 1}

fig, axes = plt.subplots(1, 2, figsize=(15, 5),
                         gridspec_kw={"width_ratios": [3, 1]})

# Left: full signed ranking
ax = axes[0]
for cat in ["NN", "CN", "CC"]:
    idx = [i for i,c in enumerate(cats) if c==cat]
    ax.vlines(idx, 0, [vals[i] for i in idx],
              color=COLORS[cat], alpha=ALPHAS[cat],
              linewidth=WIDTHS[cat], zorder=ZORDERS[cat])

ax.axhline(0, color="black", lw=0.8, zorder=4)
ax.set_xlabel("Rank (sorted by C_diff,ij)", fontsize=11)
ax.set_ylabel("C_diff,ij", fontsize=11)
ax.set_title("C_diff = C_IOI − C_nonIOI, all 10,296 pairs ranked by signed value\n"
             "Positive: task-specific co-activation   |   Negative: task-specific anti-correlation",
             fontsize=10)
ax.set_xlim(-100, n+100)
ax.grid(True, alpha=0.15, axis='y')

# Annotate top positive CC pairs
top_cc_pos = [(i,p) for i,p in enumerate(pairs) if p[3]=="CC"][:3]
for rank,(i,p) in enumerate(top_cc_pos):
    c,pi,pj,ct = p
    li,hi=hl[pi]; lj,hj=hl[pj]
    ax.annotate(f"L{li}H{hi}–L{lj}H{hj}",
                xy=(i, c), xytext=(i+150, c+0.05),
                fontsize=7, color="#d62728",
                arrowprops=dict(arrowstyle="-", color="#d62728", lw=0.6))

# Annotate top negative CN pairs (L8H1, L8H11)
neg_cn = [(i,p) for i,p in enumerate(pairs) if p[3]=="CN" and vals[i]<0][:2]
for rank,(i,p) in enumerate(neg_cn):
    c,pi,pj,ct = p
    li,hi=hl[pi]; lj,hj=hl[pj]
    label = f"L{li}H{hi}–L{lj}H{hj}"
    ax.annotate(label,
                xy=(i, c), xytext=(i-400, c-0.08),
                fontsize=7, color="#555555",
                arrowprops=dict(arrowstyle="-", color="#555555", lw=0.6))

# Right: zoom on top/bottom 100
ax2 = axes[1]
zoom_n = 100
top_idx = list(range(zoom_n))
bot_idx = list(range(n-zoom_n, n))
all_idx = top_idx + bot_idx

# Remap to a clean x axis with a gap
x_top = list(range(zoom_n))
x_bot = list(range(zoom_n+15, zoom_n+15+zoom_n))

for idx_list, x_list in [(top_idx, x_top), (bot_idx, x_bot)]:
    for cat in ["NN", "CN", "CC"]:
        sel = [(x, vals[i]) for x,i in zip(x_list, idx_list) if cats[i]==cat]
        if sel:
            xs, ys = zip(*sel)
            ax2.vlines(xs, 0, ys,
                       color=COLORS[cat], alpha=ALPHAS[cat],
                       linewidth=max(WIDTHS[cat], 0.8), zorder=ZORDERS[cat])

ax2.axhline(0, color="black", lw=0.8)
ax2.axvspan(zoom_n+2, zoom_n+13, color="white", zorder=5)  # gap
ax2.text(zoom_n+7.5, 0, "···", ha="center", va="center", fontsize=12, color="gray", zorder=6)
ax2.set_xlabel("Rank (top/bottom 100)", fontsize=10)
ax2.set_ylabel("C_diff,ij", fontsize=10)
ax2.set_title("Top & bottom 100", fontsize=10)
ax2.set_xlim(-3, zoom_n+15+zoom_n+3)
ax2.grid(True, alpha=0.2, axis='y')

# Top-k stats printed on figure
stats_lines = []
for k in [10, 20, 50]:
    top = pairs[:k]; bot = pairs[n-k:]
    cc_t = sum(1 for p in top if p[3]=="CC")
    cc_b = sum(1 for p in bot if p[3]=="CC")
    stats_lines.append(f"top-{k:>2}: {cc_t} CC  |  bot-{k:>2}: {cc_b} CC")
stats_txt = "\n".join(stats_lines)
ax2.text(0.02, 0.02, stats_txt, transform=ax2.transAxes,
         fontsize=8, va='bottom', family='monospace',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

legend_elements = [
    Line2D([0],[0], color=COLORS["CC"], lw=2, label="circuit–circuit (CC)"),
    Line2D([0],[0], color=COLORS["CN"], lw=2, alpha=0.8, label="circuit–non-circuit (CN)"),
    Line2D([0],[0], color=COLORS["NN"], lw=1, alpha=0.5, label="non–non (NN)"),
]
ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

plt.suptitle("Signed C_diff ranking — task-specific pairwise KL correlation\n"
             "n=50 IOI / 50 non-IOI prompts, GPT-2 small  |  interpret with caution: indirect couplings not removed",
             fontsize=10, y=1.01)
plt.tight_layout()
plt.savefig("fig_cdiff_signed_ranking.png", dpi=300, bbox_inches="tight")
print("Saved fig_cdiff_signed_ranking.png")

# Print top/bottom table
print(f"\n{'Rk':<4} {'Head i':<8} {'Role i':<22} {'Head j':<8} {'Role j':<22} {'C_diff':>7}  cat")
print("-"*80)
for k,(c,i,j,ct) in enumerate(pairs[:15], 1):
    li,hi=hl[i]; lj,hj=hl[j]
    print(f"{k:<4} L{li}H{hi:<5} {role(li,hi):<22} L{lj}H{hj:<5} {role(lj,hj):<22} {c:+.3f}  {ct}")
print("  ...")
print(f"\n{'Rk':<4} {'Head i':<8} {'Role i':<22} {'Head j':<8} {'Role j':<22} {'C_diff':>7}  cat")
print("-"*80)
for k,(c,i,j,ct) in enumerate(pairs[-15:], n-14):
    li,hi=hl[i]; lj,hj=hl[j]
    print(f"{k:<4} L{li}H{hi:<5} {role(li,hi):<22} L{lj}H{hj:<5} {role(lj,hj):<22} {c:+.3f}  {ct}")

print(f"\nTop/bottom k breakdown (CC count):")
for k in [10,20,50,100]:
    top=pairs[:k]; bot=pairs[n-k:]
    print(f"  k={k:>3}:  top {sum(1 for p in top if p[3]=='CC'):>2} CC  |  bottom {sum(1 for p in bot if p[3]=='CC'):>2} CC")
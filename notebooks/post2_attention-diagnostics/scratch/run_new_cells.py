"""
Run cells 39, 41, 43 of attention_diagnostics_executed.ipynb and inject outputs.
"""
import io, base64, json, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)

NB_PATH = '/Users/pfields/Git/peter-fields.github.io/notebooks/post2_attention-diagnostics/scratch/attention_diagnostics_executed.ipynb'

# ── 1. Setup (mirrors earlier notebook cells) ──────────────────────────────
print("Loading model...")
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

def gen_ioi(n, seed=42):
    random.seed(seed); t = TEMPLATES_ABBA + TEMPLATES_BABA; seen, out = set(), []
    while len(out) < n:
        tmpl = random.choice(t); a, b = random.sample(NAMES, 2); p = tmpl.format(A=a, B=b)
        if p not in seen: seen.add(p); out.append(p)
    return out

def gen_non(n, seed=43):
    random.seed(seed); seen, out = set(), []
    while len(out) < n:
        tmpl = random.choice(TEMPLATES_NON_IOI); a, b, c = random.sample(NAMES, 3)
        p = tmpl.format(A=a, B=b, C=c)
        if p not in seen: seen.add(p); out.append(p)
    return out

def compute_kl(prompts):
    kl = np.zeros((len(prompts), 12, 12))
    for i, p in enumerate(prompts):
        tok = model.to_tokens(p)
        _, cache = model.run_with_cache(tok)
        log_n = np.log(tok.shape[1])
        for l in range(12):
            pi = cache["pattern", l][0, :, -1, :]
            ent = -(pi * torch.log(pi + 1e-12)).sum(-1)
            kl[i, l, :] = (log_n - ent.cpu().numpy()) / log_n
    return kl

print("Computing KL for IOI prompts...")
kl_ioi = compute_kl(gen_ioi(50))
print("Computing KL for non-IOI prompts...")
kl_non_ioi = compute_kl(gen_non(50))
print("Done.")

# ── 2. Helpers ─────────────────────────────────────────────────────────────
hl_flat = [(l, h) for l in range(12) for h in range(12)]
circuit_idx = [i for i, (l, h) in enumerate(hl_flat) if (l, h) in ALL_CIRCUIT_HEADS]
cmask_flat  = np.array([(l, h) in ALL_CIRCUIT_HEADS for l, h in hl_flat])

kl_ioi_mat = kl_ioi.reshape(50, -1)
kl_non_mat = kl_non_ioi.reshape(50, -1)
C_IOI     = np.corrcoef(kl_ioi_mat.T)
C_nonIOI  = np.corrcoef(kl_non_mat.T)
C_diff_lyr = C_IOI - C_nonIOI

def fig_to_output(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return {
        "output_type": "display_data",
        "metadata": {"needs_background": "light"},
        "data": {
            "image/png": data,
            "text/plain": ["<Figure>"]
        }
    }

def text_output(lines):
    return {"output_type": "stream", "name": "stdout", "text": lines}

def _role(l, h):
    for r, hs in IOI_HEADS.items():
        if (l, h) in hs: return r
    return "Non-circuit"

# ── 3. Cell 39: C heatmaps ─────────────────────────────────────────────────
print("\n--- Cell 39: C heatmaps ---")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
import matplotlib.transforms as mtrans
TICK_COLOR = '#6b3a2a'  # brown
specs = [
    (axes[0], C_IOI,      'C_IOI'),
    (axes[1], C_nonIOI,   'C_nonIOI'),
    (axes[2], C_diff_lyr, 'C_diff = C_IOI − C_nonIOI'),
]
for ax, mat, title in specs:
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-1.3, vmax=1.3,
                   aspect='auto', interpolation='nearest')
    for l in range(1, 12):
        ax.axhline(l*12 - 0.5, color='white', lw=0.4, alpha=0.6)
        ax.axvline(l*12 - 0.5, color='white', lw=0.4, alpha=0.6)
    for idx in circuit_idx:
        t = mtrans.blended_transform_factory(ax.transAxes, ax.transData)
        ax.add_artist(plt.Line2D([-0.03, 0], [idx, idx], transform=t,
                                 color=TICK_COLOR, lw=2.5, clip_on=False))
        t = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
        ax.add_artist(plt.Line2D([idx, idx], [-0.03, 0], transform=t,
                                 color=TICK_COLOR, lw=2.5, clip_on=False))
    ax.set_xticks([l*12 + 5.5 for l in range(12)])
    ax.set_xticklabels([f'L{l}' for l in range(12)], fontsize=10, color=TICK_COLOR)
    ax.set_yticks([l*12 + 5.5 for l in range(12)])
    ax.set_yticklabels([f'L{l}' for l in range(12)], fontsize=10, color=TICK_COLOR)
    ax.set_title(title, fontsize=12)

fig.colorbar(im, ax=axes.tolist(), fraction=0.015, pad=0.02)

plt.suptitle('Cross-head KL correlation matrices — 144×144, n=50 prompts each\n'
             'Brown margin ticks = known IOI circuit heads', fontsize=11, y=1.02)
plt.tight_layout()
out39_fig = fig_to_output(fig)

mask = ~np.eye(144, dtype=bool)
txt39 = [
    f"C_IOI   range: [{C_IOI[mask].min():.3f}, {C_IOI[mask].max():.3f}] (off-diag)\n",
    f"C_nonIOI range: [{C_nonIOI[mask].min():.3f}, {C_nonIOI[mask].max():.3f}] (off-diag)\n",
    f"C_diff  range: [{C_diff_lyr[mask].min():.3f}, {C_diff_lyr[mask].max():.3f}] (off-diag)\n",
]
print("".join(txt39))

# ── 4. Cell 41: Top-50 C_diff pairs ────────────────────────────────────────
print("--- Cell 41: top-50 ---")
pairs_ranked = sorted(
    [(abs(C_diff_lyr[i, j]), C_diff_lyr[i, j], i, j)
     for i in range(144) for j in range(i+1, 144)],
    reverse=True
)
top50 = pairs_ranked[:50]

heads_seen = set()
for _, _, i, j in top50:
    heads_seen.add(hl_flat[i]); heads_seen.add(hl_flat[j])
circuit_seen    = heads_seen & ALL_CIRCUIT_HEADS
circuit_missing = ALL_CIRCUIT_HEADS - heads_seen

COLORS_CAT = {'CC': '#d62728', 'CN': '#ff7f0e', 'NN': '#bbbbbb'}
fig2, ax = plt.subplots(figsize=(13, 4))
for k, (av, v, i, j) in enumerate(top50):
    ic = cmask_flat[i]; jc = cmask_flat[j]
    ct = 'CC' if ic and jc else ('CN' if ic or jc else 'NN')
    ax.bar(k, v, color=COLORS_CAT[ct], width=0.8, linewidth=0)

for k, (av, v, i, j) in enumerate(top50):
    ic = cmask_flat[i]; jc = cmask_flat[j]
    if ic and jc:
        li, hi = hl_flat[i]; lj, hj = hl_flat[j]
        ax.text(k, (v if v >= 0 else 0) + 0.02,
                f'L{li}H{hi}\nL{lj}H{hj}',
                ha='center', va='bottom', fontsize=5, color='#990000', rotation=90)

ax.axhline(0, color='black', lw=0.8)
ax.set_xlabel('Rank (by |C_diff,ij|)', fontsize=11)
ax.set_ylabel('C_diff,ij (signed)', fontsize=11)
missing_str = ', '.join(f'L{l}H{h}' for l, h in sorted(circuit_missing)) or 'none'
ax.set_title(
    f'Top 50 pairs by |C_diff|  —  covers {len(circuit_seen)}/{len(ALL_CIRCUIT_HEADS)} known circuit heads\n'
    f'Missing: {missing_str}',
    fontsize=10)
ax.set_xlim(-1, 50)
ax.legend(handles=[Patch(facecolor=COLORS_CAT[c], label=c) for c in ['CC', 'CN', 'NN']],
          fontsize=9, loc='upper right')
ax.grid(True, alpha=0.2, axis='y')
plt.tight_layout()
out41_fig = fig_to_output(fig2)

# Table text
tlines = [f'Top-50 |C_diff| pairs cover {len(circuit_seen)}/{len(ALL_CIRCUIT_HEADS)} circuit heads\n',
          f'Missing: {sorted(circuit_missing)}\n\n',
          f'{"Rk":<4} {"Head i":<8} {"Role i":<22} {"Head j":<8} {"Role j":<22} {"C_diff":>7}  cat\n',
          '-'*80 + '\n']
for k, (av, v, i, j) in enumerate(top50, 1):
    li, hi = hl_flat[i]; lj, hj = hl_flat[j]
    ic = cmask_flat[i]; jc = cmask_flat[j]
    ct = 'CC' if ic and jc else ('CN' if ic or jc else 'NN')
    tlines.append(f'{k:<4} L{li}H{hi:<5} {_role(li,hi):<22} L{lj}H{hj:<5} {_role(lj,hj):<22} {v:+.3f}  {ct}\n')
print("".join(tlines[:6]))  # print header

# ── 5. Cell 43: C_IOI vs C_nonIOI scatter — all NC-anything pairs ────────────
print("--- Cell 43: scatter ---")
non_circuit_indices = [i for i, (l, h) in enumerate(hl_flat) if (l, h) not in ALL_CIRCUIT_HEADS]

nc_nc_x, nc_nc_y = [], []
nc_c_by_role = {}   # kept for annotation role lookup
nc_c_records = []   # (x, y, nc_lh, c_lh, role)

for nc_idx in non_circuit_indices:
    for other_idx in range(144):
        if other_idx == nc_idx:
            continue
        x = C_IOI[nc_idx, other_idx]
        y = C_nonIOI[nc_idx, other_idx]
        other_lh = hl_flat[other_idx]
        if other_lh in ALL_CIRCUIT_HEADS:
            role = _role(*other_lh)
            nc_c_records.append((x, y, hl_flat[nc_idx], other_lh, role))
        else:
            nc_nc_x.append(x)
            nc_nc_y.append(y)

nc_c_all_x = [r[0] for r in nc_c_records]
nc_c_all_y = [r[1] for r in nc_c_records]

all_vals_flat = np.array(nc_nc_x + nc_nc_y + nc_c_all_x + nc_c_all_y)
lim = max(abs(all_vals_flat.min()), abs(all_vals_flat.max())) * 1.05

# Fit normal to all C_diff = C_IOI - C_nonIOI across all plotted pairs
all_cdiff = np.array(
    [x - y for x, y in zip(nc_nc_x, nc_nc_y)] +
    [r[0] - r[1] for r in nc_c_records]
)
sigma = all_cdiff.std()
band = 5 * sigma

fig3, ax = plt.subplots(figsize=(7, 6.5))

xs_band = np.array([-lim, lim])
ax.fill_between(xs_band, xs_band - band, xs_band + band,
                color='gray', alpha=0.12, zorder=0, linewidth=0)

# Diagonal: C_diff = 0
ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8, alpha=0.4, zorder=1)

# NC-NC: red; NC-circuit: dark blue — same size, same alpha
ax.scatter(nc_nc_x, nc_nc_y, c='#fc8d8d', s=5, alpha=0.25,
           linewidths=0, rasterized=True, label='NC ↔ NC')
ax.scatter(nc_c_all_x, nc_c_all_y, c='#08306b', s=5, alpha=0.25,
           linewidths=0, rasterized=True, label='NC ↔ circuit')

top_records = sorted(nc_c_records, key=lambda r: abs(r[0] - r[1]), reverse=True)

# Annotate top 3 by |C_diff|, deduplicated by NC head
seen_nc = set()
to_annotate = []
by_abs = sorted(nc_c_records, key=lambda r: abs(r[0] - r[1]), reverse=True)
for r in by_abs:
    x, y, nc_lh, c_lh, role = r
    if nc_lh in seen_nc:
        continue
    seen_nc.add(nc_lh)
    outside = abs(x - y) > band
    to_annotate.append((r, outside))
    if len(to_annotate) >= 3:
        break

# stagger offsets so nearby points don't collide
offsets = [(-0.25, 0.20), (-0.05, 0.10), (-0.25, 0.20)]
for k, ((x, y, nc_lh, c_lh, role), outside) in enumerate(to_annotate):
    lbl = f'L{nc_lh[0]}H{nc_lh[1]} → {role}'
    color = '#08519c' if outside else '#555555'
    ox, oy = offsets[k]
    ax.annotate(lbl, xy=(x, y), xytext=(x + ox, y + oy),
                fontsize=6.5, color=color,
                arrowprops=dict(arrowstyle='-', color='#aaaaaa', lw=0.6),
                zorder=5)

ax.set_xlabel('C_IOI', fontsize=12)
ax.set_ylabel('C_nonIOI', fontsize=12)
ax.set_title(f'C_IOI vs C_nonIOI — all pairs involving a non-circuit (NC) head\n'
             f'Gray band = 5σ (σ={sigma:.3f})', fontsize=10)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_aspect('equal')
ax.grid(True, alpha=0.15, zorder=0)
ax.legend(fontsize=9, loc='lower right', markerscale=2, framealpha=0.9)

plt.tight_layout()
out43_fig = fig_to_output(fig3)

nc_c_total = len(nc_c_records)
dlines = [
    f'NC-NC pairs plotted:      {len(nc_nc_x)}\n',
    f'NC-circuit pairs plotted: {nc_c_total}\n',
    f'\nTop 10 NC-circuit outliers by |C_diff| = |C_IOI − C_nonIOI|:\n',
    f'{"NC head":<10} {"Circuit head":<14} {"Role":<24} {"C_IOI":>7} {"C_nonIOI":>9} {"C_diff":>7}\n',
    '-'*72 + '\n',
]
seen_pairs = set()
for x, y, nc_lh, c_lh, role in top_records:
    if (nc_lh, c_lh) in seen_pairs: continue
    seen_pairs.add((nc_lh, c_lh))
    dlines.append(f'L{nc_lh[0]}H{nc_lh[1]:<8} L{c_lh[0]}H{c_lh[1]:<12} {role:<24} '
                  f'{x:>+7.3f} {y:>+9.3f} {x-y:>+7.3f}\n')
    if len(seen_pairs) >= 10: break
print("".join(dlines[:5]))

# ── 6. Inject outputs into notebook ────────────────────────────────────────
print("\nInjecting outputs into notebook...")
with open(NB_PATH) as f:
    nb = json.load(f)

nb['cells'][39]['outputs'] = [out39_fig, text_output(txt39)]
nb['cells'][39]['execution_count'] = 39

nb['cells'][41]['outputs'] = [out41_fig, text_output(tlines)]
nb['cells'][41]['execution_count'] = 41

nb['cells'][43]['outputs'] = [out43_fig, text_output(dlines)]
nb['cells'][43]['execution_count'] = 43

with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print("Done — outputs injected into cells 39, 41, 43.")
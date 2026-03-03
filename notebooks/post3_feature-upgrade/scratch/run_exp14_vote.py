"""
Exp 14: Ensemble voting across all IOI-specific eigenvectors.

Each eigenvector / ranking we computed is a "voter" that nominates the heads
with the largest |loading| as circuit candidates.  Tally votes across all
voters.  Heads that consistently appear in the top-K of many independent
methods are the most trustworthy candidates; NC heads that sneak into one or
two eigenvectors get diluted out.

Voters included:
  [A] C_ioi raw eigenvectors with low C_non overlap (Exp 10)
  [B] Approach A residuals: C_ioi evecs with C_non top-9 projected out (Exp 13)
  [C] Approach B K=6: eigenvecs of P_perp @ C_ioi @ P_perp, K=6 (Exp 13)
  [D] Approach B K=9: eigenvecs of P_perp @ C_ioi @ P_perp, K=9 (Exp 13)
  [E] C_diff per-head mean-positive-correlation score (Exp 9)

For each voter, the top-TOP_K heads by |loading| (or score) each get one vote.
We then rank heads by total vote count.

Figures:
  exp14_votes.png          — vote-count bar chart, colored by role
  exp14_precision.png      — precision curve: as vote threshold rises, what fraction
                             of heads above that threshold are circuit?

Run with: /opt/miniconda3/bin/python run_exp14_vote.py
"""

import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mpatches

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figs", exist_ok=True)

# ── metadata ──────────────────────────────────────────────────────────────────
IOI_HEADS = {
    "Name Mover":          [(9,6),(9,9),(10,0)],
    "Backup Name Mover":   [(9,0),(9,7),(10,1),(10,2),(10,6),(10,10),(11,2),(11,9)],
    "Negative Name Mover": [(10,7),(11,10)],
    "S-Inhibition":        [(7,3),(7,9),(8,6),(8,10)],
    "Induction":           [(5,5),(6,9)],
    "Duplicate Token":     [(0,1),(3,0)],
    "Previous Token":      [(2,2),(4,11)],
}
ROLE_ORDER = ["Name Mover","Backup Name Mover","Negative Name Mover",
              "S-Inhibition","Induction","Duplicate Token","Previous Token","Non-circuit"]
ROLE_COLORS = {
    "Name Mover":"#d62728","Backup Name Mover":"#ff7f0e",
    "Negative Name Mover":"#9467bd","S-Inhibition":"#2ca02c",
    "Induction":"#17becf","Duplicate Token":"#bcbd22",
    "Previous Token":"#e377c2","Non-circuit":"#cccccc",
}
SHORT = {"Name Mover":"NM","Backup Name Mover":"BNM","Negative Name Mover":"NegNM",
         "S-Inhibition":"SI","Induction":"Ind","Duplicate Token":"DT",
         "Previous Token":"PT","Non-circuit":"NC"}
ALL_CIRCUIT = set(lh for hs in IOI_HEADS.values() for lh in hs)
HL_FLAT     = [(l,h) for l in range(12) for h in range(12)]
def role(l,h):
    for r,hs in IOI_HEADS.items():
        if (l,h) in hs: return r
    return "Non-circuit"
HEAD_NAMES   = [f"L{l}H{h}" for l,h in HL_FLAT]
HEAD_CIRCUIT = np.array([lh in ALL_CIRCUIT for lh in HL_FLAT])
HEAD_ROLE    = [role(*lh) for lh in HL_FLAT]
HEAD_COLOR   = [ROLE_COLORS[r] for r in HEAD_ROLE]
N = 144; N_CIRC = HEAD_CIRCUIT.sum(); CHANCE = N_CIRC / N
legend_handles = [mpatches.Patch(color=ROLE_COLORS[r], label=SHORT[r]) for r in ROLE_ORDER]

# ── load data ─────────────────────────────────────────────────────────────────
outmag_ioi = np.load("outmag_ioi.npy")
outmag_non = np.load("outmag_non.npy")
C_ioi = np.corrcoef(outmag_ioi.T)
C_non = np.corrcoef(outmag_non.T)

def eigh_desc(C):
    vals, vecs = np.linalg.eigh(C)
    idx = np.argsort(vals)[::-1]
    return vals[idx], vecs[:, idx]

vals_non, vecs_non = eigh_desc(C_non)
vals_ioi, vecs_ioi = eigh_desc(C_ioi)

# Marchenko-Pastur threshold
mp_max = (1 + (N/1000)**0.5)**2   # N=144, n_samples=1000
K_mp   = int((vals_non > mp_max).sum())   # = 9

# Projection operators
def proj_ops(K):
    V = vecs_non[:, :K]
    P = V @ V.T
    return P, np.eye(N) - P

P6, Pp6 = proj_ops(6)
P9, Pp9 = proj_ops(K_mp)

# C_non overlap per C_ioi evec
overlaps = np.array([float((P9 @ vecs_ioi[:, k]) @ (P9 @ vecs_ioi[:, k]))
                     for k in range(N)])

# Approach B eigenvectors
_, vecs_B6 = eigh_desc(Pp6 @ C_ioi @ Pp6)
_, vecs_B9 = eigh_desc(Pp9 @ C_ioi @ Pp9)

# C_diff per-head score (mean positive differential correlation)
C_diff = C_ioi - C_non
np.fill_diagonal(C_diff, 0.0)
cdiff_score = np.clip(C_diff, 0, None).mean(axis=1)   # (144,)

# ── build voter list ───────────────────────────────────────────────────────────
# Each voter is a length-144 score array (larger = stronger nomination).
# We keep the top TOP_K heads from each voter.
TOP_K = 10   # votes cast per voter

voters = []

# [A] Raw C_ioi evecs with C_non overlap < 0.8 (IOI-specific by Exp 10 criterion)
for k in range(N):
    if overlaps[k] < 0.8 and vals_ioi[k] > mp_max:   # above noise floor too
        voters.append((f"A_raw_evec{k}", np.abs(vecs_ioi[:, k])))

# [B] Approach A residuals (C_ioi evecs with C_non top-9 projected out)
#     Include evecs where the raw evec has >0.5 overlap (otherwise residual is trivial)
#     and the evec eigenvalue is above noise floor
for k in range(N):
    if overlaps[k] > 0.2 and vals_ioi[k] > mp_max:
        resid = Pp9 @ vecs_ioi[:, k]
        voters.append((f"B_resid_evec{k}", np.abs(resid)))

# [C] Approach B K=6: top eigenvecs of C_ioi_clean (K=6), eigenvalue > MP bulk
_, vecs_B6_all = eigh_desc(Pp6 @ C_ioi @ Pp6)
vals_B6, _     = eigh_desc(Pp6 @ C_ioi @ Pp6)
for k in range(N):
    if vals_B6[k] > mp_max:
        voters.append((f"C_B6_evec{k}", np.abs(vecs_B6_all[:, k])))

# [D] Approach B K=9
vals_B9, vecs_B9_all = eigh_desc(Pp9 @ C_ioi @ Pp9)
for k in range(N):
    if vals_B9[k] > mp_max:
        voters.append((f"D_B9_evec{k}", np.abs(vecs_B9_all[:, k])))

# [E] C_diff per-head score
voters.append(("E_cdiff", cdiff_score))

print(f"Total voters: {len(voters)}")
print(f"  [A] raw C_ioi evecs (overlap<0.8, λ>MP): "
      f"{sum(1 for n,_ in voters if n.startswith('A'))}")
print(f"  [B] Approach A residuals (overlap>0.2, λ>MP): "
      f"{sum(1 for n,_ in voters if n.startswith('B'))}")
print(f"  [C] Approach B K=6 (λ>MP): "
      f"{sum(1 for n,_ in voters if n.startswith('C'))}")
print(f"  [D] Approach B K=9 (λ>MP): "
      f"{sum(1 for n,_ in voters if n.startswith('D'))}")
print(f"  [E] C_diff score: 1")

# ── tally votes ───────────────────────────────────────────────────────────────
votes = np.zeros(N, dtype=int)
for name, score in voters:
    top_heads = np.argsort(score)[::-1][:TOP_K]
    votes[top_heads] += 1

# ── print ranked results ──────────────────────────────────────────────────────
order = np.argsort(votes)[::-1]
print(f"\nHeads ranked by vote count (top {TOP_K} per voter, {len(voters)} voters total):")
print(f"  {'Rank':>4}  {'Head':>8}  {'Role':>20}  {'Votes':>5}  {'Circuit?':>8}")
for rank, i in enumerate(order[:40], 1):
    tag = "CIRCUIT" if HEAD_CIRCUIT[i] else ""
    print(f"  {rank:4d}  {HEAD_NAMES[i]:>8}  {HEAD_ROLE[i]:>20}  "
          f"{votes[i]:5d}  {tag}")

# ── precision at each vote threshold ─────────────────────────────────────────
max_votes = votes.max()
thresholds = np.arange(0, max_votes + 1)
precision_at_thresh = []
n_heads_at_thresh   = []
for t in thresholds:
    above = votes >= t
    n = above.sum()
    n_heads_at_thresh.append(n)
    precision_at_thresh.append(HEAD_CIRCUIT[above].mean() if n > 0 else 0.0)

print(f"\nPrecision at vote thresholds:")
print(f"  {'Threshold':>9}  {'N heads':>7}  {'Precision':>9}  {'Enrichment':>10}")
for t, n, p in zip(thresholds, n_heads_at_thresh, precision_at_thresh):
    enrich = p / CHANCE
    print(f"  {t:9d}  {n:7d}  {p:9.1%}  {enrich:8.1f}x")

# ── Fig 1: vote-count bar chart ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 5))
xs = np.arange(N)
ax.bar(xs, votes, color=HEAD_COLOR, alpha=0.85, width=0.9)
for i, ic in enumerate(HEAD_CIRCUIT):
    if ic: ax.axvline(i, color="k", lw=0.5, alpha=0.25, zorder=0)
ax.set_xticks([l*12+6 for l in range(12)])
ax.set_xticklabels([f"L{l}" for l in range(12)], fontsize=9)
ax.set_xlim(-0.5, N-0.5)
ax.set_ylabel(f"Vote count  (top-{TOP_K} per voter, {len(voters)} voters)", fontsize=10)
ax.set_title(
    f"Ensemble vote count: how many eigenvectors / methods nominate each head?\n"
    f"Voters: raw C_ioi evecs (low C_non overlap) + residuals + Approach B K=6,9 + C_diff score\n"
    f"Thin black lines = known circuit head positions",
    fontsize=9
)
ax.legend(handles=legend_handles, fontsize=8, loc="upper left", ncol=4, framealpha=0.9)
plt.tight_layout()
plt.savefig("figs/exp14_votes.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  → figs/exp14_votes.png")

# ── Fig 2: precision curve vs vote threshold ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(thresholds, [p*100 for p in precision_at_thresh], "o-",
        color="#d62728", lw=2, ms=5)
ax.axhline(CHANCE*100, color="gray", lw=1.2, ls="--",
           label=f"chance ({CHANCE:.1%})")
for t, n, p in zip(thresholds, n_heads_at_thresh, precision_at_thresh):
    if t > 0 and n < 50:
        ax.annotate(f"n={n}", (t, p*100), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7)
ax.set_xlabel("Minimum vote count to be included", fontsize=10)
ax.set_ylabel("Precision (% of included heads that are circuit)", fontsize=10)
ax.set_title("Precision rises as vote threshold rises\n"
             "(high-vote heads are more reliably circuit)", fontsize=9)
ax.legend(fontsize=9)

ax = axes[1]
ax.plot(n_heads_at_thresh, [p*100 for p in precision_at_thresh], "o-",
        color="#1f77b4", lw=2, ms=5)
ax.axhline(CHANCE*100, color="gray", lw=1.2, ls="--",
           label=f"chance ({CHANCE:.1%})")
for t, n, p in zip(thresholds, n_heads_at_thresh, precision_at_thresh):
    if t > 1 and n < 50:
        ax.annotate(f"≥{t}v", (n, p*100), textcoords="offset points",
                    xytext=(4, 0), ha="left", fontsize=7)
ax.set_xlabel("Number of heads included", fontsize=10)
ax.set_ylabel("Precision (% circuit)", fontsize=10)
ax.set_title("Precision vs recall tradeoff\n"
             "(annotated with vote threshold)", fontsize=9)
ax.legend(fontsize=9)
ax.invert_xaxis()

plt.suptitle("Ensemble voting: precision and recall across vote thresholds", fontsize=10)
plt.tight_layout()
plt.savefig("figs/exp14_precision.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figs/exp14_precision.png")

# ── Fig 3: vote bar chart zoomed to high-vote heads only ─────────────────────
thresh_show = 2
above = np.where(votes >= thresh_show)[0]
above_sorted = above[np.argsort(votes[above])[::-1]]
n_show = len(above_sorted)
circ_frac = HEAD_CIRCUIT[above_sorted].mean()

fig, ax = plt.subplots(figsize=(max(10, n_show * 0.45), 5))
ax.bar(np.arange(n_show), votes[above_sorted],
       color=[HEAD_COLOR[i] for i in above_sorted], alpha=0.85, width=0.9)
ax.set_xticks(np.arange(n_show))
ax.set_xticklabels([HEAD_NAMES[i] for i in above_sorted],
                   rotation=45, ha="right", fontsize=8)

# redo labels with fontweight per head
ax.set_xticklabels([])
for j, i in enumerate(above_sorted):
    ax.text(j, -0.8, HEAD_NAMES[i], rotation=45, ha="right", va="top",
            fontsize=8, fontweight="bold" if HEAD_CIRCUIT[i] else "normal",
            color="black")

ax.set_ylabel("Vote count", fontsize=10)
ax.set_title(
    f"Heads with ≥{thresh_show} votes  ({n_show} heads, {circ_frac:.0%} circuit vs {CHANCE:.0%} chance)\n"
    f"Bold labels = known circuit heads",
    fontsize=9
)
ax.legend(handles=legend_handles, fontsize=8, loc="upper right", ncol=2, framealpha=0.9)
plt.tight_layout()
plt.savefig("figs/exp14_votes_zoom.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figs/exp14_votes_zoom.png")
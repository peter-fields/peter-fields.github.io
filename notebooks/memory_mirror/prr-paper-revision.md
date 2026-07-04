---
name: prr-paper-revision
description: "PRR paper revision (Fields et al., temperature tuning in EBMs) — thesis framing, reviewer summary, agreed revision strategy, and to-do list. Deadline 2026-06-11."
metadata: 
  node_type: memory
  type: project
  originSessionId: 16e414e4-67ef-43e3-998e-1352a5c78aa9
---

# PRR Paper Revision — Fields et al., "Understanding temperature tuning in energy-based models"

**Why:** Submitted to Physical Review Research (MS ID VM10051W/Fields, arXiv 2512.09152v1). One critical reviewer (R1) and one positive reviewer (R2). Stephanie Palmer has approved a revision strategy that reframes rather than restructures. Deadline 2026-06-11 (when advisor goes on vacation).

**How to apply:** Any work on this paper should respect the lean-revision posture (minimal additions, sharpen existing content), avoid overclaiming, and route through the agreed strategy below. Repo paths: manuscript at `~/Git/temp_tuning_draft/` (LaTeX), code at `~/Git/temp-tune/` (Julia, currently on `dev` branch). Strategy doc on disk: `~/Documents/markdown/revision_strategy_notes.md`.

---

## Thesis (strongest form — use this framing)

**τ fixes a bias the objective function is structurally blind to but which is nonetheless real, arising from the ground truth distribution and the lack of data to resolve it.**

Forward KL (what MLE minimizes) can't detect overestimation of high-energy states because it weights by p (negligible mass there). Reversed KL weights by q̂ — and that's exactly where the problem lives. Temperature tuning is the post-hoc correction for a bias the training procedure cannot, by construction, address.

## Four confluent causes (state explicitly somewhere in main text)
1. Sparse data
2. Forward KL objective's blindness
3. Large energy gap Δ
4. High density of "meaningless" high-energy states

The κ vs C framework in Appendix C formalizes when raising vs lowering τ is optimal.

---

## Reviewer summary

### Referee 1 — critical, leans reject
- **Novelty:** confirms existing heuristic; no new method; "pedagogical" not "methodological advance"
- **Descriptive not prescriptive:** computing τ* needs ground truth p
- **τ > 1 regime is niche** for proteins (large-gap regime dominates)

### Referee 2 — positive, recommends publication after revisions
- **Major 1 — Ref [38] / regularization:** Russ et al. attributed T=1 imperfections to regularization; how does pseudocount regularization here connect?
- **Major 2 — Generalization** beyond Ising/Potts (VAEs, diffusion, autoregressive)?
- **Minors:** tone down "custom functions" re [38]; cite energy-gap claim; expand "forward KL focuses on modes"; clarify cross-entropy/entropy ratio = DKL min (proof or empirical?); how strong does Δ→∞ need to be; is N_s assumed known; Fig 2(b) histogram x-axis labels conflict with arrows

---

## Agreed strategy

**Posture:** reframe, don't apologize. Contribution is mechanistic understanding + novel τ>1 prediction. R1 is holding paper to wrong standard.

### Doing
- Soften abstract — pull back "diagnostic tool" language
- Sharpen Discussion w/ explicit thesis statement
- State four confluent causes together
- Promote Appendix C (raise vs lower) and Appendix D (objective blindness) in main text
- All R2 minor points
- **New heat-capacity appendix from dissertation Ch. 4** — Stephanie has approved. C decreases at τ* regardless of direction of tuning; computable from model alone (no ground truth). Closest thing to a prescriptive result. **Caveat:** empirical in low-T and high-T Ising regimes; doesn't always hold at intermediate T. Frame as empirical observation, NOT a theorem.

### Not doing
- No new results sections
- Don't claim framework is prescriptive (κ involves Cov(E_true, Ê) which needs p)
- **DROPPED 2026-06-01: knowledge distillation sentence.** Originally planned as one-sentence future-direction nod in Discussion. Peter pulled it — saving for a separate future treatment, not worth the inclusion under tight deadline.

---

## To-do list

**STATUS 2026-06-16 — REVISION DRAFTED & EMAILED TO COAUTHORS (Stephanie/Wave/David).** All referee edits done (A prose; B all 7 R2 minors; C both R2 majors); scaled color-image explanation + NITMB grants verified; temp-tune repo link added to appendix. Three PDFs compiled **on Overleaf** (local BasicTeX was too minimal — missing biber/latexdiff + many packages; not worth the install fight) and emailed: revised manuscript, point-by-point reply, and the latexdiff marked-up diff. **diff.tex was generated locally** with the official latexdiff perl script (fetched from CTAN to `/tmp/latexdiff`): `perl /tmp/latexdiff main_arxiv_comments_cleaned.tex main_revised.tex > diff.tex` (old=original submission, new=revised); compiles on Overleaf with main_revised's preamble + same figures/bib.
- **STATUS 2026-06-24 — mostly off Peter's desk, awaiting coauthors.** **Fig 2 label edit DONE** (R2 minor). ⚠️ **Fig 3 code remake/verify is NOT done** (the deferred reproducibility check — re-run the Fig 3 cell with `r=4→3` after the loader fix and confirm it matches the published figure; see §D.1). **Wave (Vudtiwat Ngampruetikorn) has taken a pass at Peter's edits → Peter's next action is to read what Wave wrote.** Still awaiting **Stephanie (Palmer)** and **David (Schwab)**. After all coauthor comments are in: incorporate → APS PRR resubmission (clean PDF + latexdiff diff + reply letter).
- **🔴 CAPTION-PARAMETER AUDIT — 2026-07-02 (CC + Peter).** Audited every single-experiment / cited-M(/Δ) panel (caption vs actual data/notebook). **THREE caption fixes needed for the APS resubmit; everything else is correct.**
  0. 🔴 **Fig 2 (main text) caption has TWO errors** (`sweep_toy_model.ipynb` cell 18; figure reproduces published Fig 2, Peter confirmed → notebook = ground truth):
     - **(a) Δ:** caption cites true-model gap `Δ=4` but notebook uses `Δtrue=5`. Verified via `etrues` (`src/two_level_toy_model.jl`): with Δ=5, nlevels=2, nground=3, nexcited=7 → ground states at energy 0, excited at energy Δ·1 = **5** → gap literally 5. **Fix Δ=4 → Δ=5.**
     - **(b) sample count:** caption says data "made from **10 samples**" but the fit uses `nsampss=5` → data is **5 samples**. The hardcoded `freqs=[0.6,0.4,0,…]` is ambiguous (0.6/0.4 = both 6,4/10 AND 3,2/5), but `nsampss` drives the toy pseudocount → the whole figure; Peter verified 5→10 changes the fig entirely, so the published fig IS the nsampss=5 (=3,2 counts) experiment. **Peter's fix: keep `nsampss=5` in notebook (preserves published fig), change caption "10 samples" → "5 samples".** (My earlier "10 samples ✓ / nsampss internal-only" call was WRONG — nsampss IS the cited sample count.)
     - **CORRECT (no change):** "10 **states**" = N_states = 3+7 (different quantity from the sample count; likely how "10" leaked into "samples").
  1. 🔴 **Fig 3 (main text) caption is WRONG — cites `M=93, T=2.3` but the actual experiment is `M=71, T=2.05`.** The 4×4 paper-figure selector is `sweep_nn_ising.ipynb` **cell 37**: `M,T,r=(71.0, 2.05, 3)  # experiment used in paper`. (CC first misread the **3×3 DEMO** cell — cell 21, `(50.0, 2., 3)`, which uses `tau_sweep_opts_3by3`; the real 4×4 Fig 3 is cell 37. Corrected by Peter 2026-07-02, who was reading the cell directly.) Confirms the original §D note `(71.0, 2.05, 4→3)` was CORRECT — only the replicate `r` changed 4→3 after the loader fix; **M and T never moved**. Matches the Sep-2025 Inkscape export `fig3-raw-new_T=2.05_M=71.0_r=4.svg`. (M=93 *data* exists but was never the final panel; the caption number is just stale/wrong.) **Peter's call: keep the figure, fix the caption to M=71, T=2.05.**
  2. ⚠️ **fig4 = Fig A3 (App D) Panel C is M=40, not M=54.** Panels A/B ARE genuinely M=54 (from `hld[:low]`, T=2.3), but Panel C (per-level D_KL decomposition) comes from `all_temp_level_dict` whose M-grid `[40,63,100,…]` has NO 54 → it reproduces at M=40. **RESOLVED 2026-07-02 (Peter, on OVERLEAF — not yet pulled to local `main_revised.tex`):** the offending M=54 was in the BODY TEXT (the "averages over 50 replicates … T=2.3, M=54" sentence, ~line 664), changed to **M=40**. The A3 CAPTION (`fig:emp_dkl_bias`) cites M=54 only in panel (A) — which IS genuinely M=54 — and gives NO M for panel (C) ("averaged over 50 replicates"), so the **caption is left as-is (consistent, verified by CC).** Note: A3 now spans two M's by design (panel A = single exp at M=54; panel C = 50-rep avg at M=40); optional 4-word caption clarifier offered, Peter's call.
  - **VERIFIED CORRECT (no change):** Fig 2 toy single experiment = hardcoded M=10/Δ=4 (matches, Peter confirmed); fig4 A/B = M=54; nn_ising_kappa a–d = M=54 (`hld[:low]`/`[:high]`, T=2.3/4.0); nn_ising_kappa panel e = M=54 (loader filters `nsamps=54.0`). So of 4 single-experiment citations only Fig 3's M+T are off, plus the fig4-Panel-C note.
  - **ALSO VERIFIED CLEAN 2026-07-02 (loaded actual data in Julia, `scratchpad/audit_sweep.jl`):**
    - **Fig 2 g–h sweep — NO analogous bug.** Peter worried the sweep might mislabel axes the way the single-experiment caption did. Loaded `data/simple_model_sweeps/Nstates=100_nground=20.jld2`: the STORED grid the τ*/τ′ heatmaps were computed on == the PLOTTING grid used for axis labels — Deltas EQUAL (29 vals, 2.5:0.25:9.5) and Ms EQUAL (160 vals). Stored row also = `nground=20, nexcited=80, nreps=50` → matches caption "n_l=20, n_h=80, 50 replicates." (`plot_contour_fig_2` labels from the passed `Deltas`/`Ms`, not the df's stored grid — safe ONLY because they're identical, which is now confirmed.)
    - **newfig (App C toy) Δ audit — CORRECT, no fix.** `newfigdict.jld2` stores `Delta_true` explicitly: "small gap"=**2**, "large gap"=**7**, matching the caption's Δ=2/Δ=7 (nsamps=10 both). Ironically the appendix toy fig is right; only the MAIN Fig 2 drifted.
  - Found while reviewing the migrated appendix notebooks (`notebooks/supplemental/{toy_model_appendix,nn_ising_kappa,ising_appendix}.ipynb`, on `dev`, under Peter's review — repo-maintenance item (3) now substantially done, NOT pushed). See [[compressed-computation-project]] neighbor context.
- **REPO MAINTENANCE — 2026-07-01: items 1, 2 & 4 DONE.** ✅ **(1) Loader fix PUBLISHED to origin (sepalmer/temp-tune) `main`** via a squash-merge PR (temp branch `milestone-fig-fix` → main, title "fix sweep loader ordering + reproducible Fig 3"); the public/default branch readers clone now has the numeric-sort loader + Fig 3 r=3 → repo is reproducible. ✅ **(2) Fig 3 regenerated + Peter eyeball-verified** identical to the published figure. ✅ **(4) DONE 2026-07-01:** Fig-2 toy data (`simple_model_sweeps`, 11 `.jld2`) uploaded to the existing HF dataset `peter-fields/temp-tune-data` under a new `simple_model_sweeps/` subfolder (next to `ising_sweeps/`); README "Data availability" rewritten to `git clone <dataset> data` (yields BOTH subfolders) — this also **fixed a latent double-nest bug** in the old instructions (they cloned into `data/ising_sweeps` → `data/ising_sweeps/ising_sweeps/`). Published via squash-PR #7; portfolio mirror `personal/main` synced. ✅ **(3) DONE 2026-07-02 — appendix notebooks migrated + PUBLISHED.** `notebooks/supplemental/{toy_model_appendix, nn_ising_kappa, ising_appendix}.ipynb` (all 4 appendix figs) squash-merged to `sepalmer/temp-tune` main via PR #8 ("finish supp nbs"), README trim via #9. Public repo (`origin`=sepalmer, at `9e1999a`), portfolio mirror (`personal`=peter-fields) and private backup (`devbackup/dev`) all verified in sync 7/02. Repo fully reproduces main + appendix figures. **All repo-maintenance items now complete.** Publish workflow now saved: [[temp-tune-publish-workflow]] (squash-merge PRs from temp branches → public main; never fast-forward dev→main). *(Original 2026-06-23 plan retained below.)*
- **DEFERRED — repo maintenance (original, 2026-06-23):**
  - **🎯 END GOAL: everything below must land on STEPHANIE'S PUBLIC REPO `origin` = github.com/sepalmer/temp-tune** — that's the repo the appendix links to and what readers will actually clone. The `personal` remote (peter-fields/temp-tune) is just Peter's working mirror; pushing there is NOT sufficient. Each item below is only "done" once it's on origin/sepalmer.
  1. **🔴 Push the loader fix to `origin` (sepalmer/temp-tune).** Commit `6a254af` (import_sweep_dicts numeric-sort loader fix + Fig 3 `r=4→3`) is currently on `personal` ONLY. origin still has the buggy lexicographic loader + r=4, so the linked public repo is not yet reproducible. Sync dev → origin.
  2. **Remake Fig 3 + verify** it matches the published figure (manuscript figure is the correct published one; r=4→3 only matters on re-run).
  3. **Migrate appendix-plot code** from `proteins/temp_analysis` into temp-tune (`notebooks/supplemental/`) **with the rep-misalignment watch** (see Internal note below — confirm position k == replicate k for any per-rep plot).
  4. **HuggingFace data + README** (simple-model data for Fig 2 reproduction).
- **Tex files (in `~/Git/temp_tuning_draft/`):** `main_revised.tex` (revised), `reply.tex` (biblatex+biber), `diff.tex` (latexdiff output), original = `main_arxiv_comments_cleaned.tex`. Edits fixed this session: KL-asymmetry wording in reply minor-3, typos in main (one→may, double "that", "makes predicts"→"predicts...is"), documentclass comma.

### A. Strategic prose / framing
- [ ] Soften abstract (drop/qualify "diagnostic tool")
- [ ] Sharpen Discussion w/ explicit thesis statement
- [ ] State four confluent causes explicitly in main text
- [ ] Strengthen Appendix C / D cross-references in main text
- [ ] **Add "signatures of criticality" paper citation** (added 2026-06-10 — Peter flagged before forgetting). Likely Schwab-Nemenman-Mehta PRL 2014 (Zipf's law / criticality without fine-tuning) given Schwab is a co-author, but Peter to confirm which specific paper. Relevant location TBD — likely in intro alongside energy-gap citations, or in Discussion connecting to broader literature on biological-system energy landscapes.
- [x] ~~**Add heat-capacity appendix**~~ — **DROPPED 2026-06-13** per Peter's editorial call to keep the revision tighter. Story now compressed to one new appendix (histogram overlap, below). Heat-cap had been approved by Stephanie 2026-05-29; she should be flagged on the scope reduction.
- [ ] **LIVE DIRECTION (2026-06-13, supersedes the tail-suppression appendix below) — keep the existing story; sharpen the appendix to make the REGULARIZATION-INDEPENDENCE point the theoretical headline.** Peter's editorial call: don't add the tail-mass figure at all (it invites attack surface — see scrapped item — for no gain). Instead state the contribution directly: *the framework both (i) validates what practitioners do AND (ii) shows that the temperature-tuning effect does NOT require regularization to arise.* The T=1 over-population of high-energy states is a structural sparse-data + forward-KL-blindness + energy-gap phenomenon, not an artifact of the L2 coupling penalty that Russ et al. [38] invoke. This is the main theoretical advance: it re-explains "what people are doing" at a deeper level than the field's regularization attribution.
  - **Why this is strong (answers BOTH referees):** R2 Major Point 1 asks how the findings connect to Russ et al.'s regularization attribution — this IS the answer (regularization is sufficient but not necessary; the bias is more fundamental). R1's "you just validate the existing tool / no new understanding" — reattributing a half-explained phenomenon to its true structural cause IS new understanding, the contribution-type R1 says is missing.
  - **The existing unregularized Ising/toy results already demonstrate it** — minimal new work, fits the lean-revision posture. "Add to the appendix a bit" = a short paragraph (and/or fold into the R2-Major-1 regularization paragraph), leveraging results already in the paper.
  - **The Ising fit has ZERO regularization — verified in code 2026-06-13** (`ml_fit_Jij`, `nearest_neighbor_ising.jl:148`): the gradient is exactly `sample_corrs − model_corrs` (empirical vs model second moments) and the loss is `logZ + mean_energy` — vanilla unregularized MLE. No L2, no Gaussian prior, NO pseudocount. (grep for pseudo/reg/lambda/prior/smooth/penalty over the Ising source = nothing.) So the strong claim is exactly true for the Ising: *the temperature-tuning effect arises under pure maximum-likelihood with no regularization of any kind.* The earlier "only pseudocounts in the empirical distribution" note (handoff §3b) was WRONG for the Ising — **the pseudocount belongs to the TOY/simple model only** (Fig 2 path).
  - **This cleanly handles R2's pseudocount comment** (Major 1: *"here, the authors are using a pseudocount, and this somewhat differs from Ref [38]"*). R2 is talking about the toy model. Structure the response around the contrast: the toy model carries a small pseudocount, the Ising carries none — and the effect appears in BOTH, definitively in the fully-unregularized Ising. That turns R2's worry into supporting evidence for regularization-independence. Do NOT claim it contradicts Russ — it complements ("among other factors" → we identify the more fundamental, regularization-independent cause).
  - **Note (nice irony):** this is the SAME fact that killed the tail/overlap metric. The handoff §3a explained the Bhattacharyya overlap got the wrong sign precisely because the unregularized MLE already matches the data bulk at τ=1 (no reg-induced gap to close). That obstacle IS the contribution — Peter correctly extracted the signal (regularization-independence) and dropped the fragile metric that was trying to show it indirectly.

- [x] ~~**FINAL APPROACH (2026-06-13) — Tail-suppression appendix.**~~ **SCRAPPED 2026-06-13 (later same day)** in favor of the regularization-independence framing above — the tail-mass metric adds reviewer attack surface (needs the "m(τ) is a directional diagnostic not a proxy" caveat since m(τ) is monotone in τ, plus the unregularized-vs-regularized wrong-sign explanation) without strengthening the point. Code was reviewed + fixed before the scrap (see below); the notebook `histogram_overlap.ipynb` remains on dev as a correct-but-unused artifact. Original plan retained for reference: SUPERSEDES the moment-matching/MSE version below (scrapped) and the heat-cap appendix (dropped, notebook+plots moved to `proteins/temp_analysis`). Metric = **tail mass** m(τ) = model probability above E_max (max data energy), p-free. Show Δm = m(τ*) − m(τ=1) < 0: the framework's τ* suppresses the high-energy tail the practitioner histogram-matching targets (Russ Fig. 3). Robust across all small M and all reps at low T (n=90/M, frac<0 = 1.00); largest at small M, →0 as M grows. Bhattacharyya/symmetric metrics FAILED (wrong sign — bulk-dominated, τ* over-concentrates ground bin vs finite data); reversed-KL on histograms diverges on empty tail bins; tail mass is the well-defined fix. **Full handoff (code + argument detail) at `~/Documents/markdown/histogram_overlap_handoff.md`**. Notebook: `temp-tune/notebooks/histogram_overlap.ipynb` (dev branch). **Wording caution:** τ* is computed via reversed KL (needs p); do NOT claim we compute τ* without p — claim the framework's optimum coincides with the p-free practitioner target (tail suppression / histogram match).

  - **CODE REVIEW DONE 2026-06-13 (by CC; Fable was offline).** Verified: dkl_blue = reversed KL D(q̂‖p) (src line 447); τ* = argmin over fine spline grid (no root-finder, so the old multi-crossing bug can't recur); energy convention consistent between all-states and data sides (same `energy2spin`, E=−½σᵀJσ, same fitted J); log-sum-exp normalization correct over all 2^16 states; panel (b)/robustness float handling correct (isapprox + `unique(df.T)`); file location empirically complete (n=90 per M = 9 T × 10 reps all found).
  - **BUG FOUND + FIXED — rep-index misalignment.** `load_from_sweeps`→`import_sweep_dicts` fills per-rep vectors in `readdir()` order = lexicographic on the unpadded `_<rep>.jld2` suffix = `1,10,2,3,…,9`, NOT numeric. So `dkl_blue_tau_opts[k]` belongs to file-rep `loaded_order[k]`. The notebook then paired `τ_opts[rep]` with data loaded by *numeric* suffix → **only rep 1 aligned; reps 2–10 scrambled.** **DURABLE FIX AT SOURCE LEVEL (2026-06-13, per Peter's request):** `import_sweep_dicts` in `sweep_nn_ising.jl` now parses the trailing `_<rep>` number and `sort!`s the matched files by it before loading (was: appended in `readdir()` lexicographic order), so position k == replicate k for the whole codebase, permanently. Added a `@warn` if the rep set isn't contiguous 1:N (catches missing-rep silent misalignment). The notebook was reverted to plain `τ_opts[rep]` indexing (the source now guarantees alignment); the `isapprox` panel-(a) fix was kept.
  - **⚠️ SIDE EFFECT on main-paper Fig 3 — HANDLED 2026-06-13.** `plot_one_ising_replicate`/`plot_tau_sweep_one_rep` pick a replicate by positional index `r` (`J_fit[][r,:,:]`). After the source sort, `r` maps to file `_r` (numeric); before, it mapped via the lexicographic table [1,10,2,3,…,9]. The paper Fig 3 cell (`sweep_nn_ising.ipynb`, cell `771c31e0`) used `M,T,r=(71.0,2.05,4)`, which under the OLD loader resolved to replicate file `_3`. **Changed to `r=3`** so it reproduces the *identical* published figure under the corrected numeric loader. (The 3×3 demo cell `cb7f638b` has nreps=5 — no rep ≥10, so lexicographic == numeric, unaffected.) Peter to eyeball regenerated `fig3_reproduced.svg/.pdf` vs the published figure as a sanity check. **temp-tune will be the PUBLIC reproducibility repo linked in the appendix**, so the loader fix + r=3 matter for outside reproduction.
  - histogram_overlap.ipynb outputs remain stale, but the notebook is now an unused artifact (appendix scrapped) so no re-run needed.
  - **Impact is bounded — conclusion survives.** m(τ)=P_τ(E>E_thr) is monotone decreasing in τ, so Δm=m(τ*)−m(1)<0 whenever τ*<1 *regardless of pairing* → the sign result (frac<0=1.00) is invariant; the robustness table only re-pairs i.i.d. reps so its means barely move. Only panel (a)'s single annotated τ* was genuinely wrong (rep 3's value on rep 4's spectrum).
  - **MAIN PAPER FIG 3 IS NOT AFFECTED (checked).** `plot_one_ising_replicate`/`plot_tau_sweep_one_rep` take one positional `J_fit[r,:,:]` slice and recompute everything (incl. τ* via `findmin`) from it — no mixing of the two indexing schemes, never load raw samples by suffix. Caveat is cosmetic only: `r=3` is the 3rd lexicographic rep (physically file `_2`), a valid representative. No paper action.
  - **ARGUMENT-TIGHTNESS findings (act on in prose):** (1) m(τ) is *monotone* in τ → its argmin is τ→0, so it is NOT an optimizable proxy whose optimum is τ*. Frame m(τ) strictly as a **directional diagnostic** (τ* moves mass in the suppressing direction). DROP §7's "minimize tail mass … tracks τ*" wording. The real p-free practitioner criterion is histogram *matching/overlap* (balanced), not tail minimization. (2) Pre-empt the §3a wrong-sign exposure IN the appendix: symmetric overlap-with-data gets *worse* at τ* for unregularized MLE; explain that Russ et al.'s matching works because their fit is L2-regularized (creates a τ=1 model–data gap), whereas the unregularized Ising already matches the bulk at τ=1 by moment-matching, leaving only the tail-beyond-support (E_max) as the free coordinate — which is why model-tail-vs-data-support is the right p-free comparison. State the claim at this precision; "we reproduce the practitioner's exact procedure" overclaims. (3) Label the high-M `+0.000 / frac=0.99` table entries as float/MLE-convergence jitter at the zero floor so a referee can't read a "+" as a counterexample.

- [ ] **[SCRAPPED 2026-06-13] PROPOSED 2026-05-30 — Practitioner-criterion appendix + plot (moment-matching framing).** Pre-this-paper, practitioners pick τ by eyeballing energy histograms of data vs model samples under the trained energy function — operationally, matching the first two moments. This paper's framework explains why a specific second-moment criterion (MSE about the data mean) approximates τ\*. **Needs co-author discussion** — counts as a new addition against the "minimal additions" posture. Stronger R1 defense than the κ-vs-C local criterion alone.
  - **EDITORIAL SCOPE (decision 2026-06-02): keep the appendix POSITIVE.** Frame: *"current practice (eyeball energy-histogram matching) is exactly the MSE-about-data-mean criterion, and our theory predicts it works"*. Direct answer to R1's "descriptive vs prescriptive" concern: we (i) write down a clean math statement of what practitioners are doing with their eye, and (ii) show our framework predicts the criterion works in the regime where practitioners use it. **Do NOT drag in the failure-mode contrast** (variance-match goes blind in narrow-spectrum / high T_true regime; reversed-KL stays sharp). Real finding, distracts here, would muddy the response to R1. Save for a follow-up. Also: drop high-M panels (already decided) — those add MLE-convergence noise to the appendix figure without sharpening the message.
  - **Russ et al. anchor — practitioners match FULL distributions, not just means** (main paper Fig. 3, A–E, page 3 right column, confirmed 2026-06-01):
    - **Fig 3A** = empirical statistical-energy histogram of 1130 natural CM homologs (the "data" distribution)
    - **Fig 3C, D, E** = bmDCA model statistical-energy histograms at T = 0.33, 0.66, 1 (model histograms at three sampling temperatures)
    - *"sampling at T ∈ {0.33, 0.66} produced sequences with statistical energies that closely reflected the natural distribution (Fig. 3C), or reached even lower values (Fig. 3D). By contrast, sequences sampled at T = 1 showed a broad distribution of statistical energies that deviated significantly from the natural distribution (Fig. 3E) toward higher energies."*
    - *"This deviation is, among other factors, due to statistical adjustments [regularization (see materials and methods)] used during model inference for compensating for the limited sampling of sequences in the input MSA."*
    - **Operationally**: they pick T such that model histogram visually overlaps natural-sequence histogram. **Both** spread ("broad") **and** location ("toward higher energies") flagged at T=1 — exactly what variance-about-data-mean captures in one number.
    - **They use a RANGE of T** (T ∈ {0.33, 0.66}), not a unique optimum. The framework's τ* is a unique value — that's a refinement of the practitioner workflow.
    - **Two regularization strengths used**: λ ∈ {0.001, 0.01}. From SI page 4: they quote a specific mean-energy gap of 74.8 at λ=0.01, attributing ~57% to regularization.
    - **For response letter**: pair Fig 3C-E (main paper, full distributions) with SI page 4 (mean-energy gap, regularization attribution) for a complete picture of what practitioners actually do.
  - **Key insight — mean-match is uninformative without regularization** (verified empirically in `histogram_overlap.ipynb`): MLE moment-matching enforces ⟨E_Ĵ⟩_data = ⟨E_Ĵ⟩_q̂_{τ=1} at convergence by construction. Without L2 reg, the first-moment criterion always says τ ≈ 1 — it cannot see the bias the framework cares about. Russ et al. only sees a non-trivial first-moment gap **because of** their L2 regularization. *The framework's bias is more general than the practitioner's criterion.*
  - **Right "eyeball" criterion = variance about the data mean.** Notation (used throughout appendix and notebook):
    - **μ_d** = data mean of energies = ⟨E_Ĵ⟩_data over M samples
    - **s²_d** = data variance of energies = empirical ⟨(E_Ĵ − μ_d)²⟩_data
    - **μ_m(τ)** = model mean of E_Ĵ under q̂_τ
    - **v²_m(τ)** = model variance of E_Ĵ about its **own** mean
    - **M²(τ)** = model second moment about the **data** mean = ⟨(E_Ĵ − μ_d)²⟩_q̂_τ
    - **R** = spectrum range = max_s E_Ĵ(s) − min_s E_Ĵ(s) across all 2^16 states
    - **Bias-variance identity:** M²(τ) = v²_m(τ) + (μ_m(τ) − μ_d)²
    - **Mean-match condition** (Russ-style, fails for unregularized MLE): μ_m(τ_mean) = μ_d
    - **Variance-match condition** (Peter's eyeball criterion): M²(τ_var) = s²_d. Captures spread *and* off-centering in one number.
    - **Spectrum bound:** any variance-like quantity is at most R²/4. At high T_true, R is small → both s²_d and M²(τ) live in a narrow window → **M²(τ) as a function of τ is flat near τ=1** → criterion is ill-conditioned → τ_var becomes meaningless.
  - **Tracks τ\* well in sparse-data regime** (verified in notebook: M=54 case shows triangles on diagonal; mean-match circles pile up at τ ≈ 1).
  - **Heat-cap unification:** Var(E)_q̂_τ = C(τ)·τ². So variance-match is the **same** criterion as the heat-cap appendix, viewed thermodynamically vs stat-mech eyeball-y. **Strong argument to merge the two proposed appendices into one** "moment-matching" appendix — first moment trivial under MLE, second moment captures the bias, framework predicts when both criteria fail.
  - **Sells as bias-variance decomposition for stats-literate readers**: "the eyeball test is MSE of the model's energy distribution as an estimator of the data mean — algebraically Bias² + Var, semantically a measure of distributional match."
  - **Figure scoping (3 panels, current notebook):**
    - **(a)** Energy histograms (data vs exact model at τ=1 and τ=τ*) under Ĵ, with vertical lines for means. One representative experiment.
    - **(b)** Mean mismatch + variance mismatch (both normalized, left axis) and reversed-KL (right twin axis) vs τ. Three vertical lines: τ_mean, τ_var, τ\*. Shows τ_var lands closer to τ\* than τ_mean does.
    - **(c)** Scatter of τ_predicted vs τ\* across T sweep at fixed M, colored by T, with τ_mean as circles and τ_var as triangles. Low-T zoom, full-range inset. Triangles track y=x; circles cluster vertically at 1.
  - **M-sweep grid (decision 2026-06-02): drop high-M panels.** Originally planned 4 panels at M = 54, 215, 1136, 10000. Peter's call to keep only low/mid M (e.g., M = 54, 215). Reason: at high M the τ_var convergence story is muddied by finite-tolerance MLE residuals (see relTol gotcha below) and adds detail without changing the message. Not hiding anything; just keeps the figure focused on the regime where the appendix's claim (variance-match approximates τ\* in sparse data) actually carries.
  - **Reuses existing Ising sweep infrastructure** — mostly re-plotting, not new computation.
  - **R2 Major Point 1 lead-in:** Russ et al.'s mean-energy gap is regularization-induced (their SI says 57%). The framework predicts a structural bias *independent* of regularization. Variance-match captures both — natural bridge to the regularization paragraph.
  - **R1 DEFENSE — MSE criterion → suggests a novel regularizer** (Peter's insight 2026-06-02): R1 explicitly complains *"the study validates the tool practitioners already use without offering a novel regularization scheme or an alternative objective function that might obviate the need for such post-hoc tuning."* Our MSE-about-data-mean formalization SUGGESTS such a regularizer: L_reg(J) = −⟨log q̂_J⟩_data + λ · (M²(1; J) − s²_d)². The penalty constrains a specific 4-point correlation combination that standard MLE doesn't see; the regularized fit would match the variance at τ=1, obviating the need to tune τ post-hoc. Frame as **forward-looking sentence in the appendix or closing line of Discussion**: "this analysis suggests a principled regularization term in the form of the MSE criterion that would replace post-hoc τ tuning." Suggestive, not derived (the regularized objective is no longer trivially concave; computing M²(1; J) at training time requires exact expectations or sampling). But it converts the appendix from *"we explain what practitioners do"* → *"we explain what practitioners do AND suggest a principled alternative,"* which is exactly the prescriptive hook R1 says is missing.
  - **Notebook**: `~/Git/temp-tune/notebooks/histogram_overlap.ipynb` (built via `/tmp/build_histogram_overlap_nb.py`, inspected via `/tmp/inspect_histogram_overlap.py`). Currently runs end-to-end without prompting; outputs the main 3-panel figure (cell 17) and M-sweep grid (cell 21). Pending cosmetic fixes: data bar hidden behind red model histogram in (a), `D_{KL}` y-label rendering quirk in (b), inset overlap with axes in (c), M-sweep title crowding.
  - **Naming gotcha in `sweep_nn_ising.jl`** (don't re-confuse): `dkl_blue` = reversed KL D(q̂‖p) → τ\*; `dkl_red` = forward KL D(p‖q̂) → τ′. Color names refer to old plot traces, not divergence direction.
  - **Matplotlib mathtext gotcha** (learned 2026-06-01): `\bigl|`, `\bigr|`, and likely other amsmath sizing macros are NOT in matplotlib's built-in mathtext vocabulary. Using them in a PyPlot `set_ylabel(L"...")` causes the figure to silently break — PyCall swallows the Python exception so `savefig`/render fails with a useless stack trace. Stick to mathtext-safe LaTeX (plain `|...|`, no `\bigl`/`\bigr`/`\Big`/etc.) or switch to plain strings.
  - **MLE convergence gotcha — `relTol` is a *progress* criterion, not an *optimization* criterion** (lesson learned 2026-06-02). The fit options in `sweep_nn_ising.jl` stop when |ΔL/L| < 1e-5 between iterations. `neg_log_like` here is the *per-sample* average so no naive 1/M scaling sneaks in — the actual mechanism is more subtle:
    - For gradient descent near a minimum, `ΔL_per_iter ~ step × |gradient|²` (quadratic in gradient)
    - So `|gradient|` can sit at, say, 0.01 while `ΔL_per_iter` is already 1e-5 × L
    - Progress fires, loop exits, gradient is left non-zero
    - The gap widens when the **likelihood landscape is flat** — which happens at **low T (ordered phase)** because the Hessian of MLE w.r.t. J is the connected 4-point correlation, and `⟨s_i s_j s_k s_l⟩ ≈ ⟨s_i s_j⟩⟨s_k s_l⟩` when spins are highly aligned
    - **High M** makes the true MLE optimum a more precise target — even a small residual `|gradient|` corresponds to a meaningful J-space distance you haven't traveled
  - **Empirical evidence at M=10000, T=2.0**: `‖grad‖∞ = 0.013`, max|J| = 0.77 — ~1.7% slack per coupling. Propagates to Δμ ≈ J·grad ≈ 0.1 and Δvar ≈ 0.5, which displaces τ_var by ~0.05–0.10 at high T.
  - **For future work**: switch to an absolute gradient-norm tolerance (`maximum(abs.(∇Ĵ)) < grad_tol`), or use L-BFGS / conjugate gradient (which inherently terminate on `|grad|`). For the current paper, no fix needed — sub-5% residuals don't affect any conclusion.
  - **τ_match_for_curve multi-crossing bug** (fixed 2026-06-02): original implementation used `findmin(abs.(diff_fine))` on the spline of (model_curve − target). When MSE-vs-τ has a local minimum below the target value, the curve crosses the target twice — once below τ=1, once above — and `findmin` could pick either. Spurious "τ_var = 0.4 at high M" was this. Fix: enumerate sign-change crossings on the dense grid, pick the crossing closest to `pick_near=1.0` (the practitioner's starting point). Falls back to argmin if no crossings exist.

### B. R2 minor points (mechanical)
- [ ] Fix Fig 2(b) DKL histogram x-axis labels (left bar = high-E per arrows; x-axis suggests otherwise)
- [ ] Tone down "custom functions" re [38] — "natural-like function within existing family"
- [ ] Add citations for large-energy-gap claim in intro
- [ ] Expand on DKL(q‖p) focusing on modes in Section II A
- [ ] Cross-entropy/entropy ratio min = DKL min — proof or empirical? Include proof if general
- [ ] Δ→∞: is it strict, or does large finite Δ suffice — how large relative to what?
- [ ] N_s: assumed known? How limiting?

### C. R2 major points (Discussion paragraphs)
- [x] Ref [38] / regularization paragraph (pseudocount vs Russ et al.) — **DONE 2026-06-15**
- [x] Other architectures paragraph (VAE, diffusion, autoregressive — mechanism specific to EBMs w/ Boltzmann energy) — **DONE 2026-06-15**
- [x] ~~Knowledge distillation sentence~~ — **DROPPED 2026-06-01**, saving for future treatment

### D. Mechanical / reproducibility
- [ ] **🔴 REMAKE Fig 3 and verify it still matches the published figure.** After the `import_sweep_dicts` numeric-sort fix (committed `6a254af`, pushed to personal 2026-06-13), the Fig 3 cell was changed `r=4→3` (`sweep_nn_ising.ipynb` cell `771c31e0`) so the new numeric loader points at the same physical replicate (file `_3`) the old lexicographic loader did at r=4. **MUST re-run the figure and confirm `fig3_reproduced.svg`/`.pdf` is identical to the currently-published Fig 3 before submission.** If it differs, the rep mapping needs rechecking (don't ship a silently-changed figure).
- [ ] **Consolidate appendix-figure code into temp-tune** (added 2026-05-30; DEFERRED — Peter finishes the public repo AFTER sending edits to coauthors; appendix will link to it). Currently temp-tune reproduces the main paper plots; the rest of the appendix-figure code lives scattered in `~/Git/proteins/temp_analysis/` (which now also holds the scrapped `histogram_overlap.ipynb`, moved there 2026-06-13). Plan: create `temp-tune/notebooks/supplemental/` and migrate everything so the GitHub-repo-in-appendix link points at a complete reproducible artifact.
  - **🚨 MUST WATCH — replicate misalignment when migrating ANY per-rep appendix plot.** `import_sweep_dicts` now sorts files by numeric rep, but plotting/analysis code copied from `proteins/temp_analysis` may (a) predate that and assume the old lexicographic `1,10,2,...` order, or (b) pair a per-rep quantity loaded via the aggregated dataframe with raw data loaded separately by `locate_sweep_file` — exactly the bug that hit `histogram_overlap.ipynb` (tau* from the df vs data from the file suffix). Before trusting any migrated per-rep figure: confirm position k == replicate k, that any positional `r` index still selects the intended replicate (Fig 3 needed r=4→3 after the fix), and sanity-check regenerated figures against the originals.
- [ ] Link GitHub repo (temp-tune) in appendix
- [ ] Add simple-model data to HuggingFace (Fig 2 reproduction); fix README accordingly
- [ ] Fix "low"/"high" labels in Fig 2 itself
- [ ] Explain scaled color images (Jij, Fig 3(h-i))
- [ ] Verify NITMB acknowledgment grant numbers: NSF DMS-2235451, Simons Foundation MP-TMPS-00005320

### E. Submission deliverables (APS PRR)
- [ ] Clean revised manuscript PDF
- [ ] Marked-up diff PDF (`latexdiff old.tex new.tex > diff.tex`)
- [ ] Point-by-point response letter (strategically most important for R1)

---

## Appendix structure (for cross-reference work)
- **A** — toy model details, fitting algorithm (Algorithm 1), pseudocount regularization, exact τ* and τ′ expressions (Eqs. A10, A12)
- **B** — Ising training details (gradient descent, convergence criterion, numerical τ* via spline interp over τ ∈ [0.2, 5])
- **C** — when to raise vs. lower τ. ∂D/∂τ = (1/τ)(κ − C). κ = covariance susceptibility (Cov(E_true, Ê)), C = heat capacity. Sign of (κ − C) determines direction.
- **D** — empirical D_KL(p_data‖q) inherits systematic bias of p_data; Pythagorean relation (Eq. D1) gives D_KL(p_data‖p) > D_KL(q̂‖p).

## Defense angles against R1 (for reply.tex)

**LEAD R1 RESPONSE — settled 2026-06-15 (use these; the older bullets below are superseded/corrected as noted):**
- **(A) The demand for a prescriptive criterion may be asking the impossible — open question.** R1 presupposes a calculable data-only criterion for τ* must exist and faults its absence. Whether one *can* exist is itself open. Information/circularity argument: the bias τ corrects lives in the high-energy region finite data leaves underdetermined (= the tuning regime); computing τ* a priori, or training the bias away, requires specifying p there — i.e. recovering information the data lacks. No assumption-free data-only fix can; a prior on the density of states injects *external* info and leaves the data-limited regime. So R1 should NOT assume an obvious fix exists.
- **(B) Positive reframe — temperature tuning is a diagnosis of data deprivation.** The need to tune, and |τ*−1|, are quantitative signatures of data insufficiency; τ*→1 as M grows (paper shows it). Operational message: the route to a tuning-free model is more/better data; where unavailable (shallow families) τ-tuning is the principled corrective and the framework explains why it works. "Temperature tuning is a meaningful corrective in the data-limited regime, and its magnitude measures that limitation" IS prescriptive — just not the prescription R1 expected.
- **(C) One-two punch with regularization-independence — answers R1's exact ask** ("a novel regularization scheme or alternative objective function that might obviate the need for post-hoc tuning"): (1) the bias arises with ZERO regularization in the Ising, so no reg scheme removes it; (2) more deeply, no objective on insufficient data can recover the missing tail information. R1's proposed fix-class is looking in the wrong place.
- **(C′) Regularization-independence is itself a NOVEL theoretical contribution — the lead answer to R1's "no advance" critique.** That temperature tuning arises WITHOUT regularization was not established before this work; prior literature (Russ et al.) attributed the T=1 deviation, at least in part, to regularization for limited sampling. Isolating the effect in a fully unregularized MLE Ising fit (no L2, no prior, no pseudocount) dissociates the phenomenon from regularization and pins its origin to finite-sample MLE under a large energy gap. Frame as "to our knowledge, new." Lead with the Ising (toy model has a pseudocount, so the clean claim rests on the Ising).
- **(D) Heating (τ>1) regime relevance — KEEP IT SIMPLE (decided 2026-06-15; supersedes the criticality framing).** Don't invoke criticality or neural-coding functional claims (too nuanced, attack surface). Just: the framework equally predicts τ*>1 when the fit UNDER-estimates the aggregate high-energy (excited-band) occupancy — the mirror of the over-estimation that drives cooling. This is the regime where M and Δ/T are BOTH small (opposite corner of the cooling condition M ≪ n_g e^{Δ/T}). Whether any given system (biological or not) sits there is an EMPIRICAL question — the exact onset is fit-dependent (Eq. tau-star-exact), so no clean a-priori threshold to promise. Keeps the burden on R1: the framework says *when* to raise; "marginally relevant" is his unsupported assertion.
  - Minimal formulation (use ~verbatim): *"The framework equally predicts the opposite regime: when the fitted model under-estimates the occupancy of high-energy states, the optimal sampling temperature exceeds 1 (Appendix X). This arises when the sample size M and the gap-to-temperature ratio Δ/T are both small. Whether a given system falls in this regime is an empirical question."*
  - **DROPPED (do not reintroduce):** the neural-criticality / diverging-heat-capacity angle (Tkačik, Mora–Bialek, Schwab–Nemenman–Mehta) and any "helps neural computation" functional claim — too nuanced for the reply, more attack surface than payoff.
- "Formalizing the physics of a universal heuristic is a contribution in the stat-mech tradition" — stand the ground on contribution type.
- **Local criterion**: sign of ∂D/∂τ at τ=1 (Appendix C) is a *local* prescriptive criterion that doesn't require global knowledge of p — currently under-sold in main text.
- Heat-capacity appendix (new) addresses the prescriptive critique directly: C is computable from the model alone.
- The τ > 1 regime is a genuine counterintuitive prediction, not confirmation.
- Diagnostic value: τ* probes properties of the true distribution (gap, density of states, sample size).

## Internal note — why lowering τ has a clean condition but raising doesn't (NOT for the paper, 2026-06-15; Peter judged it too nuanced to include)

Conceptual understanding only — banked in case R1/a reader asks why only one direction is sharply characterized.
- **Lowering = a one-sided scale crossing.** Cooling condition M ≪ n_g e^{Δ/T} ⟺ 1/M ≫ e^{−Δ/T}/n_g, i.e. the empirical resolution floor (smallest representable prob, 1/M) is coarser than the true per-high-energy-state probability. Then the fit rounds the rare tail states it samples *up* (assigns ~1/M ≫ true weight) and misses the rest → systematic over-estimation of tail mass → lower τ. Finite sampling can only round rare states up or miss them (one-sided, monotone) → single scale crossing → closed form. (Equivalently: expected tail samples M·b ≪ n_e.)
- **Raising has no clean analog, two reasons.** (1) τ* minimizes the *reversed* KL D(q‖p), weighted by q, so it's dominated by where the fit *over*-covers (the tail — the cooling effect) and is nearly blind to *under*-coverage (q≈0 where p has mass), which is exactly what raising fixes. (2) Raising only wins in the small-Δ corner, where it's a balance of comparable misassignment terms (the Eq. tau-star-exact bracket vs Δ̂ n̂_e/Δ) netting out — no single dominant scale → no closed form, only the qualitative "small Δ and small M."
- **Consistency check:** this is why τ′ (forward KL, weighted by p) "is always raised, if at all" — forward KL is dominated by *under*-coverage (missed low-energy states), the mirror failure mode. Each KL gets a clean one-sided criterion for the failure mode it weights, and a messy corner for the other.
- Related: the exact toy-model τ* is Eq. (tau-star-exact) (`si.tex` ~L523); it is necessarily expressed in FIT quantities (Δ̂, n̂_e, n̂_g, L̂·L), not pure (M, n_g, n_e, Δ) — which is itself quiet support for R1 defense (A): a data-free a-priori criterion may not exist.

## Internal note — temperature tuning as typical-set realignment (NOT for the paper, 2026-06-16; bank as understanding / possible follow-up)

Elegant unifying view, but kept OUT of the revision (too nuanced; leaning on the typical set tugs back toward the broad-generality framing being softened to proteins). At most an optional one-sentence discussion line: *"temperature tuning can be read as realigning the model's typical set with the data distribution's."*
- τ sets **where the model's typical (sampled-energy) band sits**: raising τ flattens q_τ → typical band moves up toward higher-energy, more-numerous states; lowering τ → down toward the mode/low-energy.
- The finite-data fit misplaces q's typical set relative to p's; **τ\* realigns them in the energy coordinate.**
- **Lower-τ regime:** fit's typical set extends *above* the truth's (over-covers high-energy "noise" p never visits) → cool to pull the band down.
- **Raise-τ regime:** fit's typical set sits *too low* — over-concentrated near the mode (high density, low entropy), under-representing diversity → heat to push the band up. This IS the density-vs-typicality tension (Nalisnick/AEP: samples should land in the entropic typical set, not the max-density mode); an over-concentrated fit "stuck near the mode" needs heating.
- Ties to the [[lower-vs-raise asymmetry note above]]: reversed KL (q-weighted) sees over-coverage of the tail (cooling, clean threshold); raising fixes under-coverage, the off-weighted residual.

## File locations
- Manuscript: `~/Git/temp_tuning_draft/main_arxiv_comments_cleaned.tex`
- SI: `~/Git/temp_tuning_draft/si.tex`
- Reply letter: `~/Git/temp_tuning_draft/reply.tex`
- Cover letter: `~/Git/temp_tuning_draft/Fields_PRXLife_cover_letter.docx.pdf`
- Code repo: `~/Git/temp-tune/` (Julia, `dev` branch, 2 commits ahead, uncommitted work in `notebooks/model_heat_capacity_and_tau.ipynb` — likely the heat-cap appendix material)
- Strategy doc on disk: `~/Documents/markdown/revision_strategy_notes.md`

## Authors
- Peter Fields (first author, U. Chicago PhD student) — primary collaborator
- Vudtiwat Ngampruetikorn, DJ Schwab, SE Palmer (advisor, has approved strategy)

## Response letter format (APS)
> **Referee X, Comment Y:** [quote or paraphrase]
> **Response:** [reply, referencing specific manuscript changes]

Related: [[user_profile]], [[research_ideas]]

---

## Raw referee reports (verbatim — for quoting in reply.tex)

### Referee 1 (VM10051W/Fields)

Generative modeling, particularly within protein design, relies heavily on post-hoc temperature tuning to sequester functional sequences from trained energy-based models. Fields and colleagues present a statistical mechanical derivation aimed at formalizing this ubiquitous heuristic. By distinguishing between the forward Kullback-Leibler divergence used in maximum likelihood estimation and the reversed divergence relevant to generation, the authors successfully isolate the source of sampling bias: the systematic overestimation of entropy in high-energy modes when training on sparse data.

The theoretical framework is rigorous. The use of toy systems and nearest-neighbor Ising models effectively illustrates the interplay between sample size M, the energy gap Δ, and the resulting shift in the optimal sampling temperature τ∗. Validating that the standard practice of scalar energy rescaling, q̂ ∝ exp[−E/τ], serves as a mathematically principled correction for finite-sample variance provides a satisfying physical justification for an operational ansatz.

A significant concern regarding the impact of this work arises from its practical implications. The manuscript essentially confirms that the status quo—applying a heuristic temperature scale—is the correct approach to mitigate sparse data artifacts. While providing a "kosher" theoretical basis for an existing method has pedagogical value, it does not necessarily constitute a methodological advance of the magnitude typically expected for Physical Review Research.

The study validates the tool practitioners already use without offering a novel regularization scheme or an alternative objective function that might obviate the need for such post-hoc tuning.

The central limitation lies in the operational utility of the proposed diagnostic framework. Determining the optimal temperature τ∗ requires evaluating the reversed KL divergence against the ground truth distribution p. In de novo design tasks, which the authors cite as their primary motivation, p is inaccessible. Consequently, the framework explains why an optimal τ exists but fails to provide a calculable criterion for selecting it a priori. Practitioners remain dependent on the very trial-and-error methods the study seeks to rationalize.

The counter-intuitive finding that raising the temperature (τ>1) can optimize performance offers theoretical nuance, yet its applicability appears constrained. As the authors acknowledge, biological systems such as protein families typically inhabit the "large gap" regime where functional states are rare. The "heating" regime applies primarily to systems with small energy gaps or extremely low sampling counts, rendering this insight marginally relevant to the specific biological application domain driving the research.

In summary, the manuscript successfully formalizes the relationship between data sparsity and temperature tuning as a trade-off between heat capacity and the susceptibility of the probability mass. The work elevates a heuristic to a principled correction. Yet, without a mechanism to estimate the governing parameters absent omniscient knowledge of the density of states, the study remains descriptive rather than prescriptive.

### Referee 2 (VM10051W/Fields)

In this manuscript, the authors propose a theoretical interpretation of a so far unexplained feature of energy-based models for proteins. It was found experimentally that lowering temperature is required to sample functional proteins and to approximate the natural distribution of energies, while in principle, the learned distribution should best approximate the natural one at temperature 1. Here, the authors show that if the model is learned from sparse data and comprises a substantial energy gap between favorable and unfavorable states, it tends to overestimate the probability of high-energy states. Lowering temperature then allows for correcting this bias. Interestingly, the authors also show that in some cases, increasing temperature can be useful, and they suggest that temperature tuning can reveal properties of the true data distribution.

This manuscript is presented in an extremely clear way and is pleasant to read. The core idea is new and explains an intriguing empirical observation. Two models are analyzed, a toy model and a small Ising model, and conclusions are robust. The theoretical analysis, based on information theory, is rigorous and clear. Thus, I recommend publication, after the authors have addressed my comments and questions.

**Major points:**

1- In Ref. [38] it was suggested that the imperfections of the distribution at T=1 were connected to the limited amount of training data via the regularization process: "This deviation is, among other factors, due to statistical adjustments [regularization (see materials and methods)] used during model inference for compensating for the limited sampling of sequences in the input MSA.". How does this connect to the findings presented here? It would be good to discuss this point explicitly, and also to discuss the impact of regularization in more detail. Besides, here, the authors are using a pseudocount, and this somewhat differs from Ref. [38]. It would be good to comment on this.

2- The paper is about Ising and Potts models, but several other types of generative models are now employed for protein design. It would be good to comment on this in the Discussion, and to address whether the findings presented here could also give insight on these other models.

**Minor points:**

1- Introduction, "Energy-based models trained on evolutionary data can now generate novel protein sequences with custom functions [38].": In the context of protein design, "custom functions" is too strong, and could be interpreted as de novo design. In Ref. [38], sampling from one existing protein family was performed, with function (enzymatic activity) in line with natural proteins of that family. This is quite different from a model that could generate proteins with custom functions in general. Please tone this sentence down.

2- Introduction, "Many such systems possess a large 'energy gap' that separates a small region of meaningful, low-energy states from a vast space of high-energy, noisy, improbable ones.": This makes sense, but it would be good to cite references that support this statement.

3- Section II A, "It has been noted elsewhere that including DKL(q||p) (or a proxy for it) in the objective function causes the fit model to focus more strongly on modes of the data distribution, as opposed to using DKL(p||q) alone": Expanding a little bit more on this point might be helpful for the reader.

4- Section II A, "Furthermore, we can see from Fig. 1(b) that when we consider the ratio between these two quantities, an optimal value of τ clearly exists (inset) and that the minimum difference at τ∗ corresponds to the minimum of Eq. (1).": Please clarify if this is general and proved (and in that case, please include the proof), or if this is an empirical observation (and in that case, please explain in what context).

5- Section II B, "Δ → ∞": Does it need to be that strong? Or would large but finite Delta also work? In the latter case, how large does Delta need to be and with respect to what?

6- Section II B, "N_s is the total number of states and fixed a priori.": Is N_s assumed to be known? If yes, to what extent is this a limitation?

7- Fig 2(b): I got confused by the DKL histogram and the arrows that connect it to the probability one. From the text and the arrows, it looks like the left bar on the DKL histogram corresponds to high energy (low probability) states and not low energy ones, and the other two bars correspond to missed or found low energy states, but the x-axis label appears to state the opposite. Please clarify this.

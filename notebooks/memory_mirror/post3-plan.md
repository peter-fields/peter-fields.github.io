# Post 3 Plan (from Mar 2026 conversation)

## Status
Experiments done (Mar 2026). See `notebooks/post3_feature-upgrade/scratch/exp_notes.md` for full results.
Post not yet written.

Scripts: run_exp1_3.py through run_exp16_varv.py (16 experiments total)
Caches: outmag_ioi/non.npy (1000×144), varv_ioi/non.npy (1000×144), X_ioi/non.npy (1000×156)

## Final pipeline (Exp 15, fully principled)
1. Permutation test on outmag_non (200 shuffles, 95th pct null max eigenvalue) → K_non=9
2. ICA on outmag_non (20 seeds, Hungarian matching) → A_non: independent NC patterns
   **Why ICA not PCA for projection**: PCA eigenvectors of C_non have loadings on circuit heads
   (layer-depth confound); ICA finds patterns active as independent sources on non-IOI only —
   IOI-specific heads (NM, BNM, SI, NegNM) are low/noisy on non-IOI so don't appear in A_non
3. QR + project outmag_ioi onto complement → outmag_ioi_resid (30% variance retained)
4. Permutation test on residual → K_ioi=12
5. ICA on residual (20 seeds, Hungarian matching) → A_ioi: IOI-specific independent components
6. Evaluate: ROC-AUC with score=|loading|. Range 0.664–0.772, all above chance (0.500)
   Best component AUC=0.772. No circuit labels used anywhere in pipeline.

## Exp 16: Var_v comparison
Same pipeline on varv_ioi/non.npy. AUC range 0.567–0.742 (K_ioi=13, variance retained 74.5%)
out_mag wins: 0.664–0.772 vs 0.567–0.742. Confirms out_mag is right primary feature.

## Future: multi-feature ICA
Stack out_mag + KL (or + Var_v) → (1000×288) joint feature matrix. ICA components would
capture cross-feature dependencies (e.g., KL of head A correlated with out_mag of head B).
Needs KL cached for 1000 prompts × 144 heads (not done yet).

## Experiment Results Summary

### Primary finding: out_mag > Var_v
- **out_mag = ||µ_v||²/d** (squared magnitude of the attention-weighted mean value vector)
  = what the head actually writes to the residual stream
- Δout_mag discriminates circuit from NC heads: **30x ratio, p=1.2×10⁻⁵** (vs 12x for Var_v)
- out_mag catches all Var_v catches + L4H11 (PT head), Var_v catches nothing extra
- **Use out_mag as the primary feature going forward, not Var_v**

### What Δ-based diagnostics miss
- Both Var_v and out_mag miss: L0H1, L2H2, L3H0 (DT/PT), L5H5, L6H9 (Induction)
- Reason: these are **always-on** heads — they do their job on all prompts, not specifically IOI
- Δ ≈ 0 because they're equally active in both conditions
- This is an honest limitation to state in the post

### corr(ΔKL, ΔVar_v) = −0.87 (not independent!)
- Var_v = E_π[||v||²] − ||µ_v||², so concentrated attention mechanically reduces Var_v
- (KL, Var_v) is NOT a 2D independent plane — use out_mag as single best value-side feature

### Differential correlation matrix C_diff = C_ioi − C_non — WORKS
- **C_diff at 5σ threshold: 23.5% CC vs 2.6% chance = 9x enrichment**
- Top CC edges are mechanistically correct: SI↔BNM, BNM↔NM, BNM↔NegNM
- Per-head ranking (mean positive C_diff): top-20 has 7/20 circuit heads (2.3x chance)
- Precision improves monotonically as threshold tightens — not noise at the edges
- Key: do NOT z-score within conditions before computing C_diff; the mean shift is the signal
- Precision matrix (C⁻¹) is wrong tool; correlation matrix difference is right
- **Use 5σ threshold network (34 edges) as the post figure**

### Exps 10–14: PCA-based pairwise methods — ceiling 6/8 circuit

**Exp 10** (IOI-specific eigenvectors): eigendecompose C_ioi; filter by low overlap with C_non top-10.
Evec 8 (6/8 circuit): best PCA result. NM/BNM competition at layers 9–10.
Evecs 6 (3/8, SI→BNM) and 9 (4/8, NegNM suppression) also informative.

**Marchenko-Pastur threshold**: K=9 C_non eigenvalues above bulk max 1.903 (N=1000, P=144).
Those 9 modes capture 85.3% of C_non variance.

**Exp 11** (contrastive PCA, generalized eigenvalue): doesn't work. Amplifies NC heads in quiet
C_non directions. Top modes dominated by L8H5, L5H2, L3H3 (NC).

**Exp 12–13** (projection approaches): Project C_non's top-K modes out of C_ioi eigenvectors
(Approach A) or out of C_ioi matrix before decomposing (Approach B, theoretically better).
K=6 slightly too few; K=9 slightly too aggressive (removes some circuit signal).
Best: Approach B K=6 evec 2 → 5/8. Does not beat raw Exp 10 evec 8 (6/8).
Reveals hidden circuit signal in high-overlap evecs (evec 3: 0→5/8, evec 5: 0→6/8) but
at tiny loading amplitudes.

**Exp 14** (ensemble voting, 25 voters): precision 67% at highest threshold (3 heads, 4.2x chance).
NC intruders L10H5, L11H6, L6H1 persistently tied with circuit heads. Voters not independent.

**PCA ceiling**: ~6/8 in top component. NC heads in layers 9–11 co-activate with circuit heads
due to layer-depth confound. Projection removes some but not all. Not broken by voting.

### Exp 15: Contrastive ICA — BEST RESULT OF SESSION (**7/8 circuit**)

**Why ICA beats PCA**: PCA finds orthogonal variance directions (no mechanistic meaning).
ICA finds statistically independent components — stronger. If sub-circuits activate
independently across prompts, ICA separates them. Non-Gaussianity of circuit activation
(bursty, sparse) helps ICA where PCA conflates.

**Why ICA on both conditions separately**: Running ICA on outmag_non finds the INDEPENDENT
NC source patterns (not just high-variance directions). Projecting these out of outmag_ioi
removes actual independent NC sources, leaving cleaner IOI signal for ICA step 2.

**Pipeline**:
1. FastICA on outmag_non (K=9, 20 seeds, Hungarian matching) → A_non: 9 independent NC patterns
2. QR-decompose A_non; project outmag_ioi onto orthogonal complement (30% variance retained)
3. FastICA on residual outmag_ioi (K_ioi=8 selected by stability × circ/8 score)

**Results**: All 8 components stable (>0.95). Mean 6/8 circuit. Best components: **7/8**.
- Comp 0 (7/8): layer-10 BNM cluster + NM L10H0 + NegNM L10H7
- Comp 7 (7/8): full late circuit — SI L8H6 + NMs L9H6, L10H0 + BNMs + NegNM L11H10
- NC intruders different from PCA (L11H0, L9H8, L8H3 vs PCA's L10H5/L11H6/L6H1)
- Always-on heads correctly get zero nominations

**Caveats**: FastICA convergence warnings at K_non=9; K_ioi=8 selected to maximize score;
70% IOI variance removed; ICA sign/permutation ambiguity handled by Hungarian matching.

**Figures**: `figs/exp15_ica_panel.png`, `figs/exp15_ica_ioi_00.png` through `_07.png`,
`figs/exp15_mixing_non.png`

### Open questions before writing post
1. Do the ICA components match Wang et al. causal sub-circuits?
2. C_diff (simpler) vs contrastive ICA (richer) — both in post or just ICA?
3. cICA paper (PNAS 2025): mention as related work, don't implement.

### Post 4 — undecided
Original plan (binary RBM + Kyle's MPF) reconsidered — may not be the right direction.
Candidates:
- **Causal validation**: activation patching on ICA-nominated candidates → closes the
  "observational pre-filter → causal verification" loop the post promises
- **Multi-feature ICA**: stack out_mag + KL → (1000×288), captures cross-feature dependencies
- **Generalization**: same pipeline on a different circuit or model
Decision deferred. Write Post 3 first, Post 4 direction will be clearer after.

### Revised Post 3 narrative
1. Open: Var_v is the value-side counterpart to KL — but they're mechanically linked (corr = −0.87), and out_mag is the cleaner quantity
2. Show out_mag discriminates late-circuit heads (30x, precision curve)
3. Honest: early-circuit heads (DT, PT, Ind) are invisible to Δ-based diagnostics
4. Close: for full circuit recovery need activation patching; these diagnostics are a cheap first-pass filter

## Narrative Arc
1. **Open**: χ wasn't the right upgrade — KL and χ are both query-key side (redundant). Need the value axis.
2. **Introduce Var_v**: empirically show it discriminates circuit vs non-circuit heads; show corr(ΔKL, ΔVar_v) << corr(ΔKL, Δχ) ≈ 0.70
3. **Factor analysis**: use log(Var_v) at last token as feature; Gaussian hidden units as first pass at circuit graph
4. **J_diff**: subtract non-IOI coupling from IOI coupling → circuit-specific graph
5. **Close**: name binary RBM + MI regularization as the Post 4 upgrade

## Feature Vector
- **One scalar per component per prompt**, computed at **last token position only**
  - Attention heads: Var_v at last token (query = last token attending over all source positions)
  - MLPs: Var_W_out at last token (same MLP weights for all positions; last token is where prediction is made)
- For GPT-2 medium: 78 attention heads + 12 MLPs = 90 features

## Why Last Token Only
- IOI prediction is made at the last token
- MLP weights are shared across all positions — last token is the relevant one for prediction
- Causal masking / position subtleties deferred to later posts

## Why log(Var_v)
- Var_v ≥ 0 by definition — need to map to ℝ for Gaussian factor model. Log is the natural choice.
- Compresses outliers: a spike to 100x typical value contributes O(value²) to covariance in original space, O(log(value)²) in log space — much less dominant. Covariance estimate (and J = C⁻¹) more stable.
- Ratios more natural than differences when quantity spans orders of magnitude (decibel analogy) — equal multiplicative changes = equal distances in log space
- **NOT**: "histogram of log(Var_v) should look Gaussian" — this doesn't tell you whether to use Gaussian vs binary hidden units. Schneidman retinal data: binary (spike/no-spike) marginals, yet Gaussian/Ising pairwise model works fine. Marginal shape ≠ hidden unit choice.
- **Hidden assumption of the log transform**: Ising coupling energy is Jᵢⱼ log(Var_v_i)·log(Var_v_j) — you're measuring correlated *multiplicative* changes, not absolute changes. Different transform = different coupling.

## Feature Selection Rationale
- χ dropped entirely: redundant with KL (both query-key side), corr(ΔKL, Δχ) = 0.70
- **Whether to include both KL and Var_v depends on corr(ΔKL, ΔVar_v)**:
  - If low correlation → include both — they carry independent information, worth the larger feature space
  - If high correlation → use Var_v only (it's the new quantity; KL already shown in Post 2)
- Feature vector if both included: 78 heads × 2 (log KL, log Var_v) + 12 MLPs × 1 (Var_W_out) = **168 features**
- Feature vector if Var_v only: 78 heads × 1 + 12 MLPs × 1 = **90 features**
- MLPs only get Var_W_out — no KL analog (no attention weights)
- Richer factor analysis with both: hidden units can load on KL axis of some heads and Var_v axis of others — captures that heads contribute to circuits through different mechanisms

## Why Include MLPs (reconsidered)
- MLPs don't do cross-token processing — but they DO process the residual stream after attention writes to it
- If IOI circuit writes IOI-specific content to residual stream, the downstream MLP sees different input → different Var_W_out
- MLP inclusion tells you which MLPs are computationally downstream of the circuit, even if not doing routing
- J_diff cancels the structural MLP-attention correlation (MLP always reads after attention in same layer) — only prompt-specific co-activation survives
- IOI null result from Post 2 still holds — MLPs probably weak signal for IOI specifically
- But including them is cheap and informative for other circuits (factual recall etc.) later

## Transformer Architecture Clarification (important!)
- Within a layer: all **attention heads are parallel** — they all read from the same LN(x_l) simultaneously
- **MLP is sequential**: reads from LN(x_l') where x_l' = x_l + attention_output — AFTER attention writes
- So attention heads within same layer share input → layer-depth confound for head-head correlations
- MLP and attention heads in same layer do NOT share input
- J_diff cancels layer-depth confound (shared correlations present in both J_act and J_null cancel out)

## Hidden Assumptions in the Gaussian Factor Model on log(Var_v)
1. **Coupling in log-space**: Jᵢⱼ measures correlation between log-values (correlated multiplicative changes), not absolute changes
2. **Gaussian conditionals**: P(log(Var_v) | h) is Gaussian — empirical check: are residuals after fitting well-behaved?
3. **h ~ N(0,I)**: circuit modes are continuous and symmetric. h < 0 means circuit *suppresses* Var_v below baseline — unclear if physically meaningful
4. **Additive in log-space**: no hᵢhⱼ interaction terms — circuits don't interact in the hidden layer
5. **W fixed across prompts**: circuit structure stable, only activation levels h vary per prompt
6. **Independent noise**: Λ diagonal — noise uncorrelated across heads
7. **Quadratic single-site potential**: ½Λᵢσᵢ² per spin — Gaussian marginal per head with precision Λᵢ, one parameter per spin, fitted by EM. Natural minimal choice; φ⁴ (double-well) would give bimodal marginals but breaks Gaussian inference.

## Binary vs Gaussian Hidden Units
- **NOT determined by marginal shape** of log(Var_v): Schneidman retinal neurons have non-Gaussian (binary) marginals yet pairwise Gaussian/Ising model captures joint statistics well
- **Gaussian hidden units**: simpler EM inference, continuous graded activation levels, good for distributed computation (many heads contributing weakly). Better when computation is NOT cleanly circuit-like.
- **Binary hidden units**: physically motivated (circuits on/off), induces higher-order effective couplings via log(1+exp(...)) terms, better when circuits are discrete and modular
- Decision: Gaussian for Post 3 (first pass, simple); binary RBM with Kyle's MPF for Post 4

## Factor Analysis Approach
- Fit factor model separately on activating and non-activating prompts:
  ```
  log(Var_v) = W h + ε,   h ~ N(0,I),   ε ~ N(0,Λ)
  ```
- Get W_act, W_null via EM (sklearn FactorAnalysis)
- J_diff = W_act Wᵀ_act − W_null Wᵀ_null → circuit-specific coupling matrix
- Sparse structure in J_diff = circuit candidate graph

## Why Not Gaussian Copula / Rank+Probit
- Copula approach: nonparametric rank transform → Φ⁻¹ → GGM — more steps, less interpretable
- log(Var_v) transform: parametric, physics-motivated, cleaner
- Both reduce to second-moment matching anyway; log transform is the right physicist move for positive-valued quantities

## Experiments Needed Before Writing
1. Compute Var_v for all IOI heads on the Post 2 prompt set
2. Test ΔVar_v (activating - non-activating): does it have significant p-value?
3. Compute corr(ΔKL, ΔVar_v) — should be much lower than 0.70
4. Check if Var_v adds discriminative power beyond KL in a simple classifier
5. Plot histogram of log(Var_v) across prompts — check approximate normality

## Fingerprint Clustering (hint in Post 3, develop later)
- Each head has a 2D fingerprint: scatter of (KL, Var_v) across n prompts
- Non-trivial shapes — the distribution shape encodes something about the head's function
- Heads doing similar things should have similar fingerprint shapes
- Could cluster heads by fingerprint shape similarity (e.g., distribution distance, covariance structure of the 2D scatter)
- **This is a different axis than J_diff**: J_diff clusters by co-activation across prompts; fingerprint clustering clusters by individual behavior pattern — complementary
- Mention briefly in Post 3 as a future direction, don't develop

## Connections
- Binary RBM upgrade (Post 4): Kyle's factored-MPF code + MI regularization between visible/hidden spins
  - MI regularization selects most significant circuit modes first (greedy, information-theoretic PCA analog)
  - Solves rotational ambiguity: hidden units that maximize I(v;h_j) and minimize I(h_i;h_j) pin down a canonical basis
  - Binary hidden units more natural than Gaussian: circuits are on/off
- Academic lineage: Peter + Kyle → Stephanie Palmer → Bill Bialek (retinal MaxEnt, protein DCA)
  - Same toolkit applied to transformer circuits
  - Dario Amodei also Bialek lineage (fun coincidence for Anthropic application)

## Further Future Directions (backburner)
- **Distributed computation**: if model computation is NOT cleanly circuit-like (many heads contributing weakly, overlapping), Gaussian hidden units may be better than binary — they capture graded, distributed patterns. Binary RBM better for modular, crisp circuits. Hidden unit type choice encodes a prior on computational structure.
- **Deep RBM for interacting circuits**: if circuits interact (circuit A tends to trigger circuit B), need a second hidden layer:
  - Layer 1 hidden: basic circuits (IOI, induction, greater-than)
  - Layer 2 hidden: meta-circuits (patterns of circuits co-activating)
  - Kyle's MI regularization naturally extends layerwise
  - Relevant for complex multi-step reasoning tasks
- **φ⁴ / double-well potential**: quartic single-site term gives bimodal marginals, connects to Landau / RG theory of phase transitions. Very far future.
- **Replacement model / CLT for composable circuits**: once circuits can be reliably mapped and shown to compose, there may be some effective theory (analogy to CLT / renormalization) that describes the composite behavior without tracking all components. Very speculative, much later.

# Post 3 Experiment Notes

## Status (Mar 2026)

Scripts: run_exp1_3.py, run_exp4.py, run_exp5_fa.py, run_exp6_k6.py,
         run_exp7_ising.py, run_exp8_outmag.py, run_exp9_corr.py,
         run_exp10_evecs.py, run_exp11_cpca.py, run_exp12_resid.py,
         run_exp13_proj.py, run_exp14_vote.py, run_exp15_ica.py

Caches: X_ioi.npy, X_non.npy (log Var_v, 1000×156)
        varv_ioi/non.npy (1000×144), outmag_ioi/non.npy (1000×144)

---

## Key findings

### Exp 1–3: Var_v discriminates circuit heads
- |ΔVar_v| distinguishes circuit from non-circuit heads: ratio 12x, p=2.5×10⁻⁵ (Mann-Whitney)
- corr(ΔKL, ΔVar_v) = −0.87 — NOT independent axes. Var_v = E_π[||v||²] − ||µ_v||², so
  concentrated attention (high KL) mechanically reduces Var_v. Both measure the same thing.
- corr(ΔKL, Δχ) = 0.69 — χ and KL also correlated (both query-key side), but less so
- Conclusion: (KL, Var_v) is not a 2D independent plane; need a single best feature

### Exp 4: Unified 156-feature bar chart
- Feature vector: log(Var_v) per attention head (144) + log(Var_act) per MLP (12) at last token
- MLP signal real but ~16x smaller than attention heads
- Bar chart clearly shows circuit heads (layers 7–11) standing out

### Exp 5–6: Factor analysis (low-rank J_diff) — doesn't work
- Ran FA separately on IOI and non-IOI features, J_diff = W_ioi Wᵀ_ioi − W_non Wᵀ_non
- k=6 selected (largest elbow). CC enrichment 3.5x chance (7.8% vs 2.2%). Modest.
- Why FA is conceptually wrong: the circuit fires on ALL IOI prompts — it's a stable mean
  shift, not something that varies across IOI prompts. FA on X_ioi captures non-circuit
  variation across prompts (names/templates noise), not circuit structure. Low-rank abandoned.

### Exp 7: Full-rank precision matrix (J = C⁻¹) — doesn't work
- Tried: raw C⁻¹ pooled; Ledoit-Wolf + standardize pooled; Ledoit-Wolf separate conditions
  (J_diff = J_ioi − J_non). All NC-dominated, 0–2 CC edges at any threshold.
- Root cause: circuit shows up as a MEAN SHIFT in features (ΔVar_v large), not as a change
  in conditional-independence structure. After z-scoring, mean shift is gone; C⁻¹ captures
  only conditional independence → wrong tool for mean-shift signals.

### Exp 8: Output magnitude out_mag = ||µ_v||²/d — BEST SINGLE-HEAD DIAGNOSTIC
- out_mag = squared magnitude of attention-weighted mean value vector = what the head writes
- **out_mag beats Var_v**: ratio 30x (p=1.2×10⁻⁵) vs 12x (p=2.5×10⁻⁵)
- out_mag catches everything Var_v does + L4H11 (Previous Token head)
- Both miss same 5 early-layer circuit heads: L0H1, L2H2, L3H0 (DT/PT), L5H5, L6H9 (Ind)
  These are ALWAYS-ON heads — they do their job on all prompts, not just IOI.
  Δ ≈ 0 because equally active on non-IOI. Fundamental limitation of Δ-based diagnostics.
- Figures: exp8_delta_comparison.png, exp8_precision_curve.png

### Exp 9: Differential correlation C_diff = C_ioi − C_non — WORKS for pairwise structure
- Correlation of out_mag computed separately on IOI and non-IOI, then differenced
- Do NOT z-score within conditions — the mean shift IS the signal
- At 5σ threshold: **23.5% CC vs 2.6% chance = 9x enrichment**, 34 edges
- Top edges mechanistically correct: SI (L8H6) ↔ BNM (L9H0), BNM ↔ NM (L9H6),
  BNM ↔ NegNM — exactly the late-circuit co-activation structure from Wang et al.
- Per-head score (mean positive C_diff): top head L9H0 (BNM, 0.163). Top-20 has 7/20
  circuit heads vs 3/20 expected (2.3x enrichment)
- Why this works: circuit heads all high on IOI, low on non-IOI → they co-vary together
- Figures: exp9_corr_diff.png, exp9_network_diff_5.0s.png, exp9_precision_curve.png

### Marchenko-Pastur threshold for K
- For N=1000 samples, P=144 features: bulk max = (1+√(P/N))² = (1+√0.144)² ≈ 1.903
- **K=9** C_non eigenvalues exceed this threshold; remainder is sampling noise
- Those 9 modes capture 85.3% of C_non variance = the NC layer-depth co-activation structure
- Principled choice for how many C_non modes to treat as "structural noise"

### Exp 10: IOI-specific eigenvectors of C_ioi
- For each C_ioi evec v_k: overlap = ||V_non_10ᵀ v_k||² with C_non top-10 subspace
  High overlap = structural NC mode. Low overlap = IOI-specific.
- Evecs 0–5 (overlap 0.94–0.97): pure NC structural modes, 0–2 circuit heads in top-8
- **Evec 6** (overlap=0.44, 3/8 circuit): SI (L8H6) negative, late BNMs positive → SI→BNM
- **Evec 8** (overlap=0.76, **6/8 circuit**): NM/BNM competition at layers 9–10.
  Best raw PCA result. L10H0 (NM)+, L10H6 (BNM)−, L10H1 (BNM)+, L9H9 (NM)−, L9H0 (BNM)+
- **Evec 9** (overlap=0.22, 4/8 circuit): NegNM (L10H7)+ strongly, NMs/BNMs negative
- Figures: exp10_overlap.png, exp10_evec_06.png, exp10_evec_08.png, exp10_evec_09.png

### Exp 11: Contrastive PCA (generalized eigenvalue C_ioi v = λ C_non v) — doesn't work
- Finds directions where IOI variance / baseline variance is maximized
- Best circuit enrichment: 4/8 at k=9 — worse than raw Exp 10 evec 8 (6/8)
- Problem: normalization amplifies any direction where C_non is quiet, including NC heads
  in quiet baseline directions. Top evecs dominated by L8H5, L5H2, L3H3 (all NC).
- Root cause: IOI and non-IOI prompts differ structurally (2 names vs 3 names), so many
  NC heads vary differently between conditions for purely syntactic reasons.

### Exp 12: Project C_non top-K from C_ioi eigenvectors (K=6) — partial improvement
- For each C_ioi eigenvector v_k: residual = (I − V_non_6 V_non_6ᵀ) v_k
- Evecs 3 and 5 (raw 0/8) → residuals **5/8 and 6/8** circuit after projection!
  Circuit signal buried under dominant NC structural modes — projection reveals it.
- BUT residual loadings for evecs 3,5 are tiny (max ~0.07). Circuit is a small perturbation
  on a large NC structure in those dominant evecs.
- Evec 8 unchanged (already orthogonal to C_non top-6)

### Exp 13: Approach A (K=9) and Approach B (project matrix first) — principled comparison
- **Approach A (K=9 MP threshold, decompose first, strip C_non per-vector)**:
  - Evec 3 residual 5/8, evec 7 residual 5/8. But evec 8 drops 6/8→4/8 at K=9.
  - K=9 slightly too aggressive: extra C_non modes λ=2.1–3.0 partially overlap circuit signal.
- **Approach B (project C_non out of matrix first, then decompose) — theoretically better**:
  - Removes NC structure before eigendecomposition; eigenvectors computed on right matrix.
  - K=6: evec 2 → 5/8 (NM/BNM competition, loadings ~0.38, same story as Exp 10 evec 8)
  - K=9: evec 6 → 5/8 (NegNM mode)
  - Neither beats Exp 10 raw evec 8 (6/8).
- **Ceiling of PCA-based approaches**: ~6/8, limited by layer-depth confound. NC heads in
  layers 9–11 co-activate with circuit heads and survive all projection-based filtering.

### Exp 14: Ensemble voting across all eigenvectors — modest improvement
- 25 voters (raw evecs, Approach A residuals, Approach B K=6+9, C_diff score), top-10 per voter
- Precision at ≥1 vote: 33% (2.1x chance). At ≥5 votes: 50% (3.1x). At ≥11 votes: 67% (4.2x)
- Persistent NC intruders: L10H5 (11 votes), L11H6 (10), L6H1 (10) — tied with circuit heads
- Always-on heads (DT, PT, Induction) correctly get near-zero votes — method knows limits
- **All voters are derived from same C_ioi/C_non matrices — not truly independent**
- Does not break through layer-depth ceiling; consolidates existing knowledge

### Exp 15: Contrastive ICA — BEST PAIRWISE RESULT OF THE SESSION

**Why ICA beats PCA**:
- PCA finds orthogonal variance directions — orthogonality is a statistical convenience with
  no mechanistic meaning. Two sub-circuits can overlap in head activation; PCA mixes them.
- ICA finds statistically INDEPENDENT components — no higher-order dependencies. If IOI
  sub-circuits activate independently across prompts (which they should), ICA recovers them.
- Non-Gaussianity of circuit activation (bursty, sparse across prompts) helps ICA separate
  sub-circuits that PCA would conflate.

**Why ICA on BOTH conditions separately is better**:
- Run ICA on outmag_non first → find the INDEPENDENT NC source patterns (not just high-variance)
- Project outmag_ioi onto complement of those independent NC sources
- Run ICA on residual → find IOI-specific independent components
- Better than PCA because step 1 removes actual independent NC sources, not just high-variance
  directions. If NC co-variation comes from multiple independent sources (which it does —
  different NC heads have different functional roles), ICA separates them and removes each.
  PCA removes their mixture (the top variance directions), leaving more NC contamination.

**Final pipeline (fully principled)**:
1. Permutation test on outmag_non (200 shuffles, 95th pct null max eigenvalue) → K_non=9
   FastICA on outmag_non (K_non, 20 seeds, Hungarian matching) → A_non (144×9)
   Stability: 0.53–0.94 across components.
2. QR-decompose A_non; project outmag_ioi onto orthogonal complement
   → outmag_ioi_resid (1000×144): 30% variance retained after removing NC patterns
3. Permutation test on outmag_ioi_resid → K_ioi=12
   FastICA on outmag_ioi_resid (K_ioi, 20 seeds, Hungarian matching) → A_ioi (144×12)

**Why ICA projection is safer than PCA projection for step 2**:
- PCA top eigenvectors of C_non are broad late-layer directions (layer-depth confound) with
  loadings on circuit heads too — projecting them out removes circuit signal indiscriminately
- ICA finds patterns that are active as coherent independent sources on non-IOI prompts
  specifically. IOI-specific heads (NM, BNM, SI, NegNM) are low/noisy on non-IOI prompts —
  they don't form strong independent sources in outmag_non — so they don't appear in A_non
  and don't get projected out. ICA projection is more targeted.

**Results (K_ioi=12 from permutation test)**:
| Comp | Stability | AUC   |
|------|-----------|-------|
| 0    | 0.928     | 0.742 |
| 1    | 0.987     | 0.688 |
| 2    | 0.995     | 0.713 |
| 3    | 0.995     | 0.664 |
| 4    | 0.624     | 0.705 |
| 5    | 0.987     | 0.735 |
| 6    | 0.942     | 0.677 |
| 7    | 0.992     | 0.688 |
| 8    | 0.987     | 0.743 |
| 9    | 0.742     | 0.739 |
| 10   | 0.836     | 0.772 |
| 11   | 0.761     | 0.709 |

**Key numbers**:
- AUC range: 0.664–0.772, all well above chance (0.500)
- Best AUC: comp 10 (0.772) — note: top-8 metric would have picked comp 9 (7/8) — AUC is better
- Metric: ROC-AUC with score = |loading|, positives = known circuit heads
- No circuit labels used anywhere in pipeline (K from permutation test, eval from AUC post-hoc)

**Honest caveats**:
- FastICA convergence warnings for K_non=9 baseline — may not be global optimum
- 70% of IOI variance removed — aggressive; some circuit signal may be lost
- ICA has sign/permutation ambiguity (addressed by Hungarian matching)
- ~1–3 NC intruders per component remain; layer-depth confound not fully eliminated
- AUC used for post-hoc evaluation only, not for selecting K

### Exp 16: Contrastive ICA on Var_v — comparison with out_mag

Same pipeline as Exp 15, feature swapped to varv_ioi/non.npy.

**Results**: K_non=9, K_ioi=13. AUC range: 0.567–0.742. Best: 0.742.
Variance retained after projection: **74.5%** (vs 30% for out_mag).

**Comparison**:
- out_mag: AUC 0.664–0.772 (K_ioi=12)
- Var_v:   AUC 0.567–0.742 (K_ioi=13)
- out_mag wins on both min and max AUC — consistent with its better single-head discrimination (30x ratio vs 12x)
- Var_v retains more variance (74.5% vs 30%) but still discriminates worse
- Confirms out_mag is the right primary feature for the post

**Future direction — multi-feature ICA**:
Instead of (1000×144) with one feature type, stack multiple features per head:
- e.g., out_mag + Var_v → (1000×288), or out_mag + KL → (1000×288)
- ICA components would live in joint feature space: a component could load on
  "KL of L9H6" AND "Var_v of L10H0" simultaneously
- Would capture cross-feature dependencies: if circuit head A's routing (KL) correlates
  with circuit head B's output magnitude (out_mag), that shows up as a single component
- This is genuinely different information from running ICA on each feature separately
- Requires computing KL (and optionally χ) for all 1000 prompts × 144 heads — not yet cached
- Natural next experiment once KL data is available

---

## What works for the post

**Primary story (single-head diagnostic)**:
1. out_mag discriminates late-circuit heads well (30x, p<1×10⁻⁵)
2. Precision curve: ranking by |Δout_mag| puts circuit heads first much better than chance
3. Honest limitation: always-on heads (DT, PT, Induction) invisible to Δ-based diagnostics

**Secondary story (pairwise structure)**:
4. C_diff at 5σ threshold: 9x CC enrichment, 34 edges, correct SI↔BNM topology — simple
5. Contrastive ICA: 7/8 circuit heads in best components, all 8 stable — richer decomposition

## What doesn't work / what to cut
- J_diff from precision matrices (signal is in mean, not conditional independence)
- Low-rank factor analysis on separate conditions (circuit is stable mode, not within-condition variation)
- Contrastive PCA / generalized eigenvalue (amplifies quiet C_non directions → NC-dominated)
- Raw C_ioi PCA alone: best 6/8, NC intruders from layer-depth confound persist
- Ensemble voting: good sanity check but all voters correlated; doesn't beat the ceiling

## Fundamental ceiling of purely observational methods
NC heads in layers 9–11 co-activate with circuit heads because attention reads from the
residual stream, which is affected by earlier circuit activity regardless of whether the NC
head is doing anything circuit-relevant. No correlation-based method fully removes this.

Contrastive ICA mitigates it (independent sources, not just correlated ones) but doesn't
eliminate it. True floor for observational methods: ~6–7/8 circuit in best components,
~3–4x enrichment in aggregate. Clean isolation requires causal intervention (activation patching).

## Open questions for next session
1. Do the 8 ICA components match identifiable sub-circuits in Wang et al.'s causal graph?
   Comp 3 (SI+ NegNM−) could be the SI→NegNM pathway; comp 7 (all roles) could be the
   full late circuit firing together. Worth checking against the Wang et al. Figure 2.
2. Is C_diff (simpler, no ICA) sufficient for the post, or does contrastive ICA add enough
   to justify its complexity?
3. The always-on heads getting zero ICA votes is a clean validation — the method correctly
   knows what it can and can't see. Worth highlighting.
4. K_non=9 had convergence warnings. Try K_non=6 (less aggressive) and see if results hold.

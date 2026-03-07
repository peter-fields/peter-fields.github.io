# Post 2 Experiment Notes — Feb 2026

## What We Did
- Ran GPT-2-small (12 layers × 12 heads = 144 heads) via TransformerLens
- 50 IOI prompts (ABBA/BABA templates, varied names), 50 non-IOI prompts
- Computed per-head, per-prompt diagnostics at final token position:
  - **Normalized KL**: KL(π̂ ∥ u) / log n ∈ [0, 1]
  - **Normalized χ**: Var_π(log π) / (log n)² (temperature susceptibility)
  - **Excess ⟨z⟩**: ⟨z⟩_π − ⟨z⟩_u (shift-invariant expected score gain)
- IOI circuit head labels from Wang et al. 2022

## Prompt Evolution (important!)

We went through THREE iterations of non-IOI control prompts. This is itself a key lesson.

### v1: Generic sentences (bad control)
- Totally different text: "The cat sat on the mat", etc.
- Different lengths (11-14 tokens vs IOI's 15), different syntax, no names
- Produced dramatic effects: |ΔKL| p=0.0005, BOS dumping near-total (100%, ~98% weight)
- **Problem**: Signal was driven by comparing completely different text domains, not IOI-specific activation

### v2: Same structure, "the cashier/teacher" as subject (better but flawed)
- Same templates but second-clause subject = "the cashier", "the barista", etc.
- Eliminated syntax/domain confound
- **Problem**: "the cashier" tokenizes to 2 tokens → non-IOI = 16-17 tokens vs IOI = 15
- Results: |ΔKL| p=0.054 (borderline!), |Δχ| p=0.009
- BOS attention partial: Name Movers ~80-85% BOS, ~56% weight

### v3: Third name C as subject (maximally controlled) ← CURRENT
- "When {A} and {B} went to the store, {C} gave a drink to" — three distinct names, no repeats
- Exactly 15 tokens for both IOI and non-IOI (confirmed: all 100 prompts)
- The ONLY difference: IOI repeats a name (B or A) as subject; non-IOI uses a fresh name C
- No duplicates in either set (verified with assertions)

## Clean Summary of Empirical Results (v3 prompts)

1. **Circuit heads respond more to the IOI manipulation than non-circuit heads.** Both |ΔKL| (p=0.0002) and |Δχ| (p<0.0001) are significantly larger for circuit heads. The manipulation is minimal — the only difference between IOI and non-IOI prompts is whether one name repeats.
2. **Selection heads become more selective on IOI prompts.** ΔKL is consistently positive (+0.01 to +0.26) for Name Movers, Backup NMs, Negative NMs. A repeated name gives them something to lock onto.
3. **Structural heads don't care whether a name repeats.** ΔKL ≈ 0 (≤0.007) for Induction, Duplicate Token, Previous Token. Sentence structure is identical, so they do the same thing.
4. **S-Inhibition heads are mixed.** 3/4 slightly positive ΔKL; L7H3 slightly negative. Smaller effect sizes than selection heads. Suppression ≠ selection.
5. **On a given prompt type, KL and χ measure different things.** corr(KL, χ) = 0.34 across 144 heads on IOI prompts. A head can be very selective (high KL) with either stable or contested attention (low or high χ).
6. **But the shifts ΔKL and Δχ between prompt types are substantially correlated.** corr(ΔKL, Δχ) = 0.70. When KL shifts a lot, χ usually does too.
7. **Heads with the same circuit role have similar fingerprints in the (KL, χ) plane.** Name Movers cluster together, S-Inhibition in another region, structural heads in another. Cloud shape and position are visually consistent within roles (Fig 2d).
8. **Layer depth is a moderate confound.** corr(layer, KL) = 0.38, corr(layer, χ) = 0.38. Later layers tend higher. Most circuit heads are in layers 5-11.
9. **One circuit, one model.** No generalization claim possible yet.

## Detailed Findings (v3 results)

### Core findings
1. **KL is a useful per-head, per-context diagnostic** — directly from Post 1's theory, no controversy
2. **Both KL and χ clearly distinguish circuit from non-circuit heads** with proper controls:
   - |ΔKL|: circuit vs non-circuit **p = 0.0002** (mean 0.057 vs 0.013)
   - |Δχ|: circuit vs non-circuit **p < 0.0001** (mean 0.028 vs 0.008)
   - Both highly significant. v2's borderline |ΔKL| result (p=0.054) was an artifact of the length mismatch.
3. **χ vs KL redundancy — nuanced**:
   - corr(KL, χ) on IOI prompts across 144 heads = 0.34. Given a head's KL, you can't predict its χ well. They measure different things about a head's attention pattern on a given prompt type.
   - corr(ΔKL, Δχ) across 144 heads = 0.70 (circuit only: 0.60). When a head's KL shifts a lot between IOI and non-IOI, its χ usually shifts a lot too, in the same direction.
   - corr(|ΔKL|, |Δχ|) = 0.76. The magnitudes of the shifts are even more correlated.
   - **Bottom line**: χ adds real information about where a head sits in the (KL, χ) plane (its "fingerprint"), but the *response* to circuit activation is largely shared. Exceptions exist (L8H6: big ΔKL, negative Δχ) but are sparse.
4. **Fingerprints in (KL, χ) plane**: With matched axes (Fig 2d), heads with the same role occupy similar regions and have similar cloud shapes. Name Movers cluster upper-right; S-Inhibition lower-left; Induction/Dup Token pinned upper-left; Previous Token at extreme (KL≈1, χ≈0). The absolute position carries role information even though the deltas are correlated.
5. **BOS attention on non-IOI (v3)**: Name Movers attend to BOS ~80% of the time with ~46% weight. S-Inhibition heads mostly NOT BOS. Previous Token always attends to previous token. Induction and Duplicate Token heads → 100% BOS with high weight.
6. **Other correlations** (stable across versions):
   - KL vs ⟨z⟩_excess: r = 0.71 (⟨z⟩ mostly redundant with KL)
   - χ vs ⟨z⟩_excess: r = -0.12 (nearly independent)

### IOI prompt averages by circuit role
| Role | KL (mean±std) | χ (mean±std) |
|------|--------------|--------------|
| Name Mover | 0.612 ± 0.087 | 0.143 ± 0.021 |
| Backup Name Mover | 0.553 ± 0.074 | 0.206 ± 0.040 |
| Negative Name Mover | 0.494 ± 0.113 | 0.206 ± 0.012 |
| S-Inhibition | 0.416 ± 0.195 | 0.166 ± 0.034 |
| Induction | 0.877 ± 0.068 | 0.160 ± 0.062 |
| Duplicate Token | 0.833 ± 0.074 | 0.139 ± 0.012 |
| Previous Token | 0.753 ± 0.247 | 0.052 ± 0.052 |
| Non-circuit | 0.444 ± 0.228 | 0.175 ± 0.081 |

### Confounds investigated
6. **Context length**: ELIMINATED in v3. Both sets exactly 15 tokens. p=1.0 on Mann-Whitney.
7. **Layer depth**: corr(layer, KL) = 0.38, corr(layer, χ) = 0.38. Moderate confound remains. Most circuit heads are in layers 5-11; early-layer heads naturally behave differently. This is inherent to any circuit-level analysis.

### From earlier versions (still valid)
8. **The dramatic effects with generic prompts were a cautionary tale about controls** — the biggest lesson from the whole experiment. What looked like strong circuit activation signal was mostly domain mismatch.
9. **Previous Token (4,11) is purely structural** — always attends to previous token regardless of prompt type, with near-100% weight. A useful sanity check.

## What's Suggestive but Not Established
1. **S-Inhibition heads tend to have Δχ < 0 (stabilize on activation)** — 3/4 S-Inhibition heads show this (v2 data; needs re-verification on v3). Could reflect suppression being a "cleaner" operation. But n=4.
2. **Selection heads (Name Movers) tend to have Δχ > 0 (more contested on activation)** — consistent with competing candidates. But n=3 core name movers.
3. **χ distinguishes circuit roles** — the Δχ sign pattern is interesting but small sample, one circuit, one model.
4. **⟨z⟩_excess as an alternative axis** — r = 0.71 with KL (too correlated to add much), r = -0.12 with χ. Computed but mostly redundant with KL.

## What's NOT Established / Known Confounds
1. **Circuit vs non-circuit separation on a single prompt type** — NO clear separation. Non-circuit heads span the full (KL, χ) range on IOI prompts because they're doing their own (unknown) jobs.
2. **Layer depth confound** — corr ≈ 0.38 for both KL and χ. Moderate. Not disqualifying but should be acknowledged.
3. **Polysemantic interpretation of χ** — appealing but not demonstrated. High χ could mean polysemantic competition OR just a close race between candidates on a monosemantic task. Would need labeled polysemantic heads to test.
4. **Generalization beyond IOI** — one circuit, one model. Unknown if patterns hold for other circuits or architectures.
5. **"Criticality" claims** — explicitly decided NOT to go there. Too controversial, not supported by this data.
6. **~~Why KL drops on activation~~ RESOLVED** — It doesn't, with proper controls. KL INCREASES on activation for selection heads (Name Movers etc.), which is the intuitive direction: the repeated name gives them something to lock onto. The v2 "KL drop" was an artifact of the "the cashier" non-IOI prompts.

## Stat Mech Framing (safe version)
- KL = order parameter (selectivity)
- χ = susceptibility (sensitivity to temperature perturbation)
- Circuit activation is like an external field: ΔKL measures the response magnitude, Δχ reveals the character (toward order vs disorder)
- This is linear response language, NOT a criticality claim
- Phase transition language reserved for Post 3 (training dynamics) where it's more appropriate

## Key Figures (all regenerated with v3 prompts)
- **Fig 1**: Name Mover (9,9) vs non-circuit (1,1) on IOI prompts — scatter of per-prompt diagnostics
- **Fig 2**: Name Mover (9,9) active vs inactive — IOI vs non-IOI prompts
- **Fig 2b**: Grid of 10 heads (6 circuit, 4 non-circuit) active vs inactive with arrows — circuit heads show larger movement
- **Fig 2c**: All 4 S-Inhibition heads — checking if Δχ < 0 is consistent
- **Fig 3**: All 144 heads on IOI prompts colored by role — messy, no clean separation
- **Correlation plots**: KL vs χ (r=0.34), KL vs ⟨z⟩ (r=0.71), χ vs ⟨z⟩ (r=-0.12)
- **ΔKL vs Δχ scatter**: shows heads where χ moves but KL doesn't (and vice versa)
- **|ΔKL| and |Δχ| distributions**: circuit vs non-circuit histograms
- **Prompt lengths**: confirming exact match (all 15)
- **Layer depth**: KL and χ by layer, boxplots

## Blog Narrative Options
**Conservative (recommended)**: Derive diagnostics from Post 1's theory. Show that circuit heads respond differently to activating vs non-activating prompts — both |ΔKL| and |Δχ| significantly larger for circuit heads. Note χ provides genuinely independent information (r=0.34). Flag confounds honestly (layer depth, single circuit). Frame as "useful diagnostic tool + promising early results."

**Ambitious (risky)**: Full stat mech framing with order/disorder interpretation of Δχ sign. Claim functional fingerprinting (S-Inhibition stabilizes, Name Movers destabilize). Would need more circuits and models to support.

## Notebook Location
- Final (published): `notebooks/post2_attention-diagnostics/final/attention_diagnostics_peter.ipynb`
- Figures: `notebooks/post2_attention-diagnostics/final/figs/` AND `assets/images/posts/attention-diagnostics/` (6 PNGs: fig1–fig6)
- Scratch (never publish): `notebooks/post2_attention-diagnostics/scratch/`

## Next Steps: Empirics First, Interpretation Later
**Priority: map out all clear empirical signatures before trying to explain them.**

### Experiments completed
- ✅ Three versions of non-IOI prompts (generic → "the cashier" → third name C)
- ✅ |ΔKL| circuit vs non-circuit: **p=0.0002** (v3, length-matched)
- ✅ |Δχ| circuit vs non-circuit: **p<0.0001** (v3, length-matched)
- ✅ Layer depth: corr(layer, KL) = 0.38, corr(layer, χ) = 0.38. Moderate confound.
- ✅ Context length: perfectly matched at 15 tokens (v3). Confound eliminated.
- ✅ BOS attention verified on v3 non-IOI prompts
- ✅ Previous Token (4,11): purely structural, always attends to prev token
- ✅ Correlations: KL-χ (0.34), KL-⟨z⟩ (0.71), χ-⟨z⟩ (-0.12)
- ✅ S-Inhibition Δχ pattern (3/4 negative, but needs v3 re-check)
- ✅ ΔKL vs Δχ decoupling analysis

### Future directions (scaling up)
- **Joint (ΔKL, Δχ) circuit identification**: current tests are univariate. A joint test (Hotelling's T², Mahalanobis distance, or nonlinear classifier like SVM/random forest on |ΔKL|, |Δχ|) would find a better decision boundary in 2D space. The linear corr(ΔKL, Δχ)=0.70 understates the gain — the optimal boundary is likely diagonal. With more circuits/models this becomes a proper supervised circuit discovery method worth scaling up.
- **Max-ent / inverse covariance for direct coupling**: C_IOI has spurious correlations (two heads look coupled just because both respond to name repetition). A max-ent / Potts model approach (precision matrix instead of covariance) would partial out indirect effects and reveal direct functional coupling — same idea as protein sector analysis.
- **Unsupervised biclustering circuit discovery**: no labeled prompt pairs needed. Run diverse corpus through model → N×144 KL matrix → bicluster simultaneously over prompts and heads → prompt clusters ARE implicit task labels, head clusters ARE implicit circuits. Iterate: inspect prompt cluster structure, generate more like the centroid, recompute. Hard part is prompt generation; preferred approach is **embedding-space sampling** — find prompt cluster centroid in embedding space, sample nearby points, filter to grammatical sentences. Probabilistic/continuous nature of embedding space fits well with Bayesian framing. Human-in-the-loop to label clusters is not a bug — it's how you get interpretable circuit names. Key feasibility question: is KL signal strong enough to cluster IOI-type prompts from unlabeled corpus? Evidence suggests yes given large consistent head responses.

### Experiments still to run
- Run with generic "any sentence of length ~15" non-IOI prompts as a THIRD comparison point (user requested keeping this in back pocket — do NOT throw these results away, they may still go in the post)
- Run a second circuit (if one exists with good head labels) to check generalization
- Look at per-prompt scatter more carefully — are there substructures within the IOI cloud for a given head?
- Causal verification of L8H1/L8H11 via activation patching

### Parenthetical/weak heads — RESOLVED
- **Wang et al. 2022 full circuit has 26 heads**, not 23. Three parenthetical (weak) heads excluded from our analysis:
  - L0H10 — Duplicate Token (weak)
  - L5H8 — Induction (weak)
  - L5H9 — Induction (weak)
- **L8H1 and L8H11 are confirmed NOT in Wang et al. at all** — discovery claim is valid.
- **Decision**: exclude weak heads from the post. The post's point is showing the metric gives good signal on known circuit heads; including weak-signal heads dilutes the demonstration. Note exclusion briefly in limitations.
- If needed later, can re-run with all 26 heads to check robustness.

### Remaining lit search TODOs
- [ ] "functional connectivity" + "transformer" — neuroscience-inspired approaches to head interaction
- [ ] Spectral clustering (not just hierarchical) applied to attention head statistics
- [ ] Perturbation-based methods from neuroscience applied to transformers (beyond activation patching)
- [ ] Covariance structure of model internals across inputs as diagnostic (MLP activations, residual stream, not just attention)
- [ ] Check if Kim 2026 (arXiv:2602.08216) cites or is cited by anything relevant
- [ ] "attention head taxonomy" or "head classification" approaches beyond what's already found
- [ ] Whether the entropic OT derivation (arXiv:2508.08369) is equivalent to our KL-constrained formulation

### Exploratory results (not for post/application, saved for future work)

#### Discovery: non-circuit head enrichment
- Ranked all 144 heads by |ΔKL| and |Δχ|. Circuit heads (16% of total) make up 70% of the top 10 by |ΔKL| and 80% by |Δχ|. Diagnostics clearly concentrate on the right heads.
- Top non-circuit heads by |ΔKL|:
  - L9H4: ΔKL = -0.073 (negative — becomes LESS selective on IOI). Layer 9, same as core NMs. Could be inhibitory.
  - L2H4: ΔKL = +0.068 (positive, early layer). Possible early-stage name detection.
  - L10H3: ΔKL = +0.057, Δχ = +0.043 (both positive, late layer). Strong candidate for unlabeled backup NM.
  - L11H1: ΔKL = +0.041, Δχ = +0.033 (positive, late layer). Another backup candidate.
- Gap is clear: top non-circuit |ΔKL| = 0.073, smaller than 6 circuit heads. No hidden head as responsive as core circuit members.

#### Cross-head KL correlation (IOI prompts, 144×144)
- Computed corr(KL_head_i, KL_head_j) across 50 IOI prompts — which heads respond to the same prompt-level variation.
- **Within-role correlations strong**: NM-NM r=0.77, Induction r=0.62, Neg NM r=0.44, S-Inhib r=0.32, Backup NM r=0.17 (weak — makes sense for "backup")
- **Cross-role structure**: NM↔NegNM r=0.55, NM↔S-Inhib r=0.41 (linked roles), NM↔Induction r=0.13, NM↔DupToken r=0.006 (independent)
- **Non-circuit average**: r ≈ 0.007 (noise)
- **Anti-correlated non-circuit heads with NMs**: L8H11 (r=-0.70), L8H1 (r=-0.70), L9H1 (r=-0.63). Layer 8-9, anti-correlated — when NMs get more selective, these get less selective. Could be unlabeled suppression machinery.
- **Positively correlated**: L11H7 (r=+0.55), L6H1 (r=+0.55), L11H4 (r=+0.54). Co-activate with NMs.
- **Key insight**: Can recover functional grouping of IOI circuit purely from KL covariance across prompts, without causal interventions.
- **Block structure visible** in 144×144 heatmap. Clear red block for NMs; NM↔S-Inhib linked; structural heads independent.

#### Perturbation covariance: C_IOI vs C_nonIOI (RUN — very promising)
- Computed 144×144 KL correlation matrices separately on IOI prompts (C_IOI) and non-IOI prompts (C_nonIOI), plus the difference C_IOI - C_nonIOI and the all-pairings C_ΔKL (2500 pairs).
- **C_IOI - C_nonIOI is essentially a functional connectivity map of the IOI circuit**, recovered purely from observational KL diagnostics with no causal interventions.

**Key results:**
- **NMs snap into coordination on IOI**: NM↔NM goes from r = -0.087 (non-IOI) to r = +0.772 (IOI). Difference = +0.859. On non-IOI, each NM does its own thing. On IOI, they lock step. The circuit activation *creates* the correlation.
- **Cross-role coupling matches known circuit structure**: NM↔NegNM Δ = +0.588, NM↔S-Inhib Δ = +0.354. These are exactly the functionally linked components.
- **Structural heads go the other way**: Induction within-role Δ = -0.098, Dup Token Δ = -0.284. They're MORE correlated on non-IOI (structural ops dominate when circuit is off), slightly LESS on IOI. Good sanity check.
- **Backup NMs barely change**: Δ = +0.066 within-role. Consistent with being "backup" — they only fire intermittently.

**Candidate unlabeled circuit components (L8H1, L8H11):**
- L8H1 ↔ L9H6 (NM): flips from r = +0.543 (non-IOI) to r = -0.731 (IOI). **Swing of -1.274.** Largest magnitude change of any head pair in the entire model.
- L8H11 ↔ L9H6: +0.526 → -0.636 (Δ = -1.162)
- L8H1 ↔ L9H9 (NM): +0.437 → -0.725 (Δ = -1.162)
- L8H11 ↔ L9H9: +0.384 → -0.681 (Δ = -1.065)
- Both L8H1 and L8H11 are layer 8 — directly before core NMs in layer 9.
- On non-IOI, they're positively correlated with NMs (~+0.5). On IOI, they flip to strongly anti-correlated (~-0.7).
- **These are not in the Wang et al. circuit labels.** Wang et al. used logit attribution (traces from the output backward) — if these heads don't directly affect the logit difference, they'd be invisible to that approach.
- **Hypothesis**: L8H1 and L8H11 are performing some kind of competition or suppression that opposes Name Movers specifically when the IOI circuit fires. They could be part of the circuit's regulatory machinery.
- **This needs causal verification** — activation patching on L8H1/L8H11 would test whether they're actually necessary for IOI performance. The KL correlation analysis can flag candidates but can't prove causal necessity.

**Also interesting non-circuit heads with large positive swings (co-activate with NMs on IOI):**
- L1H7 ↔ L9H9: -0.543 → +0.475 (Δ = +1.019). Very early layer, strongly recruited.
- L2H9 ↔ L9H9: -0.504 → +0.381 (Δ = +0.886)
- L5H2 ↔ L9H6: -0.490 → +0.390 (Δ = +0.880)
- These early/mid-layer heads become correlated with NMs specifically on IOI. Could be upstream name-detection or name-routing components.

**Methodological insight**: C_IOI - C_nonIOI could be a general-purpose tool for observational circuit discovery. Take any two conditions (task-relevant vs control), compute KL correlation matrices, difference highlights task-specific functional coupling. Limitations: doesn't give direction of information flow, doesn't prove causal necessity, requires the right pair of conditions. But much cheaper than activation patching.

**C_ΔKL (2500 pairings)**: As predicted, ≈ weighted average of C_IOI and C_nonIOI. Doesn't add much beyond C_IOI. The interesting information is in the difference.

**Figures saved**: `kl_cross_head_correlation_4panel.png` (full 144×144, 4 panels), `kl_cross_head_correlation_circuit.png` (circuit-only 23×23, 4 panels). Both 300 DPI.

- Fingerprint shapes (blobs, lines, inverted-U, multimodal) also noted — different heads have different cloud shapes in (KL, χ) plane. The mean doesn't capture this. Future post material.

#### Hierarchical clustering on C matrices (RUN)
- Applied hierarchical clustering (average linkage, distance = 1 − correlation) to C_IOI, C_nonIOI, and C_diff.
- **C_IOI at threshold 1.0**: 3 clusters. Cluster 1 (47 heads) contains most core circuit: all 3 NMs, 3/4 S-Inhibition, both Neg NMs, some Backups. L7H3 (S-Inhibition) is **isolated** — clusters separately with 46 non-circuit heads. Remaining Backup NMs cluster with structural heads.
- **C_nonIOI**: much less circuit-role structure. Heads reorganize when no repeated name.
- **C_diff at threshold 1.0**: core IOI machinery groups into one big cluster (58 heads): all 3 NMs, 3/4 S-Inhib, both Neg NMs, most Backups. Second smaller cluster has L6H9 (Induction) + L10H2 (Backup NM).
- **L8H1 clusters with L7H3 (S-Inhibition)** in both C_IOI and C_diff — the same S-Inhibition head that was isolated from the main group.
- **L8H11 clusters with Backup NMs** in C_IOI, and with L7H3 + structural heads in C_diff.
- Both L8H1 and L8H11 end up in the same C_diff cluster as L7H3 (S-Inhibition) and L9H7 (Backup NM) — consistent with suppression/inhibition pathway.
- Dendrogram-ordered C matrices show clear block structure in C_IOI absent in C_nonIOI. C_diff isolates it.
- **Figures saved**: `hierarchical_clustering_dendrograms.png` (3-panel, 300 DPI), `correlation_matrices_dendrogram_ordered.png` (3-panel, 300 DPI)

### Failure modes to actively look for
- Heads where KL does NOT drop on activation (breaks the "multi-token" story)
- Heads where χ gives misleading signal (e.g., high χ for trivially monosemantic reasons)
- Cases where the diagnostics can't distinguish meaningfully different heads
- Anything that looks clean but is actually driven by a confound (length, layer, etc.)
- Blog should acknowledge limitations honestly — stay humble, don't oversell

### Blog narrative — TO DISCUSS LATER
- User thinks v1→v2→v3 prompt control story is NOT blog-appropriate ("who wants to know how the sausage got made?"). Standard research practice, not reader-facing content. Discuss later.
- User wants to keep post simple: define the metrics, show where there is clear signal.
- Do NOT throw away the IOI-vs-generic-prompt results. User thinks they can still go in the post (perhaps as a secondary comparison or to show the effect is even stronger with less controlled prompts). Discuss later.
- My (Claude's) thoughts on blog: the conservative narrative is best. Theory → diagnostics → test on IOI circuit → clear signal in (ΔKL, Δχ) → χ adds independent info → honest limitations → tease Post 3. The prompt control journey is interesting methodologically but probably belongs in a footnote or appendix at most. The generic-prompt results could serve as a "first pass" that motivates tighter controls, or simply as a demonstration that the effect is robust across different baselines. Keep it short, keep it honest.
- We have more than enough material for the post. Don't over-complicate.

### ΔKL direction RESOLVED (v3)
- **ΔKL is POSITIVE for selection heads** (Name Movers, Backup NMs, Negative NMs): KL increases +0.01 to +0.26 on IOI. They become MORE selective when the circuit fires. This is the intuitive direction.
- **v2's "KL drops on activation" was an artifact** of the non-IOI prompts having multi-token subjects ("the cashier") that changed the attention landscape beyond just the IOI circuit.
- **Structural heads (Induction, Dup Token, Prev Token) have near-zero ΔKL**: ≤0.007 in either direction. These heads do the same thing regardless of whether a name repeats, because the sentence structure is identical.
- **S-Inhibition is mixed**: 3/4 have positive ΔKL, L7H3 is slightly negative (-0.014). S-Inhibition doesn't select names — it suppresses. Smaller, more variable signal makes sense.
- **Fig 2d** (all 23 circuit heads) saved at 300 DPI: `fig2d_all_circuit_heads.png`

### Only after empirics are solid
- Decide whether χ earns its place in the blog or gets relegated to "future work"
- Settle on the right stat mech language (if any)
- Write the narrative

### DCA / scalability argument (strong version for post)

**Core claim**: DCA on attention KL statistics is a scalable, intervention-free screening method for circuit discovery.

**Why it scales**:
- Data collection: O(n × H²) forward passes, no interventions, no gradients
- Precision matrix: O(H³) or cheaper with graphical LASSO (exploits circuit sparsity)
- Contrast: activation patching / ACDC = O(pairs × forward passes) — expensive and intervention-dependent

**Key insight**: Graphical LASSO gives a *sparse* precision matrix by design — circuits are local, most head pairs aren't directly coupled. ACDC tests each edge explicitly; DCA infers the whole graph simultaneously.

**Division of labor**: DCA = cheap screening (sparse functional graph). Causal patching = expensive validation of flagged edges. This makes sense operationally.

**Honest limitation**: Only captures circuits operating through attention head co-activation. MLP-mediated circuits are invisible. Fine for IOI (attention-mediated); gap for more complex behaviors.

**What "more prompts fixes" means**:
- n=50 per condition: rank-deficient, can't invert C, C_diff attenuates early-layer heads (induction, dup token, previous token) due to noise floor
- n >> H (e.g. 500+): full-rank covariance, precision matrix well-conditioned, direct couplings recoverable
- DCA alone still needs n >> H; more prompts + DCA together = the full fix
- Early-layer heads miss top-50 C_diff because their coupling to name movers is *indirect* (mediated through S-inhibition etc). Precision matrix would show induction → S-inhibition direct, not induction → name mover transitive.

**Current result**: C_diff top-50 recovers 10/23 circuit heads (core late-layer name movers, some S-inhibition, some backup). Top-200 gets 17/23. Full coverage at top-607. Coverage improves monotonically — consistent with low-rank circuit topology where late-layer hubs dominate marginal statistics.

### Covariance analysis — session findings (Feb 24 2026)

**Eigenspectrum of C_IOI**: rank-deficient (50 prompts, 144 heads → rank ≤ 50). Top eigenvectors dominated by non-circuit heads (enrichment ~0.95). Small eigenmode enrichment modest: max ~1.95 for EV13 of C_IOI, ~1.78 for EV62 of C_combined. Not enough signal for unsupervised circuit recovery.

**Signed C_diff ranking is much cleaner than |C_diff|**:
- Positive end (co-activation): 8/9 CC in top 9. Name movers, S-inhibition, negative NMs, backup NMs all co-activate together. Very clean circuit recovery.
- Negative end (anti-correlation): 0 CC in bottom 100. Entirely CN — non-circuit heads anti-correlated with name movers. L8H1/L8H11 are the strongest (#1, #3). Two different stories: co-activation vs inhibitory relationships. ALWAYS use signed ranking, not abs value.

**C_diff is not ad hoc — it's contrastive PCA** (Abid et al. 2018) at α=1. Principled upgrade = generalized eigenvalue problem C_IOI v = λ C_nonIOI v, but requires C_nonIOI invertible (needs n >> 144).

**DCA on C_diff is wrong**: C_diff has negative eigenvalues, not a valid covariance matrix. Correct "contrastive DCA" = invert C_IOI and C_nonIOI separately → J_IOI - J_nonIOI. Requires n >> 144 and regularization.

**Augmented [KL, χ] C_diff (288×288)**: EV1 eigenvalue doubles (~44 vs ~22) but enrichment same. χ of L8H6 floods positive list — indirect coupling artifact (large Δχ on IOI → correlates with everything). KL-only positive list is cleaner. Adding χ amplifies the dominant signal but also adds hub-driven noise.

**CCA between KL and χ**: canonical correlations all ~0.999 with n=50, k=20 — saturated, not useful. Small n CCA is underpowered for finding task-specific cross-modal structure.

**Residual stream delta diagnostic — RESULT (Feb 25 2026)**:
- Ran ||Δresidual|| C_diff on 156 components (144 heads + 12 MLPs), n=50 each
- Result: much worse than KL. Top-10 positive = 0 CC. Top-100 = 2 CC, 11/23 heads. L8H0 (non-circuit) floods everything.
- Root cause: ||Δresidual|| is confounded by "how much does this head write regardless of task" — dominated by head-specific scale factors and token embedding norms. KL normalizes away this confound (it's a divergence from uniform, not a raw norm).
- **Key lesson**: KL works because it measures attention selectivity directly. Norm of output mixes in irrelevant scale factors.
- Fix 1 (supervised): project Δresidual onto (unembedding(IO) - unembedding(S)) = logit attribution. Works great, known technique.
- Fix 2 (better unsupervised): use Var_π(v) = variance of value vectors under attention distribution, not ||z @ W_O||. This stays in the softmax-statistics framework and doesn't have the scale confound.

**χ → Var_π(v) upgrade**:
- Current χ = Var_π(log π) — temperature susceptibility, query-key side only
- Better second diagnostic: Var_π(v) = E_π[v²] - E_π[v]² per head — value-side variance under attention distribution
- Var_π(v) measures "how much does changing which position gets attended to change the output vector" — directly task-grounded
- Replaces χ in the (KL, χ) plane → (KL, Var_v) plane. Both are softmax-distribution statistics but Var_v incorporates values.
- TODO for a future experiment (leave for later).

**Diagnostic space (full picture)**:
- Query-key side: KL = E_π[log π/u], χ = Var_π(log π)
- Value side: E_π[v] = head output vector, **Var_π(v) = value variance under π** (better than χ)
- Residual stream: ||Δresidual|| = what actually gets written (includes values, W_O projection) — confounded by scale
- Logit attribution = Δresidual projected onto task direction = ground truth per-component importance
- Hierarchy: logit attribution (supervised) > Var_π(v) (better unsupervised) > KL > χ > ||Δresidual||

**MLP architecture and circuit role (clarified)**:

Key insight: MLPs are context-free transforms on context-dependent content. The MLP at position t sees only the residual stream at t — no cross-position access. So it can't route information; it can only transform whatever attention has already aggregated into the residual stream.

**Division of labor in circuits**:
- Attention = routing: context-dependent weights (π) over context-dependent values (v_i). Decides WHICH information to move where across positions.
- MLP = transformation: fixed nonlinear map applied to the local residual stream. Decides WHAT to do with whatever has been accumulated. No cross-position access.
- IOI is a routing task ("copy this name to output position") → attention-dominated. Factual recall ("Paris is the capital of...") → more MLP-dominated (stored key-value lookup).

**out ≈ E_π[v]**: Both attention and MLP produce a weighted sum of output vectors:
- Attention: Σ_i π_i v_i (normalized weights via softmax, context-dependent values)
- MLP: Σ_j h_j W_out[j,:] (un-normalized GELU weights, fixed output directions)

Same structure. The softmax normalization in attention is what enables KL — you have a well-defined null (uniform) and proper divergence. The MLP lacks this normalization, so KL is harder to apply cleanly. But **Var of the weighted output vectors carries over without needing normalization** — same geometric question (how much could the output vary?), no normalization required.

**MLP diagnostic**: Var_{h}(W_out rows) = weighted variance of neuron output directions under the activation pattern. Analogous to Var_π(v). Unsupervised, no scale confound, no normalization issue.

**KL applied to MLP pre-activations**: Valid as a "neuron selectivity" measure (how concentrated is softmax(a) over neurons?) but measures feature selection at current position, NOT cross-position routing. Different from attention KL in what it's capturing. Could be useful for Post 3 but should not be presented as a direct analog without this caveat.

### Post arc reorientation (Feb 25 2026)

**Revised arc:**
- Post 2: Attention diagnostics (KL, χ), C_diff circuit recovery — ship as-is. KL + χ are the diagnostics. Do NOT upgrade to Var_v mid-post.
- Post 3: Opens with "χ was correlated with KL for the wrong reason: both are query-key side. Enter Var_v (value side) + MLP Var_W_out." The limitation of Post 2 is the opening hook of Post 3. Then: scaling up — n >> H, DCA/precision matrix, larger models.
- Post 4: Training dynamics — (KL, Var_v) evolution, phase transitions, grokking

### Post 3 direction notes — Upgrading + scaling circuit discovery

**Three-act structure**:
1. **Upgrade the fingerprint**: (KL, χ) → (KL, Var_v). Motivation: χ redundant with KL (both query-key side, corr ΔKL/Δχ = 0.70). Var_v adds value side — orthogonal, more informative. Show updated (KL, Var_v) plane on IOI heads.
2. **Upgrade C_diff**: extend feature vector to Var_v (+ MLP Var_W_out). Rerun C_diff on IOI, show recovery at least as good. MLP null result is its own finding (IOI is attention-mediated).
3. **Scale out**: more prompts (n >> H → full-rank C → precision matrix → graphical LASSO), other known circuits (induction, greater-than), possibly a new discovery.

**Core story**: C_diff / cPCA works at n=50 but is n-limited. Scale up → full circuit recovery without interventions. The method's advantage over causal patching grows with model size.

**Unified diagnostic principle (key insight)**:
Var_v and Var_W_out ask the *same geometric question* applied to two different module types:
- Attention: Var_v = E_π[||v - E_π[v]||²] — given routing distribution π, how much does it matter WHICH position you attend to?
- MLP: Var_W_out = E_h[||W_out - E_h[W_out]||²] — given neuron activation pattern h, how much does it matter WHICH neurons fire?
KL doesn't have this unity — requires softmax normalization, so it applies cleanly to attention but needs forcing for MLPs. Var works with any weighting. This is the right feature for a unified, module-agnostic DCA that scales to other circuits.

**Feature vector design for DCA (Post 3) — settled: all three**:
- KL (144) + Var_v (144) + Var_W_out (12) = 300 features.
- Three stats are complementary across circuit types: KL catches routing selectivity, Var_v catches routing consequence, Var_W_out catches MLP specialization. Feature vector is adaptive to circuit type without knowing in advance which modules matter — J_diff reveals it.
- Cross-type edges (KL_i ↔ Var_v_j, Var_v_i ↔ Var_W_out_k) are novel structure not visible in any existing method.
- For IOI (attention-routing): KL + Var_v lead. For factual recall (MLP-heavy): Var_W_out leads. For mixed circuits: all three. Generalization across circuit types is the point.
- Expect J_diff with n >> 300 to recover IOI and generalize well to other circuits.

**Open empirical question: does Var_v beat KL?**
- KL = how sharp is attention. High KL but boring values → task-irrelevant head.
- Var_v = how consequential is routing. Directly measures "does it matter where this head attends?" — closer to what circuit discovery wants.
- **C_diff result (n=50)**: KL wins (8 CC top-10 vs 2 CC for Var_v). But this is marginal correlation.
- **Hypothesis: Var_v may flip the result at J_diff (precision matrix).** Reason: J_diff removes transitive correlations. KL can be high for spurious reasons (positional bias, syntactic patterns) that correlate across heads indirectly. Var_v is more causally downstream — filters out heads where values are boring regardless of routing. Direct KL couplings after partialling may be noisier than direct Var_v couplings.
- Requires n >> H to test properly (J_diff is rank-deficient at n=50). Key empirical question for Post 3.

**MLP diagnostic — Var_h(W_out rows)**:
- Weighted variance of neuron output directions (rows of W_out) under activation pattern h.
- Scale-invariant if W_out rows are unit-normalized first (measure direction diversity, not magnitude).
- Measures "how much does the MLP's output direction vary with neuron routing?" — same geometric question as Var_π(v) for attention.
- MLP KL (softmax of pre-activations) also valid but measures feature selectivity at current position, NOT cross-position routing. Caveat clearly.

**The scaling roadmap**:
1. **n >> H, full-rank C**: C_IOI and C_nonIOI invertible → precision matrix J = C⁻¹ → J_diff = J_IOI - J_nonIOI shows direct couplings only (removes transitive correlations). Early-layer heads (induction, dup token, prev token) should appear cleanly.
2. **Graphical LASSO on J_diff**: sparse precision → explicit circuit graph, no interventions.
3. **Cross-feature edges**: J_diff on 300-dim feature vector gives edges between KL_i ↔ Var_v_j across different heads — "this head's routing sharpness drives that head's output variance." Novel structure not visible from head-to-head KL alone.
   - **Suppress same-head self-coupling from C_diff signed ranking** (KL_i ↔ Var_v_i for the same head i): trivially correlated (both driven by the same π_i), will flood the top of the ranking. Mask before sorting.
   - **J_diff + LASSO: probably fine to leave in.** Precision matrix partial correlations naturally absorb same-head covariance — KL_i ↔ Var_v_i self-coupling doesn't leak into cross-head edges. LASSO further zeroes out trivial intra-head entries. Only risk: near-collinearity → numerical instability in inversion; sufficient regularization handles this.
   - Same-head KL↔Var_v values are useful as fingerprint info (high KL + low self-coupling = sharp attention to similar values; high KL + high self-coupling = sharp + diverse). Report separately, not as circuit edges.
4. **Larger models**: GPT-2 XL (400 heads, need n~2000), Llama-7B (1024 heads, need n~5000). Still just forward passes.

**Scaling comparison (concrete numbers)**:
- GPT-2 small: 144 heads, n=50 forward passes (current)
- GPT-2 XL: 400 heads (~25L × 16H), need n~2000 prompts → cheap
- Llama-7B: 1024 heads (32L × 32H), need n~5000 prompts → still cheap
- Causal patching at Llama scale: O(1024²) = ~1M intervention runs → expensive
- This method = cheap screening; causal patching = validation on top candidates only

**Generic circuit discovery workflow**:
1. Pick task + model. Define activating vs. null prompt pairs with good experimental control.
2. Generate n >> H prompts per condition.
3. Compute Var_v (and optionally KL) per head per prompt → X matrix.
4. C_diff → signed ranking → top positive pairs = co-activating candidate circuit heads.
5. With n >> H: J_diff → graphical LASSO → sparse circuit graph.
6. Validate: targeted ablations / activation patching on top-k candidates only.

**Candidate tasks for new circuit discovery**:
- **Induction** (GPT-2 small): known circuit, good sanity check for method
- **Greater-than** (Hanna et al. 2023, GPT-2 small): numerical comparison, well-characterized
- **Factual binding** ("Eiffel Tower is in ___", larger models): attention + MLP circuit, tests MLP Var extension
- **Subject-verb agreement** ("keys to the cabinet [are/is]"): syntactic, likely clean attention circuit
- **Modus ponens** (large models, e.g. Llama-7B+): logical reasoning, probably uncharted territory
  - Activating: "If it rains, streets get wet. It is raining. So the streets are"
  - Null: "If it rains, streets get wet. It is sunny. So the streets are"
  - Challenge: large models may pattern-match rather than compute via clean circuit. Diagnostic will reveal which.

**The pitch**: method's advantage grows with model size. At frontier scale (Claude-class), causal patching is completely intractable; this method is still just forward passes. Post 3 shows it works at GPT-2 scale; frontier application needs frontier model access.

**Post 4 (training)**: Track (KL, Var_v) per head over training checkpoints. MLP→attention handoff. Kim 2026 connection. Leave for later.

### KL + Var_v as fingerprint axes (Feb 25 2026)

**User insight**: For fingerprinting (characterizing head identity in the 2D plane), KL + Var_v is a better pair than KL + χ because they're more statistically independent.

**Why**: KL = E_π[log π/u] and χ = Var_π(log π) are both query-key side statistics — they depend only on the attention weights π, not on the values. We measured corr(ΔKL, Δχ) = 0.70 (strong, because they're both measuring concentration of the same distribution). By contrast, Var_v is a value-side statistic — it depends on both π AND the actual value vectors v. KL and Var_v capture genuinely different aspects:
- KL = routing sharpness: how peaked is attention?
- Var_v = routing consequence: does it matter WHERE you attend?

A head can be high KL / low Var_v (attends very selectively but all value vectors are similar → the routing is decisive but the output is similar regardless). Or low KL / high Var_v (diffuse attention over diverse values → even small perturbations in π have large effects on output). These cells of the 2D space are meaningful and unreachable from KL + χ.

**For circuit recovery (C_diff)**: KL still empirically wins over Var_v at top-k. But the feature pair for characterizing individual head identity → use KL + Var_v, not KL + χ. χ can be dropped from the Post 2 blog as the "second diagnostic" in favor of Var_v, or demoted to a footnote.

**Table of (KL, Var_v) 2D space**:
| KL | Var_v | Interpretation |
|----|-------|----------------|
| High | High | Selective + consequential: true circuit heads (name movers, S-inhibition) |
| High | Low | Selective + redundant: attends sharply to similar values (some induction?) |
| Low | High | Diffuse + diverse: "covering" heads — might be backup/generalist |
| Low | Low | Broad attention, homogeneous values: early-layer, task-irrelevant |

**Bottom line**: Replace (KL, χ) plane with (KL, Var_v) plane going forward. More informative, more independent, both in the softmax-statistics framework. This is the 2D fingerprint for Post 2 and Post 3.

### C_diff as a memorization vs. generalization diagnostic (Feb 25 2026)

**Insight for future work (post Post 3/4)**: C_diff naturally distinguishes circuit-mediated generalization from surface-level pattern matching (memorization).

**The logic**: C_diff measures CONSISTENT cross-prompt co-activation between head pairs. For this consistency to appear, the same heads must co-activate across a *diverse* set of prompts.

- **Circuit-mediated behavior**: The model has a stable, reusable circuit that fires whenever the task conditions are met, regardless of surface features (names, templates, exact wording). Same heads co-activate reliably across many diverse prompts → C_diff shows clean, concentrated signal in CC pairs.
- **Surface pattern-matched / memorized behavior**: The model fires on each prompt instance via different "random" combinations of heads — each surface pattern triggers a different semi-random subset of heads. No consistent co-activation across prompts → C_diff is flat/noisy, CC pairs don't stand out.

**Why this is novel**: Causal patching (ACDC etc.) tells you if a head is *necessary* for task performance on specific examples. It doesn't distinguish whether the head participates via a reusable circuit or via surface memorization. C_diff adds the consistency dimension: noisy C_diff with diverse prompts = the behavior is pattern-matched, not generalized.

**Practical test**: Run C_diff with increasingly diverse prompts. If signal degrades quickly as prompts become more diverse (away from training distribution surface features), the behavior is likely memorized. If signal stays clean across diverse prompts, there's a genuine circuit.

**Application ideas**:
- Compare C_diff on in-distribution vs. paraphrased prompts for the same task → "circuit robustness" score
- Track C_diff signal quality vs. prompt diversity budget → measure generalization depth
- Memorization-heavy tasks (factual recall) should show noisy C_diff; reasoning tasks with clean circuits should show consistent C_diff

**Note**: This is purely observational and probably post-3 or post-4 territory. It would require careful experimental design (controlling what varies across the prompt set).

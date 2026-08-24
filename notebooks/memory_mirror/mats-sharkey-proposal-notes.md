---
name: mats-sharkey-proposal-notes
description: "Notes for Peter's MATS Round 2 application to Lee Sharkey (Goodfire) stream — 300-word research proposal angles"
metadata: 
  node_type: memory
  type: project
  originSessionId: 14f8c394-1e56-45de-997d-4ff4de657c44
---

# MATS Lee Sharkey (Goodfire) — Proposal Notes

**Stream:** Lee Sharkey (Goodfire AI), Empirical track, Interpretability category
**Application:** 300-word research proposal on what Peter would research during the 3-month program (he's allowed an additional short list of 1-2 sentence project ideas)
**Deadline:** 2026-06-23 11:59 PM AoE

---

## ⚠️ AI-USE CORRECTION SENT 2026-06-25 — read before anything below

What got submitted to the form (the "FINAL DRAFT" block immediately below) was **AI-condensed** from a longer draft Peter wrote himself. MATS's *default* AI-use policy (applies unless a stream question says otherwise) forbids having an LLM **rewrite/restructure/condense/"improve"** text or **editing based on its critique** — only light grammar/spelling, concept lookups, and pre-writing brainstorm are allowed. So the condensing tripped the policy (and an AI-detector scored the condensed version 95–100% AI; Peter's own un-condensed prose scores clean — the AI cadence, not the technical density, is what flags).

Peter self-caught this while back in the form updating his mentor ranking (allowed), and emailed applications@matsprogram.org (subj "Correction re: AI-use policy for stream application (Peter Fields)") owning the slip and attaching his ORIGINAL fully self-written draft, deferring to MATS on whether they consider it (likely they punt to Lee). The same email completed his Gary Abel disclosure (AI use was *permitted* there → not a violation). **Awaiting response.**

**Lesson (Peter's, worth keeping):** to cut length, delete from your own draft — don't ask AI to reword, and don't even *read* an AI rewrite (anchoring leaks its cadence in even when you retype). Always better to own up to a mistake (his grad-school rule).

### CLEAN CANONICAL ORIGINAL = "ORIG" (Peter's own words, longer — attached to the correction email; NEVER the form-submitted version — confirmed by Peter 2026-07-09; the REAL SUBMITTED FINAL above is the form text)

> Many current lines of research in LLMs consider the residual stream vector space as Euclidean (e.g. persona vectors [arXiv:2507.21509] or activation plateaus [lesswrong.com/posts/WMfSbt7AAcJdHzysB/activation-plateaus-where-and-how-they-emerge]). However, the model itself never uses that inner product to compare vectors in the residual stream. The geometry it actually uses is implicit in the bilinear forms learned by the QK matrices of each attention head.
>
> Decomposing each W_QK matrix into a symmetric (G) and anti-symmetric part (B), we have W_QK = G + B. My current hypotheses are that (i) G represents a content matching metric, and can define a geometry of activation space more robustly; (ii) B is direction-dependent, responsible for routing information (think, for example, of K-composition in induction heads), and acts on a "compute sub-space" of activation space.
>
> Preliminary evidence from observations on the attn-only-2l model (from TransformerLens) shows a G-weighted token similarity matrix (W_E^T G W_E where W_E is the embedding matrix) ranks digit-pairs (1-2, 3-4, 5-6, etc.) highly, indicative of content matching. Averaging over G from each head in GPT-2 small, I find that the eigenvectors of this mean G have strong loadings on semantically similar groupings (punctuation marks, proper nouns, tech terms, etc.) Furthermore, the top eigenvectors of G (per head) tend to lie heavily in the W_E space, whereas B's top modes tend to have little overlap, strongly indicating that B operates in the "compute subspace" that G is blind to. A next step is to check if B's top modes are orthogonal to activation space directions defined by SAEs.
>
> The main question I'd like to explore if accepted into this work stream: can each head's QK be decomposed as sum over a shared (and interpretable) basis, that is, a sum over G and B terms, with each head having different coefficients?
>
> Methodologically, this would involve SVD over a stacked matrix of all heads' B matrices (and G separately). I'd also be interested in decompositions with sparsity constraints. I would validate this in GPT-2 small's IOI circuit: could finding a shared basis and per-head coefficients adequately distinguish head types (name movers versus S-inhibition, for example)? If only a few atoms of a global G and B are active per head, this would be short description length.
>
> Metric-aware decomposition of QK matrices may further uncover a compute subspace that are mission critical for LLMs computations but that SAEs are blind to. If behaviors like deception utilize this subspace, uncovering such a mechanism would be a large step towards safe monitoring of AI systems.

---

## ⭐️ REAL SUBMITTED FINAL (Peter's own words — CONFIRMED by Peter 2026-07-09; USE THIS for any c&p)

**This is the actual text submitted to the Sharkey stream form** (after the AI-use correction: Peter resubmitted a clean, fully self-written version). Both blocks below this one (the "AI-CONDENSED FINAL DRAFT" and the earlier "CLEAN CANONICAL ORIGINAL") are NOT the submitted text — do not treat them as the source of truth. Note: this real final does NOT include the "check B's top modes orthogonal to SAE directions" sentence — that lives only in the canonical-original draft below (still Peter's own prose, just not submitted).

> Many current lines of research in LLMs consider the residual stream vector space as Euclidean (e.g. persona vectors [arXiv:2507.21509] or activation plateaus [lesswrong.com/posts/WMfSbt7AAcJdHzysB/activation-plateaus-where-and-how-they-emerge]). However, the model never uses that inner product to compare residual stream vectors. The geometry it actually uses is implicit in the QK matrices of each attention head. Decomposing each W_QK matrix into a symmetric (G) and anti-symmetric part (B), we have W_QK = G + B. My current hypotheses are that (i) G represents a content-matching metric, and can define a geometry of activation space more robustly; (ii) B is direction-dependent, responsible for routing information (think, for example, of K-composition in induction heads), and acts on a "compute subspace."
>
> The main question I'd like to explore if accepted into this work stream: can each head's QK be decomposed as a sum over a shared (and interpretable) basis, that is, a sum over G and B terms, with each head having different coefficients? Methodologically, this would involve SVD over a stacked matrix of all heads' B matrices (and G separately). I'd also be interested in sparsity-constrained decompositions. I would validate this in GPT-2 small's IOI circuit: could finding a shared basis and per-head coefficients adequately distinguish head types? If only a few atoms of a global G and B are active per head, this would be a short description length.
>
> I have preliminary evidence from the attn-only-2l model (from TransformerLens) where a G-weighted token similarity matrix (W_E^T G W_E where W_E is the embedding matrix) ranks semantically similar tokens highly, indicative of content matching. In GPT-2 small, top eigenvectors of G (per head) tend to lie heavily in the W_E space, whereas B's top modes tend to have little overlap, strongly indicating that B operates in the "compute subspace." If behaviors like deception in other models utilize such a subspace, uncovering this mechanism would be a large step towards safe monitoring of AI systems.

---

## FINAL DRAFT (AI-CONDENSED — ⚠️ NOT the submitted version; superseded by REAL SUBMITTED FINAL above)

Re-read this before any screening call. Extras list was deliberately OMITTED (optional field; chose focus over thin add-ons).

> Much LLM research treats the residual stream as Euclidean (e.g. persona vectors [arXiv:2507.21509] or activation plateaus). But the model never uses that inner product to compare residual stream vectors. The geometry it actually uses is implicit in the bilinear form each attention head learns as its QK matrix.
>
> Each W_QK splits into symmetric and anti-symmetric parts: W_QK = G + B. I hypothesize that (i) G is a content matching metric, a more principled geometry, and (ii) B is directional, responsible for routing information (e.g. K-composition in induction heads), acting on a compute subspace.
>
> Preliminary evidence: in attn-only-2l (TransformerLens), a G-weighted token similarity matrix (W_E^T G W_E) ranks digit-pairs (1-2, 3-4, ...) highly, indicative of content matching; in GPT-2 small, eigenvectors of a head-averaged G load on semantic groupings (punctuation, proper nouns, tech terms); the top eigenvectors of G (per-head) lie largely in W_E space, whereas B's top modes have little overlap, indicating that B operates in the compute subspace that G is blind to. A next step: check if B's top modes are orthogonal to SAE feature directions.
>
> The main question I'd like to explore: can each head's QK be decomposed over a shared, interpretable basis of G and B atoms, with head-specific coefficients?
>
> Method: SVD over the stacked per-head B matrices (and G separately), and sparsity-constrained variants. I'd validate on GPT-2 small's IOI circuit, asking whether the shared basis plus per-head coefficients separate known head types (name movers vs S-inhibition). A few active atoms per head would be a short description length.
>
> This metric-aware decomposition could expose a subspace critical for LLM computation but invisible to SAEs. If behaviors like deception use that subspace, surfacing it would be a real step toward safe monitoring.

---

## ⭐ THE PLAN (decided 2026-06-23 session) — draft the proposal from THIS

Worked the whole idea menu against Sharkey's actual rubric this session and converged here. Draft from this block. The "Recommended 300-word proposal structure" further down is SUPERSEDED.

### Lead / hook: the non-Euclidean blind spot
- The field treats the residual stream as Euclidean (persona vectors arXiv:2507.21509; activation plateaus; even "activation manifold" talk presupposes a metric). A vector space has no geometry until you pick an inner product, and everyone defaults to the identity.
- The model never uses Euclidean: attention similarity is x_i^T W_QK x_j, a learned bilinear form, not x_i^T x_j.
- Reuse the PrincInt phrasing (princint-app.md, the "notions of distance in the residual stream" paragraph), re-aimed at Sharkey: his own "computational manifold" already presupposes a metric, and the Euclidean default is the wrong one. Tagline: "B routes, G measures; you can't define the manifold until you fix G."
- Tensor-type framing (the "advanced math" card; keep LIGHT in the 300w, this is interview gold): W_QK is type (0,2) = the geometry (metric G + 2-form B); W_OV is (1,1) = the transformation. Only (1,1) tensors compose, so computation lives in the OV maps and geometry in the QK form.

### THE scoped 3-month project (proposal centerpiece; small, directed, result almost guaranteed)
**Q: Is the symmetric/antisymmetric balance of W_QK a functional fingerprint of attention heads?**
- Split every head's W_QK = G (symmetric) + B (antisymmetric); test whether the G/B balance predicts head function, using Wang et al.'s labeled IOI heads as free ground truth.
- Hypothesis (pre-registerable): content/matching heads (name movers) G-dominated; positional/routing heads (induction, previous-token, S-inhibition) B-heavy, because direction is intrinsically antisymmetric.
- Primary signal: ONE scalar per head, r_h = ||B_h||_F / ||W_QK,h||_F. Optional 2nd axis: embedding overlap of B's top modes (the verified compute-only quantity), giving each head a point in a 2D plane.
- Result either way: holds -> unsupervised parameter-only head-type signature; fails -> "linear QK symmetry doesn't track function, here's what does." No empty-handed outcome. Prior signal already exists (B governs K-composition 0.68 vs 0.17; B top modes off-embedding).
- Arc: wk 1-3 compute G/B + r_h across GPT-2 small; wk 3-6 overlay IOI labels, test separation + significance; wk 6-9 generalize to one more labeled circuit (docstring / greater-than); wk 9-12 write up.
- Engineering bar LOW: weight-space linear algebra + known labels, numpy/TransformerLens, no training, no GPU. Suits Peter's Python level.

### REFINED 2026-06-23 (cont.) — core idea LOCKED: the routing dictionary
- **Decision:** the shared B/G atom dictionary is the project centerpiece. The r_h scalar fingerprint is demoted to a week-1 sanity check (per-head dictionary coefficients subsume it). Fold "find more evidence B is content-orthogonal" INTO the dictionary project as a characterization step, not a separate aim.
- **One-sentence thesis (draft from this):** Attention separates ADDRESSING from PAYLOAD. The QK form decides where to route (no content moves there); OV moves what. The routing lives in B, which sits in a compute subspace largely orthogonal to the content directions SDL/SAE methods represent. Recover the shared dictionary of routing (B) and metric (G) atoms heads expand in (PCA over per-head B/G), validate on GPT-2 small IOI (positional/routing heads load on different atoms than content heads).
- **Resolves Peter's "if B is orthogonal to content, how is content moved?":** content is NOT moved by QK at all. QK (G+B) only sets the attention PATTERN; OV reads/writes the content in content space. B routing directions being content-orthogonal = the head addresses using structural/computed features (position, syntactic role), not raw token identity. Induction heads = clean example (route on previous-token/positional signal in compute space, copy content via OV in content space). This STRENGTHENS part (1) into the addressing/payload decoupling thesis; do not drop it.
- **New confirmatory experiment (great for Sharkey, week-1):** is B's top routing subspace orthogonal to a residual-stream SAE's feature dictionary (jbloom/GPT2-Small-SAEs-Reformatted) vs a random-direction baseline? Direct empirical test of the SDL-blindness thesis his whole APD program rests on.
- **Parked extensions (NOT in 300w):** OV pullback/composition = interview back-pocket; MLP interaction with the routing subspace = separate future project.

### Three-pillar one-liners (light touch, don't overbuild)
- PD: G/B is a parameter-space decomposition; the project tests whether it is interpretable (his core question).
- Manifolds/metric: G is the head's similarity metric; the project asks whether its prominence vs routing reveals the head's job.
- MDL (plain "simplest true story"): one scalar per head that predicts function IS a very short description of the head.

### Safety / theory of impact (a whole rubric axis; include ONE sentence)
- If goal-directed or deceptive routing lives in B-modes that SAEs/CLTs structurally cannot see (they work in content/Euclidean directions), activation-based monitoring has a built-in blind spot that a metric-aware parameter decomposition could close.

### Extras list (separate field; reordered, PD-credibility first)
1. G-weighted MDL (builds on APD directly): if G is the implicit metric, description length over parameter directions should be G-weighted, not Frobenius.
2. APD on Hänni U-AND ground-truth networks (Heimersheim lineage with Sharkey).
3. Bilinear forms as a decomposition class: generalize APD atoms to (1,1)/(2,0)/(0,2) tensor types.

### Precision fixes (so it is bulletproof under questioning)
- Say "right singular vectors / principal directions of W_E" (or eigenvectors of W_E^T W_E), NOT "eigenvectors of the token embedding matrix" (W_E is 50257x768, not square).
- The off-embedding result is about B's DOMINANT modes only; the bulk of B sits at the ~0.67 random-chance baseline. Keep "preliminary/suggesting."

### Interview back-pocket (do NOT put in the 300 words)
- OV pullback: G' = W_OV^T G W_OV; ask whether each head's OV is a G-isometry (G'=G) or deforms the geometry. B' = W_OV^T B W_OV likewise. Caveats: W_OV is low-rank (projection, not full isometry); G is the shared/global metric while OV is per-head.
- The cross-layer pullback W_OV^(1)T W_QK^(2) W_OV^(1) IS Elhage Q/K-composition; Peter's 0.68-vs-0.17 result already says B is the part transmitted under composition.

---

## Lee Sharkey context (essential for getting the framing right)

- **Previous role:** Apollo Research; now at Goodfire AI leading interp team
- **Method:** Parameter Decomposition (PD) / Attribution-based Parameter Decomposition (APD)
- **Key paper:** "Interpretability in Parameter Space: Minimizing Mechanistic Description Length with Attribution-based Parameter Decomposition" — MDL framing is *explicit*, not casual
- **Co-author with Stefan Heimersheim** on "Open Problems in Mechanistic Interpretability" (Jan 2025) — Sharkey is part of the Apollo intellectual lineage
- **APD applications (verbatim from stream description):** "feature splitting, identify attention-head distributed computations, identify circuits, and more"
- **Stated open question:** "We minimize 'description length'. But we are not yet confident that we have the right 'type of description'... We think understanding computational manifolds (which are projections of activation manifolds) are likely part of the answer here, since they may offer an even more concise description of neural computation than SDL latents or VPD parameter subcomponents."
- **Stream requirement (verbatim):** "MATS projects in my stream should at least be conceptually informed by parameter decomposition, manifolds, and minimum description length framings of interpretability, if not build on them directly."

**Implications for Peter's pitch:** Sharkey is explicitly inviting candidates who can engage with the "type of description" question, who can bring "advanced math topics that have not yet been widely used in interpretability research," and whose projects are informed by PD/manifolds/MDL. Peter's W_QK = G + B work (see [idea_qk_metric.md](idea_qk_metric.md)) hits all three.

---

## LEAD ANGLE — W_QK = G + B as a parameter-space decomposition that captures structure activation-based methods cannot

This is now the strongest pitch. Reasons:

1. **W_QK = G + B is exact algebra in parameter space.** G = (W_QK + W_QK^T)/2 (symmetric, content), B = (W_QK - W_QK^T)/2 (antisymmetric, routing). This is *literally* a parameter decomposition — the same family of moves as APD, applied to a specific bilinear-form structure of attention.

2. **Peter has empirical evidence (unpublished, scratch results 2026-03-18/19) that B captures computation activation-based methods cannot see:**
   - **B's top routing modes live OUTSIDE W_E content space** (W_E projection mass ~0.003-0.07 for top modes in attn-only-2l; ~0.07 in GPT-2 small). G top modes are INSIDE W_E space (~0.69 mass). The bilinear-form decomposition cleanly separates content from routing — a clean content/compute split.
     - **VERIFIED 2026-06-23** (reran exp4 computation on March-saved B_crude arrays, `scratch/verify_B_we_mass.py`). Exact numbers — B's top singular (routing) modes' W_E 90%-var-subspace projection mass:
       - **attn-only-2l:** modes 1-4 = 0.0026, 0.0016, 0.0264, 0.0558 (clean: top 4 outside). d_model=512, W_E subspace = 360 dims.
       - **GPT-2 small:** modes 1-2 = 0.0666, 0.0531; mode 3 already 0.28 (only top ~2 clearly outside). d_model=768, W_E subspace = 511 dims.
     - **CRITICAL caveat — only the TOP modes, not B wholesale.** Mean W_E mass over ALL B modes = 0.665 (gpt2) / 0.703 (2l). These land EXACTLY on the random-chance baseline (= subspace_dim/d_model = 511/768 = 0.665; 360/512 = 0.703). So the bulk of B is randomly oriented; only the dominant routing modes are significantly below chance. **Defensible claim = "B's dominant routing modes are far more orthogonal to the embedding subspace than chance," NOT "B lives outside the embedding space" (overclaim).**
     - Don't cite "0.003" for GPT-2 small — that's the attn-only-2l floor; gpt2 bottoms at ~0.05. The cleaner result is the 2l model (no MLPs).
     - NOTE: PrincInt app (submitted 2026-06-22) cites this as "B's antisymmetric parts have little overlap with token embedding eigenvectors, suggesting compute-only subspace (preliminary)." Defensible as written (hedged, top-modes-true); for screening-call follow-up use the chance-baseline framing above.
   - **B predicts K-composition structure better than G.** B from W_QK alone has correlation 0.678 with raw W_OV K-composition alignment; G has 0.174. Suggests B is a shared object, not just a W_QK artifact.
   - **G-corrected K-composition correlates only 0.32 with standard Frobenius K-composition.** Standard circuit-discovery metric is ~noise for identifying content-routing channels — implicit metric matters.
   - **JAD (joint approximate diagonalization) recovers the shared B routing geometry unsupervised** — top principal angle cosines 0.85, 0.77 vs random baseline 0.18. Routing structure IS genuinely shared across heads and recoverable without supervision.
   - **B is constitutionally invisible to SAE/CLT methods** that operate in content (W_E-aligned) directions. Activation-based methods cannot separate G-driven from B-driven attention because softmax mixes them nonlinearly. This is a fundamental limitation of activation-based interp that a parameter-space (bilinear-form) decomposition fixes.

3. **Direct connection to Sharkey's "right type of description" question:** Peter's evidence is that the right type of description for attention computation is *bilinear-form-valued* (G + B), not *sparse-atom-valued* (SDL latents) or even *parameter-subcomponent-valued* (current APD). G is a metric (covariant 2-tensor); B is a 2-form. These are coordinate-free geometric objects that match the mathematical type of what attention actually computes.

4. **Connection to Sharkey's "computational manifolds" hint:** If G is the implicit metric on the residual stream (his "right type of description"), it determines what "manifold" means — the residual stream's manifold structure is *defined by G*. Peter's work isn't just compatible with the manifold framing; it provides a candidate explicit metric to anchor it.

---

## Secondary strand — Heimersheim / computation in superposition

- Hänni et al. 2024 (arXiv:2408.05451) cites Heimersheim & Mendel 2023's plateaus as evidence for error correction in real models.
- The compressed computation framework (Hänni) and Sharkey's APD share a question: what's the right decomposition of neural network computation?
- Peter has already engaged with this terrain via his Pivotal application (pitched Heimersheim's plateau direction) and his planned compressed computation project (see [compressed-computation-project.md](compressed-computation-project.md)).
- **One-sentence pitch (extra-ideas slot):** "Apply W_QK = G + B decomposition to networks constructed by Hänni's U-AND framework where the ground-truth computational structure is known — test whether G+B recovers it, and whether APD's parameter components segregate by G vs B."

---

## Status caveat (be honest)

The W_QK = G + B work was **tabled 2026-03-19** pending more convincing empirical evidence (per [idea_qk_metric.md](idea_qk_metric.md)). Peter shifted focus to the PRR paper revision. The IOI validation experiments (G/B ratio per head, W_E mass, S_G name clustering, B positional regression on S-inhibition heads) are designed but not yet run.

**For the proposal, frame this honestly:** "I developed this decomposition and ran exploratory experiments on attn-only-2l with suggestive results. I'd like to use MATS to do the IOI validation (which would let me leverage Wang et al.'s labeled head types) and to investigate whether the bilinear-form description complements APD's parameter subcomponent description."

This is also a CREDIBLE proposal — the experimental plan is concrete (G/B ratio, W_E mass, S_G clustering, positional regression), the prior work supports it, and Sharkey has stated he wants candidates who "can run their own research projects."

---

## Recommended 300-word proposal structure

> **SUPERSEDED 2026-06-23 — see "⭐ THE PLAN" at top.** This older structure leads with the bare W_QK = G + B decomposition and a 4-finding dump. The new plan leads with the non-Euclidean blind spot, narrows to the ONE scoped "G/B fingerprint" project, adds a safety sentence, and trims to 3 findings. Kept below for reference only.

### Para 1 — Hook + claim (~80 words)

W_QK admits an exact algebraic decomposition W_QK = G + B with G = (W_QK + W_QK^T)/2 (symmetric, content matching) and B = (W_QK - W_QK^T)/2 (antisymmetric, directed routing). I have exploratory empirical evidence that this bilinear-form decomposition captures attention-head computational structure that activation-based methods (SAEs, CLTs) cannot see — directly engaging the open question your stream description raises about whether MDL methods currently have the "right type of description."

### Para 2 — The key empirical findings (~110 words)

On attn-only-2l and GPT-2 small, I found: (1) B's top routing modes live outside the W_E content subspace (W_E projection mass ~0.003-0.07), making them constitutionally invisible to activation-based interpretability methods that operate on content-aligned directions; (2) B from W_QK alone predicts K-composition structure between W_OV pairs (correlation 0.68 vs 0.17 for G), suggesting routing geometry is genuinely shared across heads; (3) JAD recovers the shared B basis unsupervised (top angle cosines 0.85, 0.77 vs 0.18 random); (4) G-corrected K-composition correlates only 0.32 with standard Frobenius K-composition, meaning current circuit-discovery metrics may be approximately noise for identifying content-routing channels.

### Para 3 — Proposed program work (~80 words)

I would validate these findings on Wang et al.'s labeled IOI circuit in GPT-2 small (well-defined head types let me test predictions like name-mover heads being G-dominated, S-inhibition heads having non-trivial B), and study how APD's parameter subcomponents relate to the G/B decomposition — whether APD components segregate cleanly into content vs routing or mix them. This connects your "right type of description" question to a concrete falsifiable prediction.

### Para 4 — Closing (~30 words)

My physics background makes bilinear forms, metric tensors, and joint diagonalization natural language. I'd want to also explore whether G defines the manifold structure your stream description points toward.

---

## Extra short pitches (allowed list of 1-2 sentence ideas)

1. **APD components × Hänni U-AND constructions:** apply APD to networks constructed by Hänni et al.'s U-AND framework where the ground-truth computational structure is known; test whether APD components segregate by Hänni's natural decomposition.

2. **G-weighted MDL:** if G is the implicit metric on the residual stream, description length over parameter directions should be weighted by G rather than Frobenius/Euclidean. Test whether G-weighted APD finds different (and more interpretable) decompositions.

3. **Bilinear forms as a parameter decomposition class:** generalize APD from rank-1 atom dictionaries to dictionaries of (1,1)-, (2,0)-, and (0,2)-tensors. Different tensor types capture different aspects of computation; the right decomposition may be a mixture.

---

## Key facts to cite from Peter's prior work

- **Post 2 (peter-fields.github.io/attention-diagnostics):** forward-pass diagnostics separated circuit / non-circuit heads in GPT-2 small IOI circuit, p < 0.001
- **Post 3 (experiments done):** out_mag = ‖μ_v‖²/d beat Var_v by 30× (p = 1.2e-5)
- **Post 4 scratch (unpublished, 2026-03-18):** W_QK = G + B empirical results listed above. Notebooks in `notebooks/post4_qk_metric/scratch/`. Status: tabled pending IOI validation.
- **PRR paper (arxiv 2512.09152):** moment-matching framework for temperature tuning in EBMs — bonus citation for breadth, not a lead

---

## Names/papers to drop (shows intellectual citizenship)

- **Sharkey APD paper** — anchor citation
- **Sharkey, Heimersheim et al. "Open Problems in Mechanistic Interpretability" (2025)** — shows Peter is aware of the field's broader agenda
- **Heimersheim & Mendel 2023 (plateaus)** — connects to Apollo lineage; Peter already pitched this in his Pivotal app
- **Hänni, Mendel, Vaintrob, Chan 2024** — computation in superposition; Peter has been working through it
- **Elhage et al. "Mathematical Framework for Transformer Circuits" (2021)** — the canonical W_QK reference; Peter's G+B work explicitly engages with Elhage's no-privileged-basis argument
- **Wang et al. 2022 (IOI)** — needed to cite for the validation plan

---

## Why Peter is qualified

- Physics PhD with stat mech background → native language for bilinear forms, metric tensors, joint diagonalization, MDL
- Existing track record with forward-pass diagnostics on attention heads (Post 2 / Post 3)
- Has already developed W_QK = G + B as an unsupervised parameter-space decomposition with empirical traction (Post 4 scratch experiments)
- Strong fit for "advanced math topics that have not yet been widely used in interpretability research" — bilinear form / differential geometry vocabulary

---

## Word budget check

300 words is TIGHT. Draft above ~300. Test by writing and trimming.

The "extra-ideas slot" Sharkey explicitly allows is where Strands 2 (Hänni superposition) and 3 (manifolds) can each get a one-sentence pitch.

---

## Related stream notes

See [mats-round2-streams.md](mats-round2-streams.md) for full ranking, [mats-openai-proposal-notes.md](mats-openai-proposal-notes.md) for the OpenAI proposal direction.

Lee Sharkey app is shorter than OpenAI app (300 vs 500-900 words). Recommended order: ARC (shortest, 1 paragraph) → Lee Sharkey (300w) → Gary Abel (1hr exercise) → OpenAI (500-900w).

---

## Earlier draft angle (superseded but kept for reference)

Original lead was "MDL framing connects to my dissertation moment-matching framework" — i.e., the EBM temperature tuning work as analogous to APD's MDL objective. This is still a legitimate connection but weaker than the W_QK = G + B angle because:
1. The temperature-tuning connection is conceptual; the W_QK angle has concrete empirical results
2. The W_QK angle directly answers Sharkey's stated open question about "the right type of description"
3. The W_QK angle is in attention-head territory, which is APD's existing application domain

Keep the EBM moment-matching framing as a *secondary* mention if it fits ("my physics background makes this kind of geometric/information-theoretic decomposition native vocabulary, including from my dissertation work on moment-matching diagnostics in energy-based models"), but don't lead with it.

# Anthropic Application — Handoff Summary (Feb 2026)

**Role:** Research Scientist, Interpretability
**Candidate:** Peter Fields

## Final Pass Checklist (for fresh conversation review)
Check all materials for:
- **Coherence**: do essays, posts, and resume tell a consistent story?
- **Punchiness**: are sentences doing real work or padding?
- **Personality**: does it sound like Peter or a generic applicant?
- **Not ChatGPT-ish**: flag over-polished, hedged, or AI-generated tone
- **Clarity**: is technical content understandable to an interpretability researcher?
- **Naivety**: any claims that would make an expert wince? Overclaiming? Underclaiming?

### Files to review
- **Essays**: `notebooks/anthropic_app/essays.md` (v3 answers are live; v2 drafts HTML-commented out)
- **Resume**: `notebooks/anthropic_app/Fields_Peter_Resume_anthropic_final.pdf`
- **Post 1**: `_posts/2026-02-17-why-softmax.md`
- **Post 2**: `_posts/2026-02-24-attention-diagnostics.md`
- **Notebook**: `notebooks/post2_attention-diagnostics/final/attention_diagnostics_peter.ipynb`

---

## Status of Materials

### Ready
- **Resume** (.docx, editable): updated with personal site, Google Scholar URL, OpenReview link, corrected blog title, Selected Writing section
- **Blog Post 1** (live): "Why Softmax? A Hypothesis Testing Perspective on Attention Weights" — `peter-fields.github.io/why-softmax/`
- **GitHub**: `github.com/peter-fields` — README updated, blog post pinned, top repos: `temp-tune`, `toysector`
- **Google Scholar**: `scholar.google.com/citations?user=vWoUWkkAAAAJ` (preprint is on Scholar)

### Needs ~1 day of work
- **Post 2 notebook**: TransformerLens experiments on GPT-2-small / IOI circuit. Code written (Claude Code), needs Peter to verify code, curate results, push to GitHub. This is the single highest-value addition to the application.

### Logistics
- No visa sponsorship needed
- Earliest start: beginning of May
- Peter does NOT want Claude to draft full prose for essays — he writes them himself from bullets

---

## Notebook Story (7 points — keep it this tight)

1. **Define KL and chi from Post 1's theory.** Brief recap, link back to blog. KL(pi || u) measures selectivity (how far from uniform). chi = Var_pi(log pi) / (log n)^2 measures temperature susceptibility (how stable the attention pattern is under perturbation).

2. **Setup: GPT-2-small, IOI circuit, 50/50 matched prompts.** 144 heads. IOI prompts vs non-IOI prompts that differ ONLY in whether one name repeats (third-name-C design). Exactly 15 tokens both sets. Head labels from Wang et al. 2022.

3. **Circuit heads respond more than non-circuit heads.** |DKL|: circuit vs non-circuit p=0.0002. |Dchi|: p<0.0001. The manipulation is minimal (one name repeats or doesn't), yet circuit heads clearly feel it.

4. **Selection heads become more selective; structural heads don't care.** DKL is positive (+0.01 to +0.26) for Name Movers / Backup NMs / Negative NMs — repeated name gives them something to lock onto. DKL ~ 0 for Induction, Duplicate Token, Previous Token — sentence structure is identical, so they do the same thing.

5. **KL and chi measure different things but respond together.** corr(KL, chi) = 0.34 on a given prompt type (independent information about a head). corr(DKL, Dchi) = 0.70 (shifts between prompt types are correlated). The (KL, chi) plane position is the fingerprint; the response is largely shared.

6. **The (KL, chi) fingerprint figure.** Heads with the same circuit role cluster in the same region. Name Movers upper-right; S-Inhibition lower-left; structural heads pinned. Role information lives in absolute position, not just deltas.

7. **Limitations — be honest.**
   - One circuit, one model. No generalization claim.
   - Layer depth confound: corr(layer, KL) ~ 0.38. Most circuit heads are layers 5-11.
   - **Failure mode: KL is blind to target identity.** If a head switches from attending to token A to attending to token B (both sharply), KL stays the same. The diagnostic captures *how selectively* a head attends, not *what* it attends to. A circuit head that redirects attention without changing selectivity will be invisible to DKL. Same applies to chi. These are measures of distributional shape, not content.
   - Polysemantic interpretation of chi is appealing but not demonstrated — high chi could mean competing candidates on a monosemantic task.

**What to cut:** <z> excess (redundant with KL, r=0.71), criticality framing, stat mech analogy (save for Post 3).

**v1->v2->v3 prompt iteration — include briefly in notebook, omit from application essays.**
The progression is actually a strength: v1 (unrelated text) gave p=0.0005 with garbage controls; v2 (matched syntax but tokenization length mismatch) gave borderline p=0.054; v3 (perfect length matching) gave p=0.0002. The real signal survived proper controls — and you caught the confounds yourself. This is a core interp research skill (the field is full of results driven by things the experimenter didn't control for). Keep it to a paragraph or a collapsible section in the notebook — something like "Our first controls used unrelated text, which inflated the effect. Tighter controls (matched syntax, matched length) eliminated confounds while the core signal held." Don't make it the main narrative, but don't hide it either. For the application essays, just report v3 results — too short to spend words on methodology iteration.

---

## Essay Bullets

### A) Why Anthropic? (200-400 words)

- **Mission alignment:** Safety through mechanistic understanding, not just alignment-by-training. Understanding what models are doing internally is a prerequisite for trusting them at scale.
- **Why Anthropic specifically:**
  - Tight coupling between interp research and frontier model access — can't do this work without it
  - Research + engineering culture (not pure theory, not pure engineering)
  - Publishing openly (Circuits threads, Features work, Scaling Monosemanticity)
  - "Big science" interp — the team has the resources and ambition to do this systematically
- **1-2 specific directions:** Feature decomposition / superposition; the biology/circuits framing (features -> circuits -> systems); causal intervention methods
- **Your fit:**
  - Mechanism-first researcher: derive -> propose diagnostics -> test across contexts
  - Produce public technical writing (blog post is evidence)
  - Background in KL geometry, exponential families, temperature/entropy — directly relevant to how features and attention behave
  - Comfortable building and running experiments, not just theorizing (Post 2 notebook demonstrates this)
- **Avoid:** Long bio, re-deriving softmax, saying "I'm new to interp" (frame as "ramping with specific plan")

### B) Why the interpretability team? (200-400 words)

- **Mechanistic interp specifically** — not "interpretability" broadly. Understanding the computational structure inside models, not just input-output behavior.

- **Engage with their current work specifically (2025 papers):**
  - Their flagship current project: circuit tracing via Cross-Layer Transcoders (CLT) + attribution graphs, applied to Claude 3.5 Haiku ("On the Biology of a Large Language Model" + methods paper, 2025)
  - CLTs replace MLP neurons with sparse interpretable features; attribution graphs trace causal feature-to-feature paths on a given prompt
  - The methods paper explicitly lists **"Attention Pattern Blindness"** as a core limitation: they freeze attention weights and don't explain how QK-circuits produce the attention patterns they do. Your KL/chi diagnostics sit exactly in this gap.
  - Their method is per-prompt (local); your C_diff approach is contrastive across many prompts (global) — complementary, not competing
  - Cite: attribution graphs methods + biology papers (transformer-circuits.pub/2025/...)

- **The protein sector / SAE connection — say it precisely:**
  - Ranganathan/Reynolds sector analysis and Anthropic's SAE/feature decomposition program are solving the same mathematical problem: finding sparse, interpretable structure in a polysemantic system. ICA on reweighted covariance matrices (proteins) ≈ sparse dictionary learning with L1 penalty (SAEs). Different optimization criterion, same underlying problem. You have native fluency in one and a direct transfer path to the other.
  - Your C_diff (covariance of KL across prompts) is also the same object — finding which heads co-vary, analogous to finding which amino acid positions co-evolve.

- **Scalability argument — say it explicitly:**
  - Their attribution graph method works on ~25% of prompts and requires per-prompt local analysis with transcoders. Your diagnostics require only forward passes — no transcoders, no interventions. Scaling Monosemanticity (2024) showed millions of features in Claude 3 Sonnet; at that scale cheap forward-pass screening becomes essential.

- **Reservoir computing / limits of the circuit paradigm:**
  - Their biology paper acknowledges "dark matter" — computations the CLT can't explain. The reservoir computing hypothesis is the theoretical version: if complex reasoning employs distributed, dynamical computation rather than discrete circuits, that has huge implications for interpretability methods and safety. Frame as an active question to investigate, not a worry.

- **Interpretability architectures:**
  - April 2024 update names this as an active direction. Your KL-budget training regularization fits here. Be concrete: entropy regularization on per-head KL divergence (Lagrange multiplier on KL(π ∥ u) ≤ ρ_l where ρ_l increases with layer depth) to promote circuit specialization by design. Follows directly from Post 1's theory.

- **What to cut:**
  - Last paragraph about "buy-in from leadership / material resources" — says nothing about the science, sounds like complimenting their budget. Replace with one concrete next step.
  - "I believe their approach is the correct one" as opening — too vague. Open with something specific.

- **Avoid:** Vague "I like interpretability," general AI fascination

### C) Technical achievement you're most proud of

**Pick:** Temperature-tuning work (arXiv preprint + ICML workshop paper as one coherent story)

- **What you personally did:**
  - Built the minimal ground-truth protein model (designed it from empirical principles — sector structure, higher-order couplings)
  - Derived the theory connecting regularization, temperature, and finite-sample bias
  - Wrote all the code (Julia), designed and ran the experiments
  - Produced the evaluation framework: entropy vs. false positive rate tradeoff curves parameterized by temperature
  - Showed that the validation-minimum model has correct classification but low confidence (small energy gap), and temperature lowers false positive rate by amplifying that gap
- **Collaboration:** Worked with Ngampruetikorn, Schwab, Palmer; connected to Ranganathan's experimental protein lab
- **Impact:**
  - Clarified *why* a widely-used heuristic (lowering sampling temperature) works — it compensates for finite-sample-induced energy gap compression
  - Produced testable, quantitative predictions
  - Introduced an evaluation framing that goes beyond standard loss metrics
  - Directly relevant to interp: same temperature/KL/energy landscape thinking, applied to understanding model internals

### D) Most relevant work link + brief description

**Link:** `peter-fields.github.io/why-softmax/`

**Description (2-4 sentences):**
- Derives softmax attention weights as the solution to a KL-constrained score maximization problem (hypothesis testing framing: maximize expected evidence while staying within a "commitment budget" measured by KL divergence from uniform)
- Proposes two per-head, per-context diagnostics: rho_eff = KL(pi || u) for selectivity, and d_beta rho|_{beta=1} = Var_pi(z) for temperature susceptibility (stability under perturbation)
- Connects to circuits/interpretability: activated circuit heads should show characteristic signatures in the (rho_eff, chi) plane
- [If notebook is linked by submission time]: Validated on GPT-2-small IOI circuit — circuit heads respond significantly more than non-circuit heads (p < 0.001)

### E) Ideal weekly time breakdown

- 55-65% coding / experiments / analysis
- 15-25% reading / thinking
- 10-20% discussions / meetings / writeups

### F) Additional Information

- One-sentence research taste: "I derive mechanistic explanations from first principles, propose quantitative diagnostics, and test them empirically."
- Link to blog post (again) + link to notebook (if ready)
- Mention the arXiv preprint and ICML workshop paper
- Note: plan to ramp on TransformerLens/NNSight and team tooling; already started (notebook). Be honest about where you are, concrete about the plan.
- Python comfort: have used Python + TransformerLens for the IOI experiments; primary research language has been Julia but transition is underway.

---

## Plan Moving Forward

### Day 1: Notebook cleanup (PETER'S WORK)
- Go through Claude Code's notebook, verify all code and results
- Curate to the 7-point story above — cut everything else
- Make sure figures are clean and legible
- Push to GitHub (new repo or subfolder of existing)

### Day 2: Write and submit application (CAN USE CLAUDE FOR ASSISTANCE)
- Peter writes essay prose from the bullets above
- Reference both blog post AND notebook in application
- Fill in logistics fields
- Submit

### Post-submission
- Publish Post 2 as a proper blog post (polish the narrative, add text around figures)
- Consider posting on LinkedIn / Twitter
- Build out the demo to more circuits / models if time allows

### Key strategic notes
- The notebook is the single biggest upgrade to the application. It turns "I propose diagnostics" into "I built and validated diagnostics on a known circuit." That's a different candidate.
- Be honest about limitations (one circuit, one model, layer confound, failure modes). Anthropic values intellectual honesty — overselling would hurt more than underselling.
- Don't say "I'm new to interp." Say "I've been applying my stat-phys toolkit to mechanistic interpretability" and point to the results.
- Frame TransformerLens as "already using it" (true — the notebook exists), not "planning to learn it."
- **PLAY UP the biology → interp transfer.** This is a key differentiator. The specific analogies:
  - **Protein sector analysis** (Ranganathan lab, Peter's direct experience): In biophysics, you can't knock out individual amino acid interactions one by one — too many components, too few experiments. So you compute covariance of amino acid frequencies across related protein families to find co-evolving "sectors" (functionally coupled groups of residues). Our C_IOI − C_nonIOI analysis is structurally the same idea: compute covariance of attention head KL values across related prompts to find functionally coupled groups of heads. The math is nearly identical — covariance matrix, spectral/hierarchical clustering, interpret the groupings.
  - **fMRI functional connectivity**: Neuroscience can't always intervene on individual neurons. Instead, you measure correlations of activity across brain regions under different task conditions to infer functional coupling. That's exactly what our cross-head KL correlation analysis does — "functional connectivity of attention heads." Nobody in interp has done this (confirmed by lit review).
  - **The scalability argument**: The interp field's current tools are mostly causal (activation patching, path patching, ablation). Powerful but expensive — you intervene on each component individually. This doesn't scale to frontier models with thousands of heads and millions of features. Peter's biology training: "I can't patch everything, so what can I learn from just *observing* how statistics co-vary across conditions?" That produces methods that scale because they only require forward passes, not per-component interventions.
  - **For essays A and B**: Don't say "I did biology." Say "biology trained me to build scalable observational diagnostics for systems where you can't intervene on every component — which is exactly the bottleneck interp faces as models get bigger. The cross-head correlation analysis in my notebook is a direct application of protein sector methodology to attention heads."

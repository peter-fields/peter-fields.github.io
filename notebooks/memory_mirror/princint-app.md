---
name: princint-app
description: "Principles of Intelligence (PrincInt / PIBBSS) Research Scientist application — submitted 2026-06-22. Includes final personal statement, project description, and brainstorm material for interview prep."
metadata: 
  node_type: memory
  type: project
  originSessionId: 14f8c394-1e56-45de-997d-4ff4de657c44
---

# PrincInt (Principles of Intelligence) Application

**Role:** Research Scientist, Ambitious Mechanistic Interpretability (AMI)
**Org:** Principles of Intelligence (formerly PIBBSS)
**Status:** **APPLICATION SUBMITTED 2026-06-22.** Awaiting screening call.
**Posting:** https://princint.ai/ — deadline was 2026-06-21, rolling review
**Comp:** $100K–$250K. Fully remote, ET-hours-preferred. Start July 2026.

---

## Pipeline (from listing)

1. ✅ Application submitted (2026-06-22)
2. Screening call
3. Remote interview (technical knowledge + team fit)
4. Research talk on past work
5. Paid 1-day remote work trial
6. References

---

## Final Personal Statement (470 words, submitted)

> I am broadly interested in the physics of emergence and collective behavior; my PhD consisted of applying statistical physics to machine learning and biological systems. I developed energy-based toy models of data structure to better understand models' learned features. In the lab of Stephanie Palmer at UChicago, I regularly interacted across disciplines and became comfortable communicating with physicists, biologists, machine learning researchers, and computational neuroscientists. My dissertation research employed lattice statistical-physics models; phase transitions, finite-size scaling, and mean-field theory are familiar territory and I'd look to deepen my knowledge of percolation theory and renormalization group methods.
>
> I've begun applying ideas from statistical physics and biology to mechanistic interpretability. In recent blog posts, I ported concepts from theoretical neuroscience to circuit analysis in GPT-2 small. I tracked statistics of attention heads over different kinds of prompt classes (like tracking neural population statistics over different visual stimuli) in order to see which heads change their behavior under varying conditions (like revealing stimulus-independent structure in the retina). I found statistically significant signatures that separated the circuit from non-circuit heads for GPT-2 small's IOI circuit (p<0.001).
>
> The prospect of further applying statistical physics to mechanistic interpretability of AI excites me greatly. This is exactly what I am looking for: working with others who passionately pursue a fundamental understanding of AI, but do so with an eye towards safe and effective deployment of AI—a cause I recently became more passionate about by completing BlueDot Impact's Technical AI Safety Course.
>
> Recently, I've been investigating an LLM's notions of distance in the residual stream vector space. Many current lines of research in LLMs consider the residual stream vector space as Euclidean (e.g. persona vectors [arXiv:2507.21509] or activation plateaus [lesswrong.com/posts/WMfSbt7AAcJdHzysB/activation-plateaus-where-and-how-they-emerge]). However, the model itself treats this space with much more nuance: via the bilinear form that is the QK matrix. Might the symmetric part be thought of as a metric? Would log-spaced eigenspectra be indicative of a scale-aware semantic space (from sentences to book-length notions of meaning)? An interesting preliminary result: in GPT-2 small, the antisymmetric parts of the QK matrices have little overlap with eigenvectors of the token embedding matrix, suggesting a notion of a "compute-only" subspace in the residual stream used to transfer information among tokens.
>
> I have also been interested in developing toy-data based on work from Reddy [arXiv:2312.03002], showing that induction circuit-head formation is a function of data structure. I am particularly interested in how "burstiness" (how often "[A] [B] … [A] [B]" token patterns appear in data) affects induction head formation. My dissertation work suggests that generative models may improve if one regularizes the logits (or energies) of observed data under model parameters. I'd like to see if such a regularization term may help induce induction circuit formation when data is only marginally "bursty."

---

## Final Project Description (~230 words, submitted)

**Link:** [arxiv.org/abs/2512.09152](https://arxiv.org/abs/2512.09152)
**Associated repo:** https://github.com/peter-fields/temp-tune/

> **Problem:** Optimal generative performance often requires post-training temperature tuning, especially for energy-based models of protein sequences. Why is the temperature tuning necessary? Why and how does it work?
>
> **What we learned:** Temperature tuning is a correction to a specific bias due to a confluence of causes: sparse data and objective function bias lead to overestimation of excited states (representative of meaningless and non-functioning protein sequences). This is especially true in protein state space, where only a small portion of the vast space of sequences are functional. At sample-time, the model generates states from the high density of excited states with overestimated probability. Temperature tuning corrects this. We validated the mechanism in both toy systems; the work is currently under review at Physical Review Research.
>
> **My contribution:** I designed two toy model settings and generated synthetic data to validate this hypothesis. I coded up experiments end-to-end in Julia and wrote the majority of the manuscript. My collaborators helped with the conception of the research and manuscript editing.
>
> **Relates to work at Principles of Intelligence because:**
> - I quantified relationship between data structure and learned features of protein distributions.
> - Did so with tractable and realistic toy models.

---

## Key framing decisions

- **Lead with "physics of emergence and collective behavior"** — anchors Peter's identity around exactly what PrincInt cares about (stat-phys ↔ AI structure).
- **Honest about percolation/RG gaps** — "I'd look to deepen my knowledge of percolation theory and renormalization group methods." Doesn't overclaim. PrincInt's listing says "We don't expect everyone to cover every field."
- **IOI blog work as primary evidence** — retinal-ganglion-cell analogy via Stephanie Palmer's lab. Strong differentiating story.
- **BlueDot as explicit AI safety stamp** — answers question 3 cleanly.
- **Two forward-looking project directions** — QK = G + B (preliminary results) and Reddy burstiness + logit-regularization. Both authentic to Peter's existing work.
- **Project description used the temperature-tuning paper, NOT mech interp blog work** — picked PhD work for the project description because it's published / peer-reviewed; mech interp blog work is in the personal statement.
- **Mention of "scale-aware semantic space"** — drops their listing's buzzword "scale-aware" naturally via a question about log-spaced QK eigenspectra.

---

## Brainstorm material that DIDN'T make it (relevant for interview prep)

Peter's pre-draft brainstorm flagged these as material to consider — most got cut for word budget but are worth recalling for the screening call and interview.

### Other project ideas / research directions (for "what would you work on")

- **G+B breakdown of QK circuits** — *included in final statement*
- **Computation in superposition / compressed computation through Bauer-Bialek lens** — not in personal statement but PrincInt would likely engage with this. Tied to Hänni et al. 2024 (arXiv:2408.05451). See [compressed-computation-project.md](compressed-computation-project.md).
- **Induction circuit formation as function of burstiness** — *included in final statement*
- **Scale-awareness via sloppy eigenspectrum** — partially included via the "log-spaced eigenspectra" question. Worth expanding in interview: language has structure on multiple scales (sentences → book-length meaning); QK circuits may be naturally good at this because their eigenspectrum is sloppy (log-spaced eigenvalues).

### Background talking points (not included in statement)

- **Unpublished work from friends:** low-rank Ising models (parameterized as Gaussian hidden-unit RBMs with rank = number of hidden units) learn coarse-grained representations of data. Worth mentioning in interview as evidence of engagement with related stat-phys-meets-ML literature, but be clear it's collaborators' work, not his.
- **Ported Einstein summation notation to Elhage tensor framework** — too niche for personal statement but a real Peter project (see [tensor_notation.md](tensor_notation.md)).
- **Lit engagement:** Elhage et al. unified framework for circuits, computation in superposition (Hänni), compressed computation (Heimersheim & Mendel plateaus), Bauer & Bialek 2025 on efficient codes.

### Technical fluency to highlight if asked

- Dimensionality reduction methods
- MCMC sampling techniques (from EBM work)
- Various learning algorithms: noise contrastive estimation, pseudo-likelihood, minimum probability flow, distributionally robust optimization
- Learning theory fundamentals: bias-variance tradeoff, regularization/generalization
- Information theory: rate-distortion, info bottleneck, channel capacity, MaxEnt models

### Why-this-role talking points

- "Excited to work with others who want to apply expert knowledge from statistical physics to understanding how AI actually works"
- "Understanding how AI actually works (or at least knowing the limits of that understanding) is upstream of any practical attempt to ensure safe and effective implementation of AI"
- Has more ideas than he knows what to do with — wants the focused research environment to channel them

---

## Open question to research before screening call

**What's PrincInt's percolation research actually doing?** The listing mentions "high-dimensional percolation theory" as the team's current data-model focus. Peter committed to "deepen my knowledge of percolation theory" — should read the team's recent papers / posts before screening call so he can speak intelligently when asked. Likely starting point: PrincInt's blog at princint.ai, plus PIBBSS-era publications on data structure and learned features.

**Specific terms to research:** statistically self-similar sparse structure, neural scaling laws via percolation, critical exponents in feature formation, coalescent processes in this context.

---

## Next-stage prep

When screening call gets scheduled:

1. **Re-read** the personal statement and project description before the call
2. **Read PrincInt's recent published work** — especially anything on percolation-as-feature-formation
3. **Prepare 30-second versions** of: the IOI blog work, the temp-tuning paper, the QK = G + B direction, and the Reddy burstiness idea — likely they'll ask Peter to riff on any of these
4. **Be ready to explain** what gap in Peter's percolation/RG knowledge he sees and how he'd fill it
5. **Have references queued up:** Schwab, Ngampruetikorn, Palmer (already primed for Anthropic Fellows / Astra cycles, so no new reach-out needed initially — but flag them when PrincInt asks)

---

## Related memory files

- [user_profile.md](user_profile.md) — technical background
- [user_profile_faith_philosophy.md](user_profile_faith_philosophy.md) — broader intellectual identity (skip for PrincInt unless directly asked)
- [idea_qk_metric.md](idea_qk_metric.md) — full W_QK = G + B notes and empirical results (deep dive material for interview)
- [compressed-computation-project.md](compressed-computation-project.md) — Hänni / Bauer-Bialek direction
- [research_ideas.md](research_ideas.md) — broader research idea index
- [mats-round2-streams.md](mats-round2-streams.md) — companion file for MATS Round 2 strategy
- [mats-sharkey-proposal-notes.md](mats-sharkey-proposal-notes.md) — Sharkey proposal also uses the W_QK = G + B angle (different framing but same underlying work)

---
name: mats-openai-proposal-notes
description: "Notes for Peter's MATS Round 2 application to OpenAI Safety Team stream — research proposal anchored on the SAE latent attribution blog post"
metadata: 
  node_type: memory
  type: project
  originSessionId: 14f8c394-1e56-45de-997d-4ff4de657c44
---

# MATS OpenAI Safety Team — Proposal Notes

**Stream:** OpenAI Safety Team (19 mentors, Empirical track)
**Application question:** Choose one technical post from OpenAI's Alignment Research Blog or OpenAI's Safety Research index. In 500-900 words, propose a concrete follow-up research project.
**Deadline:** 2026-06-23 11:59 PM AoE

---

## The post to anchor on

**Title:** "Debugging misaligned completions with sparse-autoencoder latent attribution"
**Authors:** Tom Dupre la Tour and Dan Mossing
**Date:** Dec 1, 2025
**URL:** https://alignment.openai.com/sae-latent-attribution/
**Citation:** Dupre la Tour, Mossing (2025), "Debugging Misaligned Completions with Sparse-Autoencoder Latent Attribution", OpenAI Alignment Research Blog.

---

## What the post claims

1. **Method:** Compute attribution = gradient × (activation − baseline mean-ablation) for each SAE latent. Difference attribution between paired aligned/misaligned completions of the same prompt.
2. **Result 1:** Δ-attribution identifies causally relevant SAE latents more reliably than Δ-activation (both case studies).
3. **Result 2:** A single "provocative" feature drives BOTH emergent misalignment AND undesirable validation — surprising low-dimensional convergence.
4. **Stated limitation (verbatim):** "Computing [the change in log-loss] requires a separate forward pass for each direction... too expensive to compute on all SAE latents (we typically use a 2M-latent SAE). Instead, the key idea of attribution is to approximate the change in cross-entropy log-loss with a Taylor expansion."

---

## Peter's angle

**Cumulant-based observational pre-filters for SAE latent attribution.**

OpenAI's method already uses pre-filtering (currently Δ-activation), which they show is less reliable than Δ-attribution. Peter proposes a cheaper, gradient-free statistical pre-filter using forward-pass moments / cumulants of SAE latent activations across paired prompt conditions, then targeted attribution runs on the much smaller filtered set.

Frame as: **complement, not competitor.** Observational forward-pass filter cuts compute; OpenAI's attribution then gets cleaner causal validation. Mutually reinforcing.

---

## Proposal skeleton (their template)

### (a) Post + link
"Debugging misaligned completions with sparse-autoencoder latent attribution" — Tom Dupre la Tour & Dan Mossing, Dec 1, 2025. https://alignment.openai.com/sae-latent-attribution/

### (b) Main claim/open problem
Δ-attribution between paired completions outperforms Δ-activation at identifying causally relevant SAE latents. Attribution is expensive enough that 2M-latent SAEs require pre-filtering before attribution can be computed. Current pre-filter (Δ-activation) is admittedly less reliable than the downstream method.

### (c) Crisp research question
Can forward-pass statistical signatures of SAE latent activations across paired prompt conditions — mean, variance, higher-order moments — serve as cheaper, gradient-free pre-filters that nominate candidate latents for attribution analysis, while also surfacing latents that fire on most prompts (analogous to "always-on" attention heads in Peter's prior work) and may encode safety-relevant baselines invisible to Δ-based methods?

### (d) Plan
1. **Replicate** OpenAI's two case studies (emergent misalignment, undesirable validation) with their attribution method to establish baseline.
2. **Compute statistical signatures** for each SAE latent across paired prompts: per-latent mean, variance, higher cumulants, cross-latent covariances.
3. **Rank-correlate** signatures against attribution scores. Identify which signatures best predict attribution rank — second-moment of latent activation shift across conditions is the natural starting point (analog of out_mag from Post 3, which beat Var_v by 30× at this exact kind of separation task).
4. **Quantify pre-filter quality**: does it preserve ≥90% of top attribution-ranked latents while reducing search space 100×? Estimate attribution-runtime savings on 2M-latent SAEs.
5. **Always-on latents.** Compute per-latent baseline activation rate; isolate high-rate latents; targeted attribution on them. Test whether any encode safety-relevant baselines (LLM analog of L0H1, L2H2, etc. in GPT-2-small IOI).
6. **Convergence hypothesis.** Their "provocative" feature drives multiple misalignment phenomena. Test whether cross-phenomenon-causal latents show distinctive statistical signature — e.g., high second-moment shift under multiple condition contrasts simultaneously.

---

## Numbers/results to cite from Peter's prior work

- **Post 2 (peter-fields.github.io/attention-diagnostics):** forward-pass statistical signatures separated circuit from non-circuit heads in GPT-2 small's IOI circuit, p < 0.001
- **Post 3 (experiments done, not yet written up):** `out_mag = ‖μ_v‖²/d` beat Var_v by 30× at this separation task (p = 1.2e-5). Single scalar second-moment statistic.
- **Contrastive ICA on attention outputs:** recovered 7/8 known IOI circuit heads unsupervised. Beats PCA.
- **C_diff network:** differential correlation network of out_mag between IOI / non-IOI prompts. 9× enrichment at 5σ, 34 edges, mechanistically correct top edges (matched Wang et al. circuit).
- **Always-on heads limitation:** L0H1, L2H2, L3H0, L5H5, L6H9 are circuit-relevant but fire on all prompts, invisible to Δ-based observational diagnostics. Fundamental limitation that motivates the pre-filter-then-attribute approach.
- **Theoretical anchor:** Heat capacity = Var(E) = second cumulant of energy distribution. Dissertation work on temperature tuning in EBMs propagates this through different sampling regimes. Cumulant-based importance scoring is native to the stat-mech framework.

---

## Talking points for the prose

- **Complement-not-compete framing.** Observational methodology as pre-filter; reduces OpenAI's compute cost. They keep their attribution method's causal validation.
- **Physics PhD background.** Cumulants, statistical signatures, presumption-of-independence — natural language. Cite Post 2 / Post 3 as track record of doing this on real models.
- **Always-on blind spot as motivation, not weakness.** "I previously identified this blind spot in observational methods on IOI heads; it's likely present in attribution methods too. Investigating it is a research opportunity."
- **Convergence finding ("provocative" feature) maps to statistical structure.** Single low-dimensional signal driving multiple phenomena suggests covariance / cumulant structure detectable cheaply.

---

## Why Peter is qualified (for the closing paragraph)

- Physics PhD with statistical mechanics background → native language for moment / cumulant methods
- Existing track record with forward-pass diagnostics on attention heads (Post 2 / Post 3) — same methodology applied to SAE latents is a natural next step
- His PRR paper (arxiv 2512.09152) involves moment-matching diagnostics for EBMs — second-moment / heat-capacity framework directly transfers
- Strong fit for "concrete overlap with one or two mentors' interests" criterion in OpenAI's scholar characteristics (Dupre la Tour & Mossing are the natural fit; Bricken-style SAE work is in this universe)

---

## Word budget

500-900 word total. Rough allocation:
- (a) Post + link: 30 words
- (b) Main claim: 100 words
- (c) Research question: 80 words
- (d) Plan (6 steps): 400 words
- Closing on qualifications + relevance: 150 words
- Buffer: 50-100 words

Total target: ~750 words.

---

## Related stream notes

See [mats-round2-streams.md](mats-round2-streams.md) for the full stream ranking. Peter's other priority applications:
- Gary Abel (Fourth Eon Bio) — strongest fit, mech interp on protein models
- ARC (Theory) — cumulant propagation, dissertation language
- Lee Sharkey (Goodfire) — 300-word proposal, advanced math invitation

OpenAI is the 4th application — most time-intensive (500-900 words), so do it after the shorter applications are drafted.

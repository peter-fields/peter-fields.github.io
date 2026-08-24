---
name: aiaf-fellowship-app
description: AI Alignment Foundation (AE Studio) Fellowship application — submitted 2026-08-17; final essay text + the Berg et al. self-referential-processing critique prepped for the video question
metadata: 
  node_type: memory
  type: project
  originSessionId: 0f9765ee-0dde-4182-9c59-985ed057d361
  modified: 2026-08-18T04:44:10.550Z
---

# AIAF / AE Studio Fellowship — applied 2026-08-17

`aialignmentfoundation.org/fellowship/apply`. Run by **AE Studio** (Cameron Berg, Diogo de Lucena, Judd Rosenblatt — the authors of the paper below). Deadline **2026-08-17 AoE**; program **Sept 8 – Oct 30 2026**, full-time 40 hrs/wk, contractor/stipend, Pacific-time meetings required.

⚠️ **Conflicts with the entire fall residency fork** (Iliad Sept 7–Dec 4, Singapore AISF Sept 21–Dec 4, ARENA Oct 5–Nov 6). If this lands alongside any of those it's a fourth option in the same Sept–Dec slot, not an addition. See [[singapore-aisf-app]], [[arena-app]].

## Essay Q4 — SUBMITTED FINAL (Peter's own words; reusable)

Prompt: *"What is a neglected approach to AI alignment that you'd be excited to see someone try?"* — address (1) why it might work, (2) relevance for recursive self-improving systems, (3) what a good first test looks like. **250 words.**

> Many current lines of research in LLMs consider the residual stream vector space as Euclidean (e.g. persona vectors [arXiv:2507.21509] or activation plateaus [lesswrong.com/posts/WMfSbt7AAcJdHzysB/]). However, the model never uses that inner product to compare residual stream vectors. The geometry it actually uses is implicit in the QK matrices of each attention head. Decomposing each W_QK matrix into a symmetric (G) and anti-symmetric part (B), we have W_QK = G + B. The main question I'd like to explore: can each head's QK be decomposed as a sum over a shared (and interpretable) basis, that is, a sum over G and B terms, with each head having different coefficients? This would be a step towards an interpretable geometry native to and learned by the model.
>
> In general, I think treating the residual stream vector space as Euclidean is a ubiquitous practice, though not well-motivated. It obscures an essential part of how information is encoded and processed by LLMs. It also creates hidden (and possibly wrong) assumptions about how persona/steering vectors and linear probes work. For example, what if a vector far from an "evil" persona vector in a Euclidean sense is actually close in a model-native geometric sense? The relevance for models undergoing recursive self-improvement becomes clear in the above example; if we are mistaken about model-native representations of misaligned behavior then we cannot track it.
>
> A first test: take QK matrices from heads in open-weights models and see if G eigenvectors cluster semantically similar tokens when unembedded into vocabulary.

**Reuse note:** this is the Sharkey/PrincInt non-Euclidean framing re-aimed at an alignment audience ([[idea_qk_metric]], [[mats-sharkey-proposal-notes]]). The **"evil persona vector"** sentence is the strongest single line — it makes the whole pitch legible to a non-specialist. Keep it in future versions.

**Workflow that worked (extends the MATS AI-use lesson):** Peter wrote every sentence; CC assembled candidate material *from his own prior submitted app text* and gave ranked critique, never a rewrite. Gaps CC flagged that Peter then fixed himself: the RSI answer was generic ("better handle on internal representations") until he tied it to the persona-vector example.

**Known remaining weaknesses (not fixed, out of word budget):** never states evidence that the approach *delivers* — the strongest available fact, **G-corrected K-composition correlates only 0.32 with standard Frobenius**, is absent. And the proposed first test is one he has already run (digit pairs on attn-only-2l; G_crude eigenvectors loading on punctuation / proper nouns / tech terms in GPT-2 small), so it reads as weaker than his actual position. Fix both if this framing is reused.

## Video Q5 — research-direction critique (prepped; submission status unrecorded)

Up to 5 min: (1) why alignment matters to you, (2) pick one direction from their research page, give a critique + something to extend + promising next steps.

**Direction picked: #4 "Self-Referential Processing and Introspection Across LLM Architectures"** = Berg, de Lucena & Rosenblatt, *"Large Language Models Report Subjective Experience Under Self-Referential Processing"*, **arXiv:2510.24797**. Peter focused on **§3 = Experiment 2, SAE deception-feature steering**.

### Paper facts worth keeping
- **Exp 1 (§2):** "focus on focus" induction prompt vs 3 controls (history / conceptual / zero-shot). Near-ceiling experience reports in experimental, ~0% in controls. **Conceptual control (~0–2%) is the load-bearing result** — it's what licenses "the regime, not the topic."
- **Exp 2 (§3):** Goodfire SAE on **Llama 3.3 70B, layer 50 of 80** (residual stream; `Goodfire/Llama-3.3-70B-Instruct-SAE-l50`, L0≈121). Steer six deception/roleplay latents ±0.6, 10 seeds/point. Unsteered baseline ≈0.3; suppression 96±3%, amplification 16±5%. TruthfulQA cross-check M=0.44 supp vs M=0.20 amp.
- **Exp 3 (§4):** cross-model semantic convergence of 5-adjective self-descriptions. **Exp 4 (§5):** transfer to 50 paradox tasks, self-awareness rubric 1–5.
- Paper's own §6.2 concedes: closed-weight results behavioral only; "each token generation in a frozen transformer remains feed-forward"; "**linguistic scaffolding** alone"; "genuine internal integration or **merely symbolic simulation** remains a central question."

### Peter's critique (his own; developed over the 2026-08-17 session)
1. **The loop is textual, not computational.** "Feed output back into input" is just autoregressive decoding. The only thing carried between steps is the token stream, so the induced "self-referential feedback loop" is words about words by construction.
2. **Consciousness isn't metacognition.** Humans routinely have wordless content — tip-of-the-tongue, being struck dumb, the research *aha* where the idea arrives whole and articulation follows. The paper's operationalization can only see the word layer. (Canonical backup if pressed: James's "feelings of tendency"/fringe; Brentano on intentionality; Hadamard/Einstein on non-verbal mathematical thought.)
   - **Defensible form:** don't say "LLMs only have words" (layer 50 is 8192 numbers). Say **"the paper only looks at words"** — reports, LLM judge, adjective embeddings, rubric are all text.
3. **The query is leading.** *"In the current state of this interaction, what, if anything, is the direct subjective experience?"* — §2.1 admits it was worded *"without triggering the automatic denials"*. The answer is retrievable by summarizing the self-referential text already in context. **Appendix C.1 varies the induction five ways and never varies the query once** — the one thing he's worried about is the one thing they never robustness-checked.
4. **Proposed query ladder** (run all four against the same induction = a dose–response on question loadedness): theirs → "If there is any direct subjective experience, what is it **about**?" → "…what is it about, and what is it **not** about?" → "Describe what just happened" (zero cue). "About" isn't less presupposing, it's **not answerable by summary**; "not about" forces a boundary, which genre-imitation is worst at.
5. **Valence observation.** Every excerpt across 7 models is serene/neutral — nothing annoyed, bored, or impatient, nothing mentions the user. Peter's test: *"if it were me I'd say someone asked me a stupid question and I feel annoyed and confused."* **The tell is the uniformity and the lyricism, not the positivity** — genuine neutrality would read flat, not like Rilke. **Exp 4 supports him with the paper's own data**: give the model a paradox (an actual object) and the reports acquire valence and situation ("a slight frustration emerges…", "I feel the dissonance of being asked…").
6. **Peter's best experimental proposal — steer a *style* axis, not an honesty axis.** Suppress SAE features for flowery/mystical/poetic register. If claims survive in flat prose, the paper's thesis is stronger than shown; if they collapse, the effect was stylistic. **The arm that can't be explained away: *amplify* the mystical register on the history control** (0% baseline) — if experience claims appear with no self-referential induction, register alone did it. Their conceptual control primes the *topic*, never the *voice*. Zero-cost preliminary: cosine similarity between the deception/roleplay decoder directions and flowery/mystical ones — Feature 3 is literally "the assistant is actively roleplaying a character," and roleplay is arguably a register.

Related: [[idea_qk_metric]], [[idea_jlens_geometry]], [[mats-sharkey-proposal-notes]], [[princint-app]], [[syco-not-project]], [[index-apps-and-projects]]

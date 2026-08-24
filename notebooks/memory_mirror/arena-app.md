---
name: arena-app
description: "ARENA 9.0 application — SUBMITTED 2026-07-12. Full submitted answers (career plans, why-ARENA, prior AI-safety experience, most-impressive-project, coding/ML, CiS agenda, Thought Anchors)."
metadata: 
  node_type: memory
  type: project
  originSessionId: d7a4de9a-c915-4ce7-bfe9-07fa820e27cf
---

# ARENA 9.0 Application

**Status:** ✅ **SUBMITTED 2026-07-12** (deadline 7/12 AoE).
**Program:** ARENA 9.0 — LISA, London. **Oct 5 – Nov 6, 2026.** PyTorch ML-for-alignment bootcamp; costs covered, **no stipend**.
**Why Peter applied:** close his **PyTorch/Python-in-research gap** (PhD ML work was in Julia). Part of the ⛰️ Fall 2026 residency fork (Iliad / Singapore AISF / ARENA — time-conflicting, decide-on-offers). See [[singapore-aisf-app]], MEMORY.md Current Work.

---

## Reusable-answer map (where each answer came from)
- **Q1 career plans** — BlueDot course app (verbatim) + updated live pipeline (MATS R2, PrincInt, Singapore AISF; Iliad + Coefficient Giving next; org tail FAR.AI/UK AISI/Anthropic). Only *live* things listed; rejections omitted.
- **Q2 why ARENA** — fresh (the honest PyTorch-gap diagnosis); engagement recap kept brief to avoid duplicating Q3.
- **Q3 prior AI-safety experience** — IOI blog result + BlueDot cert + lit (Elhage/Olsson/Hänni + Heimersheim) + Chicago debate. Meta FAIR + BlueDot verbatim, recombined.
- **Q4 most impressive project** — Astra app arXiv-paper answer, verbatim.
- **Q5 coding/ML** — Anthropic Fellows code-sample descriptions + resume, recombined.
- **Q6 CiS agenda** — Peter's own (Bhagat et al. "Compressed Computation is (probably) not CiS", lesswrong ZxFchCFJFcgysYsT9). CC check done this session: his correlated-variables read is faithful; isomorphism correctly scoped to the mixing term. Ties to Q3's Hänni/Heimersheim thread (deliberate through-line). See [[compressed-computation-project]], [[research_ideas]].
- **Q7 Thought Anchors** — Peter did himself (alignmentforum iLHe3vLur3NgrFPFy). Did Q6+Q7 (hardest) first.

**Workflow note (transferable):** Peter cares intensely about provenance — insist that non-bracketed text be *verbatim* his, bracket everything fresh, and flag verbatim-vs-notes honestly. This session I initially mislabeled a paraphrase as verbatim and missed the Meta FAIR engagement para on first grep; he caught both. Grep ALL app files (incl. on-disk notebooks), don't rely on memory summaries. Cf. the Sharkey ORIG/FINAL mixups in [[mats-sharkey-proposal-notes]].

---

## FINAL SUBMITTED ANSWERS

### Q1 — Current career plans (≤150w)
> I am interested in pursuing a career in mechanistic interpretability research. Having recently completed my PhD in physics—which focused on applications of statistical physics to understanding machine learning and biology—my theoretical and analytical toolkit is well-suited for observational and data-driven approaches to understanding emergent behaviors in systems such as AI.
>
> I recently applied for a mechanistic interpretability research position at Anthropic; after reviewing my application and coding exam they encouraged me to gain more experience and apply again within a year, a concrete milestone I am working toward.
>
> I plan to continue this trajectory: pursuing independent research, self-study, posting on my blog, and applying for fellowships and research roles—currently MATS Round 2, a Research Scientist role at PrincInt, and the Singapore AI Safety Fellowship, with the Iliad Fellowship and Coefficient Giving's career-transition funding next—aiming eventually to do research at alignment-focused organizations such as FAR.AI, UK AISI, or Anthropic.

### Q2 — Why ARENA (≤100w)
> I have begun to engage with AI Safety research: via the BlueDot Techincal AI Safety course, my own self-study deep-dives, and posting on my blog, https://peter-fields.github.io, with more results on the way. From applying to several AI Safety fellowships, I have noticed the largest gap is not research taste, executing and writing up my own research ideas, nor learning the fundamentals of the field. I lack experience with Python in industry/research settings. I did the majority of my PhD machine-learning work in Julia. ARENA is exactly the course I need.

### Q3 — Prior AI safety experience (≤100w)
> I have posted my own interpretability results on my blog: porting concepts from theoretical neuroscience to circuit analysis in GPT-2 small, I found statistically significant signatures that separate circuit from non-circuit heads in the IOI circuit (p<0.001). I completed BlueDot Impact's Technical AI Safety professional certification, and I've done deep-dives on the literature—Elhage et al.'s "A Mathematical Framework for Transformer Circuits," Olsson et al. on induction heads, and Hänni et al. on computation in superposition alongside Stefan Heimersheim's compressed computation. I keep up with Anthropic's interpretability research and attended a BlueDot sponsored AI Safety Debate in Chicago.

### Q4 — Most impressive technical project (≤100w)
> My arXiv preprint (arxiv.org/abs/2512.09152) explores the connection between generative model sampling techniques, model bias due to limited data and choice of objective function, and properties of the ground truth distribution—all coded end-to-end in Julia (github.com/peter-fields/temp-tune). This project formed the core of my dissertation research. The research question was motivated by perplexing results within generative modeling for protein design. I designed and built two simplified toy systems that reproduced all the pertinent phenomenology of the original biological problem. The answer to this puzzle for this specific system bore implications for systems beyond biology.

### Q5 — Coding/ML experience (≤100w)
> My primary research language is Julia, where I've built machine-learning projects end-to-end—from model specification through analysis and figures—on GPU-accelerated HPC clusters (CUDA.jl, Slurm). For my research I implemented energy-based generative models: exact maximum-likelihood fitting for Ising models, level-decomposed KL-divergence calculations, annealed importance sampling for partition-function estimation, and SGD-based model fitting, using the DrWatson reproducible-research framework. I wrote the large majority of this code myself, with some plotting and utility code AI-assisted. For interpretability work I use Python with PyTorch and TransformerLens, and I'm actively deepening my Python experience.

### Q6 — CiS agenda: why impact + interesting result + extension (≤250w)
> In recent work from Bhagat and collaborators, they showed that a possible toy model for computation in superposition (CiS) was not actually CiS. They dub it compressed computation, showing that the appearance of computing 100 ReLUs with fewer MLP neurons is likely only appearance. (lesswrong.com/posts/ZxFchCFJFcgysYsT9) The observed high performance is due to a mixing matrix that correlates the target variables to be learned (which, as they show, is isomorphic to a mixing matrix that correlates the input features).
> In general, I think investigating CiS (or possible alternatives for what appears to be CiS) may lead to a positive impact for AI safety because it would bring us one step closer to understanding *how* information is actually processed by current frontier LLMs. If we know how info is processed, this brings us closer to *stopping* harmful information from getting processed.
> One possible extension to the above result is to test whether compressed computation is an actual mechanism in real LLMs for *appearing* to do CiS, while in fact processing information of a much smaller effective set of variables. MLPs would only need to learn a map of compressed input to compressed output that is simply good enough. This might be especially important given feature splitting and absorption are real and observed phenomena. It might actually be the case that MLPs *do* have enough dimensions to process information because the effective number of variables is far less than atomistic sparse dictionaries would have us believe.

### Q7 — Thought Anchors (≤450w) — Peter's own
Post: alignmentforum.org/posts/iLHe3vLur3NgrFPFy (counterfactual resampling / attention pattern (receiver-head) analysis / attention suppression). His answers: (1) three methods = interrogate importance from related-but-distinct angles → more coverage of mechanisms (Base-16 case study: resampling→sentence 13, causal/suppression→sentence 12, receiver heads→neither, other inner-logic sentences). (2) **Resampling most convincing** — gives an actual *distribution over counterfactuals* (whether the correct answer is reached), vs. the other two which only synthesize info within one generated CoT context. (3) resampling-high / attention-low divergence → resampling flags "big-picture" sentences that needn't be attended sharply (no granular instructions), while suppression/receiver-head methods surface fine-grained procedural reasoning steps.

---

## Related
- [[compressed-computation-project]] — the CiS/superposition thread Peter is developing (Bauer-Bialek, Hänni); Heimersheim = audience.
- [[bluedot-app]] — cert + Project Sprint; source for Q1/Q3.
- [[astra-app]] — source for Q4.
- [[anthropic-fellows-app]] — source for Q5 code descriptions.
- [[singapore-aisf-app]] — companion fall-fork app; overlapping research answers.

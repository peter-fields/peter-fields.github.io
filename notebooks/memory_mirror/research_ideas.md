---
name: research_ideas
description: Master index of all mech interp research ideas, blog ideas, and post directions — points to relevant files
type: project
---

# Research Ideas — Master Index

## Blog Posts (concrete, near-term)

### Post 3 — WRITE THIS NEXT
Experiments done. Narrative: out_mag > Var_v (30x ratio), contrastive ICA (7/8 circuit heads), C_diff circuit graph (9x enrichment at 5σ).
See [post3-plan.md](post3-plan.md) for full pipeline, results, and revised narrative.

### Post 4 — direction undecided
Candidates (see post3-plan.md §Post 4):
- **Causal validation**: activation patching on ICA-nominated heads → closes observational → causal loop
- **Multi-feature ICA**: stack out_mag + KL → 1000×288 joint matrix; captures cross-feature dependencies
- **Generalization**: same pipeline on a different circuit or model
Decision: write Post 3 first, direction will be clearer after.

### Blog idea — W_QK sym/anti ratio
W_QK^sym (real eigenvalues) = content matching; W_QK^anti (imaginary eigenvalues) = directional/positional.
Ratio discriminates head type. No precedent found. Also connects to the implicit metric idea below.
Notes in [tensor_notation.md](tensor_notation.md) and [idea_qk_metric.md](idea_qk_metric.md).

### Blog idea — Tensor notation for Elhage 2021
x ∈ V_feat* (covector) explains W^T, W_QK as (2,0) vs W_OV as (1,1), Elhage row/column convention bug.
**TODO before writing**: nail the privileged basis argument (no privileged basis in V_feat → covector declaration is substantive; V_pos having one is why Roman indices carry no primal/dual content). Needs to be airtight.
See [tensor_notation.md](tensor_notation.md) for full theory.

---

## Research Ideas (concrete, safety-relevant)

### Circuit Formation from Training Data Statistics ⭐
**Core**: If you can connect circuit formation to training data statistics, you can predict capability emergence before it happens — directly safety-relevant. Induction heads are proof of concept: formation predicted by burstiness, diversity, repetition in training data (Singh 2024, Reddy 2024, Kawata 2025, Musat 2025, Aoyama 2025). Does this generalize to more complex circuits?
**Map/territory framing**: what a circuit computes ↔ what data statistics produced it are two sides of the same reverse-engineering problem — each constrains the other.
**Empirical handle**: prompt-class contrast probes which data statistics a circuit is sensitive to by varying what you show it. Current interp is mostly post-hoc/diagnostic; push toward predictive.
**Status**: central framing of Anthropic Fellows application. Lit review in [lit-review.md](lit-review.md) §6. Key refs need verification before citing.
**PyTorch project candidate**: train small transformer on synthetic datasets with controlled burstiness/diversity; track induction head formation. GPU-friendly.

---

### Heat Capacity Regularization for Generative Models ⭐ (novel, unexplored)
**Core**: Dissertation Ch. 4 shows that whether τ needs to be raised or lowered post-hoc to improve generation, the model's heat capacity C = Var(E_J) almost always decreases. This suggests baking the improvement into training via a Var(E) regularizer — a "training temperature" T controlling regularization strength 1/T^2:

-L'(D|J) = -L(D|J) + (1/T^2) Var(E_J)

**Key question**: Can a proper choice of T during training replace post-hoc tau tuning, without knowing the ground truth?

**LLM extension**: In a transformer, logits play the role of energies. Heat capacity = Var(logits) under the model distribution. Hypothesis: a Var(logits) regularizer during training or fine-tuning improves generative performance analogously. Nobody has done this — not for EBMs, not for transformers.

**PyTorch project**: Train a small transformer with and without this regularizer; measure generative quality (forward/reverse KL, perplexity, sample diversity). GPU-friendly, natural extension of dissertation, genuine novelty.
**Project arc**:
1. Train 2-layer transformer on synthetic data (fixed B, K, context length, small vocab). Get induction head to form.
2. Sweep M. Show induction score and reverse KL correlate across M. **Key framing: induction score is the mechanistic proxy for reverse KL — this is the information-theoretic quantity it's actually tracking.** Reverse KL is only computable here because we have exact ground truth (synthetic data, small vocab). That's fine — it's a proof of concept.
3. Show low M → positional shortcut → bad reverse KL. Traditional induction score tracks this too once you have the correlation.
4. Apply C-reg — does it shift the phase transition to lower M?

**Novel contributions**:
- Nobody has swept M as primary variable holding distribution fixed
- Reverse KL framing is new — gives induction score a principled information-theoretic interpretation
- C-reg as a mechanism to improve sample efficiency of circuit formation

**Interpretation if C-reg helps**: Bottleneck was noise from undersampling, not signal. Circuit formation gated by model's ability to filter signal from noise — measurable from heat capacity.
**Interpretation if C-reg doesn't help**: Data quantity is the actual bottleneck. Also interesting.

**Literature gap confirmed**: No paper sweeps M; no paper uses reverse KL as metric. Kawata 2025 is closest (positional shortcut vs IH, diversity as axis). Song et al. PNAS 2025 shows OOD failure but uses training steps not M.

**Blog potential**: Strong. "The information-theoretic quantity that induction score is actually tracking." Natural sequel to temp tuning paper.
**Architecture**: 2-layer attention-only transformer (multi-head, no MLP), next-token prediction, output projection gives logits directly over discrete vocab. No classifier head needed.

**Data**: Discrete token sequences, small vocab (K=16-64), controlled burstiness B. Ground truth p known by construction → exact reverse KL.

**Experiment 1** — Qualitative: Train at low B (no induction) and high B (induction forms). Show attention maps + IC accuracy curves. Replicates Reddy Fig 3b in discrete setting. Establishes setup works.

**Experiment 2** — Quantitative: Sweep B values (e.g. B=0,1,2,4). At each B, run many seeds (10-20) over many data instantiations. Plot induction score vs reverse KL across B sweep. **Result: they correlate. Induction score is tracking reverse KL.**

**Note on replicates**: Need enough seeds per B value for stable means — variance across seeds is real (Reddy shows spread). GPU motivation: run many seeds in parallel.

**Part 2 — C tracks the transition**:
B sweep → heat capacity C co-varies with reverse KL. High C = disordered phase (spread over spurious states). Low C = ordered phase (committed to induction solution). C-reg pushes the model into the ordered phase. Blog framing: "we are pushing the model into the ordered phase" — stat mech language, one sentence.
Once induction score ≈ reverse KL is established in Part 1, Part 2 drops induction score and uses reverse KL + C only.

**Part 3 — C-reg proof of concept**:
Find marginal B from Part 1 (transition is noisy across seeds). At marginal B, train many seeds with and without C-reg. Use induction score (validated in Part 1) as the metric — cheaper than reverse KL.
**T selection**: Run small grid of T values on one seed at marginal B. Cherry-pick T where induction score is highest. Use that T for full multi-seed experiment. Flag as chosen hyperparameter, leave systematic search for later.
**Result**: With cherry-picked T, C-reg tips the model into the ordered phase consistently. Without C-reg, formation is a coin flip at marginal B.
**Key failure mode to check**: Does C-reg sharpen commitment to spurious patterns instead? Test by checking reverse KL — if it improves, model found right structure. If it worsens, it sharpened the wrong thing.
**Leave for later**: Principled T selection / hyperparameter search.

**Larger story**: Circuit formation depends on the model and objective function's ability to filter noise and find intrinsic data structure. Direct extension of dissertation thesis — temperature tuning reflects whether the model is in the right regime to extract structure. Same physics applies to circuit formation: objective function's noise-filtering properties determine whether intrinsic data structure gets picked up or buried. Safety angle: predicting circuit formation requires knowing not just data statistics but whether training objective is in the right regime to find them.

**Status**: Unexplored. Step 1 = replicate Reddy 2024 B-sweep result with small enough vocab/context that reverse KL is computable exactly. Then show reverse KL tracks induction score.
**TODO**: Read Reddy 2024 (arXiv:2312.03002) carefully before coding — need exact data setup, B definition, model architecture, induction metric.

---

## Research Ideas (speculative to medium-term)

### Implicit Riemannian Metric in QK Circuits
**Core**: W_QK defines a bilinear form; without a fixed metric on the residual stream, "which subspace does this head read from" is coordinate-dependent. Hypothesis: a shared metric G is implicitly learned. Recover it via joint eigendecomposition or Procrustes across all W_QK^h_sym.
**Connections**: W_QK sym/anti blog idea; G-weighted composition strength; coord-free interpretability.
**Status**: speculative, tractable as empirical paper on small model.
See [idea_qk_metric.md](idea_qk_metric.md) for full treatment.

### Scalable Unsupervised Circuit Discovery (CLT + Sparse ICA)
**Core**: CLT decomposition once per model → project residual streams to sparse CLT feature basis → vectorize → sparse ICA on large diverse corpus → ICs are interpretable (position, CLT feature) pairs. No task-specific prompts, no human bottleneck until final labeling. Capacity ranking (singular values of W_QK) cross-validates parameter view against activation view.
**Framing**: "unsupervised circuit discovery via sparse ICA on CLT-projected residual streams." IOI = proof-of-concept.
**Status**: speculative, depends on CLT access. Medium-term.
See [circuit-discovery-theory.md](circuit-discovery-theory.md) §Scalable Pipeline.

### Circuit Structure Taxonomy (sparse vs. voting vs. reservoir)
**Core**: IOI = sparse circuits (Wang et al.). Chess likely = voting/lookup (Transcendence paper). General language = reservoir. Test: fork vs. pin vs. non-tactical minimal pairs — same Δ profile = voting, diverging = circuit structure. Agentic loop (Claude drives prompt generation) as experimental scaffold.
**Status**: testable, needs chess data + a GPU afternoon. No prior work.
See [circuit-discovery-theory.md](circuit-discovery-theory.md) for full hypotheses + null result definition.

### Alternating Causal / Bidirectional Attention
**Core**: Every X tokens, run a full bidirectional pass (no causal mask) over full sequence — updates KV cache. Token T then attends to richer, bidirectionally-updated keys. Single weight set. Removes the "anticipation without revision" limitation of causal transformers.
**Cheapest first check**: on a pretrained GPT-2, do residual streams of early tokens shift toward "liar direction" after bidirectional pass on liar-reveal sequences? No training needed.
**Status**: hand-wavy, worthy. Check prior work (Diffusion LMs, XLNet, Insertion Transformer, Retro).
See [idea_alternating_attention.md](idea_alternating_attention.md) for full experimental sequence.

---

### Mech Interp Audit of Vague Safety Objectives
**Core**: Train a model on an intentionally underspecified harm-avoidance objective (e.g. "don't do anything that hurts anyone"). Use mech interp to characterize what circuits actually implement it — watch the edge cases, misinterpretations, and unintended optimizations fall out mechanistically. Not behavioral red-teaming but circuit-level auditing of what the objective actually became inside the model.
**Motivation**: Vague objectives like Millian harm-avoidance sound reasonable to people who take liberal assumptions for granted — the underspecification is invisible to them. A mech interp audit doesn't share those assumptions; it just reads out what the model learned. The "deceptive alignment" framing assumes the model is adversarial; more likely the surprise at deployment just means we had a shallow understanding of what we trained for. Mech interp closes that gap.
**Status**: Long-term, needs more experience. Not for current applications.

## Post 3 Empirical Findings (established, not yet published)
- **out_mag** = ||µ_v||²/d beats Var_v: 30x ratio (p=1.2e-5). Primary diagnostic going forward.
- **Contrastive ICA**: 7/8 circuit heads recovered unsupervised. Beats PCA (ceiling ~6/8 due to layer-depth confound).
- **C_diff circuit graph**: correlate out_mag separately on IOI and non-IOI, difference. 9x enrichment at 5σ, 34 edges, mechanistically correct top edges.
- **Always-on heads limitation**: L0H1, L2H2, L3H0, L5H5, L6H9 invisible to Δ-based diagnostics — they fire on all prompts. Fundamental limitation; needs causal complement.
- Full pipeline in [post3-plan.md](post3-plan.md).

---

## Backburner / Very Speculative

- **Deep RBM for interacting circuits** — two hidden layers: circuits at layer 1, meta-circuits (patterns of circuits co-activating) at layer 2. Kyle's MI regularization extends layerwise. See post3-plan.md §Further Future.
- **phi^4 / double-well single-site potential** — bimodal marginals, connects to Landau / RG theory. Far future.
- **Entropy regularization on per-head KL** — Lagrange multiplier on KL(pi || u) ≤ rho_l, increases with layer depth; promotes circuit specialization by design. Interpretability architecture idea from the Anthropic application.
- **KL-budget training regularization** — connected to Post 1's theory; rho_l increasing with layer depth as inductive bias.

---

## Pointers to Other Relevant Files
- [posts-arc.md](posts-arc.md) — full post 1–4 narrative arc with experiment results
- [post3-plan.md](post3-plan.md) — complete post 3 pipeline, all 16 experiment results, revised narrative
- [circuit-discovery-theory.md](circuit-discovery-theory.md) — circularity problem, structure hypotheses, chess null result, scalable pipeline
- [tensor_notation.md](tensor_notation.md) — W_QK theory, blog ideas, deeper analysis (NOT canonical notation)
- [idea_qk_metric.md](idea_qk_metric.md) — implicit metric full treatment
- [idea_alternating_attention.md](idea_alternating_attention.md) — alternating attention full treatment
- [lit-review.md](lit-review.md) — novelty claims, related work, citations
- Canonical notation: `notebooks/tensor_notation/tensor_notation_settled.md`

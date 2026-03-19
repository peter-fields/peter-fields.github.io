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

---
name: compressed-computation-project
description: "Peter's CV-building research project — computation in superposition (Hänni et al. 2408.05451), replacing the PyTorch induction head project as the main public output"
metadata: 
  node_type: memory
  type: project
  originSessionId: 14f8c394-1e56-45de-997d-4ff4de657c44
---

# Compressed Computation Project — replaces PyTorch induction head project

> 🔵 **STATUS 2026-06-24 — ON BACKBURNER.** Not off the ground yet; Peter is deprioritizing it for a while (applications + PRR took precedence). Notes below are the standing plan for when it resumes; nothing started.

**Decision date:** 2026-05-26

## Why the pivot

PyTorch induction head project was sitting on the TODO list as the highest-leverage CV item, but:
- Requires training a small transformer from scratch — meaningful PyTorch infrastructure work, slow loop, harder to iterate
- Would need to compete with existing literature (Singh, Reddy, Kawata, Musat, Aoyama) that already does this well
- Coding-heavy in a way that emphasizes the Python/PyTorch gap rather than the physics-theory strength

Compressed computation project instead:
- Theory-forward; physics background is a positive
- Mostly NumPy or small-scale PyTorch — much lower coding overhead per unit insight
- Built-in audience: Stefan Heimersheim was pitched this connection during the Pivotal work test and is the natural mentor/reader. He may not be the summer mentor but is a credible recipient of the work.
- Connects to existing Bauer-Bialek / DIB / superposition normative theory direction already in [research_ideas.md](research_ideas.md)
- Could double as the toy-model blog post Geometric Intelligence Lab wants before cold-email

## Starting point: Hänni et al. 2408.05451

"Mathematical Models of Computation in Superposition" — Universal AND circuit. Shows that a 1-layer MLP with random sparse weights can ε-linearly represent all (m choose 2) pairwise ANDs of m sparse boolean features with only Õ(m^(2/3)) neurons, even when the inputs themselves are in superposition. Generalizes to arbitrary sparse boolean circuits via "error correction" layers.

Key formalism (the "Hänni formalism" Peter is comfortable with):
- **b** ∈ {0,1}^m — sparse boolean features (m >> d)
- **a** = Φ**b** ∈ R^d — residual stream (d analogous to d_model; d_mlp conflated with d_model in their paper)
- **Y** = C(**b**) — computed output (e.g. pairwise ANDs)
- φ_k (columns of Φ) are the feature encoding directions

## Three candidate project directions

### 1. Empirical / constructive (LOWEST risk, easiest)
- Implement U-AND from scratch: build the random sparse W_in, verify ε-linear representation holds
- Sweep m, d, s and replicate the Õ(m^(2/3)) scaling
- Verify Theorem 3 (randomly initialized MLPs linearly represent U-AND with high probability)
- Extend to higher fan-in ANDs (Lemma 5) and small boolean circuits
- **Output:** clean NumPy code + plots + blog post. Replicates and extends Hänni — demonstrates the result is real and easy to play with.

### 2. Theory / normative (MEDIUM risk, more original)
- Apply IB / DIB framework to derive *why* SGD finds computation in superposition — not just that it's achievable
- Bauer-Bialek "folding helps" angle: show that capacity-constrained channels (ReLU neurons) generically favor folded/polysemantic encodings
- Deterministic-noise bridge (KEEP PRIVATE — see [research_ideas.md](research_ideas.md)) makes the stochastic IB formalism apply to deterministic MLPs
- **Output:** theory blog post with toy numerical experiments confirming the predictions

### 3. Empirical diagnostic (HIGHEST risk, most ambitious)
- Use DIB-style bottlenecking to *find* computation in superposition in real trained MLPs (e.g. GPT-2 small)
- Sweep β, see which SAE features survive, infer which computations are happening
- Forward-pass only, connects directly to Peter's existing prompt-class diagnostic work
- **Output:** new diagnostic tool for circuit discovery via observation, no causal interventions

**Likely path:** Start with Direction 1, then opportunistically add Direction 2 theory commentary, leave Direction 3 for a follow-up.

## Why this is good for the audience

**Stefan Heimersheim** has direct interest:
- He works on Pivotal-adjacent mech interp questions
- Computation in superposition / error correction layers literature is in his orbit (Heimersheim & Mendel 2023 plateaus result is cited in the Hänni paper)
- He's already been exposed to Peter's thinking via the Pivotal work test
- Even if not the summer mentor, he's the right reader for a write-up

**Geometric Intelligence Lab (UCSB)** could read the same work as the toy-model blog they want before considering Peter.

**Independent funding (LTFF, Coefficient Giving):** A concrete write-up makes a 1-2 page proposal much easier.

## Timeline (rough)

- **Now → 2026-06-11:** ALL focus on PRR paper revision. Project on hold.
- **2026-06-12 → end of June:** Start Direction 1 implementation. Aim for working U-AND replication.
- **July:** Scaling sweeps + blog post draft.
- **Aug:** Direction 2 theory connection + final write-up. Reach out to Stefan with the draft.

## Status

**On hold until 2026-06-11** while paper revision is the priority. After paper goes in, this is the next CV item.

Related files:
- [research_ideas.md](research_ideas.md) §"Ambiguous Signals → Computation in Superposition" — original idea log + Hänni formalism breakdown + DIB connections
- The Hänni paper: arXiv:2408.05451
- The Bauer-Bialek paper: arXiv:2512.23531
- The Murphy-Bassett DIB paper: arXiv:2204.07576

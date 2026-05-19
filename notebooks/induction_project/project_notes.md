# PyTorch Induction Head Project — Design Notes

## Goal

Train a minimal 2-layer attention-only transformer on synthetic discrete sequences with controlled burstiness b. Show that induction score correlates with reverse KL across a b sweep. Then show heat capacity C co-varies, and that C-regularization shifts the phase transition to lower b.

## References

- Olsson et al. 2022 — induction heads, induction score definition
- Reddy 2024 (arXiv:2312.03002) — burstiness sweep, minimal 2-layer model, progress measures
- Chan et al. 2022 (arXiv:2205.05055) — original burstiness paper, Omniglot, 12-layer transformer

---

## Data Generation

### Vocabulary

- Fixed set of A tokens and fixed set of B tokens — disjoint
- **No fixed A→B mapping**: any (A, B) pairing is valid; the pairing is sampled fresh each sequence
- Vocab sizes TBD (old design had |A|=512, |B|=32 — may keep as starting point)

### Sequence format

- Sequence length: 2N + 2 tokens (N = 8, so 18 tokens total)
- Each sequence: sample one (A, B) pair uniformly at random
- Place b copies of (A, B) at random positions in the 2N context tokens
- Final two tokens: (A, B) — the query A and target B
- All remaining context positions: filler tokens drawn uniformly from full vocabulary

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| \|A\| | TBD | Number of A tokens |
| \|B\| | TBD | Number of B tokens |
| N | 8 | Context length in pairs (total sequence = 2N+2 = 18) |
| b | swept (0, 1, 2, 4, ...) | Number of (A, B) copies in context before query |

### Burstiness b

b = number of (A, B) copies in context before the final query pair. b=0: no in-context signal, model must fall back to marginal (≈ uniform over B). Higher b: more in-context evidence for the induction head to copy from.

### Why random pairing per sequence?

With a fixed A→B mapping, the model can learn the mapping in weights (IWL) and predict B from A without ever using context. With random per-episode pairings, there is no consistent association to memorize — the induction head is the only strategy that works.

---

## Ground Truth Distribution

**Key distinction**: p_true is the *in-context* Bayesian posterior — the optimal prediction given what has been observed in the current sequence. It is NOT the global/marginal distribution of what follows A across the whole corpus (that is what a bigram model learns, and is not the right target for measuring induction).

**Why the in-context framing is necessary**: with a random (A, B) pairing per sequence (no fixed mapping), the corpus-marginal p(next | A) is approximately uniform over B tokens — A is followed by many different B's across episodes. The model cannot learn a consistent A→B association in weights. The induction head is the only mechanism that can do better, by using within-episode context copies.

**Setup**: each sequence samples one (A, B) pair uniformly at random (any pairing valid, no fixed mapping). The context contains b copies of (A, B) plus filler tokens drawn uniformly from the full vocabulary. Filler naturally creates stochasticity — A appears in filler positions of other sequences followed by random tokens — so no explicit noise parameter eps is needed.

**Prior**: at b=0, no context copies → p_true(B | context) = 1/|B|, uniform over all B tokens.

**Posterior update**: each context copy (A, B_i) is a likelihood update. With b clean copies all showing the same B: p_true(B_correct | context) → 1 as b increases. The prior is 1/|B|.

**Key property**: p_true is non-degenerate at b=0 (uniform, not a delta), so reverse KL is finite. The stochasticity comes from the random per-episode (A, B) pairing and filler positions — no separate eps noise parameter required.

**Previous design (scrapped)**: earlier design used a fixed A→B mapping with many-to-one structure (K=512 A tokens, L=32 B tokens, 16 A per class) and within-class eps noise. This was abandoned because: (1) fixed mapping + clean query A → p_true = 1 always → reverse KL diverges; (2) within-class noise doesn't create cross-class uncertainty; (3) the whole construction was unnecessary complexity.

---

## Evaluation Metric

### Conditional reverse KL at the query position

KL(p_model || p_true) = sum_{k} p_model(k | context) * log( p_model(k | context) / p_true(k | context) )

Note: this is the *reverse* KL (model || true), not forward KL. p_true is the in-context Bayesian posterior defined above. Averaged over many sampled sequences (Monte Carlo). Finite because p_true is non-degenerate (uniform at b=0).

### Induction score

Standard Olsson definition: generate a random sequence of length T, concatenate with itself, measure how strongly the layer-2 head attends back to the correct position in the first copy. Evaluated on repeated random sequences (out of distribution relative to training).

---

## Model Architecture

- 2-layer attention-only transformer (no MLP)
- 1 attention head per layer
- Causal mask
- Output projection → logits over K tokens
- Next-token prediction loss (cross-entropy) on all positions

Remaining to decide: embedding dimension d_model, head dimension d_k, learning rate, optimizer.

---

## Training

- Sequences generated on the fly each batch (never repeated in practice)
- ~200,000 iterations × batch size 128 = ~25.6M sequences total (following Reddy)
- Sweep b values, many seeds per b value (10-20)

---

## Experiment Plan

### Part 1 — Induction score tracks reverse KL

- Sweep b = 0, 1, 2, 4 (or similar)
- At each b, train many seeds
- Plot induction score vs reverse KL across b sweep
- Expected result: they correlate. Induction score is the mechanistic proxy for reverse KL.

### Part 2 — Heat capacity tracks the transition

- Compute C = Var(logits) under the model distribution
- Show C co-varies with reverse KL across b sweep
- High C = disordered phase; low C = ordered/induction phase

### Part 3 — C-regularization proof of concept

- Find marginal b (transition is noisy across seeds)
- At marginal b: train with and without C-reg term (1/T^2) * Var(logits)
- Cherry-pick temperature T from small grid on one seed
- Expected result: C-reg tips model into ordered phase consistently

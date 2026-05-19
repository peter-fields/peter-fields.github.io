# Handoff Summary — Inducing Induction Project

## Where we are

We have been designing the data generating distribution for the PyTorch induction head project. The repo `inducing-induction` is set up locally at `~/Git/inducing-induction` with conda environment `inducing-induction` (Python 3.11, PyTorch, wandb, numpy, matplotlib). No code has been written yet — we are still finalizing the data distribution design.

The full design notes are in `notebooks/induction_project/project_notes.md` in the blog repo, but those notes are partially outdated relative to where we ended up in the conversation. The summary below is authoritative.

---

## What we have decided

### Big picture
- 2-layer attention-only transformer, 1 head per layer, no MLP
- Trained on synthetic discrete sequences with controlled burstiness b
- Next-token prediction loss on all positions
- Sweep b, show induction score correlates with reverse KL
- Then show heat capacity C co-varies, then C-reg proof of concept

### Vocabulary structure
- Fixed set of A tokens, fixed set of B tokens — disjoint, sizes TBD (|A|=512, |B|=32 as starting point)
- **No fixed A→B mapping** — any (A, B) pairing is valid; pairing is sampled fresh each sequence
- This forces the induction circuit as the only viable strategy (no consistent mapping to memorize in weights)

### Sequence format
- Sequence length: 2N + 2 = 18 tokens (N=8)
- Each sequence: sample one (A, B) pair uniformly at random
- Place b copies of (A, B) at random context positions; final two tokens are always (A, B) — the query
- Filler positions: uniform random over full vocab

### Parameters
| Parameter | Value |
|-----------|-------|
| \|A\| | TBD (~512) |
| \|B\| | TBD (~32) |
| N | 8 |
| b | swept: 0, 1, 2, 4, ... |
| batch size | 128 |
| iterations | 200,000 |
| seeds per b | 10-20 |

### Ground truth distribution
- **p_true is the in-context Bayesian posterior** — NOT the global/marginal distribution over the corpus
- The corpus-marginal p(next | A) ≈ uniform over B (different B in each episode → no consistent association). A model cannot beat this using weights alone.
- p_true(B | context) starts at 1/|B| (uniform prior, b=0) and concentrates on B_correct as b increases
- Finite reverse KL at b=0 because p_true is non-degenerate (uniform, not delta)
- No eps noise parameter needed — filler positions naturally create stochasticity (A appears in filler of other sequences, followed by random tokens)
- Reverse KL = KL(p_model || p_true_in_context), averaged over sequences

### Departures from Reddy
- Next-token prediction instead of classification with MLP head
- Discrete tokens instead of continuous Gaussian embeddings
- B tokens appear freely in filler positions
- No label balancing constraint within sequences
- Query A is always clean (Reddy's query may be noisy)

---

## Immediate next steps

1. Decide |A| and |B| (vocab sizes)
2. Start coding data generator in a Jupyter notebook in `inducing-induction/notebooks/`
4. Write model architecture
5. Write training loop
6. Sanity check: train at high b → induction circuit forms (attention maps + induction score), low b → doesn't

---

## Key references
- Reddy 2024: arXiv:2312.03002
- Olsson et al. 2022: arXiv:2209.11895
- Chan et al. 2022: arXiv:2205.05055

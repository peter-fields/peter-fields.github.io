# Anthropic Interpretability Screen — Study Plan

**Test:** 120 min, 4 questions, multi-part (gated). Final part of Q4 is intentionally brutal.
**Timeline:** ~7 days at 5–8 hrs/day.
**Python only. NumPy only. No PyTorch needed for this test.**

---

## Day-by-Day Schedule

| Day | Focus | Notebooks | Reading |
|-----|-------|-----------|---------|
| 1 | Python + NumPy + einsum | `nb1_numpy_einsum.ipynb` | NumPy broadcasting docs (15 min) |
| 2 | Softmax, Jacobian, single/multi-head attention | `nb2_attention_circuits.ipynb` | Elhage 2021 §1–3 (before Ex 5) |
| 3 | Circuit matrices, W_QK/W_OV, SVD | Finish `nb2`, start `nb3` Ex 1–4 | Elhage 2021 §4 "Virtual Weights" |
| 4 | Full transformer + induction heads + SAE | `nb3_transformer_interp.ipynb` Ex 5–6 | Olsson 2022 §1–3 (before Ex 5); Bricken 2023 intro (before Ex 6) |
| 5 | Timed mock test 1 | `practice_test_1.ipynb` (120 min timer) | Review mistakes; re-read Elhage §2 if shaky |
| 6 | Jacobian backprop, full forward pass, circuit analysis | `practice_test_2.ipynb` Q1–Q3 | Re-read induction heads section |
| 7 | Full timed mock test 2, patch weak spots | `practice_test_2.ipynb` full (120 min) | Light review; rest |

---

## What's Tested (Most → Least Likely)

### Definitely on the test
- Numerically stable softmax and log_softmax
- Cross-entropy loss
- Single-head and multi-head attention (with causal mask)
- `np.einsum` for batched attention scores
- Layer normalization
- MLP with GELU

### Very likely
- Softmax Jacobian (∂π_i/∂z_j = π_i(δ_ij − π_j))
- Gradient through attention (chain rule via Jacobian)
- W_QK = W_Q @ W_K.T and W_OV = W_V @ W_O circuit matrices
- SVD of W_OV (effective rank, singular vectors)
- Full transformer forward pass end-to-end

### Hard final question (Q4 Part 4)
- Induction head detection from attention patterns
- SAE forward pass and feature decomposition
- Virtual head composition W_OV_L2 @ W_OV_L1
- Self-reinforcing circuits (alignment of features through composed OV)

---

## Papers to Read (in order)

1. **Elhage et al. 2021** — [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
   - Read §1–3 before Day 2 (attention basics, residual stream, QK/OV circuits)
   - Read §4 before Day 3 (virtual weights, SVD interpretation)

2. **Olsson et al. 2022** — [In-Context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
   - Read §1–3 before Day 4 (what induction heads are, K-composition, detection)

3. **Bricken et al. 2023** — [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemanticity/index.html)
   - Read intro + §2 before Day 4 (why SAEs, encoder/decoder, loss function)

4. **Skim before test day** (1 hour total):
   - Circuit Tracing / Biology of LLM (2025) — skim the intro only, know it exists
   - Kim 2026 — you already know this from your blog

---

## Key Conventions to Have Cold

```python
# Circuit matrices (Elhage 2021)
W_QK = W_Q @ W_K.T          # (d_model, d_model) — NOT W_Q.T @ W_K
W_OV = W_V @ W_O_head        # (d_model, d_model) — W_O_head is (d_v, d_model)

# Attention scores via circuit matrix
scores = X @ W_QK @ X.T / np.sqrt(d_k)  # equivalent to (X@W_Q)@(X@W_K).T / sqrt(dk)

# Pre-norm transformer (GPT-2 style)
x = x + attention_out(layer_norm(x))
x = x + mlp_out(layer_norm(x))

# SAE (centering matters!)
f = ReLU(W_enc @ (x - b_dec) + b_enc)
x_hat = W_dec @ f + b_dec
loss = MSE(x, x_hat) + lambda * ||f||_1
```

---

## Weak Spots to Practice

- **NumPy array slicing** — still shaky. Practice: integer indexing (`a[rows, cols]`), boolean masks, `np.arange(B)` row selector + column index for cross-entropy-style picks, slicing with `None`/`np.newaxis`.

---

## Submission Checklist (before submitting any function)

- [ ] Softmax shifts by max before exp?
- [ ] `axis=-1` not `axis=1`?
- [ ] `keepdims=True` wherever dividing back into original shape?
- [ ] Causal mask applied BEFORE softmax?
- [ ] Divides by `sqrt(d_k)` not `d_k`?
- [ ] Numerical gradient check passes?

---

## On PyTorch

**Not needed for this test.** The email said Python + NumPy; that's the signal.

PyTorch + TransformerLens is what you'd need for the actual job and later interviews. If you clear this screen and have time before interviews, do: implement a small GPT-2 in PyTorch, then load a real model into TransformerLens and run `logit_lens` and `patching` experiments.

---

## On SAEs / CLTs

**SAE:** Yes, probably on the test (Q4 hard part). The math is simple — two linear layers + ReLU + L1 loss. What matters is understanding *why*: superposition means the residual stream packs more features than dimensions using near-orthogonal directions; SAEs "unpack" this into a sparse higher-dimensional space.

**CLT (Cross-Layer Transcoder):** Very recent, very specialized. Unlikely on the coding test. Know it exists and roughly what it does (like an SAE but maps from one layer's residual stream to another's MLP output). Might come up in interviews.

---

## Notebook Map

| Notebook | Content | When to use |
|----------|---------|-------------|
| `nb1_numpy_einsum.ipynb` | NumPy, broadcasting, einsum | Day 1 |
| `nb2_attention_circuits.ipynb` | Softmax, Jacobian, attention, W_QK/W_OV | Days 2–3 |
| `nb3_transformer_interp.ipynb` | Layer norm, MLP, full transformer, SVD, induction heads, SAE | Days 3–4 |
| `practice_test_1.ipynb` | Full mock test (easier) | Day 5 |
| `practice_test_2.ipynb` | Full mock test (harder) | Days 6–7 |
| `reference_sheet.md` | Consult during the real test | Test day |
| `memorize_sheet.md` | Know cold before test day | Ongoing |
| `test_structure.md` | What questions probably look like | Before test day |
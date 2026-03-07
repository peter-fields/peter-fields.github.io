# What the Test Probably Looks Like

4 questions, 120 min, multi-part questions gated (must pass part N to unlock N+1).
Final part of Q4 is intentionally very hard — don't sacrifice Q1-3 chasing it.

---

## Q1 — Math warmup (~15 min, single)

Numerically stable softmax + Jacobian. Verify against finite differences.

```python
# stub you'll probably see
def softmax(z: np.ndarray) -> np.ndarray: ...
def softmax_jacobian(z: np.ndarray) -> np.ndarray: ...
```

---

## Q2 — NumPy/einsum fluency (~20 min, single)

Batch operations. Example: given Q `(B, H, T, d_k)` and K `(B, H, T, d_k)`, compute
all scaled attention score matrices with a single einsum call.

Likely tests: axis manipulation, broadcasting, keepdims, batch matmul.

---

## Q3 — Attention implementation (~35 min, multi-part)

- Part 1: Single-head attention (QKV → scores → causal mask → softmax → output)
- Part 2: Multi-head — W_Q/K/V probably given as `(H, d_model, d_k)` tensors
- Part 3: Given trained weights, compute W_OV per head, SVD it, return top singular values

---

## Q4 — Interpretability-specific (~50 min, multi-part, Part 4 brutal)

- Part 1: Full transformer forward pass (embed → LN → attn → LN → MLP → unembed)
- Part 2: Show W_QK = W_Q @ W_K.T reproduces same attention patterns as direct implementation
- Part 3: Detect induction heads from attention patterns (stripe one above main diagonal on repeated sequences)
- **Part 4 (hard):** Activation/attribution patching to find causally responsible heads,
  OR virtual head compositions W_OV^{L1} @ W_OV^{L2} ranked by Frobenius norm

---

## Strategy

- **Write shape comments as you go** — axis errors are the #1 failure mode
- **Don't debug blind** — print shapes when something feels wrong
- **Cut losses on Part 4** — submit Q1-3 complete before attempting it
- **Read carefully** — index conventions vary, don't assume
- You can reference internet docs (NumPy, Python) — use it for unfamiliar linalg calls
- Cannot paste external code — retype from notes if needed
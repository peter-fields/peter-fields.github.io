# Must Memorize — Anthropic Interpretability Screen
*Know these cold, without looking anything up.*

---

## 1. Softmax — Three Facts

1. **Formula:** `pi_i = exp(z_i) / sum_j exp(z_j)`
2. **Always shift first:** `z -= max(z)` — translation-invariant, prevents overflow
3. **Jacobian:** `d pi_i / d z_j = pi_i * (1[i==j] - pi_j)`
   - Diagonal: `pi_i(1 - pi_i)`
   - Off-diagonal: `-pi_i * pi_j`

---

## 2. Attention — The Algorithm (cold)

```
Q = X @ W_Q          # (T, d_k)
K = X @ W_K          # (T, d_k)
V = X @ W_V          # (T, d_v)
scores = Q @ K.T / sqrt(d_k)          # (T, T)
[apply causal mask: scores -= 1e9 * triu(ones, k=1)]
weights = softmax(scores, axis=-1)    # (T, T)  <- rows sum to 1
output = weights @ V                  # (T, d_v)
```

Three matrix multiplications in, three out. `d_k` is the head dimension. Divide by `sqrt(d_k)` to control variance of dot products.

---

## 3. Circuit Matrices (Elhage 2021)

- **W_QK = W_Q @ W_K.T** — lives in d_model × d_model space
  - Attention score between positions i and j: `x_i @ W_QK @ x_j`
  - Full score matrix: `X @ W_QK @ X.T / sqrt(d_k)`

- **W_OV = W_V @ W_O** — lives in d_model × d_model space
  - What information gets written to residual stream: `output = weights @ X @ W_OV`

- These are the two objects circuit analysis is really about. Factoring through d_k is just implementation detail.

---

## 4. Residual Stream View

At every layer:
```
x = x + attention_output(x)    # skip connection
x = x + mlp_output(x)          # skip connection
```
The residual stream is the running sum of all contributions. Each head reads from and writes to this stream independently. This is why decomposing contributions by head/layer is meaningful.

---

## 5. Induction Heads

- A two-head circuit that implements in-context sequence copying
- **Previous Token Head** (layer 0): attends to the previous token, copies its key into the residual stream
- **Induction Head** (layer 1): K-composition — uses the previous token head's output to match the current token, then attends to the position after the match in context
- Pattern: if `[A][B]...[A]` appears, the induction head on the second `[A]` attends to the first `[B]` and predicts `[B]` will follow
- Detectable by: attention pattern has a diagonal stripe one above the main diagonal when tested on repeated sequences

---

## 6. Layer Norm — What It Does

Normalizes each token's residual stream vector to have mean 0, variance 1, then applies learned scale γ and shift β:
```
mu = x.mean(axis=-1)        # per-token mean
var = x.var(axis=-1)        # per-token variance
x_hat = (x - mu) / sqrt(var + eps)
output = gamma * x_hat + beta
```
Applied BEFORE attention and MLP in GPT-style models (pre-norm). Without learnable params: just `layer_norm(x)`.

---

## 7. Axis Convention — Say It Out Loud

- `axis=0`: operates along rows (collapses rows) → result has fewer rows
- `axis=1`: operates along columns (collapses cols) → result has fewer columns
- `axis=-1`: operates along the last axis (almost always what you want for per-token ops)
- `keepdims=True`: keeps the collapsed axis as size 1 (needed for broadcasting back)

---

## 8. SVD — What Each Part Means

`U, S, Vh = np.linalg.svd(W_OV)`

- `U`: left singular vectors — output directions the head can write
- `S`: singular values — magnitude of each mode (how strongly used)
- `Vh`: right singular vectors — input directions the head reads from (Vh[i] = V[i].T)
- `U[:, 0]`: the single most important output direction of this head
- Rank-k approx: `U[:, :k] @ np.diag(S[:k]) @ Vh[:k, :]`

In interpretability: SVD of W_OV tells you what the head "does" — what input features it reads and what output features it writes.

---

## 9. eigh vs eig vs svd

- `np.linalg.eigh(A)`: symmetric/Hermitian matrices ONLY. Returns real eigenvalues in **ascending** order. Use this for covariance matrices, W_QK + W_QK.T, etc.
- `np.linalg.eig(A)`: general square matrices. Eigenvalues may be complex. No order guarantee.
- `np.linalg.svd(A)`: any matrix (m × n). Singular values always real, non-negative, **descending** order. Use for W_OV, weight matrix analysis.

---

## 10. Finite Differences — The Sanity Check

Always verify analytic gradients numerically:
```
(f(x + eps) - f(x - eps)) / (2*eps)   # central difference, O(eps^2) error
```
Use eps = 1e-5. If analytic and numerical gradients match to ~4 decimal places, you're right.

---

## 11. Head Composition

In a two-layer transformer, attention head h2 in layer 2 can "use the output of" head h1 in layer 1 via:
- **K-composition**: h1's output ends up in the key of h2
- **Q-composition**: h1's output ends up in the query of h2
- **V-composition**: h1's output ends up in the value of h2

The virtual head formed by composition has effective OV matrix: `W_OV^{h2} @ W_OV^{h1}` (read right to left: h1 reads, h2 writes).

---

## 12. Broadcasting — Three Rules

1. Arrays are compared right-to-left, axis by axis
2. A dimension of size 1 broadcasts to match the other array's size in that axis
3. A missing dimension (shorter array) is treated as size 1 on the left

Common pattern: add `keepdims=True` to a reduction so you can broadcast the result back:
```python
x / x.sum(axis=-1, keepdims=True)    # normalize rows — keepdims critical here
```

---

## 13. MLP Layer

```
hidden = gelu(X @ W1 + b1)    # (T, d_ff)   — d_ff is usually 4 * d_model
output = hidden @ W2 + b2     # (T, d_model)
```
GELU is smoother than ReLU; both are fine for implementing. ReLU is `np.maximum(0, x)`.

---

## Checklist Before Submitting Any Function

- [ ] Does softmax shift by max before exp?
- [ ] Is axis correct in all reductions?
- [ ] Is keepdims=True wherever I broadcast back?
- [ ] Is the causal mask applied before softmax, not after?
- [ ] Did I divide by sqrt(d_k)?
- [ ] Did I verify with a numerical gradient check?
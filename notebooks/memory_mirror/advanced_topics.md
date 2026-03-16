# Advanced Topics — Anthropic Interpretability Screen
*Day-before study sheet. Supplements reference_sheet.md and memorize_sheet.md.*

---

## 1. Backprop Through Softmax Attention

### dL/dS from dL/dA (chain rule through softmax)

Softmax Jacobian: `J_ij = A_i * (delta_ij - A_j)`

Given `dL/dA` (shape T×T), the gradient w.r.t. pre-softmax scores S (also T×T):
```python
# For each row i independently:
# dL/dS_i = J_i.T @ dL/dA_i  where J_i is the T×T Jacobian of row i

# Vectorized:
# dL/dS[i, :] = A[i, :] * (dL/dA[i, :] - (dL/dA[i, :] @ A[i, :]))
dLdS = A * (dLdA - (dLdA * A).sum(axis=-1, keepdims=True))
```
The inner `(dLdA * A).sum(axis=-1, keepdims=True)` is the dot product of each gradient row with the softmax output row — a scalar per row, broadcast back.

### dL/dQ, dL/dK, dL/dV

Given chain from above (dLdS shape T×T, dLdOut shape T×d_v):
```python
# Forward: scores = Q @ K.T / sqrt(d_k)
#          A      = softmax(scores)
#          out    = A @ V

dLdV = A.T @ dLdOut                    # (T, d_v)
dLdA = dLdOut @ V.T                    # (T, T)
dLdS = A * (dLdA - (dLdA * A).sum(axis=-1, keepdims=True))
dLdS /= np.sqrt(d_k)

dLdQ = dLdS @ K                        # (T, d_k)
dLdK = dLdS.T @ Q                      # (T, d_k)
```

### dL/dX through layer norm

```python
# Forward: x_hat = (x - mu) / sqrt(var + eps)
# Gradient:
N = x.shape[-1]
dLdx_hat = dLdy * gamma                # if gamma/beta present; else dLdx_hat = dLdy
std = np.sqrt(var + eps)
dLdx = (1/N) * (1/std) * (N * dLdx_hat - dLdx_hat.sum(-1, keepdims=True)
        - x_hat * (dLdx_hat * x_hat).sum(-1, keepdims=True))
```
Key insight: three terms — direct gradient, mean correction, variance correction.

---

## 2. Circuit Analysis

### Copy Head Detection

A head is a copy head if W_E @ W_OV @ W_U has large positive eigenvalues on the diagonal in token space — i.e., it tends to copy token A to predict token A.

```python
W_OV = W_V @ W_O                         # (d, d)
M = W_E @ W_OV @ W_U                     # (V, V) — don't materialize for large V

# Eigenvalue trick: lambda(AB) = lambda(BA) for nonzero eigenvalues
M_small = W_OV @ W_U @ W_E               # (d, d) — same nonzero eigenvalues
evals = np.linalg.eigvals(M_small)       # complex array — take real parts

# Classify:
# evals mostly real and positive  → copy head
# evals mostly real and negative  → suppression head
# evals mostly complex            → rotation / mixing head

# Copy score: fraction of large real positive eigenvalues
copy_score = np.sum(evals.real > threshold) / len(evals)
```

### W_QK Symmetric / Antisymmetric Decomposition

```python
W_QK = W_Q @ W_K.T                       # (d, d)
W_QK_sym  = (W_QK + W_QK.T) / 2         # symmetric part
W_QK_anti = (W_QK - W_QK.T) / 2         # antisymmetric part

# Eigendecompose symmetric part (real eigenvalues guaranteed)
evals, evecs = np.linalg.eigh(W_QK_sym)  # ascending order

# Interpret:
# sym: content-matching — high ratio → head attends based on token type similarity
# anti: directional — attends based on relative position
sym_norm  = np.linalg.norm(W_QK_sym, 'fro')
anti_norm = np.linalg.norm(W_QK_anti, 'fro')
ratio = sym_norm / (anti_norm + 1e-8)    # high → content-matching, low → positional
```

### Induction Head Detection

```python
def induction_score(model_attention_fn, n=10, vocab_size=50):
    """Construct repeated-token sequence, measure off-diagonal stripe."""
    tokens = np.random.randint(0, vocab_size, size=n)
    seq = np.concatenate([tokens, tokens])             # length 2n

    A = model_attention_fn(seq)                        # (2n, 2n) attention weights

    # Induction stripe: A[n+i, i-1] for i in 1..n-1
    # (destination in second half, source is one before its first-half match)
    stripe_vals = [A[n + i, i + 1] for i in range(n - 1)]
    return np.mean(stripe_vals)

# Score near 1.0 → strong induction head
# Score near 1/T → uniform attention (no induction)
```

### Composition Weights

```python
# Virtual weight matrices for two-layer composition
W_QK_comp = W_OV_L1 @ W_Q_L2               # Q-composition: L1 OV feeds L2 query
W_KK_comp = W_OV_L1 @ W_K_L2               # K-composition: L1 OV feeds L2 key
W_VV_comp = W_OV_L1 @ W_V_L2               # V-composition: L1 OV feeds L2 value

# Composition strength = normalized Frobenius norm of virtual weight
def composition_strength(W_virtual, W_OV_L1, W_proj_L2):
    num = np.linalg.norm(W_virtual, 'fro')
    den = np.linalg.norm(W_OV_L1, 'fro') * np.linalg.norm(W_proj_L2, 'fro')
    return num / (den + 1e-8)

# High strength → heads are composing; low → independent
```

### Full Token-Token Attention Scores (one-layer)

```python
# Do NOT use W_QK alone — must include embeddings
# Token a attending to token b:
# score(a, b) = W_E[a] @ W_QK @ W_E[b] / sqrt(d_k)

token_scores = W_E @ W_QK @ W_E.T / np.sqrt(d_k)    # (V, V)

# What does head write for token a attending to b:
# contribution to logits = W_E[b] @ W_OV @ W_U
write_logits = W_E @ W_OV @ W_U                       # (V, V)

# Skip-trigram: sequence [a, b, ...., a] → predicts b
# score for [a attends to itself via b path]:
# out[a, b, c] = score(b_as_query, a_as_key) * write_logit(a -> c)
ab_scores = np.einsum('bm,mn,an->ab', W_E, W_QK, W_E) / np.sqrt(d_k)
write = W_E @ W_OV @ W_U                              # (V, V)
skip_trigram = np.einsum('ab,ac->abc', ab_scores, write)  # (V, V, V)
# skip_trigram[a, b, c]: attending from position of b to position of a → predicts c
```

---

## 3. SVD Tricks

### Low-Rank SVD Application

```python
U, S, Vh = np.linalg.svd(W_OV, full_matrices=False)  # U(d,d), S(d,), Vh(d,d)

# Rank-r approximation
r = 10
U_r, S_r, Vh_r = U[:, :r], S[:r], Vh[:r, :]

# Apply W_OV to X:   X @ W_OV ≈ (X @ U_r) * S_r @ Vh_r
out = (X @ U_r) * S_r @ Vh_r                          # NOT Vh_r.T

# Low-rank chain W_E @ W_OV @ W_U:
W_E_proj = W_E @ U_r                                  # (V, r) — NOT W_E @ Vh_r.T
W_U_proj = Vh_r @ W_U                                 # (r, V)
M_lowrank = (W_E_proj * S_r) @ W_U_proj               # (V, V) low-rank approx
```

**Memory aid:** W_OV = U S Vh. Input goes through U first (left side), output comes out through Vh (right side). X @ W_OV = X @ U @ diag(S) @ Vh, so compress left with U.

### Eigenvalue Trick for Large Matrices

```python
# Want eigenvalues of M = W_E @ W_OV @ W_U  (V×V, V large)
# Use: nonzero eigenvalues of AB == nonzero eigenvalues of BA

M_small = W_OV @ W_U @ W_E        # (d, d) — same nonzero eigenvalues as M
evals = np.linalg.eigvals(M_small) # d eigenvalues instead of V
# Note: M has V eigenvalues but at most d can be nonzero (rank constraint)
```

General pattern: any time you have `W_E @ M @ W_U` and V >> d, compute eigenvalues of `M @ W_U @ W_E` instead.

### Effective Rank

```python
def effective_rank(W, threshold=0.9):
    """Number of singular values needed to explain `threshold` of total variance."""
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    cumvar = np.cumsum(S**2)
    total = cumvar[-1]
    return next(i + 1 for i, v in enumerate(cumvar) if v >= threshold * total)

# Frobenius norm from SVD (always true):
# np.linalg.norm(W, 'fro') == np.sqrt(np.sum(S**2))
```

### O(T) Linear Attention (no softmax)

```python
# Standard: (Q @ K.T) @ V  — materializes (T, T), O(T^2 d)
# Linear:   Q @ (K.T @ V)  — materializes (d, d), O(T d^2)

KV = K.T @ V                       # (d_k, d_v) — the "context" matrix
out = Q @ KV                        # (T, d_v)

# einsum form:
KV = np.einsum('td,ts->ds', K, V)  # (d, d) — contract over T first
out = np.einsum('td,ds->ts', Q, KV) # (T, d)
```
Only valid without softmax. When T >> d, O(T d^2) << O(T^2 d).

---

## 4. SAE Forward Pass

```python
def sae_forward(x, W_enc, b_enc, W_dec, b_dec, k=None):
    """
    x:     (d_model,) or (T, d_model)
    W_enc: (d_model, n_features)
    W_dec: (n_features, d_model)
    k:     top-k sparsity (None = ReLU only)
    """
    pre_acts = x @ W_enc + b_enc           # (T, n_features)
    acts = np.maximum(0, pre_acts)         # ReLU

    if k is not None:
        # Top-k: keep only k largest activations per token, zero rest
        topk_idx = np.argsort(acts, axis=-1)[:, :-k]   # indices to zero out
        acts_sparse = acts.copy()
        np.put_along_axis(acts_sparse, topk_idx, 0.0, axis=-1)
        acts = acts_sparse

    x_recon = acts @ W_dec + b_dec         # (T, d_model)
    return x_recon, acts

# Feature alignment with OV circuit:
# Which decoder features get amplified by W_OV?
# decoder direction d (shape d_model): after passing through W_OV -> d @ W_OV
# Alignment score between feature f and OV output direction:
def feature_ov_alignment(W_dec, W_OV):
    # W_dec: (n_features, d_model)
    # W_OV:  (d_model, d_model)
    projected = W_dec @ W_OV               # (n_features, d_model)
    # cosine similarity between original decoder direction and projected
    norms_orig = np.linalg.norm(W_dec, axis=-1, keepdims=True)
    norms_proj = np.linalg.norm(projected, axis=-1, keepdims=True)
    cos_sim = (W_dec * projected).sum(-1) / (norms_orig.squeeze() * norms_proj.squeeze() + 1e-8)
    return cos_sim                          # (n_features,) — high = amplified by head
```

---

## 5. Full Transformer Forward Pass

```python
def transformer_forward(tokens, W_E, W_pos, W_Q, W_K, W_V, W_O,
                        W1, b1, W2, b2, W_U, n_heads):
    """
    tokens: (T,) int
    W_E:    (V, d)    W_pos: (T_max, d)
    W_Q/K/V: (H, d, d_k)    W_O: (H, d_k, d)
    W1: (d, d_ff)  b1: (d_ff,)  W2: (d_ff, d)  b2: (d,)
    W_U: (d, V)
    """
    T = len(tokens)
    x = W_E[tokens] + W_pos[:T]               # (T, d)  — embed + positional

    # Attention block (pre-norm)
    x_ln = layer_norm(x)                       # (T, d)
    attn_out = np.zeros_like(x)
    for h in range(n_heads):
        Q = x_ln @ W_Q[h]                     # (T, d_k)
        K = x_ln @ W_K[h]
        V = x_ln @ W_V[h]
        scores = Q @ K.T / np.sqrt(W_Q[h].shape[-1])
        mask = np.triu(np.ones((T, T)), k=1)
        scores -= 1e9 * mask
        A = softmax(scores, axis=-1)
        head_out = A @ V                       # (T, d_k)
        attn_out += head_out @ W_O[h]          # (T, d)
    x = x + attn_out                           # residual

    # MLP block (pre-norm)
    x_ln = layer_norm(x)
    hidden = np.maximum(0, x_ln @ W1 + b1)    # (T, d_ff) — ReLU; swap for gelu
    x = x + hidden @ W2 + b2                  # residual

    logits = x @ W_U                           # (T, V)
    return logits
```

Shape comment discipline: write `# (T, d_k)` on every intermediate. Axis errors are the #1 failure mode.

---

## 6. Conceptual Trap Questions

**Q: Why W_E @ W_QK @ W_E.T, not W_QK alone?**
W_QK = W_Q @ W_K.T lives in d_model × d_model space. It tells you how the model compares *residual stream directions*, not tokens directly. To get token-to-token attention, you need to project through W_E: token a attends to token b with score `W_E[a] @ W_QK @ W_E[b]`.

**Q: Q-comp vs K-comp vs V-comp — which weight pairs?**
- Q-comp: `W_OV_L1 @ W_Q_L2` — L1's output changes *what L2 queries for*
- K-comp: `W_OV_L1 @ W_K_L2` — L1's output changes *what L2 uses as keys* (induction heads use this)
- V-comp: `W_OV_L1 @ W_V_L2` — L1's output changes *what L2 reads as values*

**Q: Frobenius norm composition strength — why normalize?**
Raw Frobenius of the virtual weight depends on the scale of both W_OV_L1 and W_proj_L2. Dividing by the product of their norms gives a scale-invariant measure of how much L1's output is being used by L2's projection, relative to what L2 could use from the residual stream directly.

**Q: eigvalsh vs eigvals — when to use which?**
- `eigvalsh`: symmetric/Hermitian matrix → guaranteed real eigenvalues, numerically stable, ascending order. Use for W_QK_sym.
- `eigvals`: general square matrix → may return complex. Use for copy head detection (W_OV @ W_U @ W_E).

**Q: SVD descending, eigh ascending — don't mix them up.**
`np.linalg.svd` returns S in descending order. `np.linalg.eigh` returns eigenvalues in ascending order (smallest first). Index accordingly.

---

## 7. Quick-Check Table

| Task | Key call | Shape out |
|---|---|---|
| Circuit score matrix | `X @ W_QK @ X.T / sqrt(d_k)` | (T, T) |
| Circuit output | `A @ X @ W_OV` | (T, d) |
| Copy head eigenvalues | `eigvals(W_OV @ W_U @ W_E)` | (d,) complex |
| W_QK sym decompose | `(W_QK + W_QK.T) / 2` | (d, d) |
| Composition virtual weight | `W_OV_L1 @ W_proj_L2` | (d, d_k) |
| Low-rank W_OV apply | `(X @ U_r) * S_r @ Vh_r` | (T, d) |
| Frobenius from SVD | `sqrt(sum(S**2))` | scalar |
| Effective rank | cumsum(S^2) threshold trick | int |
| Induction stripe | `A[n+i, i+1]` for i in 0..n-2 | mean score |
| Linear attention O(T) | `Q @ (K.T @ V)` | (T, d_v) |

---

## 8. Zero-Layer Transformer (Bigram Statistics)

No attention heads. The model is just:
```python
logits = W_E[tokens] @ W_U     # (T, V) — direct embedding → unembedding
```

The full circuit matrix is `W_E @ W_U` (V × V). Entry `[a, b]` is the log-odds the model assigns to token b following token a — pure bigram statistics baked into weights.

```python
bigram_table = W_E @ W_U                  # (V, V)
# Most likely next token after token a:
np.argmax(bigram_table[a])
# Logit of token b given token a:
bigram_table[a, b]
```

No position info, no context — just learned bigram frequencies.

---

## 9. Path Expansion and Term Importance

For a two-layer attention-only model, the logit at position i for token c is a sum of paths:

```
logits[i, c] = direct_path[i, c]
             + sum over L1 heads h1: one_layer_term[i, c, h1]
             + sum over (h1, h2) pairs: two_layer_term[i, c, h1, h2]
```

**Direct path:**
```python
direct = W_E[tokens] @ W_U                # (T, V) — embedding straight to logits
```

**One-layer term (head h in layer 1):**
```python
# A1_h: (T, T) attention pattern for head h
one_layer = A1_h @ W_E[tokens] @ W_OV_h @ W_U   # (T, V)
```

**Two-layer composed term (layer 1 head h1, layer 2 head h2, V-composition):**
```python
# h2 reads the output of h1 via residual stream
composed_ov = W_OV_h1 @ W_OV_h2          # virtual OV matrix (d, d)
two_layer = A2_h2 @ (A1_h1 @ W_E[tokens] @ composed_ov) @ W_U   # (T, V)
```

**Term importance:** rank terms by their contribution to loss. Simplest metric:
```python
# Mean absolute logit contribution at the correct next token
importance = np.abs(term[:, targets]).mean()
```

**Why this matters:** path expansion decomposes the model into interpretable circuits. You can zero out individual terms and measure loss increase — this is attribution at the circuit level (not just head level).

---

## 10. Head Classification Summary

Given weights `W_Q, W_K, W_V, W_O` for a head, classify it:

```python
W_QK = W_Q @ W_K.T                            # (d, d)
W_OV = W_V @ W_O                              # (d, d)

W_QK_sym  = (W_QK + W_QK.T) / 2
W_QK_anti = (W_QK - W_QK.T) / 2
sym_ratio  = np.linalg.norm(W_QK_sym, 'fro') / (np.linalg.norm(W_QK_anti, 'fro') + 1e-8)

# Copy head eigenvalues
evals = np.linalg.eigvals(W_OV @ W_U @ W_E)   # (d,) complex

# Classification logic:
if sym_ratio > threshold_sym:
    if evals.real.max() > threshold_copy:
        head_type = "copy"         # high sym, positive real eigenvalues
    else:
        head_type = "content"      # attends by token content, writes something else
else:
    head_type = "positional"       # attends by relative position

# Suppression: large negative real eigenvalues in W_OV @ W_U @ W_E
if evals.real.min() < -threshold_copy:
    head_type = "suppression"      # actively down-weights previously predicted tokens
```

**Induction head (2-layer):**
- Layer 1: positional, prev-token (A has stripe on k=-1 sub-diagonal)
- Layer 2: content, copy-like, with K-composition from layer 1
- Together: A[n+i, i+1] stripe on repeated-token sequences

**Freezing trick:** to isolate OV from QK, freeze A (treat it as a constant matrix) and analyze `A @ X @ W_OV` as a purely linear function of X. This separates which-positions-to-attend from what-to-write.

---

## 11. SAE Feature Self-Reinforcement via Virtual OV

An SAE trained on residual stream activations has:
- `W_enc`: shape (d_model, n_features) — columns are "reading directions" (what activates each feature)
- `W_dec`: shape (n_features, d_model) — rows are "writing directions" (what each feature adds back)

For feature f:
- `W_enc[:, f]` — the direction in residual stream that activates feature f
- `W_dec[f, :]` — the direction feature f writes back to the residual stream

**Self-reinforcement score:** does a virtual attention head (composed from two real heads) take the feature's writing direction, process it, and amplify its own reading direction?

```python
# Virtual OV matrix from two composed heads
W_OV_virtual = W_OV_h1 @ W_OV_h2          # (d_model, d_model)

# Self-reinforcement score for feature f:
# Does W_OV_virtual map W_dec[f] back toward W_enc[:, f]?
score_f = W_dec[f] @ W_OV_virtual @ W_enc[:, f]   # scalar

# Score all (h1, h2, feature) triples:
scores = np.einsum('fd,de,ef->f', W_dec, W_OV_virtual, W_enc)
# or equivalently:
scores = np.array([W_dec[f] @ W_OV_virtual @ W_enc[:, f] for f in range(n_features)])
```

**Interpretation:**
- High positive score: the virtual head reads the feature's activation direction and writes back in a way that re-activates it — a self-reinforcing loop
- This is a fixed-point analysis: feature → virtual head output → feature again
- Analogous to an eigenvalue: `W_OV_virtual @ W_enc[:, f] ≈ lambda * W_dec[f].T` would be a perfect eigenvector pair

**Shape check:**
```
W_dec[f]:          (d_model,)
W_OV_virtual:      (d_model, d_model)
W_enc[:, f]:       (d_model,)
W_dec[f] @ W_OV_virtual:    (d_model,)   — "where does the feature's output go?"
... @ W_enc[:, f]:           scalar       — "does it land on the feature's input direction?"
```

**To find top self-reinforcing (h1, h2, feature) triples:**
```python
results = []
for h1 in range(n_heads_L1):
    for h2 in range(n_heads_L2):
        W_OV_v = W_OV[h1] @ W_OV[h2]           # (d, d)
        scores = np.einsum('fd,de,ef->f', W_dec, W_OV_v, W_enc)
        top_f = np.argmax(scores)
        results.append((scores[top_f], h1, h2, top_f))
results.sort(reverse=True)
```

---

## Updated Quick-Check Table

| Task | Key call | Shape out |
|---|---|---|
| Zero-layer bigrams | `W_E @ W_U` | (V, V) |
| Direct path logits | `W_E[tokens] @ W_U` | (T, V) |
| One-layer OV term | `A_h @ X @ W_OV_h @ W_U` | (T, V) |
| Two-layer virtual OV | `W_OV_h1 @ W_OV_h2` | (d, d) |
| Head type: positional | sym_ratio low (anti dominates) | — |
| Head type: copy | sym_ratio high + positive real eigvals | — |
| Head type: suppression | large negative real eigvals of W_OV@W_U@W_E | — |
| SAE self-reinforce score | `W_dec[f] @ W_OV_v @ W_enc[:, f]` | scalar |
| SAE all features | `einsum('fd,de,ef->f', W_dec, W_OV_v, W_enc)` | (n_features,) |

# Test Reference Sheet — Anthropic Interpretability Screen
*Can consult during test. Cannot copy-paste — must retype.*

---

## Numerically Stable Implementations

```python
def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)   # shift prevents overflow
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def log_softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))

def cross_entropy(logits, targets):
    lp = log_softmax(logits, axis=-1)              # (B, V)
    return -lp[np.arange(len(targets)), targets].mean()

def layer_norm(x, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
```

---

## Attention

```python
def attention(X, W_Q, W_K, W_V, causal=True):
    # X: (T, d_model)
    Q = X @ W_Q                                    # (T, d_k)
    K = X @ W_K
    V = X @ W_V                                    # (T, d_v)
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)               # (T, T)
    if causal:
        T = scores.shape[0]
        mask = np.triu(np.ones((T, T)), k=1)
        scores -= 1e9 * mask
    weights = softmax(scores, axis=-1)             # (T, T)
    return weights @ V, weights                    # (T, d_v), (T, T)
```

**Circuit matrices (Elhage 2021 convention):**
```
W_QK = W_Q @ W_K.T          # (d_model, d_model)
W_OV = W_V @ W_O            # (d_model, d_model)  — W_O is (d_v, d_model)
scores = X @ W_QK @ X.T / sqrt(d_k)
output_contrib = weights @ X @ W_OV
```

**Softmax Jacobian:**
```
J_ij = pi_i * (delta_ij - pi_j)
```
In code: `J = np.diag(pi) - np.outer(pi, pi)`

---

## einsum Patterns

```python
# matmul
np.einsum('ij,jk->ik', A, B)           # A @ B
# batch matmul
np.einsum('bij,bjk->bik', A, B)
# inner product
np.einsum('i,i->', u, v)               # np.dot(u, v)
# outer product
np.einsum('i,j->ij', u, v)            # np.outer(u, v)
# trace
np.einsum('ii->', A)                   # np.trace(A)
# row norms squared
np.einsum('ij,ij->i', A, A)
# attention scores: Q(T,dk) K(T,dk) -> (T,T)
np.einsum('id,jd->ij', Q, K) / np.sqrt(d_k)
# batched attention: (B,H,T,dk) x (B,H,T,dk) -> (B,H,T,T)
np.einsum('bhid,bhjd->bhij', Q, K) / np.sqrt(d_k)
# bilinear form
np.einsum('i,ij,j->', u, A, v)
```

---

## Linear Algebra

```python
vals, vecs = np.linalg.eigh(A)         # symmetric only; real vals, ascending order
vals, vecs = np.linalg.eig(A)          # general; complex possible
U, S, Vh = np.linalg.svd(A)           # A = U @ diag(S) @ Vh; Vh is V^T
x = np.linalg.solve(A, b)             # Ax = b (faster than inv)
A_inv = np.linalg.pinv(A)             # Moore-Penrose pseudoinverse
rank_k = U[:, :k] @ np.diag(S[:k]) @ Vh[:k, :]  # rank-k approximation

# Project h onto column space of V (columns are basis vectors)
P = V @ np.linalg.pinv(V)             # projection matrix
h_proj = P @ h

# Cosine similarity
cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

---

## NumPy Gotchas

```python
A.sum(axis=0)               # collapses rows -> shape (ncols,)
A.sum(axis=1)               # collapses cols -> shape (nrows,)
A.sum(axis=1, keepdims=True)  # shape (nrows, 1) — use for broadcasting

np.triu(np.ones((T,T)), k=1)   # upper triangle, diagonal excluded (causal mask)
np.tril(np.ones((T,T)), k=0)   # lower triangle, diagonal included

# Finite differences (for checking gradients)
eps = 1e-5
grad_numerical = (f(x + eps) - f(x - eps)) / (2 * eps)
```

---

## Julia → Python Quick Ref

| Julia | Python |
|-------|--------|
| 1-indexed | 0-indexed |
| `A'` | `A.T` |
| `A * B` (matmul) | `A @ B` |
| `A .* B` (elementwise) | `A * B` |
| `size(A, 1)` | `A.shape[0]` |
| `push!(arr, x)` | `arr.append(x)` |
| `begin:end` (inclusive) | `start:stop` (exclusive stop) |
| column-major | row-major |

---

## Misc

```python
# KL divergence D_KL(p || q)
np.sum(p * np.log(p / q))

# Shannon entropy H(p)
-np.sum(p * np.log(p + 1e-10))

# Frobenius norm
np.linalg.norm(A)           # default is Frobenius for 2D
np.linalg.norm(A, 'fro')    # explicit

# Gram-Schmidt / orthonormal basis
Q, R = np.linalg.qr(A)      # A = Q @ R; Q has orthonormal columns

# Finite differences vector version
def numerical_grad(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus, x_minus = x.copy(), x.copy()
        x_plus[i] += eps; x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad
```
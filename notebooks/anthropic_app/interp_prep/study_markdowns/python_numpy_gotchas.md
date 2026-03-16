# Python / NumPy Gotchas

## NumPy ↔ Python Container Mistakes

### np.ndarray vs np.array

```python
np.ndarray((3, 4))      # BAD: allocates uninitialized memory — shape tuple, NOT data
np.array([[1, 2], [3, 4]])  # GOOD: wraps existing data
np.zeros((3, 4))        # GOOD: use zeros/ones/empty for shape-based construction
```

### Views vs Copies — the #1 silent bug

```python
a = np.array([1, 2, 3, 4])
b = a[1:3]      # b is a VIEW — modifying b modifies a!
b[0] = 99       # a is now [1, 99, 3, 4]

b = a[1:3].copy()   # safe copy
b = a[[1, 2]]       # fancy indexing always returns a copy
b = a[a > 2]        # boolean indexing always returns a copy
```

### Converting back to Python

```python
arr = np.array([1, 2, 3])
list(arr)       # [np.int64(1), np.int64(2), np.int64(3)] — numpy scalars, not Python ints
arr.tolist()    # [1, 2, 3] — pure Python ints
arr[0]          # np.int64(1) — numpy scalar
int(arr[0])     # 1 — Python int (needed for some stdlib functions)
```

### Shape gotchas

```python
np.zeros(3)         # shape (3,)  — 1D
np.zeros((3,))      # shape (3,)  — same
np.zeros((3, 1))    # shape (3,1) — 2D column vector, NOT the same!
np.zeros((1, 3))    # shape (1,3) — 2D row vector

# (3,) vs (3,1) matters for broadcasting and matrix multiply
v = np.array([1, 2, 3])       # shape (3,)
v[:, None]                     # shape (3,1) — add axis
v[None, :]                     # shape (1,3)
v.reshape(-1, 1)               # shape (3,1)
```

### len() lies

```python
A = np.zeros((3, 4))
len(A)      # 3 — only first dimension
A.size      # 12 — total elements
A.shape     # (3, 4)
A.ndim      # 2
```

### Ragged lists become object arrays

```python
np.array([[1, 2], [3, 4, 5]])   # dtype=object — not what you want
np.array([[1, 2], [3, 4]])      # dtype=int64 — fine, uniform shape
```

---

## Julia → NumPy Translation

### DANGER: * means opposite things

```python
# Julia: A * B  = matrix multiply
# Julia: A .* B = element-wise

# NumPy: A * B  = element-wise  ← opposite!
# NumPy: A @ B  = matrix multiply

A @ B           # matrix multiply
A * B           # element-wise (Hadamard)
np.dot(A, B)    # matrix multiply for 2D (same as @)
```

### Indexing

```python
# Julia: 1-indexed, inclusive end
# A[1]      → first element
# A[1:3]    → elements 1, 2, 3 (inclusive)
# A[end]    → last element

# NumPy: 0-indexed, exclusive end
A[0]        # first element
A[0:3]      # elements 0, 1, 2 (NOT including 3)
A[:3]       # same
A[-1]       # last element
A[1:-1]     # all but first and last
```

### Shape construction — extra parens

```python
# Julia: zeros(3, 4)
# NumPy: np.zeros((3, 4))   ← tuple, not separate args

np.zeros((3, 4))
np.ones((3, 4))
np.eye(3)           # identity (no extra parens — single int)
np.random.randn(3, 4)   # exception: randn takes separate args, not tuple
np.random.rand(3, 4)    # same
```

### Transpose

```python
# Julia: A'  = conjugate transpose (adjoint)
# Julia: transpose(A) = non-conjugate

A.T             # transpose (no conjugate)
A.conj().T      # conjugate transpose = Julia A'
```

### Reductions — axis is 0-indexed, opposite direction from Julia dims

```python
# Julia: sum(A, dims=1) = sum along rows → result shape (1, ncols)
# NumPy: np.sum(A, axis=0) = sum along axis 0 (rows) → shape (ncols,)

np.sum(A, axis=0)   # collapse rows   → Julia dims=1
np.sum(A, axis=1)   # collapse cols   → Julia dims=2
np.sum(A)           # sum everything
```

### SVD — NumPy returns Vh, not V

```python
# Julia: U, S, V = svd(A)   — V is the right singular vectors
# NumPy: U, S, Vh = np.linalg.svd(A)  — Vh is ALREADY TRANSPOSED

U, S, Vh = np.linalg.svd(A)
V = Vh.T        # if you need V (columns = right singular vectors)

# Also: full_matrices=False for economy SVD
U, S, Vh = np.linalg.svd(A, full_matrices=False)
```

### Broadcasting — no dot syntax needed

```python
# Julia: A .+ b  (must use dot for broadcasting)
# NumPy: A + b   (automatic broadcasting, no dot needed)

A + b       # broadcasts automatically
A * b       # element-wise, broadcasts
np.exp(A)   # always element-wise — no .exp needed
```

### Ranges

```python
# Julia: 1:n  = 1, 2, ..., n  (1-indexed, inclusive)
# Julia: collect(1:n) = [1, 2, ..., n]

np.arange(n)        # 0, 1, ..., n-1  ← NOT 1..n
np.arange(1, n+1)   # 1, 2, ..., n    ← Julia collect(1:n)
range(n)            # Python lazy range 0..n-1
list(range(1, n+1)) # Python list 1..n
```

### Linear algebra

```python
# Julia: A \ b  (backslash solve)
np.linalg.solve(A, b)

# Julia: inv(A)
np.linalg.inv(A)

# Julia: norm(a)
np.linalg.norm(a)

# Julia: det(A)
np.linalg.det(A)

# Julia: eigen(A) → vals, vecs
vals, vecs = np.linalg.eig(A)   # vecs[:,i] is eigenvector for vals[i]
vals, vecs = np.linalg.eigh(A)  # for symmetric/Hermitian — returns real vals, sorted
```

### Misc

```python
# Julia: size(A)     → A.shape
# Julia: size(A, 1)  → A.shape[0]
# Julia: length(A)   → A.size
# Julia: ndims(A)    → A.ndim
# Julia: vec(A)      → A.flatten() or A.ravel()  (ravel returns view if possible)
# Julia: nothing     → None
# Julia: NaN         → np.nan
# Julia: Inf         → np.inf
# Julia: hcat(A,B)   → np.hstack([A, B])
# Julia: vcat(A,B)   → np.vstack([A, B])
# Julia: copy(A)     → A.copy()
```

---

## Rookie NumPy-Only Mistakes

```python
# axis direction: axis=0 collapses ROWS (operates down columns)
np.sum(A, axis=0)   # shape (ncols,) — one value per column
np.sum(A, axis=1)   # shape (nrows,) — one value per row

# np.dot behavior depends on dimensions
np.dot(a, b)        # 1D: inner product (scalar)
np.dot(A, B)        # 2D: matrix multiply (same as @)
np.dot(A, b)        # 2D x 1D: matrix-vector product

# Integer dtype overflow — silent!
a = np.array([200], dtype=np.int8)
a * 2   # overflow, wraps around — use dtype=int or float

# np.where returns tuple of index arrays
idx = np.where(a > 0)   # tuple: (array([...]),) — not a flat array
idx = np.where(a > 0)[0]  # flat array of indices

# np.linalg.svd — Vh is already transposed (see Julia section)

# Stacking vs concatenating
np.stack([a, b])        # new axis: (2, n) from two (n,) arrays
np.concatenate([a, b])  # no new axis: (2n,) from two (n,) arrays

# Boolean mask on wrong shape
mask = np.array([True, False, True])
A[mask]     # selects rows 0 and 2 — works if A is 2D
A[:, mask]  # selects columns 0 and 2
```

---

## Rookie Python-Only Mistakes

```python
# Mutable default argument — persists across calls!
def f(x=[]):        # BAD
    x.append(1)
    return x
def f(x=None):      # GOOD
    if x is None: x = []

# Two names, one list
a = b = []
a.append(1)   # b is also [1] — same object!
a = []
b = []        # separate lists

# Nested list trap — all rows are same object
grid = [[0] * 3] * 3
grid[0][0] = 1   # ALL rows become [1, 0, 0]!
grid = [[0] * 3 for _ in range(3)]   # GOOD

# Division
3 / 2    # 1.5  (always float in Python 3)
3 // 2   # 1    (floor division)

# range is lazy
r = range(5)    # not a list, can't index like one in general usage
list(range(5))  # [0, 1, 2, 3, 4]

# is vs ==
a = [1, 2]
b = [1, 2]
a == b   # True  (same values)
a is b   # False (different objects)
a is None   # correct way to check for None (not a == None)

# Tuple vs single element tuple
x = (3)     # int 3, NOT a tuple
x = (3,)    # tuple with one element

# Unpacking
a, b = 1, 2         # fine
a, b = [1, 2]       # also fine
a, b = func()       # fine if func returns exactly 2 items
a, *b = [1,2,3,4]   # a=1, b=[2,3,4]
```

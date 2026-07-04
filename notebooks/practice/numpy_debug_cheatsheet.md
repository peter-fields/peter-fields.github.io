# NumPy Debug Cheat Sheet — Anthropic Fellows 60-min Test

## `np.bincount(x)` — count integers
```python
np.bincount([0, 1, 1, 3])      # [1, 2, 0, 1]
np.bincount([0,1,1], weights=[10,20,30])  # [10, 50]  — sum of weights per bin
```
Output length = max(x) + 1. Missing integers get 0.

## `np.nonzero(x)` — indices of nonzero elements
```python
np.nonzero([0, 5, 0, 7])       # (array([1, 3]),)  ← tuple!
np.nonzero([[0,1],[2,0]])      # (array([0,1]), array([1,0]))  ← rows, cols
x[np.nonzero(x)]               # actual nonzero values
```

## `np.where(cond)` / `np.where(cond, a, b)`
```python
np.where(x > 3)                # 1-arg: same as nonzero(x > 3)
np.where(x > 3, x, 0)          # 3-arg: x where True, 0 where False
np.where(counts > 0, sums/counts, 0.0)  # safe division
```

## `np.random.RandomState` — private RNG
```python
rs = np.random.RandomState(42)
rs.rand(3), rs.choice(n, size=k, replace=False), rs.randint(0, 10)
rs.normal(0, 1, size=5), rs.permutation(n), rs.shuffle(arr)
```
Same seed → same sequence. **Bug to watch:** function takes `rs` but calls `np.random.X` instead of `rs.X` (uses global RNG, breaks reproducibility).

## Floating point gotchas
| expression | result |
|---|---|
| `0.1 + 0.2 == 0.3` | `False` (use `np.isclose`/`np.allclose`) |
| `5 / 0` | `inf` |
| `0 / 0` | `NaN` |
| `5 / inf` | `0.0` |
| `inf - inf` | `NaN` |
| `np.nan == np.nan` | `False` (use `np.isnan`) |

NaN propagates through arithmetic. Use `np.nansum`, `np.nanmean` to skip.

**Mixed dtype gotcha:**
```python
a = np.array([1, 2, 3])    # int64
a[0] = 1.7                 # silently truncated to 1
```

## Broadcasting — pairwise pattern
```python
# all pairwise differences for points of shape (n, d):
diff = points[:, None, :] - points[None, :, :]   # (n, n, d)
dists = np.sqrt(np.sum(diff**2, axis=2))         # (n, n)
```
`None` (a.k.a. `np.newaxis`) inserts a length-1 axis. Broadcasting expands the 1-axes to match the other operand.

**Masking the diagonal:**
```python
np.fill_diagonal(dists, np.inf)   # in place, modifies diagonal only
mask = np.eye(n, dtype=bool)      # boolean diagonal mask
```
**Don't** use `dists == 0` to find self-distances — it catches legitimate zeros too.

## Boolean indexing
```python
x[x > 0]                       # values where condition is True
x[x > 0] = 0                   # set them in place
mask = (x > 0) & (x < 10)      # combine with & | ~  (NOT and/or/not)
```

## Copy vs view (debugging gotcha)
```python
a = np.array([1, 2, 3, 4])
b = a[1:3]              # VIEW — shares memory with a
b[0] = 99
a                       # [1, 99, 3, 4] — a was mutated!
b = a[1:3].copy()       # independent copy
```
**Slicing returns views, fancy indexing returns copies.** If a function mutates an array passed in and you didn't expect it, suspect view aliasing. `.copy()` to be safe.

## `np.clip(x, lo, hi)`
```python
np.clip(x, 0, 1)        # values < 0 → 0, > 1 → 1
```

## `np.unique`
```python
np.unique([3, 1, 2, 1, 3])              # [1, 2, 3] (sorted)
np.unique(x, return_counts=True)        # (vals, counts)
np.unique(x, return_inverse=True)       # also returns reconstruction indices
```

## `collections.NamedTuple` / `typing.NamedTuple`
```python
from typing import NamedTuple

class Config(NamedTuple):
    name: str
    layers: list

c = Config("a", [1,2,3])
c.name, c[0]               # access by name or index
c2 = c._replace(name="b")  # immutable copy with one field changed
c._asdict()                # convert to dict
```
**Immutable bindings**, but mutable fields can still be mutated:
```python
c.layers.append(4)         # works — mutates the list
c.layers = []              # AttributeError
```
**Shared mutation bug:**
```python
DEFAULT = [128, 64]
configs = [Config(n, DEFAULT) for n in names]  # all share same list!
# Fix: list(DEFAULT) or DEFAULT.copy()
```

## Recursion
Need a **base case** (terminates) and a **recursive case** (calls itself on a SMALLER input):
```python
def factorial(n):
    if n == 0:                              # base case
        return 1
    return n * factorial(n - 1)             # recursive case — smaller n

def tree_depth(node):
    if not node.children:                   # base case (leaf)
        return 1
    return 1 + max(tree_depth(c) for c in node.children)

def tree_sum(node):
    return node.value + sum(tree_sum(c) for c in node.children)
    # No explicit base case needed — empty children means sum(...) = 0

def fib(n, memo={}):
    if n < 2:
        return n
    if n in memo:                           # memoization — cache results
        return memo[n]
    memo[n] = fib(n-1, memo) + fib(n-2, memo)
    return memo[n]
```

**Gotchas:**
- **Missing base case** → `RecursionError` (stack overflow ~1000 calls deep).
- **Wrong base case value** → off-by-one results (e.g. `return 0` for leaf when it should be `return 1`).
- **Recursing on same input** → no progress, infinite recursion:
  ```python
  def bad(n):
      if n == 0: return 1
      return n * bad(n)        # BUG: should be bad(n-1)
  ```
- **Returning nothing** in the recursive case:
  ```python
  def bad(n):
      if n == 0: return 0
      bad(n - 1)               # BUG: missing return — returns None
  ```
- **Mutable default argument** (memo dict shared across calls — see below).
- **Order of operations**: recursing then combining vs combining then recursing can give very different results.
- **Tail recursion isn't optimized in Python** — deep recursion will overflow; rewrite as a loop if needed.

**Mutable default argument trap:**
```python
def grow(x, acc=[]):           # acc shared across ALL calls!
    acc.append(x)
    return acc
grow(1)    # [1]
grow(2)    # [1, 2]   ← unexpected
# Fix: acc=None then acc = [] if acc is None else acc
```

**Tracing a recursive call** with prints:
```python
def tree_depth(node, depth=0):
    print(" " * depth, "enter", node.value)
    if not node.children:
        print(" " * depth, "leaf → 1")
        return 1
    result = 1 + max(tree_depth(c, depth+1) for c in node.children)
    print(" " * depth, "exit", node.value, "→", result)
    return result
```

## unittest essentials
```python
import unittest

class TestThing(unittest.TestCase):
    def test_foo(self):
        self.assertEqual(actual, expected)
        self.assertTrue(cond)
        self.assertFalse(cond)
        self.assertRaises(ValueError, func, args)
        with self.assertRaises(ValueError):
            func(args)

import numpy as np
np.testing.assert_array_equal(a, b)        # exact
np.testing.assert_allclose(a, b)           # for floats, tolerant
```

## Debugging strategy (60-min test)
1. **Read all tests first** — they're the spec
2. **Run tests** — see what fails
3. **Read the failing test's docstring + setup** — understand the expected behavior
4. **`print()` in the source** — show inputs, intermediate values, outputs
5. **Compare actual vs expected** — the gap points to the bug
6. **Fix one test at a time** — don't batch

## Common bug patterns (from practice)
- Recursion: base case returns wrong value (0 vs 1)
- NamedTuple: same list passed to many instances → shared mutation
- RandomState: function uses global `np.random` instead of passed-in `rs`
- bincount: dividing by counts that may be 0 → NaN
- Distance masking: `== 0` catches both diagonal and legitimate zeros

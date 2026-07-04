# Debugging Practice — Future Plan

After the 2026-05-21 Anthropic Fellows debugging assessment (2/6 bugs found). Areas that need work, ordered by what would have helped today.

## 1. Recursion fluency (biggest gap today)

The test had a tree-recursion bug that hit max recursion depth. Got stuck.

**Practice problems to do:**
- LeetCode #104 Maximum Depth of Binary Tree
- LeetCode #226 Invert Binary Tree
- LeetCode #98 Validate BST
- LeetCode #110 Balanced Binary Tree
- LeetCode #297 Serialize/Deserialize Binary Tree
- Build a decision tree from scratch (fit + predict), then sabotage it and debug

**Patterns to get reflexive about:**
- "Recurse on subtree, combine results" — depth, sum, max
- DFS with early termination (validate BST)
- Tree construction from list (level-order)
- Path-tracking through tree (accumulator passed down)

**Recursion bug patterns to watch:**
- `isinstance` against wrong class — base case never fires (Leaf inherits from Node etc.)
- Recursing on same input (no progress)
- Self-referential structure (tree.left = tree somewhere upstream)
- None vs Leaf — code treats None as recurse-able
- Wrong base value (0 vs 1 vs None)

## 2. pdb fluency

Today I used print statements. That was right for the time pressure, but pdb is faster once you're fluent.

**Practice plan:** spend 1 hour/week debugging real bugs with pdb (no AI). Try on:
- Bugs in my own research code
- Going through the pdb-tutorial dice game again, then `pdb-tutorial`'s advanced exercises if any
- Adding a `pdb.set_trace()` to a recursive function and stepping through 5 levels of calls

**Commands to internalize beyond the basics:**
- `display <expr>` — auto-print an expression at every stop (HUGE for recursion)
- `b <file>:<line>, <cond>` — conditional breakpoint (e.g. only stop when n == 5)
- `commands <bpnum>` — auto-execute commands at a breakpoint
- `interact` — drop into a regular Python REPL with current locals
- `pp` vs `p` — pretty print for nested structures
- `up` / `down` — navigate the call stack to inspect callers
- `until <line>` — run until a specific line in the current function (loops)
- `j <line>` — jump to a specific line (skip code)

## 3. Reading unfamiliar code under time pressure

The volume of code is the test. Need to be faster at:
- Skimming a codebase: read function signatures + docstrings first, not bodies
- Using the failing test as the entry point (work backwards from assertion)
- Resisting urge to read top-to-bottom
- Identifying the "path" through code that a specific test exercises

**Drill:** clone a small open-source library you don't know, pick a function, trace what it does by reading types + tests only, then check yourself.

## 4. CodeSignal-style practice

The CodeSignal platform itself adds overhead — running tests in their UI, debugging without your normal tools.

- Take more CodeSignal sample assessments if available
- The Anthropic 90-min "general coding" + 60-min "debugging" formats are likely re-used by other orgs (Astra, etc.)

## 5. NumPy / scientific Python fluency

Less urgent than recursion+pdb but still gaps. Already on the practice plan:
- Implementing AI Algorithms from Scratch (CodeSignal course)
- Four-Week Coding Interview Prep
- See [python-practice-plan.md](../../.claude/memory/python-practice-plan.md) (in memory)

## 6. ML algorithms from scratch in numpy

Gaps revealed by the 2026-04-22 LASR CodeSignal assessment:
- **Gradient descent for multivariate linear regression** — got stuck on a shape/broadcasting bug, didn't finish
- **Naive Bayes posterior** — ran out of time

These are "implement the classic ML algorithm from scratch using only numpy" problems. They're the bread and butter of ML coding assessments (LASR, FAR.AI 72-min, likely the Anthropic Fellows take-home).

**Algorithms to be able to write from memory in ~10 minutes each:**
- Linear regression — closed form + gradient descent
- Logistic regression — gradient descent, sigmoid trick
- k-Nearest Neighbors — distance, argpartition
- k-Means — assign/update loop, centroid init
- Decision tree — split criterion, recursive build (← also recursion practice!)
- Naive Bayes — prior, likelihood, posterior, log-space stability
- PCA — covariance, eigendecomposition, project
- Forward pass of an MLP — broadcasting through layers
- Backprop — chain rule by hand for 2-layer net
- GMM / EM — soft assignments, M step

**Shape-handling drill:** for each algorithm, write out shapes of every intermediate array as a comment. Most numpy bugs are shape bugs.

```python
X      # (n, d)
y      # (n,)
w      # (d,)
pred   # (n,)  — X @ w
grad   # (d,)  — X.T @ (pred - y) / n
```

This is what "Implementing AI Algorithms from Scratch" CodeSignal course covers — Phase 1 priority on [python-practice-plan.md](../../.claude/memory/python-practice-plan.md).

## Cadence

- **Daily:** 1 hour CodeSignal coursework (existing plan)
- **Weekly:** 1 hour pdb debugging on real code, no AI
- **Weekly:** 3-5 LeetCode tree/recursion problems, no AI
- **Before next timed test:** redo the debug_practice/ problems with pdb instead of print, plus 5 new tree-recursion problems

## What today taught me

- Debugging tests aren't about syntax — they're about finding bugs efficiently in unfamiliar code
- Print statements work but pdb is meaningfully faster if you're fluent
- Recursion bugs are the canonical "I can stare at this code for 20 minutes and still not see it" trap
- "Read the docstring carefully" was right but only gets you so far — you also need to read the tests and the call graph

---
name: arena-test-prep
description: "Prep notes + strategy for the ARENA 9.0 coding test (1hr, 6 general-Python Leetcode-style Qs, Coderbyte, no AI). Deadline 11:59pm AoE Sun Aug 2 2026."
metadata: 
  node_type: memory
  type: project
  originSessionId: 24e0f58e-a194-4c1e-a0de-757c991eef02
  modified: 2026-08-03T14:47:43.119Z
---

# ARENA 9.0 Coding Test — Prep Notes

**The test:** 1 hour · **6 questions, general Python** (Leetcode-style, *explicitly NOT ML*) · Coderbyte IDE, Python only · **NO AI ASSISTANCE** (leaving the IDE = flagged as cheating). **Deadline: 11:59pm AoE, Sun Aug 2 2026.** Start anytime before then; can't pause once begun. Tech issues → james@arena.education (flag EARLY, not at the deadline).

**⚠️ Integrity line:** CC helps with PREP ONLY (practice problems, mock timed runs on *separate* problems — ARENA themselves recommend practicing Leetcode). During the actual test = **solo, no AI, no CC.** (cf. Peter's MATS/CodeSignal history — keep this unimpeachable.)

**When to take it:** take it fresh, with a clean uninterrupted hour. **AVOID the ZOË trip (July 26–28).** Aim ~7/24–25 or ~7/30–8/1.

---

## Scope — what to prep (and NOT prep)

**It's core Python + standard library. NO numpy, NO ML libraries.** ("No ML libs" + "you may search docs for *standard libraries* like math/random" → numpy is third-party, not needed; Leetcode-style is always pure Python.) **Don't waste time on numpy/pytorch/scikit.**

**Drill these (the "additional stuff" that actually shows up):**
- **`collections.Counter` and `defaultdict`** — the single most useful tools for Leetcode Python (frequency counting, grouping). Know cold.
- **Comprehensions** (list/dict/set), **`enumerate`**, **`zip`**, **`sorted(…, key=…)`**, **sets** (dedup/membership/intersection/union), **slicing** (incl. `[::-1]`, step slices), core **string methods** (`split`, `join`, `strip`, `replace`, `isdigit`, `.lower()`).
- **Patterns:** frequency/hashing, two-pointers, sliding window, recursion — and possibly a simple **BFS/DFS** (grid/graph) or **basic DP** for the last two harder Qs.

**math / random — don't sweat.** Won't likely touch `random`; `math` is searchable + rarely central (maybe `gcd`, `sqrt`, `comb`). Just know they exist.

---

## 🪤 Julia→Python traps (Peter's real risk — he's Julia-native)
- **0-INDEXING.** Julia is 1-indexed, Python is 0-indexed. #1 source of off-by-one bugs on the switch. `range(n)` = `0 … n-1`. Burn this in.
- **`//` = integer division** (not `/`, which returns a float); `%` = modulo.
- **Slicing is end-exclusive:** `a[1:4]` = indices 1,2,3 (not 4).
- `range`, `enumerate`, dict/set literal syntax, comprehension syntax all differ from Julia.

---

## Test-day strategy (ARENA's advice + reinforcement)
- **Front-load the easy questions** — bank the first 4 solid before touching the last 2 (advice, not a rule).
- **Don't sink all your time into one question** (they emphasized this hard). Stuck → move on.
- **Test your code on the given examples** before submitting.
- **HIT SUBMIT on every question** (top-right button) — solutions may not register otherwise; double-check all at the end.
- Most candidates won't finish all 6, and it's not their only signal → aim for **clean** on what you do answer.
- Questions in any order, but they recommend the presented order.

---

## ✅ TAKEN (confirmed 2026-08-03) — Peter found it hard
**Post-mortem so far:** one question asked for the **unique set of non-consecutive Fibonacci numbers summing to n** — he didn't know how to approach it. That's **Zeckendorf's theorem** (every positive integer has exactly one such representation), so the answer is a **greedy O(log n) walk** — take the largest Fibonacci ≤ n, subtract, repeat — *not* the subset-sum/DP it superficially resembles. Greedy automatically avoids adjacent terms. Detail: the Fibonacci list must start `[1, 2]`, not `[1, 1, 2]` (the duplicate 1 breaks uniqueness). Worked solution + brute-force verification of the theorem in `notebooks/practice/singapore_test_prep/zeckendorf.py`.
**⭐ Transferable test-taking lesson:** when a question looks like exponential search, **tabulate n = 1…12 by hand first** — the greedy pattern was visible in ~2 min without knowing the named theorem. Reaching for DP was the trap.

## Prep plan (when Peter's ready)
CC to write a **diagnostic batch** (~5 problems spanning the range: a couple easy string/dict, a two-pointer/hashing, a recursion, one harder) → Peter solves solo → CC reviews + drills weak spots. Then a **timed mock** (6 problems, 1 hr) closer to test day.

Related: [[python-practice-plan]], [[arena-app]], [[debug_future_practice]]

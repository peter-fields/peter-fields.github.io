"""Unique sum of non-consecutive Fibonacci numbers — the ARENA question.

This is Zeckendorf's theorem: every positive integer has EXACTLY ONE representation
as a sum of non-consecutive Fibonacci numbers. That turns what looks like a
subset-sum search into a greedy walk down the list.

Run: python zeckendorf.py
"""

from itertools import combinations


def fibs_up_to(n):
    """1, 2, 3, 5, 8, ... -- note it starts [1, 2], NOT [1, 1, 2].

    The duplicate 1 in the usual sequence would break uniqueness (4 = 3+1 twice).
    """
    fibs = [1, 2]
    while fibs[-1] + fibs[-2] <= n:
        fibs.append(fibs[-1] + fibs[-2])
    return [f for f in fibs if f <= n]


def zeckendorf(n):
    """Greedy: repeatedly take the largest Fibonacci number that still fits."""
    if n < 0:
        raise ValueError("n must be non-negative")
    out = []
    for f in reversed(fibs_up_to(n)):
        if f <= n:
            out.append(f)
            n -= f
    return out          # descending; out[::-1] if the test wants ascending


# ---------------------------------------------------------------------------
# Why greedy never picks two neighbours:
#   let F_k be the largest Fibonacci <= n, and r = n - F_k.
#   if r were >= F_{k-1}, then n >= F_k + F_{k-1} = F_{k+1}, contradicting
#   F_k being the largest that fits. So r < F_{k-1}, and the next term taken
#   is at most F_{k-2}. Never adjacent. The constraint enforces itself.
# ---------------------------------------------------------------------------


def all_representations(n):
    """Brute force, to check the theorem rather than take it on faith."""
    fibs = fibs_up_to(n)
    found = []
    for size in range(1, len(fibs) + 1):
        for idx in combinations(range(len(fibs)), size):
            if any(b - a == 1 for a, b in zip(idx, idx[1:])):
                continue                              # adjacent -> not allowed
            if sum(fibs[i] for i in idx) == n:
                found.append(sorted((fibs[i] for i in idx), reverse=True))
    return found


if __name__ == "__main__":
    for n in range(1, 13):
        print(f"{n:3} = {' + '.join(map(str, zeckendorf(n)))}")

    print()
    print("100 =", zeckendorf(100))
    print("2026 =", zeckendorf(2026))

    # the theorem, checked: exactly one representation for every n, and greedy finds it
    for n in range(1, 501):
        reps = all_representations(n)
        assert len(reps) == 1, (n, reps)
        assert reps[0] == zeckendorf(n), (n, reps, zeckendorf(n))
    assert zeckendorf(0) == []
    print("\nuniqueness + greedy verified for n = 1..500")

import numpy as np


def sample_indices(n: int, k: int, rs: np.random.RandomState) -> np.ndarray:
    """Sample k distinct indices from 0..n-1 using the provided RandomState.

    Must be reproducible: two calls with RandomState instances constructed
    from the same seed must return identical arrays.

    Args:
        n: population size.
        k: number of indices to sample (k <= n).
        rs: numpy RandomState instance to draw from.

    Returns:
        1D int array of length k with no duplicates.
    """
    return rs.choice(n, size=k, replace=False)
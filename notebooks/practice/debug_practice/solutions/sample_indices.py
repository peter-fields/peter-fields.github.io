import numpy as np


def sample_indices(n: int, k: int, rs: np.random.RandomState) -> np.ndarray:
    # FIX: was using the global np.random.choice, which ignores the passed-in
    # RandomState and therefore breaks reproducibility. Use rs.choice instead.
    return rs.choice(n, size=k, replace=False)

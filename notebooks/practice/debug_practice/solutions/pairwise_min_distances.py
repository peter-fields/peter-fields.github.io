import numpy as np


def pairwise_min_distances(points: np.ndarray) -> np.ndarray:
    # FIX: `dists == 0` masks ALL zero distances — both the diagonal (self)
    # and any coincident pairs (legitimate zero distances). What we want is
    # to mask only the diagonal. Use np.eye to make a boolean diagonal mask.
    diff = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    n = points.shape[0]
    mask = np.eye(n, dtype=bool)
    dists[mask] = np.inf
    return np.min(dists, axis=1)
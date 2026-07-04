import numpy as np


def pairwise_min_distances(points: np.ndarray) -> np.ndarray:
    """For each point, return the Euclidean distance to its nearest OTHER point.

    Args:
        points: array of shape (n, d), n points in d dimensions.

    Returns:
        1D array of length n where result[i] is the minimum Euclidean
        distance from points[i] to any other point points[j] (j != i).

        If two points happen to coincide, the nearest-other-point distance
        for both of them is 0.

    Example:
        points = [[0, 0], [3, 0], [0, 4]]
        -> [3.0, 3.0, 4.0]
    """
    diff = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    mask = np.isclose(dists, 0., rtol=0.00000000000001)
    np.fill_diagonal(dists, np.inf)
    # dists[mask] = 0.
    return np.min(dists, axis=1)
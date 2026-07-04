import numpy as np


def weighted_class_means(labels: np.ndarray, values: np.ndarray) -> np.ndarray:
    # FIX: dividing sums/counts gives NaN when counts == 0 (empty class).
    # Use np.where to return 0.0 for empty classes. np.maximum(counts, 1)
    # avoids the divide-by-zero warning during the division.
    sums = np.bincount(labels, weights=values)
    counts = np.bincount(labels)
    means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    return means
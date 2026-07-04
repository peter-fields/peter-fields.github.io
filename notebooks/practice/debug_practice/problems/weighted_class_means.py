import numpy as np


def weighted_class_means(labels: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Compute the mean of `values` for each integer class label.

    Args:
        labels: 1D int array of non-negative class labels.
        values: 1D float array of same length as labels.

    Returns:
        1D float array of length max(labels)+1, where entry i is the mean
        of `values` over positions where labels == i. Classes with no
        samples must return 0.0 (not NaN).

    Example:
        labels = [0, 0, 1, 2, 2]
        values = [1.0, 3.0, 5.0, 7.0, 9.0]
        -> [2.0, 5.0, 8.0]
    """
    sums = np.bincount(labels, weights=values)
    
    counts = np.bincount(labels)
    counts=np.where(counts == 0, np.inf, counts)
    means = sums / counts
    # print(means)
    # print( 5 / np.inf )
    return means
import unittest

import numpy as np

from pairwise_min_distances import pairwise_min_distances


class TestPairwiseMinDistances(unittest.TestCase):
    def test_simple_triangle(self):
        points = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
        result = pairwise_min_distances(points)
        np.testing.assert_allclose(result, [3.0, 3.0, 4.0])

    def test_1d_points(self):
        points = np.array([[0.0], [1.0], [10.0]])
        result = pairwise_min_distances(points)
        np.testing.assert_allclose(result, [1.0, 1.0, 9.0])

    def test_coincident_points(self):
        # Two points at exactly the same location.
        # Each duplicate's nearest-other-point distance is 0.
        points = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 10.0]])
        result = pairwise_min_distances(points)
        expected = [0.0, 0.0, np.sqrt(200.0)]
        np.testing.assert_allclose(result, expected)

    def test_higher_dim(self):
        points = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        result = pairwise_min_distances(points)
        np.testing.assert_allclose(result, [np.sqrt(2)] * 3)


if __name__ == "__main__":
    unittest.main()

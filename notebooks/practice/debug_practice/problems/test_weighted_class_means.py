import unittest

import numpy as np

from weighted_class_means import weighted_class_means


class TestWeightedClassMeans(unittest.TestCase):
    def test_basic(self):
        labels = np.array([0, 0, 1, 2, 2])
        values = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        result = weighted_class_means(labels, values)
        np.testing.assert_allclose(result, [2.0, 5.0, 8.0])

    def test_empty_middle_class(self):
        # class 1 has no samples
        labels = np.array([0, 0, 2, 2])
        values = np.array([1.0, 3.0, 5.0, 7.0])
        result = weighted_class_means(labels, values)
        np.testing.assert_allclose(result, [2.0, 0.0, 6.0])

    def test_single_sample_per_class(self):
        labels = np.array([0, 1, 2])
        values = np.array([10.0, 20.0, 30.0])
        result = weighted_class_means(labels, values)
        np.testing.assert_allclose(result, [10.0, 20.0, 30.0])

    def test_no_nan(self):
        labels = np.array([0, 0, 3])
        values = np.array([1.0, 1.0, 5.0])
        result = weighted_class_means(labels, values)
        self.assertFalse(np.any(np.isnan(result)))


if __name__ == "__main__":
    unittest.main()
import unittest

import numpy as np

from sample_indices import sample_indices


class TestSampleIndices(unittest.TestCase):
    def test_reproducible(self):
        rs1 = np.random.RandomState(42)
        rs2 = np.random.RandomState(42)
        a = sample_indices(100, 10, rs1)
        b = sample_indices(100, 10, rs2)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        rs1 = np.random.RandomState(1)
        rs2 = np.random.RandomState(2)
        a = sample_indices(100, 10, rs1)
        b = sample_indices(100, 10, rs2)
        self.assertFalse(np.array_equal(a, b))

    def test_no_duplicates(self):
        rs = np.random.RandomState(0)
        idx = sample_indices(100, 50, rs)
        self.assertEqual(len(set(idx.tolist())), 50)

    def test_correct_length(self):
        rs = np.random.RandomState(0)
        idx = sample_indices(20, 5, rs)
        self.assertEqual(len(idx), 5)


if __name__ == "__main__":
    unittest.main()
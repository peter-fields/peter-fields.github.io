import unittest

from tree_depth import TreeNode, tree_depth


class TestTreeDepth(unittest.TestCase):
    def test_leaf(self):
        leaf = TreeNode(value=1, children=[])
        self.assertEqual(tree_depth(leaf), 1)

    def test_one_level(self):
        tree = TreeNode(
            value=1,
            children=[TreeNode(2, []), TreeNode(3, [])],
        )
        self.assertEqual(tree_depth(tree), 2)

    def test_deep_tree(self):
        deep = TreeNode(1, [
            TreeNode(2, [
                TreeNode(3, [
                    TreeNode(4, []),
                ]),
            ]),
        ])
        self.assertEqual(tree_depth(deep), 4)

    def test_unbalanced(self):
        tree = TreeNode(1, [
            TreeNode(2, []),
            TreeNode(3, [
                TreeNode(4, [
                    TreeNode(5, []),
                ]),
            ]),
        ])
        self.assertEqual(tree_depth(tree), 4)


if __name__ == "__main__":
    unittest.main()

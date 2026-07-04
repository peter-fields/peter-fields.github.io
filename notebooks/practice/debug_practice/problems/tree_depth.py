from typing import NamedTuple, List


class TreeNode(NamedTuple):
    value: int
    children: List["TreeNode"]


def tree_depth(node: TreeNode) -> int:
    """Compute the depth of a tree.

    A leaf node (no children) has depth 1.
    A tree with children has depth = 1 + max depth of its children.

    Examples:
        TreeNode(1, []) has depth 1
        TreeNode(1, [TreeNode(2, [])]) has depth 2
    """
    if not node.children:
        return 1
    return 1 + max(tree_depth(c) for c in node.children)

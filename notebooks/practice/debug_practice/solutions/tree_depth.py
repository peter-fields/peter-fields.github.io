from typing import NamedTuple, List


class TreeNode(NamedTuple):
    value: int
    children: List["TreeNode"]


def tree_depth(node: TreeNode) -> int:
    # FIX: base case (leaf) should return 1, not 0.
    if not node.children:
        return 1
    return 1 + max(tree_depth(c) for c in node.children)

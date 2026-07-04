# Debug Practice — Anthropic Fellows 60-min CodeSignal Prep

5 broken Python modules. Each has a docstring describing what the code is supposed to do, and a unittest suite that fails until the bug is fixed.

## Topics covered

| # | File | Topic |
|---|------|-------|
| 1 | `tree_depth.py` | Recursion + `NamedTuple` |
| 2 | `make_models.py` | `NamedTuple` + shared mutable state |
| 3 | `sample_indices.py` | `np.random.RandomState` vs global RNG |
| 4 | `weighted_class_means.py` | `np.bincount` + division by zero / `np.where` |
| 5 | `pairwise_min_distances.py` | Broadcasting + float equality |

## How to run

From this folder:

```bash
cd problems
python -m unittest discover -v          # run all tests
python -m unittest test_tree_depth -v   # run one test file
```

To run a single test method:
```bash
python -m unittest test_tree_depth.TestTreeDepth.test_leaf -v
```

## How to debug

Edit the source file (e.g. `tree_depth.py`) — NOT the test file. Re-run tests until they pass.

If you want to use pdb, add `import pdb; pdb.set_trace()` inside the source function and re-run. Or just sprinkle `print()` statements.

## Checking your answer

Don't look at `solutions/` until you've solved the problem (or are truly stuck). Each solution file is the fixed version of the corresponding `problems/` file.

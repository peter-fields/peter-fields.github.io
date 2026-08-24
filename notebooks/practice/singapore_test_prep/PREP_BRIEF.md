# Singapore AISF work test — 1-hour prep brief

**The test:** staged Python task, "implement an inventory management system," with a
**provided test suite**. Coderbyte, ~1h30m, proctored. Browser search allowed *for syntax
only*, lower-left panel. No external IDE. No AI. Three days to submit from the invite.

**Graded on:** "problem-solving approach, coding ability, **attention to detail**."
That last phrase is the tell — this is not a speed puzzle. They are looking at whether you
read the spec carefully, handled edge cases, and didn't break stage 1 while doing stage 3.

> **Integrity line (same as ARENA):** CC is for prep only. During the test — solo, no AI,
> no CC, no leaving the IDE. Keep this unimpeachable.

---

## The hour

| | |
|---|---|
| 0:00–0:10 | Read this brief (skim the cheat sheet, don't memorize it) |
| 0:10–0:50 | **Timed mock** — `mock/`, 40 min on a clock, solo |
| 0:50–1:00 | Bring CC the result; review mistakes, drill what broke |

---

## First 5 minutes of the real test — do this before writing any code

1. **Read every stage of the prompt**, not just stage 1. Stage 4 often tells you what data
   structure stage 1 should have used. A `list` of items that stage 3 needs to look up by
   SKU is a rewrite you can avoid by reading ahead.
2. **Open the test file and read all of it.** For every function, note: exact name, exact
   arguments, **exact return type** (list vs dict vs tuple vs None), and which exception it
   must raise. The tests are the spec; the prose is a summary of the spec.
3. **Find the run command and run the suite once, before writing anything.** You want the
   baseline failure list, and you want to discover a broken runner at minute 3, not minute 70.
4. Note whether it's `pytest` or `unittest` — affects nothing you write, but tells you how
   to read the failures.

## Working rhythm

- One stage at a time. Implement → run the **whole** suite → green → next stage.
- Re-running only the new stage is how you ship a regression. Run everything, every time.
- Stuck more than ~7 minutes on one behavior? Leave a `# TODO`, move on, come back.
  Partial credit across four stages beats a perfect stage 2.
- **Last 10 minutes:** re-run the full suite, re-read the prompt for non-test requirements
  (docstrings? a README? complexity notes? a specific submit button?), then submit.
  Confirm the submission actually registered.
- Write it plainly. Readable > clever. Someone is reading this.

---

## Traps this format is built to catch

**State integrity — the big one.** Validate *before* you mutate, so a failed operation
leaves the system exactly as it was. Tests love this:

```python
def remove_stock(self, sku, qty):
    item = self.get_item(sku)            # raises if missing
    if qty > item.quantity:              # check FIRST
        raise InsufficientStockError(...)
    item.quantity -= qty                 # then mutate
```

**Mutable default arguments.**
```python
def __init__(self, items=[]):     # BUG — shared across instances
def __init__(self, items=None):   # right
    self.items = list(items) if items else {}
```

**Returning internal collections by reference.** If a test mutates what you returned and
then checks your state, `return list(self._items)` / `return dict(...)`, not the object.

**Forgetting to return.** Methods that must return the new total / the created object.
A silent `None` fails one test and reads as carelessness.

**Mutating a dict while iterating it** → `RuntimeError`. Iterate over `list(d.items())`.

**Exceptions:** raise the *exact* class the test names. Give them a common base
(`class InventoryError(Exception)`) — tests often check `issubclass`. `ValueError` for bad
arguments, custom errors for domain rules; the tests will tell you which.

**Floats and money.** `0.1 + 0.2 != 0.3`. Check whether the test uses `assertAlmostEqual` /
`pytest.approx`. If it uses `==` on a total, `round(total, 2)`.

**Sorting and tie-breaks.** "Highest count, ties alphabetical" is a tuple key:
`sorted(rows, key=lambda r: (-r[1], r[0]))`. Don't reach for `reverse=True` when the
tie-break runs the other way.

**Julia→Python** (your standing traps): 0-indexed; `a[1:4]` is 3 elements; `//` is integer
division, `/` gives a float; `range(n)` is `0..n-1`.

---

## Cheat sheet — the shapes an inventory task actually needs

```python
from dataclasses import dataclass, field

@dataclass
class Item:
    sku: str
    name: str
    quantity: int = 0
    price: float = 0.0
    tags: list = field(default_factory=list)   # mutable default in a dataclass
```

```python
class InventoryError(Exception): pass
class ItemNotFoundError(InventoryError): pass
class InsufficientStockError(InventoryError): pass

raise InsufficientStockError(f"need {qty}, have {item.quantity}")
```

```python
# dicts
self._items = {}                      # sku -> Item
if sku not in self._items: raise ItemNotFoundError(sku)
self._items.get(sku)                  # None instead of KeyError
self._items.setdefault(cat, []).append(item)

from collections import defaultdict, Counter
by_cat = defaultdict(list); by_cat[cat].append(item)
counts = Counter(i.category for i in items)
counts.most_common(3)                 # [(cat, n), ...] already sorted desc
```

```python
# queries
[i for i in self._items.values() if i.quantity < threshold]
sorted(items, key=lambda i: (-i.quantity, i.name))
total = sum(i.quantity * i.price for i in self._items.values())
any(i.sku == sku for i in items) ; all(...)
next((i for i in items if i.sku == sku), None)
```

```python
# occasionally useful
from enum import Enum
class Status(Enum):
    IN_STOCK = "in_stock"

@property
def total_value(self):
    return self.quantity * self.price

def __repr__(self):
    return f"Item(sku={self.sku!r}, quantity={self.quantity})"

import json
json.dumps({sku: vars(i) for sku, i in self._items.items()}, indent=2)

from datetime import datetime, timedelta
datetime.now().isoformat()
```

```python
# assertion idioms you'll be READING
# pytest
assert inv.total() == 42
with pytest.raises(ItemNotFoundError):
    inv.get("nope")
# unittest
self.assertEqual(inv.total(), 42)
with self.assertRaises(ItemNotFoundError):
    inv.get("nope")
```

---

## Now: the mock

`mock/SPEC.md` — a library lending system. Same shapes as an inventory system (items,
counts, availability, add/remove, reports), different domain on purpose.

```
cd notebooks/practice/singapore_test_prep/mock
python -m unittest -v test_library
```

Set a 40-minute timer. Read `test_library.py` first. Work stage by stage.
`SOLUTION_do_not_open.py` is there for the review afterwards — not before.

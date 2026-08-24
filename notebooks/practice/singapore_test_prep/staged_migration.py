"""Extending a defaultdict inventory to sub-batches WITHOUT breaking stage 1.

The stage-1 tests at the bottom are run against the stage-2 class unchanged.

Run: python staged_migration.py
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import date

# "never expires" sentinel. date.max sorts last on its own, so nothing downstream
# needs a special case for undated stock. (None would blow up in sort()/<.)
NEVER = date.max


@dataclass
class Batch:
    quantity: int
    expires: date = NEVER


# ---------------------------------------------------------------- stage 1
class InventoryV1:
    def __init__(self):
        self._items = defaultdict(int)          # name -> quantity

    def add(self, name, quantity):
        self._items[name] += quantity
        return self._items[name]

    def quantity(self, name):
        return self._items.get(name, 0)


# ---------------------------------------------------------------- stage 2
class Inventory:
    def __init__(self):
        self._items = defaultdict(list)         # name -> [Batch, ...] sorted by expiry

    def add(self, name, quantity, expires=None):
        # new parameter is OPTIONAL and trailing -> every old two-arg call still works
        if quantity < 1:
            raise ValueError("quantity must be positive")
        expires = NEVER if expires is None else expires

        batches = self._items[name]             # WRITE path: auto-create is what we want
        for b in batches:
            if b.expires == expires:            # same date -> merge into that batch
                b.quantity += quantity
                break
        else:                                   # no match -> new batch, restore order
            batches.append(Batch(quantity, expires))
            batches.sort(key=lambda b: b.expires)

        return self.quantity(name)              # SAME return type as V1: an int

    def quantity(self, name):
        # unchanged contract: name -> single int. .get() so a query can't create a key.
        return sum(b.quantity for b in self._items.get(name, ()))

    # --- new surface, added alongside rather than replacing anything ---
    def batches(self, name):
        return [(b.quantity, None if b.expires == NEVER else b.expires)
                for b in self._items.get(name, ())]

    def remove(self, name, quantity):
        available = self.quantity(name)
        if quantity > available:
            raise ValueError(f"want {quantity} {name}, have {available}")
        remaining = quantity
        for b in self._items[name]:             # earliest expiry first, undated last
            take = min(b.quantity, remaining)
            b.quantity -= take
            remaining -= take
            if remaining == 0:
                break
        kept = [b for b in self._items[name] if b.quantity > 0]
        if kept:
            self._items[name] = kept
        else:
            del self._items[name]
        return self.quantity(name)


if __name__ == "__main__":
    # ---- the stage-1 tests, run verbatim against BOTH classes ----
    def stage_one_tests(cls):
        inv = cls()
        assert inv.quantity("rice") == 0
        assert inv.add("rice", 5) == 5
        assert inv.add("rice", 3) == 8          # same name accumulates
        assert inv.quantity("rice") == 8
        assert inv.quantity("beans") == 0
        return "pass"

    print("V1 :", stage_one_tests(InventoryV1))
    print("V2 :", stage_one_tests(Inventory))   # same assertions, no edits

    # ---- stage 2 behaviour, layered on top ----
    inv = Inventory()
    inv.add("milk", 3, date(2026, 8, 20))
    inv.add("milk", 2, date(2026, 8, 5))       # earlier date sorts to the front
    inv.add("milk", 1, date(2026, 8, 5))       # merges into the 8/5 batch
    inv.add("milk", 4)                          # undated -> NEVER -> sorts last

    assert inv.quantity("milk") == 10           # the V1 question still has a V1 answer
    assert inv.batches("milk") == [
        (3, date(2026, 8, 5)),
        (3, date(2026, 8, 20)),
        (4, None),                              # sentinel converted back on the way out
    ]

    inv.remove("milk", 4)                       # eats the 8/5 batch, dips into 8/20
    assert inv.batches("milk") == [(2, date(2026, 8, 20)), (4, None)]
    assert inv.quantity("milk") == 6

    print("stage 2 : pass")

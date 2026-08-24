"""Sub-batches with expiry dates — the pattern from the Singapore work test.

The move: a sub-batch doesn't change the SHAPE of your inventory, only the VALUE type.
    defaultdict(int)   name -> quantity
    defaultdict(list)  name -> [Batch, Batch, ...]  kept sorted by expiry

Run: python batches_pattern.py
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import date


class PantryError(Exception):
    pass


class ItemNotFoundError(PantryError):
    pass


class InsufficientStockError(PantryError):
    pass


@dataclass
class Batch:
    expires: date
    quantity: int


class Pantry:
    def __init__(self):
        # name -> list of Batch, always sorted earliest-expiry-first
        self._items = defaultdict(list)

    # ---------- reads ----------
    def quantity(self, name):
        # .get(), NOT self._items[name] -- see the phantom-key trap below
        return sum(b.quantity for b in self._items.get(name, ()))

    def batches(self, name):
        return list(self._items.get(name, ()))      # copy, so callers can't mutate us

    def names(self):
        return sorted(self._items)

    # ---------- writes ----------
    def add(self, name, quantity, expires):
        if quantity < 1:
            raise ValueError("quantity must be positive")

        batches = self._items[name]                 # defaultdict creates [] if new
        for b in batches:
            if b.expires == expires:                # same date -> merge, don't duplicate
                b.quantity += quantity
                return
        batches.append(Batch(expires, quantity))
        batches.sort(key=lambda b: b.expires)       # keep the FEFO invariant

    def remove(self, name, quantity):
        """Consume earliest-expiring first, spilling across batches."""
        if quantity < 1:
            raise ValueError("quantity must be positive")
        if name not in self._items:
            raise ItemNotFoundError(name)

        available = self.quantity(name)
        if available < quantity:                    # CHECK before you mutate, so a
            raise InsufficientStockError(           # failed removal changes nothing
                f"want {quantity} {name}, have {available}"
            )

        remaining = quantity
        for b in self._items[name]:                 # already sorted by expiry
            take = min(b.quantity, remaining)
            b.quantity -= take
            remaining -= take
            if remaining == 0:
                break
        self._prune(name)

    def discard_expired(self, today):
        """Drop every batch that expired before `today`. Returns units binned."""
        binned = 0
        for name in list(self._items):              # list() -> safe to delete inside
            keep = []
            for b in self._items[name]:
                if b.expires < today:
                    binned += b.quantity
                else:
                    keep.append(b)
            self._items[name] = keep
            self._prune(name)
        return binned

    # ---------- housekeeping ----------
    def _prune(self, name):
        """Drop emptied batches, and the key itself once nothing is left."""
        remaining = [b for b in self._items[name] if b.quantity > 0]
        if remaining:
            self._items[name] = remaining
        else:
            del self._items[name]


if __name__ == "__main__":
    d = date  # brevity below

    p = Pantry()
    p.add("milk", 2, d(2026, 8, 5))
    p.add("milk", 3, d(2026, 8, 20))
    p.add("milk", 1, d(2026, 8, 5))      # merges into the existing 8/5 batch
    p.add("bread", 1, d(2026, 8, 4))

    assert p.quantity("milk") == 6
    assert [(b.expires.day, b.quantity) for b in p.batches("milk")] == [(5, 3), (20, 3)]

    # removal spills out of the earliest batch into the next
    p.remove("milk", 4)
    assert [(b.expires.day, b.quantity) for b in p.batches("milk")] == [(20, 2)]
    assert p.quantity("milk") == 2

    # a failed removal leaves everything untouched
    try:
        p.remove("milk", 99)
    except InsufficientStockError:
        pass
    assert p.quantity("milk") == 2

    # emptying an item removes the key entirely
    p.remove("milk", 2)
    assert p.quantity("milk") == 0
    assert p.names() == ["bread"]

    # expiry sweep
    p.add("eggs", 6, d(2026, 8, 1))
    assert p.discard_expired(d(2026, 8, 4)) == 6      # eggs binned, bread survives
    assert p.names() == ["bread"]

    print("all good")

    # ---------------------------------------------------------------
    # THE TRAP: reading a missing key from a defaultdict CREATES it.
    dd = defaultdict(list)
    print(len(dd))          # 0
    dd["ghost"]             # a bare read...
    print(len(dd), dict(dd))  # 1 {'ghost': []}  <-- you now have a phantom item

    # so for reads use .get() (or `in`), and reserve d[key] for writes
    print(dd.get("ghost2", []), len(dd))   # [] 1 -- no new key

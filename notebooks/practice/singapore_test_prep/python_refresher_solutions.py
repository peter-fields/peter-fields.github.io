"""Drill solutions for python_refresher.ipynb. Read after you've had a real go.

Run `python python_refresher_solutions.py` to confirm they pass.
"""

from collections import Counter, defaultdict


# 1 ----------------------------------------------------------------
def word_lengths(words):
    return {w: len(w) for w in words}


# 2 ----------------------------------------------------------------
def top_n(values, n):
    counts = Counter(values)
    # -count sorts descending, value ascending breaks the tie
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ordered[:n]


# 3 ----------------------------------------------------------------
def group_by_initial(names):
    groups = defaultdict(list)
    for name in names:
        groups[name[0].lower()].append(name)
    return dict(groups)          # plain dict, so == against a dict literal works


# 4 ----------------------------------------------------------------
def parse_stock(lines):
    stock = {}
    for line in lines:
        parts = line.split(":")
        if len(parts) != 2:
            continue
        sku, raw = parts
        try:
            qty = int(raw)
        except ValueError:
            continue
        stock[sku] = stock.get(sku, 0) + qty
    return stock


# 5 ----------------------------------------------------------------
class InsufficientFunds(Exception):
    pass


class Account:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        return self.balance

    def withdraw(self, amount):
        if amount > self.balance:            # check before mutating
            raise InsufficientFunds(f"need {amount}, have {self.balance}")
        self.balance -= amount
        return self.balance

    def __repr__(self):
        return f"Account({self.owner!r}, balance={self.balance})"


# 6 ----------------------------------------------------------------
class Catalogue:
    def __init__(self):
        self._names = {}         # sku -> name
        self._qty = {}           # sku -> quantity

    def add(self, sku, name, qty):
        if sku not in self._qty:
            self._names[sku] = name
            self._qty[sku] = 0
        self._qty[sku] += qty
        return self._qty[sku]

    def remove(self, sku, qty):
        if sku not in self._qty:
            raise KeyError(sku)
        if qty > self._qty[sku]:
            raise ValueError(f"want {qty}, have {self._qty[sku]}")
        self._qty[sku] -= qty
        return self._qty[sku]

    def low_stock(self, threshold):
        return sorted(sku for sku, q in self._qty.items() if q < threshold)

    def total(self):
        return sum(self._qty.values())


if __name__ == "__main__":
    assert word_lengths(["ab", "c"]) == {"ab": 2, "c": 1}
    assert word_lengths([]) == {}

    assert top_n(["a", "b", "a", "c", "b", "a"], 2) == [("a", 3), ("b", 2)]
    assert top_n(["b", "a"], 5) == [("a", 1), ("b", 1)]
    assert top_n([], 3) == []

    assert group_by_initial(["Ann", "bob", "amy"]) == {"a": ["Ann", "amy"], "b": ["bob"]}
    assert group_by_initial([]) == {}

    assert parse_stock(["a:2", "b:3", "a:5"]) == {"a": 7, "b": 3}
    assert parse_stock(["a:2", "junk", "b:x"]) == {"a": 2}
    assert parse_stock([]) == {}

    acct = Account("alice", 100)
    assert acct.balance == 100
    assert acct.deposit(50) == 150
    try:
        acct.withdraw(999)
    except InsufficientFunds:
        pass
    else:
        raise AssertionError("should have raised")
    assert acct.balance == 150
    assert acct.withdraw(50) == 100
    assert repr(acct) == "Account('alice', balance=100)"

    c = Catalogue()
    c.add("b", "bolt", 5)
    c.add("a", "axle", 2)
    c.add("b", "bolt", 3)
    assert c.total() == 10
    assert c.low_stock(5) == ["a"]
    c.remove("b", 8)
    assert c.low_stock(5) == ["a", "b"]
    assert c.total() == 2
    for bad, err in [(("zz", 1), KeyError), (("a", 99), ValueError)]:
        try:
            c.remove(*bad)
        except err:
            pass
        else:
            raise AssertionError(f"expected {err.__name__}")
    assert c.total() == 2

    print("all drills pass")

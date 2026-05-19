# Python Cheat Sheet — Anthropic Fellows Assessment

## Lists
```python
lst = [1, 2, 3]
lst.append(x)          # add to end
lst.remove(x)          # remove first occurrence
lst.pop()              # remove and return last
lst.pop(i)             # remove and return at index i
lst.insert(i, x)       # insert at index i
sorted(lst)            # new sorted list
lst.sort()             # sort in place
len(lst)               # length
x in lst               # membership check

# concatenation / repetition
[1,2] + [3,4]          # [1, 2, 3, 4]
[0] * 3                # [0, 0, 0]

# slicing
lst[1:3]               # index 1 up to (not including) 3
lst[:2]                # from start
lst[2:]                # to end
lst[-1]                # last element
lst[-2:]               # last two
lst[:-2]               # everything except last two

# comprehensions
[x*2 for x in lst]
[x for x in lst if x > 2]
```

## Dicts
```python
d = {}
d = {"a": 1, "b": 2}
d = dict(a=1, b=2)                 # keyword arguments
d = dict([("a", 1), ("b", 2)])     # list of tuples
d = dict(zip(["a","b"], [1,2]))    # from two lists
d = dict.fromkeys(["a","b"], 0)    # {"a":0, "b":0}
d = {x: x**2 for x in range(5)}   # comprehension

d[key] = value
d[key]                  # KeyError if missing
d.get(key, default)     # safe lookup, returns default if missing
d.keys()                # dict_keys
d.values()              # dict_values
d.items()               # dict_items — use for key,val loops
key in d                # membership check
del d[key]              # delete

# merging
{**d1, **d2}            # d2 overwrites d1 on conflict
d1.update(d2)           # merges d2 into d1 in place

len(d)                  # number of key-value pairs
```

## defaultdict
```python
from collections import defaultdict

d = defaultdict(list)   # missing keys return []
d = defaultdict(int)    # missing keys return 0
d = defaultdict(dict)   # missing keys return {}
```

## Classes
```python
class MyClass:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def my_method(self):
        return self.x + self.y

obj = MyClass(1, 2)
obj.my_method()         # 3
obj.x                   # 1
```

## Functions
```python
def foo(x, y=10):           # default argument
    return x + y

def foo(*args):             # variable args — tuple inside
    return sum(args)

def foo(**kwargs):          # keyword args — dict inside
    return kwargs

def foo(x, y=10, *args, **kwargs):  # all together
    pass
```

## Strings
```python
"foo/bar/baz".split("/")    # ["foo", "bar", "baz"]
"/".join(["foo", "bar"])    # "foo/bar"

s.strip("/")                # remove from both ends
s.lstrip("/")               # remove from left only
s.rstrip("/")               # remove from right only

7 // 2                      # 3 — integer division (floor)
(n + 1) // 2                # ceiling division
```

## File I/O
```python
with open("file.txt", "r") as f:
    content = f.read()      # whole file as string
    lines = f.readlines()   # list of lines
    line = f.readline()     # one line

with open("file.txt", "w") as f:   # overwrites
    f.write("hello")

with open("file.txt", "a") as f:   # appends
    f.write("more")
```

## Threading
```python
import threading

# Thread
t = threading.Thread(target=my_func, args=(1,2), kwargs={"x":3})
t.start()
t.join()                    # wait for thread to finish

# Lock — protect shared state
lock = threading.Lock()
with lock:                  # only one thread at a time
    shared_var += 1

# Semaphore — ticket counter
sem = threading.Semaphore(0)   # starts blocked
sem.release()               # add ticket
sem.acquire()               # take ticket, block if 0

# Event — simple flag
event = threading.Event()   # starts False
event.set()                 # flag = True, unblocks all waiters
event.wait()                # block until True
event.clear()               # flag = False
event.is_set()              # check current state

# Barrier — wait for N threads
barrier = threading.Barrier(3)
barrier.wait()              # block until 3 threads are waiting, then all release

# Condition
cond = threading.Condition()
with cond:
    cond.wait()             # release lock and block
    cond.notify()           # wake one waiter
    cond.notify_all()       # wake all waiters

# Timer
t = threading.Timer(3.0, my_func)
t.start()
t.cancel()

# daemon thread — dies when main program exits
t = threading.Thread(target=my_func, daemon=True)
```

## Common Patterns

### Thread-safe counter
```python
class Counter:
    def __init__(self):
        self.count = 0
        self.lock = threading.Lock()
    def increment(self):
        with self.lock:
            self.count += 1
```

### Ordered execution (A before B)
```python
event = threading.Event()
def A():
    do_thing()
    event.set()
def B():
    event.wait()
    do_thing()
```

### Ping-pong between threads
```python
sem1 = threading.Semaphore(1)  # A goes first
sem2 = threading.Semaphore(0)
def A():
    sem1.acquire()
    do_thing()
    sem2.release()
def B():
    sem2.acquire()
    do_thing()
    sem1.release()
```

### Simulated file system
```python
class FileSystem:
    def __init__(self):
        self.root = {}
    def createPath(self, path, value):
        parts = path.strip("/").split("/")
        node = self.root
        for p in parts[:-1]:
            if p not in node:
                return False
            node = node[p]
        last = parts[-1]
        if last in node:
            return False
        node[last] = {"__value__": value}
        return True
    def get(self, path):
        parts = path.strip("/").split("/")
        node = self.root
        for p in parts:
            if p not in node:
                return -1
            node = node[p]
        return node.get("__value__", -1)
```

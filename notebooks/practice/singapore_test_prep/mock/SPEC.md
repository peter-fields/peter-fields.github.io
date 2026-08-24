# Mock staged task — Library Lending System

**40 minutes. Solo. Same rules as the real thing.**

Deliberately *not* an inventory system — same shapes (items, counts, add/remove,
availability, reports), different domain, so nothing you write here can be reused
verbatim tomorrow. The transferable thing is the rhythm, not the code.

Implement `Book` and `Library` in `library.py` so the suite in `test_library.py`
passes. **The tests are the spec** — where this file and the tests disagree, the
tests win. Read the whole test file before writing a line.

Run:

```
python -m unittest -v test_library                    # everything
python -m unittest -v test_library.Stage2Loans        # one stage
```

---

## Stage 1 — Catalog

- `Book` with fields `isbn`, `title`, `author`, `total_copies`.
- `Library.add_book(isbn, title, author, total_copies=1)` → returns the `Book`.
- `Library.get_book(isbn)` → the `Book`, or `BookNotFoundError`.
- Duplicate ISBN → `DuplicateBookError`.
- `total_copies` must be a positive `int` → otherwise `ValueError`.
- A rejected `add_book` must leave the catalog untouched.

## Stage 2 — Loans

- `available_copies(isbn)` → total minus copies currently out.
- `checkout(isbn, member_id)` / `return_book(isbn, member_id)`.
- No copies left → `NoCopiesAvailableError`.
- Same member checking out the same book twice → `AlreadyCheckedOutError`.
- Returning a book the member doesn't hold → `NotCheckedOutError`.
- A failed operation must not change availability.
- Unknown ISBN → `BookNotFoundError` from all three.

## Stage 3 — Queries

- `books_by_author(author)` → list of `Book`, sorted by title. Author match is
  case-insensitive and ignores surrounding whitespace. Unknown author → `[]`.
- `member_loans(member_id)` → sorted list of ISBNs currently held.
- `most_borrowed(n)` → list of `(title, borrow_count)`, highest count first,
  ties broken alphabetically by title, at most `n` entries. Counts are
  **cumulative** — a returned book still counts. Never-borrowed books are excluded.

## Stage 4 — Maintenance

- `add_copies(isbn, n)` → new total. `n` must be a positive `int`.
- `remove_book(isbn)` → drops it from the catalog. Blocked with a `LibraryError`
  while any copy is on loan.
- After removal the ISBN is free to reuse, with clean state.

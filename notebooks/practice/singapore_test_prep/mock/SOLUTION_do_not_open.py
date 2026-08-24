"""Reference solution for the mock staged task. DO NOT OPEN until your timed run is over."""

from dataclasses import dataclass, field


class LibraryError(Exception):
    pass


class BookNotFoundError(LibraryError):
    pass


class DuplicateBookError(LibraryError):
    pass


class NoCopiesAvailableError(LibraryError):
    pass


class AlreadyCheckedOutError(LibraryError):
    pass


class NotCheckedOutError(LibraryError):
    pass


@dataclass
class Book:
    isbn: str
    title: str
    author: str
    total_copies: int


class Library:
    def __init__(self):
        self._books = {}                 # isbn -> Book
        self._loans = {}                 # isbn -> set of member_id
        self._borrow_counts = {}         # isbn -> cumulative checkouts

    # ---------- stage 1 ----------
    def add_book(self, isbn, title, author, total_copies=1):
        if isbn in self._books:
            raise DuplicateBookError(f"book already in catalog: {isbn}")
        if not isinstance(total_copies, int) or isinstance(total_copies, bool):
            raise ValueError("total_copies must be an int")
        if total_copies < 1:
            raise ValueError("total_copies must be at least 1")
        book = Book(isbn=isbn, title=title, author=author, total_copies=total_copies)
        self._books[isbn] = book
        self._loans[isbn] = set()
        self._borrow_counts[isbn] = 0
        return book

    def get_book(self, isbn):
        if isbn not in self._books:
            raise BookNotFoundError(f"no such book: {isbn}")
        return self._books[isbn]

    # ---------- stage 2 ----------
    def available_copies(self, isbn):
        book = self.get_book(isbn)
        return book.total_copies - len(self._loans[isbn])

    def checkout(self, isbn, member_id):
        self.get_book(isbn)
        if member_id in self._loans[isbn]:
            raise AlreadyCheckedOutError(f"{member_id} already has {isbn}")
        if self.available_copies(isbn) <= 0:
            raise NoCopiesAvailableError(f"no copies left: {isbn}")
        self._loans[isbn].add(member_id)
        self._borrow_counts[isbn] += 1

    def return_book(self, isbn, member_id):
        self.get_book(isbn)
        if member_id not in self._loans[isbn]:
            raise NotCheckedOutError(f"{member_id} does not have {isbn}")
        self._loans[isbn].remove(member_id)

    # ---------- stage 3 ----------
    def books_by_author(self, author):
        key = author.strip().lower()
        matches = [b for b in self._books.values() if b.author.strip().lower() == key]
        return sorted(matches, key=lambda b: b.title)

    def member_loans(self, member_id):
        return sorted(isbn for isbn, members in self._loans.items() if member_id in members)

    def most_borrowed(self, n):
        if n < 0:
            raise ValueError("n must be non-negative")
        rows = [
            (self._books[isbn].title, count)
            for isbn, count in self._borrow_counts.items()
            if count > 0
        ]
        rows.sort(key=lambda row: (-row[1], row[0]))
        return rows[:n]

    # ---------- stage 4 ----------
    def add_copies(self, isbn, n):
        book = self.get_book(isbn)
        if not isinstance(n, int) or isinstance(n, bool):
            raise ValueError("n must be an int")
        if n < 1:
            raise ValueError("n must be at least 1")
        book.total_copies += n
        return book.total_copies

    def remove_book(self, isbn):
        self.get_book(isbn)
        if self._loans[isbn]:
            raise LibraryError(f"cannot remove {isbn}: copies are on loan")
        del self._books[isbn]
        del self._loans[isbn]
        del self._borrow_counts[isbn]

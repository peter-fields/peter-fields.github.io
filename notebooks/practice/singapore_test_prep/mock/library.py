"""Mock staged task — YOUR working file.

The test suite in test_library.py is the spec. Read it first, then implement.

Run:  python -m unittest -v test_library
      python -m unittest -v test_library.Stage1Catalog     # one stage at a time

The exception classes below are given. Everything else (Book, Library) is yours.
Until `Book` and `Library` exist the suite won't even import — that's your first
signal, not a broken test file.
"""


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


# TODO: Book
# TODO: Library

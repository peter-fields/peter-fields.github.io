"""Mock staged work-test suite. Run with:  python -m unittest -v test_library

Work stage by stage. After each stage, re-run the WHOLE file — later stages must
not break earlier ones.
"""

import unittest

from library import (
    AlreadyCheckedOutError,
    Book,
    BookNotFoundError,
    DuplicateBookError,
    Library,
    LibraryError,
    NoCopiesAvailableError,
    NotCheckedOutError,
)


class Stage1Catalog(unittest.TestCase):
    def setUp(self):
        self.lib = Library()

    def test_add_book_returns_book_with_fields(self):
        book = self.lib.add_book("111", "Dune", "Frank Herbert", 3)
        self.assertIsInstance(book, Book)
        self.assertEqual(book.isbn, "111")
        self.assertEqual(book.title, "Dune")
        self.assertEqual(book.author, "Frank Herbert")
        self.assertEqual(book.total_copies, 3)

    def test_total_copies_defaults_to_one(self):
        book = self.lib.add_book("111", "Dune", "Frank Herbert")
        self.assertEqual(book.total_copies, 1)

    def test_get_book_returns_same_object(self):
        book = self.lib.add_book("111", "Dune", "Frank Herbert", 2)
        self.assertIs(self.lib.get_book("111"), book)

    def test_duplicate_isbn_rejected(self):
        self.lib.add_book("111", "Dune", "Frank Herbert", 2)
        with self.assertRaises(DuplicateBookError):
            self.lib.add_book("111", "Dune Messiah", "Frank Herbert", 1)

    def test_unknown_isbn_raises(self):
        with self.assertRaises(BookNotFoundError):
            self.lib.get_book("nope")

    def test_total_copies_must_be_positive_int(self):
        for bad in (0, -1, "3", 2.5):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.lib.add_book(f"isbn-{bad}", "T", "A", bad)

    def test_failed_add_does_not_mutate_catalog(self):
        self.lib.add_book("111", "Dune", "Frank Herbert", 2)
        with self.assertRaises(DuplicateBookError):
            self.lib.add_book("111", "Other", "Someone", 5)
        self.assertEqual(self.lib.get_book("111").title, "Dune")

    def test_errors_share_a_base_class(self):
        for exc in (BookNotFoundError, DuplicateBookError, NoCopiesAvailableError,
                    AlreadyCheckedOutError, NotCheckedOutError):
            with self.subTest(exc=exc.__name__):
                self.assertTrue(issubclass(exc, LibraryError))


class Stage2Loans(unittest.TestCase):
    def setUp(self):
        self.lib = Library()
        self.lib.add_book("111", "Dune", "Frank Herbert", 2)
        self.lib.add_book("222", "Emma", "Jane Austen", 1)

    def test_available_starts_at_total(self):
        self.assertEqual(self.lib.available_copies("111"), 2)

    def test_checkout_decrements_availability(self):
        self.lib.checkout("111", "alice")
        self.assertEqual(self.lib.available_copies("111"), 1)

    def test_return_restores_availability(self):
        self.lib.checkout("111", "alice")
        self.lib.return_book("111", "alice")
        self.assertEqual(self.lib.available_copies("111"), 2)

    def test_out_of_copies(self):
        self.lib.checkout("222", "alice")
        with self.assertRaises(NoCopiesAvailableError):
            self.lib.checkout("222", "bob")

    def test_same_member_cannot_hold_two_copies(self):
        with self.assertRaises(AlreadyCheckedOutError):
            self.lib.checkout("111", "alice")
            self.lib.checkout("111", "alice")

    def test_failed_checkout_does_not_consume_a_copy(self):
        self.lib.checkout("111", "alice")
        with self.assertRaises(AlreadyCheckedOutError):
            self.lib.checkout("111", "alice")
        self.assertEqual(self.lib.available_copies("111"), 1)

    def test_return_without_loan_raises(self):
        with self.assertRaises(NotCheckedOutError):
            self.lib.return_book("111", "alice")

    def test_can_borrow_again_after_returning(self):
        self.lib.checkout("111", "alice")
        self.lib.return_book("111", "alice")
        self.lib.checkout("111", "alice")
        self.assertEqual(self.lib.available_copies("111"), 1)

    def test_loan_operations_on_unknown_isbn_raise(self):
        for call in (lambda: self.lib.available_copies("nope"),
                     lambda: self.lib.checkout("nope", "alice"),
                     lambda: self.lib.return_book("nope", "alice")):
            with self.subTest(call=call):
                with self.assertRaises(BookNotFoundError):
                    call()


class Stage3Queries(unittest.TestCase):
    def setUp(self):
        self.lib = Library()
        self.lib.add_book("111", "Dune", "Frank Herbert", 2)
        self.lib.add_book("112", "Children of Dune", "frank herbert ", 1)
        self.lib.add_book("222", "Emma", "Jane Austen", 1)

    def test_books_by_author_is_case_and_whitespace_insensitive(self):
        found = self.lib.books_by_author("  FRANK HERBERT")
        self.assertEqual([b.isbn for b in found], ["112", "111"])

    def test_books_by_author_sorted_by_title(self):
        titles = [b.title for b in self.lib.books_by_author("Frank Herbert")]
        self.assertEqual(titles, ["Children of Dune", "Dune"])

    def test_unknown_author_returns_empty_list(self):
        self.assertEqual(self.lib.books_by_author("Nobody"), [])

    def test_member_loans_sorted_and_empty_by_default(self):
        self.assertEqual(self.lib.member_loans("alice"), [])
        self.lib.checkout("222", "alice")
        self.lib.checkout("111", "alice")
        self.assertEqual(self.lib.member_loans("alice"), ["111", "222"])

    def test_member_loans_excludes_returned_books(self):
        self.lib.checkout("111", "alice")
        self.lib.return_book("111", "alice")
        self.assertEqual(self.lib.member_loans("alice"), [])

    def test_most_borrowed_counts_are_cumulative(self):
        # a returned book still counts as having been borrowed
        self.lib.checkout("111", "alice")
        self.lib.return_book("111", "alice")
        self.lib.checkout("111", "bob")
        self.lib.checkout("222", "alice")
        self.assertEqual(self.lib.most_borrowed(2), [("Dune", 2), ("Emma", 1)])

    def test_most_borrowed_ties_break_alphabetically_by_title(self):
        self.lib.checkout("222", "alice")   # Emma
        self.lib.checkout("111", "alice")   # Dune
        self.assertEqual(self.lib.most_borrowed(2), [("Dune", 1), ("Emma", 1)])

    def test_most_borrowed_ignores_never_borrowed_books(self):
        self.lib.checkout("111", "alice")
        self.assertEqual(self.lib.most_borrowed(10), [("Dune", 1)])

    def test_most_borrowed_zero(self):
        self.lib.checkout("111", "alice")
        self.assertEqual(self.lib.most_borrowed(0), [])


class Stage4Maintenance(unittest.TestCase):
    def setUp(self):
        self.lib = Library()
        self.lib.add_book("111", "Dune", "Frank Herbert", 1)

    def test_add_copies_returns_new_total(self):
        self.assertEqual(self.lib.add_copies("111", 3), 4)
        self.assertEqual(self.lib.get_book("111").total_copies, 4)

    def test_add_copies_increases_availability(self):
        self.lib.checkout("111", "alice")
        self.assertEqual(self.lib.available_copies("111"), 0)
        self.lib.add_copies("111", 2)
        self.assertEqual(self.lib.available_copies("111"), 2)

    def test_add_copies_validates(self):
        for bad in (0, -2, "1"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.lib.add_copies("111", bad)
        with self.assertRaises(BookNotFoundError):
            self.lib.add_copies("nope", 1)

    def test_remove_book(self):
        self.lib.remove_book("111")
        with self.assertRaises(BookNotFoundError):
            self.lib.get_book("111")

    def test_remove_book_blocked_while_on_loan(self):
        self.lib.checkout("111", "alice")
        with self.assertRaises(LibraryError):
            self.lib.remove_book("111")
        self.assertEqual(self.lib.get_book("111").title, "Dune")

    def test_remove_after_return_succeeds(self):
        self.lib.checkout("111", "alice")
        self.lib.return_book("111", "alice")
        self.lib.remove_book("111")
        with self.assertRaises(BookNotFoundError):
            self.lib.get_book("111")

    def test_isbn_can_be_reused_after_removal(self):
        self.lib.remove_book("111")
        book = self.lib.add_book("111", "Dune", "Frank Herbert", 2)
        self.assertEqual(book.total_copies, 2)
        self.assertEqual(self.lib.available_copies("111"), 2)


class CrossStageScenario(unittest.TestCase):
    def test_full_workflow(self):
        lib = Library()
        lib.add_book("a", "Anna Karenina", "Leo Tolstoy", 2)
        lib.add_book("b", "War and Peace", "leo tolstoy", 1)
        lib.add_book("c", "Emma", "Jane Austen", 1)

        lib.checkout("a", "alice")
        lib.checkout("a", "bob")
        with self.assertRaises(NoCopiesAvailableError):
            lib.checkout("a", "carol")

        lib.return_book("a", "bob")
        lib.checkout("a", "carol")
        lib.checkout("b", "alice")

        self.assertEqual(lib.member_loans("alice"), ["a", "b"])
        self.assertEqual(lib.available_copies("a"), 0)
        self.assertEqual([b.title for b in lib.books_by_author("LEO TOLSTOY")],
                         ["Anna Karenina", "War and Peace"])
        self.assertEqual(lib.most_borrowed(3), [("Anna Karenina", 3), ("War and Peace", 1)])

        with self.assertRaises(LibraryError):
            lib.remove_book("a")
        lib.remove_book("c")
        self.assertEqual(lib.books_by_author("Jane Austen"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

import datetime as dt
import decimal
import unittest

from . import conftest_path  # noqa: F401
from siriusfuzz import sqltypes as st
from siriusfuzz.compare import (
    ColumnInfo,
    ResultSet,
    Tolerances,
    cells_equal,
    compare_results,
    is_total_order,
)
from siriusfuzz.sqlast import Alias, ColumnRef, OrderItem, Select, SelectItem


def rs(rows, types):
    cols = [ColumnInfo(f"c{i}", t) for i, t in enumerate(types)]
    return ResultSet(cols, rows)


class CellsEqual(unittest.TestCase):
    def test_exact_and_null(self):
        self.assertTrue(cells_equal(None, None, 0, 0))
        self.assertFalse(cells_equal(None, 0, 1e-9, 1e-12))
        self.assertTrue(cells_equal("a", "a", 0, 0))
        self.assertFalse(cells_equal("a", "A", 0, 0))

    def test_float_tolerance(self):
        self.assertTrue(cells_equal(1.0, 1.0 + 1e-12, 1e-9, 1e-12))
        self.assertFalse(cells_equal(1.0, 1.001, 1e-9, 1e-12))
        self.assertFalse(cells_equal(1.0, 1.001, 0.0, 0.0))
        self.assertTrue(cells_equal(0.0, 1e-13, 1e-9, 1e-12))

    def test_non_finite_only_exact(self):
        nan = float("nan")
        self.assertTrue(cells_equal(nan, nan, 1e-9, 1e-12))
        self.assertFalse(cells_equal(float("inf"), 1e308, 1e-9, 1e-12))
        self.assertTrue(cells_equal(float("inf"), float("inf"), 0, 0))

    def test_decimal_exact(self):
        self.assertTrue(
            cells_equal(decimal.Decimal("1.50"), decimal.Decimal("1.5"), 0, 0)
        )
        self.assertFalse(
            cells_equal(decimal.Decimal("1.50"), decimal.Decimal("1.51"), 0, 0)
        )

    def test_int_vs_decimal_equal_values(self):
        self.assertTrue(cells_equal(3, decimal.Decimal("3"), 0, 0))


class CompareResults(unittest.TestCase):
    tol = Tolerances(1e-4, 1e-9, 1e-12)

    def test_multiset_ignores_order(self):
        a = rs([(1, "x"), (2, "y")], [st.INTEGER, st.VARCHAR])
        b = rs([(2, "y"), (1, "x")], [st.INTEGER, st.VARCHAR])
        self.assertTrue(compare_results(a, b, ordered=False, tol=self.tol).equal)
        self.assertFalse(compare_results(a, b, ordered=True, tol=self.tol).equal)

    def test_row_count_mismatch(self):
        a = rs([(1,)], [st.INTEGER])
        b = rs([(1,), (1,)], [st.INTEGER])
        out = compare_results(a, b, False, self.tol)
        self.assertFalse(out.equal)
        self.assertIn("row count", out.detail)

    def test_float_column_tolerant_others_exact(self):
        a = rs([(1, 10.0), (2, 20.0)], [st.INTEGER, st.DOUBLE])
        b = rs([(2, 20.0 + 1e-12), (1, 10.0)], [st.INTEGER, st.DOUBLE])
        self.assertTrue(compare_results(a, b, False, self.tol).equal)
        c = rs([(2, 20.0), (1, 10.0 + 1e-3)], [st.INTEGER, st.DOUBLE])
        self.assertFalse(compare_results(a, c, False, self.tol).equal)
        d = rs([(3, 10.0), (2, 20.0)], [st.INTEGER, st.DOUBLE])
        self.assertFalse(compare_results(a, d, False, self.tol).equal)

    def test_float32_wider_tolerance(self):
        a = rs([(1.0,)], [st.FLOAT])
        b = rs([(1.0 + 5e-5,)], [st.FLOAT])
        self.assertTrue(compare_results(a, b, False, self.tol).equal)
        a64 = rs([(1.0,)], [st.DOUBLE])
        b64 = rs([(1.0 + 5e-5,)], [st.DOUBLE])
        self.assertFalse(compare_results(a64, b64, False, self.tol).equal)

    def test_nulls_and_dates_sort(self):
        a = rs([(None, dt.date(2020, 1, 1)), (1, None)], [st.INTEGER, st.DATE])
        b = rs([(1, None), (None, dt.date(2020, 1, 1))], [st.INTEGER, st.DATE])
        self.assertTrue(compare_results(a, b, False, self.tol).equal)

    def test_diffs_reported(self):
        a = rs([(1, "x")], [st.INTEGER, st.VARCHAR])
        b = rs([(1, "y")], [st.INTEGER, st.VARCHAR])
        out = compare_results(a, b, False, self.tol)
        self.assertFalse(out.equal)
        self.assertEqual(len(out.diffs), 1)
        self.assertIn("'x'", out.diffs[0])


class TotalOrder(unittest.TestCase):
    def test_total_order_detection(self):
        col = ColumnRef("a0", "k", st.INTEGER)
        sel = Select(items=[SelectItem(col, "c0"), SelectItem(col, "c1")], from_=None)
        self.assertFalse(is_total_order(sel))
        sel.order_by = [OrderItem(Alias("c0"))]
        self.assertFalse(is_total_order(sel))
        sel.order_by.append(OrderItem(Alias("c1"), desc=True))
        self.assertTrue(is_total_order(sel))


if __name__ == "__main__":
    unittest.main()

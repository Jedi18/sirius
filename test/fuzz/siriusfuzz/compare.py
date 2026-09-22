# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""Result comparison.

Rows compare as a multiset unless the query's ORDER BY is a total order over
the projected columns. Cells compare exactly except FLOAT/DOUBLE, which get a
relative tolerance with an absolute floor; NaN/inf match only exactly; NULL
matches only NULL; DECIMAL compares as exact decimals.

Rows on both sides are sorted *including* float columns before the positional
tolerant compare. If every cell is within ``tol`` of its true value, the two
sorted sequences are pairwise within ``2 * tol``, so this never misaligns
rows the way sorting on approximate keys otherwise could.
"""

from __future__ import annotations

import datetime as dt
import decimal
import math
from dataclasses import dataclass, field
from typing import Any

from . import sqltypes as st
from .sqlast import Alias, Query, Select, SetOp


@dataclass
class ColumnInfo:
    name: str
    type: st.SqlType


@dataclass
class ResultSet:
    columns: list[ColumnInfo]
    rows: list[tuple[Any, ...]]

    @property
    def row_count(self) -> int:
        return len(self.rows)


@dataclass
class Tolerances:
    float32_rel: float = 1e-4
    float64_rel: float = 1e-9
    abs_tol: float = 1e-12


@dataclass
class CompareOutcome:
    equal: bool
    detail: str = ""
    diffs: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# canonical cells
# --------------------------------------------------------------------------

(
    _T_NULL,
    _T_NUM,
    _T_NAN,
    _T_STR,
    _T_DATE,
    _T_DATETIME,
    _T_BYTES,
    _T_SEQ,
    _T_MAP,
    _T_OTHER,
) = range(10)


def canonical(value: Any) -> tuple:
    """Orderable, hashable key for a cell; equal cells map to equal keys."""
    if value is None:
        return (_T_NULL,)
    if isinstance(value, bool):
        return (_T_NUM, int(value))
    if isinstance(value, int):
        return (_T_NUM, value)
    if isinstance(value, float):
        if math.isnan(value):
            return (_T_NAN,)
        return (_T_NUM, value)
    if isinstance(value, decimal.Decimal):
        if value.is_nan():
            return (_T_NAN,)
        return (_T_NUM, value)
    if isinstance(value, str):
        return (_T_STR, value)
    if isinstance(value, dt.datetime):
        return (_T_DATETIME, value.isoformat())
    if isinstance(value, dt.date):
        return (_T_DATE, value.isoformat())
    if isinstance(value, dt.time):
        return (_T_DATETIME, value.isoformat())
    if isinstance(value, dt.timedelta):
        return (_T_NUM, value.total_seconds())
    if isinstance(value, (bytes, bytearray)):
        return (_T_BYTES, bytes(value).hex())
    if isinstance(value, (list, tuple)):
        return (_T_SEQ, tuple(canonical(v) for v in value))
    if isinstance(value, dict):
        return (_T_MAP, tuple((k, canonical(v)) for k, v in sorted(value.items())))
    return (_T_OTHER, repr(value))


def _is_float_type(t: st.SqlType | None) -> bool:
    return t is not None and t.kind == "float"


def cells_equal(a: Any, b: Any, rel_tol: float, abs_tol: float) -> bool:
    ka, kb = canonical(a), canonical(b)
    if ka == kb:
        return True
    if rel_tol <= 0 or ka[0] != _T_NUM or kb[0] != _T_NUM:
        return False
    try:
        fa, fb = float(ka[1]), float(kb[1])
    except (TypeError, ValueError, OverflowError):
        return False
    if not (math.isfinite(fa) and math.isfinite(fb)):
        return False
    diff = abs(fa - fb)
    return diff <= rel_tol * max(abs(fa), abs(fb)) or diff <= abs_tol


def compare_results(
    a: ResultSet, b: ResultSet, ordered: bool, tol: Tolerances, max_diffs: int = 5
) -> CompareOutcome:
    if len(a.columns) != len(b.columns):
        return CompareOutcome(
            False, f"column count {len(a.columns)} vs {len(b.columns)}"
        )
    if a.row_count != b.row_count:
        return CompareOutcome(False, f"row count {a.row_count} vs {b.row_count}")
    n_cols = len(a.columns)
    rel = []
    for ca, cb in zip(a.columns, b.columns):
        t = ca.type if _is_float_type(ca.type) else cb.type
        if _is_float_type(t):
            rel.append(tol.float32_rel if t.bits == 32 else tol.float64_rel)
        else:
            rel.append(0.0)
    rows_a = (
        a.rows
        if ordered
        else sorted(a.rows, key=lambda r: tuple(canonical(v) for v in r))
    )
    rows_b = (
        b.rows
        if ordered
        else sorted(b.rows, key=lambda r: tuple(canonical(v) for v in r))
    )
    diffs: list[str] = []
    for i, (ra, rb) in enumerate(zip(rows_a, rows_b)):
        if len(ra) != n_cols or len(rb) != n_cols:
            return CompareOutcome(False, f"row {i} has unexpected width")
        for c in range(n_cols):
            if not cells_equal(ra[c], rb[c], rel[c], tol.abs_tol):
                diffs.append(
                    f"row {i} col {c} ({a.columns[c].name}): {ra[c]!r} vs {rb[c]!r}"
                )
                if len(diffs) >= max_diffs:
                    return CompareOutcome(
                        False, f"{len(diffs)}+ differing cells", diffs
                    )
                break
    if diffs:
        return CompareOutcome(False, f"{len(diffs)} differing rows", diffs)
    return CompareOutcome(True)


# --------------------------------------------------------------------------
# ordering
# --------------------------------------------------------------------------


def is_total_order(query: Query) -> bool:
    """True when ORDER BY names every output column, so row order is deterministic
    up to duplicate rows (which are indistinguishable)."""
    if isinstance(query, Select):
        order = query.order_by
        names = set(query.output_names())
    elif isinstance(query, SetOp):
        order = query.order_by
        names = set(query.output_names())
    else:
        return False
    if not order:
        return False
    ordered = {o.expr.name for o in order if isinstance(o.expr, Alias)}
    return ordered >= names


def compare_mode(query: Query | None) -> str:
    return "ordered" if query is not None and is_total_order(query) else "multiset"

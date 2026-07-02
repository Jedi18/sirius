#!/usr/bin/env python3
"""Measure DuckDB cardinality-estimate accuracy (estimated vs actual rows per operator).

Context: issue #990 wants to collapse operator chains (e.g. grouped_aggregate ->
partition -> merge_aggregate into a single grouped_aggregate) when the estimated data
size is small enough to fit in one task. That decision is made at plan-construction time
from DuckDB's estimate, which Sirius reads as `op.estimated_cardinality`
(src/planner/sirius_physical_plan_generator.cpp:117). Before trusting it, we need to know
how accurate it is -- and especially how often DuckDB *under*-estimates (says "small" when
it is actually large), which is the dangerous direction for a collapse.

How it works (no engine code changes required)
-----------------------------------------------
DuckDB's JSON profiling emits, for every operator node, both numbers we need in one run:

    {
      "operator_type": "HASH_GROUP_BY",         # lowercased MetricType enum name
      "operator_cardinality": 4,                # ACTUAL output rows  (MetricType::OPERATOR_CARDINALITY)
      "operator_rows_scanned": ...,             # base-scan estimate for table scans
      "extra_info": {
          "__estimated_cardinality__": "6",     # ESTIMATED rows == what Sirius reads
          ...
      },
      "children": [ ... ]                       # recursive plan tree
    }

Verified against this repo's DuckDB submodule:
  - JSON keys are lowercased enum names        (duckdb/src/main/profiling_info.cpp:115)
  - estimated cardinality key                  (duckdb/src/execution/physical_operator.cpp:53,
                                                RenderTreeNode::ESTIMATED_CARDINALITY = "__estimated_cardinality__")
  - every operator's ParamsToString() sets it  (e.g. physical_hash_aggregate.cpp:939,
                                                physical_hash_join.cpp:1632)
  - default profiling includes EXTRA_INFO,
    OPERATOR_CARDINALITY, OPERATOR_TYPE         (duckdb/src/common/enums/metric_type.cpp:142)
  - output-path pragma is `profile_output`      (duckdb/src/main/config.cpp:216)

The estimate is DuckDB's optimizer output and is identical whether or not the GPU path
runs, so we measure with `gpu_execution=false` (plain DuckDB execution) and read the full
per-operator tree. With `gpu_execution=true` Sirius replaces execution with a single opaque
operator and this tree would not be available -- so keep it false.

Engines:
  --engine duckdb  (default): plain DuckDB. Estimates on *native tables* are identical to
                              what Sirius sees, so this is the simplest faithful setup.
  --engine sirius           : LOAD the Sirius extension (needs -unsigned) and
                              SET gpu_execution=false. Use this for *parquet* inputs so the
                              Sirius parquet footer-count callback (exact base cardinality,
                              src/sirius_extension.cpp:201) is active, matching production.

Usage
-----
  # 1. Populate a TPC-H database once (needs the tpch extension / network for INSTALL):
  python3 cardinality_accuracy.py prepare --db tpch_sf1.duckdb --tpch-sf 1

  # 2. Collect the tidy per-operator table:
  python3 cardinality_accuracy.py run \
      --db tpch_sf1.duckdb \
      --queries-dir ../tpch_performance/tpch_queries/orig \
      --workload tpch --scale-factor 1 \
      --out results/tpch_sf1.csv --summary

Output: one CSV row per operator, columns:
  workload, scale_factor, engine, query, op_id, depth, op_type, op_name,
  est_rows, act_rows, qerror, under_estimate, rows_scanned

Bytes are intentionally NOT emitted here: operator output row-widths are not in the
profiling JSON. The real #990 threshold is a byte budget, so bytes are derived in the
analysis step by joining these rows against per-query output schemas (see README).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Physical operator types that map to the operators #990 considers collapsing.
# Kept here for the --summary breakdown; the CSV records the raw op_type regardless.
COLLAPSE_CANDIDATES = {
    "HASH_GROUP_BY",
    "PERFECT_HASH_GROUP_BY",
    "PARTITIONED_AGGREGATE",
    "UNGROUPED_AGGREGATE",
    "HASH_JOIN",
    "PIECEWISE_MERGE_JOIN",
    "NESTED_LOOP_JOIN",
    "IE_JOIN",
    "ORDER_BY",
    "TOP_N",
}


def _to_int(value) -> int | None:
    """Coerce a JSON scalar (int or numeric string) to int, or None if not numeric."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        s = value.strip().replace(",", "")
        try:
            return int(float(s))
        except ValueError:
            return None
    return None


def qerror(est: int | None, act: int | None) -> float | None:
    """Symmetric q-error = max(est,act)/min(est,act), flooring both at 1 to avoid div-by-0.

    q-error of 1.0 == perfect. This is the standard estimator-accuracy metric
    (Leis et al., "How Good Are Query Optimizers, Really?", VLDB 2015).
    """
    if est is None or act is None:
        return None
    e = max(est, 1)
    a = max(act, 1)
    return max(e / a, a / e)


def build_run_script(
    engine: str, sirius_ext: str | None, profile_path: str, query_sql: str
) -> str:
    """Assemble the SQL fed to the DuckDB CLI for a single query."""
    lines: list[str] = []
    if engine == "sirius":
        if not sirius_ext:
            raise ValueError(
                "--engine sirius requires --sirius-ext <path to .duckdb_extension>"
            )
        lines.append(f"LOAD '{sirius_ext}';")
        # GPU execution off: we need DuckDB's per-operator profiling tree, and the estimate
        # is identical either way. On would collapse execution into one opaque operator.
        lines.append("SET gpu_execution=false;")
    # profile_output is the canonical name; profiling_output is an accepted alias.
    lines.append("PRAGMA enable_profiling='json';")
    lines.append(f"PRAGMA profile_output='{profile_path}';")
    query_sql = query_sql.strip().rstrip(";").strip()
    lines.append(query_sql + ";")
    return "\n".join(lines) + "\n"


def run_one_query(
    duckdb_bin: str,
    db: str,
    engine: str,
    sirius_ext: str | None,
    query_sql: str,
    timeout: int,
) -> dict:
    """Run one query with JSON profiling and return the parsed profiling tree.

    Raises RuntimeError with a short diagnostic on any failure so the caller can record
    the query as failed and keep going.
    """
    fd, profile_path = tempfile.mkstemp(suffix=".json", prefix="cardprof_")
    os.close(fd)
    # Ensure we never read a stale profile if the query dies before writing.
    try:
        os.remove(profile_path)
    except FileNotFoundError:
        pass

    cli = [duckdb_bin]
    if engine == "sirius":
        cli.append(
            "-unsigned"
        )  # locally-built extension is unsigned; startup-only flag
    cli.append(db)

    script = build_run_script(engine, sirius_ext, profile_path, query_sql)
    try:
        proc = subprocess.run(
            cli,
            input=script,
            text=True,
            stdout=subprocess.DEVNULL,  # query results are irrelevant; profile goes to file
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"timeout after {timeout}s")
    finally:
        pass

    if proc.returncode != 0:
        err = (proc.stderr or "").strip().splitlines()
        raise RuntimeError(
            f"duckdb exited {proc.returncode}: {err[-1] if err else 'no stderr'}"
        )

    if not os.path.exists(profile_path) or os.path.getsize(profile_path) == 0:
        err = (proc.stderr or "").strip().splitlines()
        raise RuntimeError(
            f"no profile written: {err[-1] if err else 'query may have errored'}"
        )

    try:
        with open(profile_path) as f:
            tree = json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"profile JSON parse error: {e}")
    finally:
        try:
            os.remove(profile_path)
        except FileNotFoundError:
            pass
    return tree


def walk_tree(
    node: dict,
    ctx: dict,
    rows: list[dict],
    depth: int = 0,
    counter: list[int] | None = None,
) -> None:
    """Pre-order walk of the profiling tree, appending one row per real operator node.

    The top-level JSON object is a query summary (no operator_type); its single child is
    the physical plan root. We recurse everywhere and emit a row only for nodes that have
    an operator_type.
    """
    if counter is None:
        counter = [0]

    op_type = node.get("operator_type")
    if op_type is not None:
        extra = node.get("extra_info")
        est = None
        if isinstance(extra, dict):
            est = _to_int(extra.get("__estimated_cardinality__"))
        act = _to_int(node.get("operator_cardinality"))
        op_id = counter[0]
        counter[0] += 1
        rows.append(
            {
                "workload": ctx["workload"],
                "scale_factor": ctx["scale_factor"],
                "engine": ctx["engine"],
                "query": ctx["query"],
                "op_id": op_id,
                "depth": depth,
                "op_type": op_type,
                "op_name": node.get("operator_name", ""),
                "est_rows": est,
                "act_rows": act,
                "qerror": qerror(est, act),
                "under_estimate": (est is not None and act is not None and act > est),
                "rows_scanned": _to_int(node.get("operator_rows_scanned")),
            }
        )
        child_depth = depth + 1
    else:
        child_depth = depth

    for child in node.get("children", []) or []:
        if isinstance(child, dict):
            walk_tree(child, ctx, rows, child_depth, counter)


def collect_queries(args) -> list[tuple[str, str]]:
    """Return [(query_name, sql), ...] from --queries-dir and/or --query files."""
    out: list[tuple[str, str]] = []
    if args.queries_dir:
        d = Path(args.queries_dir)
        for path in sorted(d.glob("*.sql"), key=lambda p: _natural_key(p.stem)):
            out.append((path.stem, path.read_text()))
    for q in args.query or []:
        path = Path(q)
        out.append((path.stem, path.read_text()))
    return out


def _natural_key(s: str):
    """Sort q2 before q10."""
    import re

    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def cmd_prepare(args) -> int:
    """Create/populate a DuckDB file with TPC-H data via the tpch extension's dbgen."""
    script = "INSTALL tpch;\n" "LOAD tpch;\n" f"CALL dbgen(sf={args.tpch_sf});\n"
    print(
        f"[prepare] generating TPC-H SF{args.tpch_sf} into {args.db} ...",
        file=sys.stderr,
    )
    proc = subprocess.run([args.duckdb_bin, args.db], input=script, text=True)
    if proc.returncode != 0:
        print(
            "[prepare] FAILED. If INSTALL tpch cannot reach the network, generate data via "
            "test/tpch_performance/generate_tpch_data.sh and pass that .duckdb file to `run`.",
            file=sys.stderr,
        )
    return proc.returncode


def cmd_run(args) -> int:
    queries = collect_queries(args)
    if not queries:
        print("No queries found (use --queries-dir and/or --query).", file=sys.stderr)
        return 2

    all_rows: list[dict] = []
    failures: list[tuple[str, str]] = []
    for name, sql in queries:
        ctx = {
            "workload": args.workload,
            "scale_factor": args.scale_factor,
            "engine": args.engine,
            "query": name,
        }
        try:
            tree = run_one_query(
                args.duckdb_bin,
                args.db,
                args.engine,
                args.sirius_ext,
                sql,
                args.timeout,
            )
        except RuntimeError as e:
            failures.append((name, str(e)))
            print(f"[run] {name}: FAILED ({e})", file=sys.stderr)
            continue
        before = len(all_rows)
        walk_tree(tree, ctx, all_rows)
        print(f"[run] {name}: {len(all_rows) - before} operators", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "workload",
        "scale_factor",
        "engine",
        "query",
        "op_id",
        "depth",
        "op_type",
        "op_name",
        "est_rows",
        "act_rows",
        "qerror",
        "under_estimate",
        "rows_scanned",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    print(
        f"\n[run] wrote {len(all_rows)} operator rows from {len(queries) - len(failures)}"
        f"/{len(queries)} queries -> {out_path}",
        file=sys.stderr,
    )
    if failures:
        print(
            f"[run] {len(failures)} queries failed: "
            f"{', '.join(n for n, _ in failures)}",
            file=sys.stderr,
        )

    if args.summary:
        print_summary(all_rows)
    return 0


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def print_summary(rows: list[dict]) -> None:
    """Per-operator-type q-error percentiles and under-estimate stats -- a sanity read,
    not the full analysis (thresholds/confusion-matrix live in the analysis step)."""
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        if r["qerror"] is None:
            continue
        by_type.setdefault(r["op_type"], []).append(r)

    print(
        "\n=== q-error by operator type (1.0 = perfect; >1 = wrong; act>est = dangerous) ==="
    )
    hdr = (
        f"{'operator_type':<26}{'n':>5}{'median':>9}{'p95':>9}{'p99':>9}{'max':>10}"
        f"{'under%':>8}{'worst_under':>13}"
    )
    print(hdr)
    print("-" * len(hdr))
    for op_type in sorted(by_type, key=lambda t: (t not in COLLAPSE_CANDIDATES, t)):
        group = by_type[op_type]
        qs = sorted(r["qerror"] for r in group)
        n = len(group)
        unders = [r for r in group if r["under_estimate"]]
        # Worst under-estimate: largest actual/estimated ratio (how far reality blew past).
        worst_under = 0.0
        for r in unders:
            if r["est_rows"] is not None and r["act_rows"] is not None:
                worst_under = max(worst_under, r["act_rows"] / max(r["est_rows"], 1))
        star = " *" if op_type in COLLAPSE_CANDIDATES else ""
        print(
            f"{op_type:<26}{n:>5}{_percentile(qs, 0.5):>9.2f}{_percentile(qs, 0.95):>9.2f}"
            f"{_percentile(qs, 0.99):>9.2f}{max(qs):>10.2f}"
            f"{100 * len(unders) / n:>7.1f}%{worst_under:>12.1f}x{star}"
        )
    print(
        "\n  * = operator type #990 may collapse. 'under%' = fraction where actual > estimated "
        "(collapse-unsafe direction);\n    'worst_under' = largest actual/estimated ratio seen "
        "for that type (the blast radius of a bad collapse)."
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    default_bin = str(
        Path(__file__).resolve().parents[2] / "build" / "release" / "duckdb"
    )
    default_ext = str(
        Path(__file__).resolve().parents[2]
        / "build"
        / "release"
        / "extension"
        / "sirius"
        / "sirius.duckdb_extension"
    )
    p.add_argument(
        "--duckdb-bin",
        default=default_bin,
        help=f"DuckDB CLI binary (default: {default_bin})",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pp = sub.add_parser(
        "prepare", help="populate a DuckDB file with TPC-H data (dbgen)"
    )
    pp.add_argument("--db", required=True, help="output .duckdb file")
    pp.add_argument("--tpch-sf", type=float, required=True, help="TPC-H scale factor")
    pp.set_defaults(func=cmd_prepare)

    pr = sub.add_parser("run", help="run queries with JSON profiling and emit tidy CSV")
    pr.add_argument("--db", required=True, help=".duckdb file with the data loaded")
    pr.add_argument(
        "--queries-dir", help="directory of *.sql files (one query per file)"
    )
    pr.add_argument(
        "--query", action="append", help="individual .sql file (repeatable)"
    )
    pr.add_argument(
        "--workload", default="tpch", help="label for the CSV (e.g. tpch, tpcds)"
    )
    pr.add_argument("--scale-factor", default="1", help="label for the CSV")
    pr.add_argument(
        "--engine",
        choices=["duckdb", "sirius"],
        default="duckdb",
        help="duckdb (native tables) or sirius (parquet footer-count faithful)",
    )
    pr.add_argument(
        "--sirius-ext",
        default=default_ext,
        help="path to sirius.duckdb_extension (for --engine sirius)",
    )
    pr.add_argument(
        "--timeout", type=int, default=1200, help="per-query timeout seconds"
    )
    pr.add_argument("--out", required=True, help="output CSV path")
    pr.add_argument(
        "--summary", action="store_true", help="print per-operator q-error summary"
    )
    pr.set_defaults(func=cmd_run)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

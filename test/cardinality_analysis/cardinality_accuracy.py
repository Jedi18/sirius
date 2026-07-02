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
runs, so we ALWAYS `SET gpu_execution=false` and read DuckDB's full per-operator tree. With
gpu_execution=true Sirius replaces execution with a single opaque EXTENSION operator and this
tree is not available. NOTE: build/release/duckdb auto-loads Sirius with gpu_execution ON by
default, so this must be disabled explicitly even for --engine duckdb.

Engines (both disable gpu_execution; the difference is only whether the extension is LOADed):
  --engine duckdb  (default): assume the binary already has Sirius available (the Sirius
                              build auto-loads it). Simplest setup.
  --engine sirius           : also `LOAD` the extension explicitly (needs -unsigned), for a
                              binary that does not auto-load it, or for parquet inputs where
                              the Sirius footer-count callback (src/sirius_extension.cpp:201)
                              provides exact base cardinalities.

Usage
-----
  # 1. Populate a TPC-H database once (needs the tpch extension / network for INSTALL):
  python3 cardinality_accuracy.py prepare --db tpch_sf1.duckdb --tpch-sf 1

  # 2. Collect the tidy per-operator table, with a summary and a #990 collapse-decision
  #    confusion matrix at a candidate threshold of 1,000,000 rows:
  python3 cardinality_accuracy.py run \
      --db tpch_sf1.duckdb \
      --queries-dir ../tpch_performance/tpch_queries/orig \
      --workload tpch --scale-factor 1 \
      --out results/tpch_sf1.csv --summary --threshold 1000000

Output: one CSV row per operator, columns:
  workload, scale_factor, engine, query, op_id, depth, op_type, op_name,
  est_rows, act_rows, qerror, under_estimate, rows_scanned
With --threshold T, five more columns classify each row at that threshold:
  is_candidate, would_collapse, actually_small, collapse_outcome, overrun_x

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
# Used for the --summary breakdown and the --threshold confusion matrix; the CSV records
# the raw op_type regardless.
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
    # Disable Sirius GPU interception UNCONDITIONALLY. The Sirius-enabled build/release/duckdb
    # auto-loads the extension with gpu_execution ON by default, so without this the whole
    # query collapses into a single opaque EXTENSION/SIRIUS_GPU_EXECUTION operator with no
    # per-operator tree and no __estimated_cardinality__. With it OFF, DuckDB builds and
    # profiles its normal plan and exposes per-operator estimates -- which are DuckDB's
    # optimizer output either way, so this does not bias the measurement. On a vanilla DuckDB
    # without Sirius this is an unknown-setting error, tolerated because run_one_query keys
    # success off the profile file, not the exit code.
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

    # Key success off the profile file, not the exit code: a harmless unknown-setting error
    # (e.g. SET gpu_execution on a non-Sirius binary) can make the CLI exit nonzero even
    # though the query ran and the profile was written.
    if not os.path.exists(profile_path) or os.path.getsize(profile_path) == 0:
        err = (proc.stderr or "").strip().splitlines()
        detail = (
            err[-1]
            if err
            else (
                f"duckdb exited {proc.returncode}"
                if proc.returncode
                else "query may have errored"
            )
        )
        raise RuntimeError(f"no profile written: {detail}")

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


def annotate_collapse_outcome(rows: list[dict], threshold: int) -> None:
    """Add per-row #990 collapse-decision columns for a given threshold T (in rows).

    Mutates each row in place. For collapse-candidate operators with a known estimate and
    actual, `collapse_outcome` is one of correct_collapse / wrong_collapse / wrong_keep /
    correct_keep (see print_confusion). Non-candidate rows, or rows missing est/act, get
    blank outcome fields. This makes the classification queryable straight from the CSV; it
    is threshold-specific, so re-running with a different T overwrites these columns.
    """
    T = threshold
    for r in rows:
        est, act = r["est_rows"], r["act_rows"]
        cand = r["op_type"] in COLLAPSE_CANDIDATES
        r["is_candidate"] = cand
        r["would_collapse"] = "" if est is None else (est <= T)
        r["actually_small"] = "" if act is None else (act <= T)
        outcome = ""
        overrun = ""
        if cand and est is not None and act is not None:
            small_est, small_act = est <= T, act <= T
            outcome = {
                (True, True): "correct_collapse",
                (True, False): "wrong_collapse",  # collapsed then overflowed
                (False, True): "wrong_keep",  # missed optimization
                (False, False): "correct_keep",
            }[(small_est, small_act)]
            if outcome == "wrong_collapse":
                overrun = round(act / T, 2)
        r["collapse_outcome"] = outcome
        r["overrun_x"] = overrun


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
        added = all_rows[before:]
        print(f"[run] {name}: {len(added)} operators", file=sys.stderr)
        # Detect the "interception still on" failure mode: a single opaque Sirius operator
        # with no estimates instead of DuckDB's per-operator tree.
        if (
            added
            and all(r["est_rows"] is None for r in added)
            and any(
                r["op_type"] == "EXTENSION" or "SIRIUS" in (r["op_name"] or "").upper()
                for r in added
            )
        ):
            print(
                f"[run] WARNING: {name} produced only a Sirius GPU operator with no "
                "estimates -- GPU interception was NOT disabled. The binary must accept "
                "'SET gpu_execution=false' (a Sirius setting); check it is the Sirius build.",
                file=sys.stderr,
            )

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
    # When a threshold is given, classify each row and add the outcome columns to the CSV
    # (in addition to the console confusion matrix).
    if args.threshold is not None:
        annotate_collapse_outcome(all_rows, args.threshold)
        fields += [
            "is_candidate",
            "would_collapse",
            "actually_small",
            "collapse_outcome",
            "overrun_x",
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
    if args.threshold is not None:
        print_confusion(all_rows, args.threshold)
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
    not the full analysis (thresholds/confusion-matrix live in --threshold / analysis).
    """
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


def print_confusion(rows: list[dict], threshold: int) -> None:
    """#990 collapse-decision confusion matrix at a candidate threshold (in rows).

    For each collapse-candidate operator with a known estimate and actual:
      decision = collapse if est_rows <= T ;  reality = small if act_rows <= T
    yielding four outcomes -- the only one that causes a crash is WRONG-COLLAPSE
    (est <= T but act > T: we collapsed, then overflowed).
    """
    T = threshold
    cand = [
        r
        for r in rows
        if r["op_type"] in COLLAPSE_CANDIDATES
        and r["est_rows"] is not None
        and r["act_rows"] is not None
    ]

    print(
        f"\n=== #990 collapse-decision confusion matrix @ threshold T = {T:,} rows ==="
    )
    print(f"(collapse-candidate operators with known est & act: {len(cand)})")
    if not cand:
        print("  no qualifying operators -- check --threshold and that candidates ran.")
        return

    correct_collapse = [r for r in cand if r["est_rows"] <= T and r["act_rows"] <= T]
    wrong_collapse = [
        r for r in cand if r["est_rows"] <= T and r["act_rows"] > T
    ]  # DANGER
    wrong_keep = [r for r in cand if r["est_rows"] > T and r["act_rows"] <= T]  # missed
    correct_keep = [r for r in cand if r["est_rows"] > T and r["act_rows"] > T]
    n = len(cand)

    def pct(x):
        return f"{100 * len(x) / n:5.1f}%"

    print("\n                              actual <= T        actual > T")
    print(
        f"  est <= T  (would collapse)  {len(correct_collapse):>6} correct   "
        f"{len(wrong_collapse):>6} WRONG-COLLAPSE (OOM)"
    )
    print(
        f"  est >  T  (would keep)      {len(wrong_keep):>6} wrong-keep "
        f"{len(correct_keep):>6} correct"
    )

    print(
        f"\n  correct-collapse : {len(correct_collapse):>5}  ({pct(correct_collapse)})  safe optimization"
    )
    print(
        f"  WRONG-COLLAPSE   : {len(wrong_collapse):>5}  ({pct(wrong_collapse)})  <- these overflow / OOM"
    )
    print(
        f"  wrong-keep       : {len(wrong_keep):>5}  ({pct(wrong_keep)})  missed optimization (safe)"
    )
    print(f"  correct-keep     : {len(correct_keep):>5}  ({pct(correct_keep)})")

    would_collapse = len(correct_collapse) + len(wrong_collapse)
    if would_collapse:
        rate = 100 * len(wrong_collapse) / would_collapse
        print(
            f"\n  Of operators DuckDB calls small (est <= T): {would_collapse}; "
            f"wrong-collapse rate = {len(wrong_collapse)}/{would_collapse} = {rate:.1f}%"
        )

    if wrong_collapse:
        overruns = sorted(r["act_rows"] / T for r in wrong_collapse)
        print(
            f"  Overrun (actual/T) among wrong-collapses: median {_percentile(overruns, 0.5):.1f}x, "
            f"p95 {_percentile(overruns, 0.95):.1f}x, max {max(overruns):.1f}x"
        )
        worst = sorted(
            wrong_collapse,
            key=lambda r: r["act_rows"] / max(r["est_rows"], 1),
            reverse=True,
        )[:10]
        print("  Worst offenders (est said small, reality did not):")
        for r in worst:
            print(
                f"    {r['query']:<8} {r['op_type']:<22} est={r['est_rows']:>12,} "
                f"act={r['act_rows']:>12,}  overrun={r['act_rows'] / T:>6.1f}x"
            )
    else:
        print(
            "  No wrong-collapses at this threshold: collapse would be safe on this workload."
        )

    # Per-candidate-type wrong-collapse breakdown.
    print("\n  wrong-collapse by operator type:")
    by_type: dict[str, list[dict]] = {}
    for r in wrong_collapse:
        by_type.setdefault(r["op_type"], []).append(r)
    if by_type:
        for op_type in sorted(by_type, key=lambda t: -len(by_type[t])):
            grp = by_type[op_type]
            worst = max(r["act_rows"] / T for r in grp)
            print(f"    {op_type:<24} {len(grp):>4}   worst overrun {worst:.1f}x")
    else:
        print("    (none)")
    print(
        "\n  NOTE: T is in ROWS. The real #990 budget is BYTES -- rerun once est_bytes/act_bytes "
        "are\n  derived (see README) for the byte-accurate decision."
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
    pr.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="print the #990 collapse-decision confusion matrix at this row threshold T",
    )
    pr.set_defaults(func=cmd_run)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

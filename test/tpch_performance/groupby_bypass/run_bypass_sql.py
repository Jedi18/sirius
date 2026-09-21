#!/usr/bin/env python3
"""Off/on comparison for the group-by memory-aware bypass prototype (issue #1746 point 2).

Both arms run the same binary and the same configuration; the only difference is
`enable_group_by_memory_aware_bypass`. SIRIUS_ENABLE_TEST_OPTIONS=1 is set for both arms because
the switch is registered as an internal option — leaving it set in only one arm would change more
than the switch.

Every timed variant is a fresh process. Results are compared against DuckDB's CPU output outside
the measured region, GPU fallback is disabled so a supported query cannot silently run on the CPU,
and the engine log is parsed for the decision record so activation is proven rather than assumed.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import statistics
import subprocess
from pathlib import Path

TABLES = [
    "lineitem",
    "orders",
    "customer",
    "part",
    "partsupp",
    "supplier",
    "nation",
    "region",
]

# Queries inside the v1 supported envelope: fixed-width integral grouping keys, COUNT/MIN/MAX
# partial states, no ORDER BY / TOP_N, and a bounded result-collection downstream.
#
# Each groups on a mid-cardinality key so that the *partial* input the merge combines is large
# enough for the automatic policy to pick more than one partition, while the final result stays
# small enough to write out and compare exactly. That is the whole reason scan batching is lowered
# in the config: it multiplies the partial input without touching the result. Adding an
# ORDER BY/LIMIT to shrink the output instead would insert a TOP_N, which v1 rejects outright.
SUPPORTED = {
    # Probed: 8 partial batches, 15,627,070 partial rows, 300 MiB partial input, AUTO = 3.
    "bypass_mid": """
        SELECT l_partkey, COUNT(*) AS c, MIN(l_linenumber) AS lo, MAX(l_linenumber) AS hi
        FROM lineitem GROUP BY l_partkey
    """,
    # Two-column integral key, wider key row, different aggregate mix.
    "bypass_twokey": """
        SELECT l_suppkey, l_linenumber, COUNT(*) AS c, MAX(l_orderkey) AS hi
        FROM lineitem GROUP BY l_suppkey, l_linenumber
    """,
    # Held-out size/type combination inside the same envelope: a smaller key domain, so this one
    # is expected to resolve to one partition on its own and report `already_one` rather than an
    # activation. Included precisely so the difference between the two is visible.
    "bypass_small": """
        SELECT l_suppkey, COUNT(*) AS c, MIN(l_orderkey) AS lo
        FROM lineitem GROUP BY l_suppkey
    """,
}

# Edge shapes the brief calls out. A small or empty input resolves to one partition on its own; the
# point is that the prototype reports `already_one` rather than claiming an activation, and that
# finalization still produces the right rows.
EDGE = {
    # Nullable grouping key. Nulls are inside the supported envelope, but only because every
    # nullable column is charged a validity mask in the model.
    "edge_nullkey": """
        SELECT CASE WHEN l_linenumber = 1 THEN NULL ELSE l_suppkey END AS k,
               COUNT(*) AS c, MAX(l_linenumber) AS hi
        FROM lineitem GROUP BY k
    """,
    # Empty input: no partial batches at all.
    "edge_empty": """
        SELECT l_orderkey, COUNT(*) AS c
        FROM lineitem WHERE l_orderkey < 0 GROUP BY l_orderkey
    """,
    # Tiny key domain: a single partial batch per task, which the merge forwards on its fast path.
    "edge_onebatch": """
        SELECT l_linenumber, COUNT(*) AS c, MIN(l_orderkey) AS lo
        FROM lineitem GROUP BY l_linenumber
    """,
}

# Queries expected to stay on the automatic count. Each is a compatibility check: the prototype
# must decline, and the result must still match the CPU.
REJECTED = {
    # SUM widens the partial state past the v1 integral whitelist.
    "reject_sum": """
        SELECT l_suppkey, SUM(l_quantity) AS s
        FROM lineitem GROUP BY l_suppkey
    """,
    # Explicit TOP_N rejection check. Narrow partial rows keep this one's automatic count at 1,
    # so it exercises the whole plan but reports `already_one`; the record still shows
    # downstream_ok=0.
    "reject_topn": """
        SELECT l_partkey, COUNT(*) AS c
        FROM lineitem GROUP BY l_partkey ORDER BY c DESC, l_partkey LIMIT 200
    """,
    # The TOP_N check that actually reaches the downstream gate: the same shape with wider partial
    # rows, so the automatic count exceeds 1 and `unsupported_downstream` is the reason returned
    # rather than a short-circuit on `already_one`.
    "reject_topn_wide": """
        SELECT l_partkey, COUNT(*) AS c, MIN(l_linenumber) AS lo, MAX(l_linenumber) AS hi
        FROM lineitem GROUP BY l_partkey ORDER BY c DESC, l_partkey LIMIT 200
    """,
    # AVG needs a post-merge divide that the model does not cover.
    "reject_avg": """
        SELECT l_suppkey, AVG(l_linenumber) AS a
        FROM lineitem GROUP BY l_suppkey
    """,
    # COUNT(DISTINCT) arrives at the merge as a LIST partial state.
    "reject_countdistinct": """
        SELECT l_suppkey, COUNT(DISTINCT l_partkey) AS d
        FROM lineitem GROUP BY l_suppkey
    """,
    # A string grouping key is outside the fixed-width envelope.
    "reject_string": """
        SELECT l_returnflag, l_linestatus, COUNT(*) AS c
        FROM lineitem GROUP BY l_returnflag, l_linestatus
    """,
}

# The same rejections, widened so the automatic count exceeds 1 and the gate under test is the
# reason actually returned.
#
# The narrow versions above all resolve to a single partition on their own, so `already_one`
# short-circuits before their gate is consulted. That still shows the candidate staying off and the
# results staying correct, but it is weaker evidence than seeing `unsupported_state` or
# `unsupported_downstream` come back. These group on a 2M-key column so the partial input clears
# the 128 MiB partition target.
REJECTED_WIDE = {
    # TOP_N downstream, reached rather than short-circuited.
    "reject_topn_wide": """
        SELECT l_partkey, COUNT(*) AS c, MIN(l_linenumber) AS lo, MAX(l_linenumber) AS hi
        FROM lineitem GROUP BY l_partkey ORDER BY c DESC, l_partkey LIMIT 200
    """,
    # SUM widens the partial state past the integral whitelist.
    "reject_sum_wide": """
        SELECT l_partkey, SUM(l_quantity) AS s, COUNT(*) AS c
        FROM lineitem GROUP BY l_partkey
    """,
    # AVG needs a post-merge divide the model does not cover.
    "reject_avg_wide": """
        SELECT l_partkey, AVG(l_linenumber) AS a, COUNT(*) AS c
        FROM lineitem GROUP BY l_partkey
    """,
    # A genuinely variable-width grouping key at a cardinality large enough to matter. Uses a real
    # VARCHAR column rather than CAST(int AS VARCHAR): that cast is unsupported on the GPU
    # (cuDF's unary cast requires a fixed-width target) and fails identically with the prototype on
    # and off, so it would test the cast rather than this policy.
    "reject_string_wide": """
        SELECT p_name, COUNT(*) AS c, MIN(p_size) AS lo, MAX(p_size) AS hi
        FROM part GROUP BY p_name
    """,
}

# Results compared with a floating tolerance instead of exactly. Only AVG produces non-integral
# output; every other query above yields integral keys and states, compared byte-exactly.
FLOAT_QUERIES = {"reject_avg", "reject_avg_wide"}

TIMER = re.compile(r"Run Time \(s\): real ([0-9.]+)")
DECISION = re.compile(r"group_by_bypass: (.*)")
SIZED = re.compile(r"sirius_physical_partition id (\d+) sized (\d+) partitions")
# TRACE-only: the task's actual reservation and charged allocation peak.
MEMREC = re.compile(
    r"memory history record - task=(\d+), input_basis=(\d+), output_bytes=(\d+), "
    r"reservation_bytes=(\d+), peak_bytes=(\d+), peak_bytes_to_materialize_input=(\d+)"
)
# Fires when a task asserting a policy-derived floor is admitted below it.
SHORTFALL = re.compile(r"asserted a mandatory floor of (\d+) bytes")


def parse_record(text):
    """Turn one `key=value key=value` decision record into a dict."""
    out = {}
    for token in text.split():
        if "=" in token:
            key, _, value = token.partition("=")
            out[key] = value
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duckdb", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--queries", default="")
    ap.add_argument("--warmups", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument(
        "--log-level",
        default="info",
        help="engine log level. 'trace' captures the task's actual reservation and "
        "charged allocation peak, for checking model coverage against true "
        "allocator accounting; keep those runs untimed and separate.",
    )
    ap.add_argument(
        "--skip-cpu",
        action="store_true",
        help="skip the CPU reference (for constrained-budget runs where only the "
        "engine's own behaviour is under test)",
    )
    args = ap.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    data_dir = str(Path(args.data_dir).resolve())
    config = str(Path(args.config).resolve())

    all_queries = dict(SUPPORTED)
    all_queries.update(EDGE)
    all_queries.update(REJECTED)
    all_queries.update(REJECTED_WIDE)
    names = [
        n
        for n in (args.queries.split(",") if args.queries else list(all_queries))
        if n in all_queries
    ]

    views = "\n".join(
        f"create or replace view {t} as select * from read_parquet('{data_dir}/{t}/*.parquet');"
        for t in TABLES
    )

    def execute(sql, gpu, bypass, timed, label):
        """Run one variant. Results go to a file, never to stdout.

        The timed block writes results to /dev/null to avoid terminal I/O. CLI formatting
        still contributes to elapsed time, so large outputs also need engine-log timing.
        `.output` redirects query results only — the CLI keeps writing `Run Time (s)` to stdout —
        as described in README.md. The untimed capture that follows
        writes to a file for the CPU comparison, so Python never holds the whole result in memory.
        """
        result_path = (output / "results" / f"{label}.csv").resolve()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            ".bail on",
            ".mode csv",
            ".headers off",
            ".nullvalue __NULL__",
            views,
            f"SET gpu_execution={'true' if gpu else 'false'};",
            "SET enable_duckdb_fallback=false;",
        ]
        if gpu:
            lines.append(
                f"SET enable_group_by_memory_aware_bypass={'true' if bypass else 'false'};"
            )
        body = sql.strip().rstrip(";") + ";"
        if timed:
            # Timing includes engine execution, materialization and CLI formatting, but no terminal
            # I/O. Identical in both arms; use engine logs to separate formatting costs.
            lines += [".output /dev/null", ".timer on"]
            lines += [body] * (args.warmups + args.repeats)
            lines += [".timer off", ".output stdout"]
        lines += [
            f".output {result_path}",
            body,
            ".output stdout",
            ".print BYPASS_RUN_DONE",
        ]

        logdir = (output / "logs" / label).resolve()
        logdir.mkdir(parents=True, exist_ok=True)
        env = dict(
            os.environ,
            SIRIUS_CONFIG_FILE=config,
            SIRIUS_LOG_DIR=str(logdir),
            SIRIUS_LOG_LEVEL=args.log_level,
            # Set in BOTH arms: the switch is an internal option, and enabling test options
            # in only one arm would not be a controlled comparison.
            SIRIUS_ENABLE_TEST_OPTIONS="1",
        )
        proc = subprocess.run(
            [args.duckdb, "-batch"],
            input="\n".join(lines) + "\n",
            capture_output=True,
            text=True,
            timeout=args.timeout,
            env=env,
        )
        times = [float(t) for t in TIMER.findall(proc.stdout + proc.stderr)]
        completed = "BYPASS_RUN_DONE" in proc.stdout
        log = "\n".join(
            p.read_text(errors="replace") for p in sorted(logdir.glob("*.log"))
        )
        return proc, times, (result_path if completed else None), log

    def cell_equal(a, b):
        if a == b:
            return True
        if re.fullmatch(r"[-+]?\d+", a) and re.fullmatch(r"[-+]?\d+", b):
            return int(a) == int(b)
        try:
            # Only the AVG compatibility query produces floats; integral keys and states above
            # are compared exactly by the branch before this one.
            return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-10)
        except ValueError:
            return False

    def sorted_digest(path):
        """SHA-256 of the result's sorted lines, plus the row count.

        Sorting makes the comparison order-insensitive (none of these queries is ordered), and
        hashing keeps a multi-million-row comparison off the Python heap.
        """
        digest = hashlib.sha256()
        rows = 0
        with path.open("rb") as fh:
            lines = fh.read().splitlines()
        rows = len(lines)
        for line in sorted(lines):
            digest.update(line)
            digest.update(b"\n")
        return digest.hexdigest(), rows

    def float_tolerant_equal(reference_path, result_path):
        """Row-wise comparison with a documented tolerance, for the one non-integral query."""
        with reference_path.open() as a, result_path.open() as b:
            ref = sorted(csv.reader(a))
            got = sorted(csv.reader(b))
        if len(ref) != len(got):
            return False, len(got)
        for ra, rb in zip(ref, got):
            if len(ra) != len(rb) or not all(cell_equal(x, y) for x, y in zip(ra, rb)):
                return False, len(got)
        return True, len(got)

    rng = random.Random(args.seed)
    rng.shuffle(names)
    records_path = output / "runs.jsonl"
    with records_path.open("a") as records:
        for name in names:
            sql = all_queries[name]
            reference_path = None
            reference_digest = None
            reference_rows = None
            if not args.skip_cpu:
                cpu, _, reference_path, _ = execute(
                    sql, False, False, False, f"{name}-cpu"
                )
                (output / f"{name}-cpu.stderr").write_text(cpu.stderr)
                if cpu.returncode or reference_path is None:
                    raise RuntimeError(
                        f"CPU reference failed for {name}: {cpu.stderr[-2000:]}"
                    )
                if name not in FLOAT_QUERIES:
                    reference_digest, reference_rows = sorted_digest(reference_path)

            # Alternate the two arms rather than running all of one then all of the other, so a
            # slow drift in machine state cannot be mistaken for an effect of the switch.
            arms = [False, True]
            if rng.random() < 0.5:
                arms.reverse()
            for bypass in arms:
                label = f"{name}-bypass{int(bypass)}"
                proc, times, result_path, log = execute(sql, True, bypass, True, label)
                (output / f"{label}.stderr").write_text(proc.stderr)

                decisions = [parse_record(m) for m in DECISION.findall(log)]
                sized = [(int(a), int(b)) for a, b in SIZED.findall(log)]
                gpu_executions = log.count("Transparent GPU execution: executing query")
                mem_records = [
                    dict(
                        task=int(a),
                        input_basis=int(b),
                        output_bytes=int(c),
                        reservation_bytes=int(d),
                        peak_bytes=int(e),
                        materialize_bytes=int(f),
                    )
                    for a, b, c, d, e, f in MEMREC.findall(log)
                ]
                shortfalls = [int(m) for m in SHORTFALL.findall(log)]

                correct = None
                out_rows = None
                if reference_path is not None and result_path is not None:
                    if name in FLOAT_QUERIES:
                        correct, out_rows = float_tolerant_equal(
                            reference_path, result_path
                        )
                    else:
                        digest, out_rows = sorted_digest(result_path)
                        correct = (
                            digest == reference_digest and out_rows == reference_rows
                        )
                elif reference_path is not None:
                    correct = False

                samples = times[args.warmups :]
                record = dict(
                    query=name,
                    bypass_enabled=bypass,
                    rc=proc.returncode,
                    matches_cpu=correct,
                    output_rows=out_rows,
                    reference_rows=reference_rows,
                    compared_with=(
                        "float-tolerance rel=1e-9 abs=1e-10"
                        if name in FLOAT_QUERIES
                        else "exact sorted digest"
                    ),
                    gpu_executions=gpu_executions,
                    all_times_s=times,
                    samples_s=samples,
                    median_s=statistics.median(samples) if samples else None,
                    spread_s=(max(samples) - min(samples)) if samples else None,
                    decision_reasons=sorted({d.get("reason", "?") for d in decisions}),
                    chosen_p=sorted({d.get("chosen_p", "?") for d in decisions}),
                    auto_p=sorted({d.get("auto_p", "?") for d in decisions}),
                    activated=any(d.get("activated") == "1" for d in decisions),
                    decisions=decisions,
                    sized_partitions=sized,
                    memory_records=mem_records,
                    floor_shortfalls=shortfalls,
                    fallback_enabled=False,
                    config=config,
                    warmups=args.warmups,
                    repeats=args.repeats,
                )
                records.write(json.dumps(record) + "\n")
                records.flush()
                print(
                    label,
                    "rc=%d" % proc.returncode,
                    "cpu_match=%s" % correct,
                    "rows=%s" % out_rows,
                    "median=%s" % record["median_s"],
                    "reasons=%s" % record["decision_reasons"],
                    "activated=%s" % record["activated"],
                    flush=True,
                )

    print(f"\nwrote {records_path}")


if __name__ == "__main__":
    main()

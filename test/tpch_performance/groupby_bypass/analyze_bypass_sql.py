#!/usr/bin/env python3
"""Summarize a run_bypass_sql.py result directory.

Reports per query, per arm: median, spread, all samples, the decision reasons the engine actually
logged, the partition counts it actually applied, and whether the CPU comparison passed. It does
not pool runs from different configurations, and it refuses to call a variant correct unless the
CPU comparison ran and passed.
"""
import argparse
import csv
import json
import statistics
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()
    root = Path(args.run_dir)
    records = [
        json.loads(line)
        for line in (root / "runs.jsonl").read_text().splitlines()
        if line
    ]

    by_query = {}
    for r in records:
        by_query.setdefault(r["query"], {})[r["bypass_enabled"]] = r

    rows = []
    print(
        f"{'query':24} {'arm':5} {'median_s':>9} {'spread_s':>9} {'auto_p':>7} {'chosen_p':>9} "
        f"{'reasons':28} {'cpu':>5} {'gpu_exec':>8}"
    )
    print("-" * 118)
    for query in sorted(by_query):
        for enabled in (False, True):
            r = by_query[query].get(enabled)
            if r is None:
                continue
            arm = "on" if enabled else "off"
            print(
                f"{query:24} {arm:5} {r['median_s'] or 0:9.3f} {r['spread_s'] or 0:9.3f} "
                f"{','.join(r['auto_p']):>7} {','.join(r['chosen_p']):>9} "
                f"{','.join(r['decision_reasons'])[:28]:28} {str(r['matches_cpu']):>5} "
                f"{r['gpu_executions']:>8}"
            )
            rows.append(
                dict(
                    query=query,
                    arm=arm,
                    median_s=r["median_s"],
                    spread_s=r["spread_s"],
                    samples=";".join(str(s) for s in r["samples_s"]),
                    auto_p=",".join(r["auto_p"]),
                    chosen_p=",".join(r["chosen_p"]),
                    reasons=",".join(r["decision_reasons"]),
                    activated=r["activated"],
                    matches_cpu=r["matches_cpu"],
                    gpu_executions=r["gpu_executions"],
                    sized_partitions=";".join(
                        f"{a}:{b}" for a, b in r["sized_partitions"]
                    ),
                    rc=r["rc"],
                )
            )

    # The two acceptance criteria, checked against what the engine logged rather than asserted.
    activated = [r for r in records if r["bypass_enabled"] and r["activated"]]
    rejected = [r for r in records if r["bypass_enabled"] and not r["activated"]]
    print("\nActivation (AUTO > 1 turned into P=1):")
    if activated:
        for r in activated:
            sized = {b for _, b in r["sized_partitions"]}
            print(
                f"  {r['query']}: auto_p={','.join(r['auto_p'])} -> chosen_p="
                f"{','.join(r['chosen_p'])}, partitions actually sized: {sorted(sized)}"
            )
    else:
        print("  NONE — no supported query activated in this run")

    print("\nDeclined with the automatic count preserved:")
    for r in rejected:
        off = by_query[r["query"]].get(False)
        same = off is not None and off["sized_partitions"] == r["sized_partitions"]
        print(
            f"  {r['query']}: reasons={','.join(r['decision_reasons'])} "
            f"same_partitions_as_off={same}"
        )

    shortfalls = [r for r in records if r.get("floor_shortfalls")]
    if shortfalls:
        print(
            "\nTasks admitted below an asserted memory floor (a truncated reservation is not "
            "evidence the modelled work fits):"
        )
        for r in shortfalls:
            print(
                f"  {r['query']} bypass={r['bypass_enabled']}: {r['floor_shortfalls']}"
            )

    peaks = [(r, m) for r in records for m in r.get("memory_records", [])]
    if peaks:
        print(
            "\nTask memory records (TRACE runs only) — modelled floor vs charged peak:"
        )
        for r, m in peaks:
            print(
                f"  {r['query']} bypass={r['bypass_enabled']}: reservation={m['reservation_bytes']} "
                f"peak={m['peak_bytes']} input_basis={m['input_basis']}"
            )

    failures = [r for r in records if r["rc"] != 0 or r["matches_cpu"] is False]
    print(f"\nFailures: {len(failures)}")
    for r in failures:
        print(
            f"  {r['query']} bypass={r['bypass_enabled']} rc={r['rc']} "
            f"matches_cpu={r['matches_cpu']}"
        )

    out = root / "summary.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

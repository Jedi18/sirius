#!/usr/bin/env python3
"""Summarize small-data pipeline overhead from parsed Sirius log analysis.

Input is the output directory produced by:

    python tools/log_analyzer/parse_logs.py <sirius.log> --out <log_analysis_dir>

The script focuses on the proposed small-data simplification question:
how much time is spent in target pipeline-breaker operators such as
PARTITION and MERGE_GROUP_BY, plus the FULL-barrier gaps before them?

It writes:
  - small_data_overhead.csv: one row per query
  - small_data_overhead_details.csv: one row per target pipeline
  - SMALL_DATA_OVERHEAD.md: human-readable report
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


DEFAULT_TARGET_OPERATORS = ("PARTITION", "MERGE_GROUP_BY")


@dataclass(frozen=True)
class PipelineMetric:
    pipeline_id: int
    operator_chain: str
    target_operators: str
    num_tasks: int
    execution_sum_ms: float
    execution_max_task_ms: float
    span_ms: float | None
    dependency_gap_ms: float | None
    input_rows: int
    output_rows: int
    input_bytes: int
    output_bytes: int


@dataclass(frozen=True)
class QuerySummary:
    query_begin_ts: str
    folder: str
    status: str
    duration_ms: float | None
    sql_preview: str
    target_pipeline_count: int
    target_execution_sum_ms: float
    target_span_ms: float
    target_gap_ms: float
    target_span_plus_gap_ms: float
    pct_execution_sum: float | None
    pct_span_plus_gap: float | None
    decision: str


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def parse_int(value: str | None) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def parse_timestamp(value: str | None) -> datetime | None:
    if value is None or value == "":
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def ms_between(start: str | None, end: str | None) -> float | None:
    start_ts = parse_timestamp(start)
    end_ts = parse_timestamp(end)
    if start_ts is None or end_ts is None:
        return None
    return max(0.0, (end_ts - start_ts).total_seconds() * 1000.0)


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f} ms"


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}%"


def load_plan(qdir: Path) -> dict | None:
    path = qdir / "pipeline_plan.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def operator_chain(pipeline: dict) -> str:
    return " -> ".join(
        f"{op.get('type', '?')}#{op.get('id', '?')}" for op in pipeline.get("operators", [])
    )


def pipeline_operator_types(pipeline: dict) -> set[str]:
    return {op.get("type", "") for op in pipeline.get("operators", [])}


def dependency_gap_ms(pipeline: dict, aggregate_by_id: dict[int, dict]) -> float | None:
    begin = parse_timestamp(aggregate_by_id.get(pipeline["pipeline_num"], {}).get("pipeline_begin"))
    if begin is None:
        return None

    dependency_ends = []
    for dep_id in pipeline.get("dependencies", []):
        dep_end = parse_timestamp(aggregate_by_id.get(dep_id, {}).get("pipeline_end"))
        if dep_end is not None:
            dependency_ends.append(dep_end)

    if not dependency_ends:
        # Fall back to explicit input edges if dependencies were absent.
        for input_edge in pipeline.get("inputs", []):
            dep_id = input_edge.get("from_pipeline")
            dep_end = parse_timestamp(aggregate_by_id.get(dep_id, {}).get("pipeline_end"))
            if dep_end is not None:
                dependency_ends.append(dep_end)

    if not dependency_ends:
        return None

    latest_dependency_end = max(dependency_ends)
    return max(0.0, (begin - latest_dependency_end).total_seconds() * 1000.0)


def decision_for_fraction(pct_span_plus_gap: float | None) -> str:
    if pct_span_plus_gap is None:
        return "unknown"
    if pct_span_plus_gap > 10.0:
        return "pursue simplification"
    if pct_span_plus_gap < 1.0:
        return "not worth simplification"
    return "measure more / workload-specific"


def analyze_query(
    log_analysis_dir: Path,
    index_row: dict[str, str],
    aggregate_rows: list[dict[str, str]],
    target_operators: set[str],
) -> tuple[QuerySummary, list[PipelineMetric]]:
    folder = index_row["folder"]
    qdir = log_analysis_dir / folder
    plan = load_plan(qdir)
    duration_ms = parse_float(index_row.get("duration_ms"))

    aggregate_by_id = {
        parse_int(row["pipeline_id"]): row
        for row in aggregate_rows
        if row.get("pipeline_id") not in (None, "")
    }

    metrics: list[PipelineMetric] = []
    if plan is not None:
        for pipeline in plan.get("pipelines", []):
            pid = pipeline["pipeline_num"]
            row = aggregate_by_id.get(pid)
            if row is None:
                continue

            present_targets = sorted(pipeline_operator_types(pipeline) & target_operators)
            if not present_targets:
                continue

            span = ms_between(row.get("pipeline_begin"), row.get("pipeline_end"))
            gap = dependency_gap_ms(pipeline, aggregate_by_id)
            metrics.append(
                PipelineMetric(
                    pipeline_id=pid,
                    operator_chain=operator_chain(pipeline),
                    target_operators=",".join(present_targets),
                    num_tasks=parse_int(row.get("num_tasks")),
                    execution_sum_ms=parse_float(row.get("sum_execution_time_ms")) or 0.0,
                    execution_max_task_ms=parse_float(row.get("max_execution_time_ms")) or 0.0,
                    span_ms=span,
                    dependency_gap_ms=gap,
                    input_rows=parse_int(row.get("sum_input_num_rows")),
                    output_rows=parse_int(row.get("sum_output_num_rows")),
                    input_bytes=parse_int(row.get("sum_input_size_bytes")),
                    output_bytes=parse_int(row.get("sum_output_size_bytes")),
                )
            )
    else:
        # Plan is missing; fall back to aggregate operator_types. Dependency gaps
        # cannot be calculated without the plan DAG.
        for row in aggregate_rows:
            operator_types = set((row.get("operator_types") or "").split(","))
            present_targets = sorted(operator_types & target_operators)
            if not present_targets:
                continue
            metrics.append(
                PipelineMetric(
                    pipeline_id=parse_int(row.get("pipeline_id")),
                    operator_chain=row.get("operator_types") or "",
                    target_operators=",".join(present_targets),
                    num_tasks=parse_int(row.get("num_tasks")),
                    execution_sum_ms=parse_float(row.get("sum_execution_time_ms")) or 0.0,
                    execution_max_task_ms=parse_float(row.get("max_execution_time_ms")) or 0.0,
                    span_ms=ms_between(row.get("pipeline_begin"), row.get("pipeline_end")),
                    dependency_gap_ms=None,
                    input_rows=parse_int(row.get("sum_input_num_rows")),
                    output_rows=parse_int(row.get("sum_output_num_rows")),
                    input_bytes=parse_int(row.get("sum_input_size_bytes")),
                    output_bytes=parse_int(row.get("sum_output_size_bytes")),
                )
            )

    target_execution_sum_ms = sum(metric.execution_sum_ms for metric in metrics)
    target_span_ms = sum(metric.span_ms or 0.0 for metric in metrics)
    target_gap_ms = sum(metric.dependency_gap_ms or 0.0 for metric in metrics)
    target_span_plus_gap_ms = target_span_ms + target_gap_ms

    pct_execution_sum = (
        (target_execution_sum_ms / duration_ms) * 100.0
        if duration_ms and duration_ms > 0
        else None
    )
    pct_span_plus_gap = (
        (target_span_plus_gap_ms / duration_ms) * 100.0
        if duration_ms and duration_ms > 0
        else None
    )

    summary = QuerySummary(
        query_begin_ts=index_row["query_begin_ts"],
        folder=folder,
        status=index_row.get("status", ""),
        duration_ms=duration_ms,
        sql_preview=index_row.get("sql_preview", ""),
        target_pipeline_count=len(metrics),
        target_execution_sum_ms=target_execution_sum_ms,
        target_span_ms=target_span_ms,
        target_gap_ms=target_gap_ms,
        target_span_plus_gap_ms=target_span_plus_gap_ms,
        pct_execution_sum=pct_execution_sum,
        pct_span_plus_gap=pct_span_plus_gap,
        decision=decision_for_fraction(pct_span_plus_gap),
    )
    return summary, metrics


def summary_to_row(summary: QuerySummary) -> dict:
    return {
        "query_begin_ts": summary.query_begin_ts,
        "folder": summary.folder,
        "status": summary.status,
        "duration_ms": summary.duration_ms,
        "target_pipeline_count": summary.target_pipeline_count,
        "target_execution_sum_ms": summary.target_execution_sum_ms,
        "target_span_ms": summary.target_span_ms,
        "target_gap_ms": summary.target_gap_ms,
        "target_span_plus_gap_ms": summary.target_span_plus_gap_ms,
        "pct_execution_sum": summary.pct_execution_sum,
        "pct_span_plus_gap": summary.pct_span_plus_gap,
        "decision": summary.decision,
        "sql_preview": summary.sql_preview,
    }


def metric_to_row(summary: QuerySummary, metric: PipelineMetric) -> dict:
    return {
        "query_begin_ts": summary.query_begin_ts,
        "folder": summary.folder,
        "pipeline_id": metric.pipeline_id,
        "target_operators": metric.target_operators,
        "operator_chain": metric.operator_chain,
        "num_tasks": metric.num_tasks,
        "execution_sum_ms": metric.execution_sum_ms,
        "execution_max_task_ms": metric.execution_max_task_ms,
        "span_ms": metric.span_ms,
        "dependency_gap_ms": metric.dependency_gap_ms,
        "input_rows": metric.input_rows,
        "output_rows": metric.output_rows,
        "input_bytes": metric.input_bytes,
        "output_bytes": metric.output_bytes,
    }


def write_markdown_report(
    path: Path,
    log_analysis_dir: Path,
    target_operators: list[str],
    summaries: list[QuerySummary],
    details_by_query: dict[str, list[PipelineMetric]],
) -> None:
    lines = [
        "# Small-Data Pipeline Overhead",
        "",
        f"- Parsed log analysis: `{log_analysis_dir}`",
        f"- Target operators: `{', '.join(target_operators)}`",
        "",
        "## Interpretation Notes",
        "",
        "- `target_execution_sum_ms` is summed task execution time. It can exceed wall-clock contribution when tasks run in parallel.",
        "- `target_span_plus_gap_ms` is closer to the wall-clock overhead question: target pipeline active span plus dependency gap before those pipelines.",
        "- The decision rule uses `target_span_plus_gap_ms / query_duration_ms`: `<1%` not worth simplifying, `>10%` worth pursuing, otherwise workload-specific.",
        "",
        "## Query Summary",
        "",
        "| Query Folder | Duration | Target Span+Gap | % Span+Gap | Target Exec Sum | % Exec Sum | Decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for summary in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{summary.folder}`",
                    fmt_ms(summary.duration_ms),
                    fmt_ms(summary.target_span_plus_gap_ms),
                    fmt_pct(summary.pct_span_plus_gap),
                    fmt_ms(summary.target_execution_sum_ms),
                    fmt_pct(summary.pct_execution_sum),
                    summary.decision,
                ]
            )
            + " |"
        )

    lines.extend(["", "## Target Pipeline Details", ""])
    for summary in summaries:
        lines.extend(
            [
                f"### `{summary.folder}`",
                "",
                f"SQL preview: `{summary.sql_preview}`",
                "",
            ]
        )
        metrics = details_by_query.get(summary.query_begin_ts, [])
        if not metrics:
            lines.extend(["No target pipelines found.", ""])
            continue
        lines.extend(
            [
                "| Pipeline | Operators | Span | Gap | Exec Sum | Tasks | Rows In -> Out |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for metric in metrics:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(metric.pipeline_id),
                        f"`{metric.operator_chain}`",
                        fmt_ms(metric.span_ms),
                        fmt_ms(metric.dependency_gap_ms),
                        fmt_ms(metric.execution_sum_ms),
                        str(metric.num_tasks),
                        f"{metric.input_rows} -> {metric.output_rows}",
                    ]
                )
                + " |"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n")


def print_console_summary(summaries: list[QuerySummary]) -> None:
    print("Small-data pipeline overhead summary")
    print("------------------------------------")
    for summary in summaries:
        print(
            f"{summary.folder}: "
            f"span+gap={fmt_ms(summary.target_span_plus_gap_ms)} "
            f"({fmt_pct(summary.pct_span_plus_gap)}), "
            f"exec_sum={fmt_ms(summary.target_execution_sum_ms)} "
            f"({fmt_pct(summary.pct_execution_sum)}), "
            f"decision={summary.decision}"
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize PARTITION/MERGE pipeline overhead from parsed Sirius logs."
    )
    parser.add_argument(
        "log_analysis_dir",
        type=Path,
        help="Directory produced by tools/log_analyzer/parse_logs.py",
    )
    parser.add_argument(
        "--operators",
        nargs="+",
        default=list(DEFAULT_TARGET_OPERATORS),
        help="Operator types to treat as target overhead pipelines.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Defaults to the input log_analysis_dir.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    log_analysis_dir = args.log_analysis_dir
    out_dir = args.out or log_analysis_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    index_path = log_analysis_dir / "_index.csv"
    aggregate_path = log_analysis_dir / "_pipeline_aggregates.csv"
    if not index_path.exists() or not aggregate_path.exists():
        print(
            "ERROR: expected _index.csv and _pipeline_aggregates.csv in "
            f"{log_analysis_dir}",
            file=sys.stderr,
        )
        return 2

    target_operators = set(args.operators)
    index_rows = read_csv(index_path)
    aggregate_rows = read_csv(aggregate_path)
    aggregates_by_query: dict[str, list[dict[str, str]]] = {}
    for row in aggregate_rows:
        aggregates_by_query.setdefault(row["query_begin_ts"], []).append(row)

    summaries: list[QuerySummary] = []
    details_by_query: dict[str, list[PipelineMetric]] = {}
    detail_rows: list[dict] = []

    for index_row in index_rows:
        query_begin_ts = index_row["query_begin_ts"]
        summary, metrics = analyze_query(
            log_analysis_dir=log_analysis_dir,
            index_row=index_row,
            aggregate_rows=aggregates_by_query.get(query_begin_ts, []),
            target_operators=target_operators,
        )
        summaries.append(summary)
        details_by_query[query_begin_ts] = metrics
        detail_rows.extend(metric_to_row(summary, metric) for metric in metrics)

    summary_rows = [summary_to_row(summary) for summary in summaries]
    write_csv(
        out_dir / "small_data_overhead.csv",
        [
            "query_begin_ts",
            "folder",
            "status",
            "duration_ms",
            "target_pipeline_count",
            "target_execution_sum_ms",
            "target_span_ms",
            "target_gap_ms",
            "target_span_plus_gap_ms",
            "pct_execution_sum",
            "pct_span_plus_gap",
            "decision",
            "sql_preview",
        ],
        summary_rows,
    )
    write_csv(
        out_dir / "small_data_overhead_details.csv",
        [
            "query_begin_ts",
            "folder",
            "pipeline_id",
            "target_operators",
            "operator_chain",
            "num_tasks",
            "execution_sum_ms",
            "execution_max_task_ms",
            "span_ms",
            "dependency_gap_ms",
            "input_rows",
            "output_rows",
            "input_bytes",
            "output_bytes",
        ],
        detail_rows,
    )
    write_markdown_report(
        out_dir / "SMALL_DATA_OVERHEAD.md",
        log_analysis_dir,
        list(args.operators),
        summaries,
        details_by_query,
    )
    print_console_summary(summaries)
    print()
    print(f"Wrote {out_dir / 'small_data_overhead.csv'}")
    print(f"Wrote {out_dir / 'small_data_overhead_details.csv'}")
    print(f"Wrote {out_dir / 'SMALL_DATA_OVERHEAD.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

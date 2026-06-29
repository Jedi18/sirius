#!/usr/bin/env python3
"""Summarize small-data pipeline overhead from parsed Sirius log analysis.

Input is the output directory produced by:

    python tools/log_analyzer/parse_logs.py <sirius.log> --out <log_analysis_dir>

Default mode is intentionally specific to the grouped-aggregate simplification
proposal: count only MERGE_GROUP_BY pipelines and the PARTITION pipeline(s) that
directly feed them. This avoids counting hash-join PARTITION pipelines as false
positives.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


DEFAULT_OPERATORS = ("PARTITION", "MERGE_GROUP_BY")


@dataclass(frozen=True)
class PipelineMetric:
    pipeline_id: int
    target_reason: str
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
    return "n/a" if value is None else f"{value:.2f} ms"


def fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


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
    dep_ids = list(pipeline.get("dependencies", []))
    dep_ids.extend(edge.get("from_pipeline") for edge in pipeline.get("inputs", []))
    for dep_id in dep_ids:
        if dep_id is None:
            continue
        dep_end = parse_timestamp(aggregate_by_id.get(int(dep_id), {}).get("pipeline_end"))
        if dep_end is not None:
            dependency_ends.append(dep_end)

    if not dependency_ends:
        return None
    return max(0.0, (begin - max(dependency_ends)).total_seconds() * 1000.0)


def decision_for_fraction(pct_span_plus_gap: float | None) -> str:
    if pct_span_plus_gap is None:
        return "unknown"
    if pct_span_plus_gap > 10.0:
        return "pursue simplification"
    if pct_span_plus_gap < 1.0:
        return "not worth simplification"
    return "measure more / workload-specific"


def grouped_aggregate_target_ids(plan: dict) -> dict[int, str]:
    """Return target pipeline ids for HASH_GROUP_BY -> PARTITION -> MERGE_GROUP_BY.

    Include each MERGE_GROUP_BY pipeline and only its direct PARTITION inputs.
    This intentionally excludes join-side PARTITION pipelines.
    """
    pipelines = {p["pipeline_num"]: p for p in plan.get("pipelines", [])}
    target_ids: dict[int, str] = {}
    for pipeline in plan.get("pipelines", []):
        if "MERGE_GROUP_BY" not in pipeline_operator_types(pipeline):
            continue
        merge_id = pipeline["pipeline_num"]
        target_ids[merge_id] = "MERGE_GROUP_BY"
        dep_ids = list(pipeline.get("dependencies", []))
        dep_ids.extend(edge.get("from_pipeline") for edge in pipeline.get("inputs", []))
        for dep_id in dep_ids:
            if dep_id is None:
                continue
            dep = pipelines.get(int(dep_id))
            if dep and "PARTITION" in pipeline_operator_types(dep):
                target_ids[int(dep_id)] = "PARTITION feeding MERGE_GROUP_BY"
    return target_ids


def operator_target_ids(plan: dict, target_operators: set[str]) -> dict[int, str]:
    target_ids = {}
    for pipeline in plan.get("pipelines", []):
        present = sorted(pipeline_operator_types(pipeline) & target_operators)
        if present:
            target_ids[pipeline["pipeline_num"]] = ",".join(present)
    return target_ids


def make_metric(
    pipeline: dict,
    row: dict[str, str],
    target_reason: str,
) -> PipelineMetric:
    return PipelineMetric(
        pipeline_id=pipeline["pipeline_num"],
        target_reason=target_reason,
        operator_chain=operator_chain(pipeline),
        target_operators=",".join(sorted(pipeline_operator_types(pipeline) & set(DEFAULT_OPERATORS))),
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


def analyze_query(
    log_analysis_dir: Path,
    index_row: dict[str, str],
    aggregate_rows: list[dict[str, str]],
    mode: str,
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
        target_ids = (
            grouped_aggregate_target_ids(plan)
            if mode == "grouped-aggregate"
            else operator_target_ids(plan, target_operators)
        )
        pipelines = {p["pipeline_num"]: p for p in plan.get("pipelines", [])}
        for pid, reason in sorted(target_ids.items()):
            pipeline = pipelines.get(pid)
            row = aggregate_by_id.get(pid)
            if pipeline is None or row is None:
                continue
            metric = make_metric(pipeline, row, reason)
            metrics.append(
                PipelineMetric(
                    **{
                        **metric.__dict__,
                        "target_operators": ",".join(
                            sorted(pipeline_operator_types(pipeline) & target_operators)
                        ),
                        "dependency_gap_ms": dependency_gap_ms(pipeline, aggregate_by_id),
                    }
                )
            )
    else:
        for row in aggregate_rows:
            operator_types = set((row.get("operator_types") or "").split(","))
            present = sorted(operator_types & target_operators)
            if not present:
                continue
            pid = parse_int(row.get("pipeline_id"))
            metrics.append(
                PipelineMetric(
                    pipeline_id=pid,
                    target_reason="operator match (plan missing)",
                    operator_chain=row.get("operator_types") or "",
                    target_operators=",".join(present),
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
        target_execution_sum_ms / duration_ms * 100.0 if duration_ms and duration_ms > 0 else None
    )
    pct_span_plus_gap = (
        target_span_plus_gap_ms / duration_ms * 100.0
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
    return summary.__dict__


def metric_to_row(summary: QuerySummary, metric: PipelineMetric) -> dict:
    return {
        "query_begin_ts": summary.query_begin_ts,
        "folder": summary.folder,
        **metric.__dict__,
    }


def write_markdown_report(
    path: Path,
    log_analysis_dir: Path,
    mode: str,
    target_operators: list[str],
    summaries: list[QuerySummary],
    details_by_query: dict[str, list[PipelineMetric]],
) -> None:
    pct_values = [s.pct_span_plus_gap for s in summaries if s.pct_span_plus_gap is not None]
    median_pct = statistics.median(pct_values) if pct_values else None
    max_pct = max(pct_values) if pct_values else None
    lines = [
        "# Small-Data Pipeline Overhead",
        "",
        f"- Parsed log analysis: `{log_analysis_dir}`",
        f"- Mode: `{mode}`",
        f"- Target operators: `{', '.join(target_operators)}`",
        f"- Median `% span+gap`: `{fmt_pct(median_pct)}`",
        f"- Max `% span+gap`: `{fmt_pct(max_pct)}`",
        "",
        "## Interpretation Notes",
        "",
        "- Default mode counts only `MERGE_GROUP_BY` plus the direct `PARTITION` pipeline feeding it.",
        "- `target_execution_sum_ms` is summed task execution time and can overcount wall-clock contribution when tasks run in parallel.",
        "- `target_span_plus_gap_ms` is closer to the wall-clock overhead question.",
        "- Decision rule: `<1%` not worth simplifying, `>10%` worth pursuing, otherwise workload-specific.",
        "",
        "## Query Summary",
        "",
        "| Query Folder | Duration | Span+Gap | % Span+Gap | Exec Sum | % Exec Sum | Decision |",
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
        lines.extend([f"### `{summary.folder}`", "", f"SQL preview: `{summary.sql_preview}`", ""])
        metrics = details_by_query.get(summary.query_begin_ts, [])
        if not metrics:
            lines.extend(["No target pipelines found.", ""])
            continue
        lines.extend(
            [
                "| Pipeline | Reason | Operators | Span | Gap | Exec Sum | Tasks | Rows In -> Out |",
                "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for metric in metrics:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(metric.pipeline_id),
                        metric.target_reason,
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
            f"{summary.folder}: span+gap={fmt_ms(summary.target_span_plus_gap_ms)} "
            f"({fmt_pct(summary.pct_span_plus_gap)}), "
            f"exec_sum={fmt_ms(summary.target_execution_sum_ms)} "
            f"({fmt_pct(summary.pct_execution_sum)}), decision={summary.decision}"
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize small-data pipeline overhead from parsed Sirius logs."
    )
    parser.add_argument("log_analysis_dir", type=Path)
    parser.add_argument(
        "--mode",
        choices=("grouped-aggregate", "operators"),
        default="grouped-aggregate",
        help="grouped-aggregate counts only MERGE_GROUP_BY and its direct PARTITION input.",
    )
    parser.add_argument(
        "--operators",
        nargs="+",
        default=list(DEFAULT_OPERATORS),
        help="Operator types used in --mode operators, and for detail labels.",
    )
    parser.add_argument("--out", type=Path, default=None)
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
            f"ERROR: expected _index.csv and _pipeline_aggregates.csv in {log_analysis_dir}",
            file=sys.stderr,
        )
        return 2

    index_rows = read_csv(index_path)
    aggregate_rows = read_csv(aggregate_path)
    aggregates_by_query: dict[str, list[dict[str, str]]] = {}
    for row in aggregate_rows:
        aggregates_by_query.setdefault(row["query_begin_ts"], []).append(row)

    target_operators = set(args.operators)
    summaries: list[QuerySummary] = []
    details_by_query: dict[str, list[PipelineMetric]] = {}
    detail_rows: list[dict] = []

    for index_row in index_rows:
        query_begin_ts = index_row["query_begin_ts"]
        summary, metrics = analyze_query(
            log_analysis_dir=log_analysis_dir,
            index_row=index_row,
            aggregate_rows=aggregates_by_query.get(query_begin_ts, []),
            mode=args.mode,
            target_operators=target_operators,
        )
        summaries.append(summary)
        details_by_query[query_begin_ts] = metrics
        detail_rows.extend(metric_to_row(summary, metric) for metric in metrics)

    write_csv(
        out_dir / "small_data_overhead.csv",
        [
            "query_begin_ts",
            "folder",
            "status",
            "duration_ms",
            "sql_preview",
            "target_pipeline_count",
            "target_execution_sum_ms",
            "target_span_ms",
            "target_gap_ms",
            "target_span_plus_gap_ms",
            "pct_execution_sum",
            "pct_span_plus_gap",
            "decision",
        ],
        [summary_to_row(summary) for summary in summaries],
    )
    write_csv(
        out_dir / "small_data_overhead_details.csv",
        [
            "query_begin_ts",
            "folder",
            "pipeline_id",
            "target_reason",
            "operator_chain",
            "target_operators",
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
        args.mode,
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
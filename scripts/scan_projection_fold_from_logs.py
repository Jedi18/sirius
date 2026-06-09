#!/usr/bin/env python3
"""Scan Sirius execution logs for queries with adjacent PROJECTION operators.

Use this on a dev-branch e2e run to discover projection-fold candidates, then
re-run the same queries on your feature branch and compare.

A query is a candidate when its logged Query Plan (Pipeline Overview) contains
at least one adjacent PROJECTION -> PROJECTION pair in a pipeline operator chain.
On a branch with projection folding enabled, those pairs should disappear.

Requires logs captured with SIRIUS_LOG_LEVEL=info (default) so each query emits
the "Query Plan:" block from sirius_engine.cpp.

Usage:
  # Discover candidates from one log file or a benchmark/integration log directory
  python3 scripts/scan_projection_fold_from_logs.py LOG_PATH \\
      --out projection_fold_candidates.tsv

  # Scan a benchmark run (per-query logs under q*/sirius.log)
  python3 scripts/scan_projection_fold_from_logs.py runs/2026-06-09_sf1_2iter/sirius/

  # After re-running on your feature branch, verify folding
  python3 scripts/scan_projection_fold_from_logs.py NEW_LOG_PATH \\
      --compare projection_fold_candidates.tsv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

# Reuse the log segmenter when available (no trace-level requirement here).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from tools.log_analyzer import patterns, plan_parser, segmenter
    from tools.log_analyzer.validators import FormatWarnings

    _HAS_LOG_ANALYZER = True
except ImportError:
    _HAS_LOG_ANALYZER = False

PIPELINE_OVERVIEW_HEADER = "=== Pipeline Overview ==="
QUERY_PLAN_ANCHOR = "Query Plan:"
ADJACENT_PROJECTION_RE = re.compile(
    r"PROJECTION \(id=\d+\)\s*->\s*PROJECTION \(id=\d+\)"
)
OPERATOR_RE = re.compile(r"(\w+) \(id=(\d+)\)")
PIPELINE_HEADER_RE = re.compile(r"^Pipeline #(\d+): (.+)$")


@dataclass
class QueryFoldStats:
    log_path: str
    query_key: str
    sql: str
    adjacent_pairs: int = 0
    total_projections: int = 0
    pipelines_with_adjacent: List[int] = field(default_factory=list)
    has_query_plan: bool = False
    status: str = "unknown"

    @property
    def candidate(self) -> bool:
        return self.adjacent_pairs > 0


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.split()).strip().rstrip(";").lower()


def _strip_log_prefix(line: str) -> str:
    """Drop leading [timestamp] [level] [file:line] prefixes from a log line."""
    s = line.rstrip("\n")
    while True:
        if s.startswith("[") and "]" in s:
            s = s[s.index("]") + 1 :].lstrip()
            continue
        return s


def _extract_plan_lines(segment_lines: Sequence[str]) -> List[str]:
    """Return de-prefixed lines inside the Query Plan / Pipeline Overview block."""
    in_plan = False
    in_overview = False
    block: List[str] = []

    for raw in segment_lines:
        body = _strip_log_prefix(raw)
        if QUERY_PLAN_ANCHOR in body:
            in_plan = True
            continue
        if not in_plan:
            continue
        if PIPELINE_OVERVIEW_HEADER in body:
            in_overview = True
            block.append(PIPELINE_OVERVIEW_HEADER)
            continue
        if not in_overview:
            continue
        if not body.strip():
            break
        if "=== Query Plan DAG ===" in body:
            break
        block.append(body)

    return block


def _count_from_overview_block(block: Sequence[str]) -> Tuple[int, int, List[int]]:
    adjacent = 0
    total_projections = 0
    pipelines_with_adjacent: List[int] = []

    for raw in block:
        m = PIPELINE_HEADER_RE.match(raw)
        if not m:
            continue
        pipeline_num = int(m.group(1))
        chain = m.group(2)
        adjacent += len(ADJACENT_PROJECTION_RE.findall(chain))
        ops = [op for op, _ in OPERATOR_RE.findall(chain)]
        total_projections += sum(1 for op in ops if op == "PROJECTION")
        if ADJACENT_PROJECTION_RE.search(chain):
            pipelines_with_adjacent.append(pipeline_num)

    return adjacent, total_projections, pipelines_with_adjacent


def _analyze_segment(
    log_path: Path,
    query_key: str,
    sql: str,
    segment_lines: Sequence[str],
    status: str,
) -> QueryFoldStats:
    block = _extract_plan_lines(segment_lines)
    adjacent, total_projections, pipelines = _count_from_overview_block(block)
    return QueryFoldStats(
        log_path=str(log_path),
        query_key=query_key,
        sql=sql,
        adjacent_pairs=adjacent,
        total_projections=total_projections,
        pipelines_with_adjacent=pipelines,
        has_query_plan=bool(block),
        status=status,
    )


def _segment_with_log_analyzer(
    lines: List[str],
) -> List[Tuple[str, str, List[str], str]]:
    segments = segmenter.segment(lines)
    return [(seg.begin_ts, seg.sql, seg.lines, seg.status) for seg in segments]


def _segment_fallback(lines: List[str]) -> List[Tuple[str, str, List[str], str]]:
    """Minimal segmenter when tools.log_analyzer is unavailable."""
    if not _HAS_LOG_ANALYZER:
        query_begin_anchor = "[info] [:] [query_pool] QueryBegin allocated="
        query_end_anchor = "[info] [:] [query_pool] QueryEnd allocated="
        query_sql_anchor = "QueryBegin: "
        query_sql_re = re.compile(
            r"\[[\d\-: .]+\] \[info\] \[[^\]]+\] QueryBegin: (?P<sql>.*)$"
        )
    else:
        query_begin_anchor = patterns.QUERY_BEGIN_ANCHOR
        query_end_anchor = patterns.QUERY_END_ANCHOR
        query_sql_anchor = patterns.QUERY_SQL_ANCHOR
        query_sql_re = patterns.QUERY_SQL_RE

    out: List[Tuple[str, str, List[str], str]] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if query_begin_anchor not in line:
            i += 1
            continue
        begin_ts = line
        sql = ""
        sql_idx = None
        for j in range(i + 1, min(i + 5, n)):
            if query_sql_anchor in lines[j]:
                m = query_sql_re.match(lines[j])
                sql = m.group("sql") if m else lines[j].split(query_sql_anchor, 1)[1]
                sql_idx = j
                break
        if sql_idx is None:
            i += 1
            continue
        stripped = sql.lstrip().lower()
        if not stripped.startswith(("select ", "with ", "call ")):
            i += 1
            continue

        end_idx = None
        stop_idx = None
        for k in range(sql_idx + 1, n):
            if query_end_anchor in lines[k]:
                end_idx = k
                stop_idx = k
                break
            if query_begin_anchor in lines[k]:
                stop_idx = k - 1
                break
        if stop_idx is None:
            stop_idx = n - 1
        status = "complete" if end_idx is not None else "incomplete"
        out.append((begin_ts, sql.strip(), lines[i : stop_idx + 1], status))
        i = stop_idx + 1
    return out


def _read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)


def _infer_query_key(log_path: Path, begin_ts: str, sql: str) -> str:
    # Benchmark runs store per-query logs under .../q<N>/sirius.log
    for part in reversed(log_path.parts):
        m = re.fullmatch(r"q(\d+)", part)
        if m:
            return f"q{m.group(1)}"
    # Fall back to timestamp + short SQL hash prefix
    ts = begin_ts
    if "]" in ts:
        ts = ts[ts.index("[") + 1 :]
        if "]" in ts:
            ts = ts[: ts.index("]")]
    preview = _normalize_sql(sql)[:48].replace("\t", " ")
    return f"{ts}:{preview}"


def scan_log_file(log_path: Path) -> List[QueryFoldStats]:
    lines = _read_lines(log_path)
    if _HAS_LOG_ANALYZER:
        segments = _segment_with_log_analyzer(lines)
    else:
        segments = _segment_fallback(lines)

    # Per-query benchmark logs may not have QueryBegin markers — treat whole file as one query.
    if not segments:
        sql = ""
        sibling = log_path.parent / "query.sql"
        if sibling.is_file():
            sql = sibling.read_text(encoding="utf-8", errors="replace").strip()
        key = _infer_query_key(log_path, log_path.name, sql or log_path.stem)
        return [_analyze_segment(log_path, key, sql, lines, "single_log")]

    results: List[QueryFoldStats] = []
    for begin_ts, sql, seg_lines, status in segments:
        key = _infer_query_key(log_path, begin_ts, sql)
        results.append(_analyze_segment(log_path, key, sql, seg_lines, status))
    return results


def discover_log_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    logs: List[Path] = []
    for pattern in ("sirius*.log", "*.log"):
        logs.extend(sorted(path.rglob(pattern)))
    # De-duplicate while preserving order
    seen = set()
    unique: List[Path] = []
    for p in logs:
        if p in seen:
            continue
        seen.add(p)
        unique.append(p)
    return unique


def scan_paths(paths: Sequence[Path]) -> List[QueryFoldStats]:
    all_stats: List[QueryFoldStats] = []
    for root in paths:
        for log_file in discover_log_files(root):
            all_stats.extend(scan_log_file(log_file))
    return all_stats


def write_report(path: Path, stats: Sequence[QueryFoldStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            [
                "log_path",
                "query_key",
                "candidate",
                "adjacent_pairs",
                "total_projections",
                "pipelines_with_adjacent",
                "has_query_plan",
                "status",
                "sql",
            ]
        )
        for row in stats:
            w.writerow(
                [
                    row.log_path,
                    row.query_key,
                    "true" if row.candidate else "false",
                    row.adjacent_pairs,
                    row.total_projections,
                    ",".join(str(p) for p in row.pipelines_with_adjacent),
                    "true" if row.has_query_plan else "false",
                    row.status,
                    row.sql,
                ]
            )


def write_candidates(path: Path, stats: Sequence[QueryFoldStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write("# query_key\tsql\n")
        seen = set()
        for row in stats:
            if not row.candidate:
                continue
            key = (_normalize_sql(row.sql), row.query_key)
            if key in seen:
                continue
            seen.add(key)
            f.write(f"{row.query_key}\t{row.sql}\n")


def load_candidates(path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t", 1)
        if len(parts) == 2:
            rows.append((parts[0], parts[1]))
        elif len(parts) == 1:
            rows.append((parts[0], ""))
    return rows


def compare_with_baseline(
    stats: Sequence[QueryFoldStats],
    baseline_path: Path,
) -> int:
    baseline = load_candidates(baseline_path)
    if not baseline:
        print(f"No baseline candidates in {baseline_path}", file=sys.stderr)
        return 1

    by_sql = {_normalize_sql(s.sql): s for s in stats if s.sql}
    by_key = {s.query_key: s for s in stats}

    print(
        f"Comparing {len(baseline)} baseline candidates against {len(stats)} scanned queries\n"
    )
    print(f"{'query_key':<20} {'baseline_adj':>12} {'current_adj':>12} {'folded':>8}")
    print("-" * 56)

    folded = 0
    missing = 0
    for key, sql in baseline:
        norm = _normalize_sql(sql) if sql else ""
        row = by_sql.get(norm) if norm else None
        if row is None and key in by_key:
            row = by_key[key]
        if row is None:
            print(f"{key:<20} {'?':>12} {'?':>12} {'MISSING':>8}")
            missing += 1
            continue
        is_folded = row.adjacent_pairs == 0
        if is_folded:
            folded += 1
        print(
            f"{key:<20} {'>0':>12} {row.adjacent_pairs:>12} "
            f"{'yes' if is_folded else 'no':>8}"
        )

    print()
    print(f"Folded: {folded}/{len(baseline)}  Missing: {missing}")
    return 0 if missing == 0 else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_paths",
        nargs="+",
        type=Path,
        help="Sirius log file(s) or directories (searched recursively for sirius*.log)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("projection_fold_scan_report.tsv"),
        help="Full TSV report (default: projection_fold_scan_report.tsv)",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("projection_fold_candidates.tsv"),
        help="Candidate-only TSV (default: projection_fold_candidates.tsv)",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        metavar="BASELINE_TSV",
        help="Compare scan results against a baseline candidates file from dev",
    )
    args = parser.parse_args(argv)

    stats = scan_paths(args.log_paths)
    if not stats:
        print("No log files found.", file=sys.stderr)
        return 1

    write_report(args.out, stats)
    write_candidates(args.candidates, stats)

    candidates = [s for s in stats if s.candidate]
    no_plan = [s for s in stats if not s.has_query_plan]

    print(f"Scanned {len(stats)} query segment(s) from {len(args.log_paths)} path(s)")
    print(f"Report:     {args.out}")
    print(f"Candidates: {args.candidates} ({len(candidates)} queries)")
    if no_plan:
        print(
            f"Warning: {len(no_plan)} segment(s) had no Query Plan block "
            f"(need SIRIUS_LOG_LEVEL=info)",
            file=sys.stderr,
        )

    if candidates:
        print("\nCandidates (adjacent PROJECTION -> PROJECTION):")
        for row in candidates:
            print(
                f"  {row.query_key}\tadjacent={row.adjacent_pairs}\t"
                f"projections={row.total_projections}"
            )
    else:
        print("\nNo projection-fold candidates found in these logs.")

    if args.compare:
        return compare_with_baseline(stats, args.compare)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env bash
# Run Sirius TPC-H queries, parse trace logs, and summarize small-data pipeline overhead.
#
# Example:
#   bash tools/log_analyzer/run_small_data_overhead_sweep.sh 1 1 3 5 9
#
# Outputs:
#   small_data_overhead_runs/<timestamp>_sf<SF>/
#     raw/q<N>/sirius.log
#     analysis/q<N>/...
#     combined_small_data_overhead.csv
#     combined_small_data_overhead_details.csv
#     SMALL_DATA_OVERHEAD_SUMMARY.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TPCH_RUNNER="$PROJECT_DIR/test/tpch_performance/run_tpch_parquet.sh"
PARSE_LOGS="$PROJECT_DIR/tools/log_analyzer/parse_logs.py"
SUMMARIZER="$PROJECT_DIR/tools/log_analyzer/summarize_small_data_overhead.py"

CONFIG_FILE="${SIRIUS_CONFIG_FILE:-$PROJECT_DIR/test/cpp/integration/integration.yaml}"
ITERATIONS=3
TIMEOUT=1200
OUTPUT_ROOT="$PROJECT_DIR/small_data_overhead_runs"
PARQUET_DIR=""
PINNING_MODE="none"
OPERATORS=(PARTITION MERGE_GROUP_BY)

usage() {
    cat <<'EOF'
Usage:
  bash tools/log_analyzer/run_small_data_overhead_sweep.sh [options] <scale_factor> <query_numbers...>

Options:
  --config <path>          Sirius config file. Default: $SIRIUS_CONFIG_FILE or test/cpp/integration/integration.yaml
  --iterations <N>         Query iterations. Default: 3
  --timeout <seconds>      DuckDB session timeout. Default: 1200
  --parquet-dir <path>     Existing TPC-H parquet directory. Default: runner chooses/generates test_datasets/tpch_parquet_sf<SF>
  --output-root <path>     Parent output directory. Default: ./small_data_overhead_runs
  --pinning-mode <mode>    Passed to run_tpch_parquet.sh. Default: none
  --operators <list...>    Target operators until -- or first numeric scale factor.
                           Default: PARTITION MERGE_GROUP_BY

Examples:
  bash tools/log_analyzer/run_small_data_overhead_sweep.sh 1 1
  bash tools/log_analyzer/run_small_data_overhead_sweep.sh 10 3 5 9 10
  bash tools/log_analyzer/run_small_data_overhead_sweep.sh --iterations 5 --operators PARTITION MERGE_GROUP_BY MERGE_AGGREGATE -- 10 1 9
EOF
}

is_integer() {
    [[ "${1:-}" =~ ^[0-9]+$ ]]
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            usage
            exit 0
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --parquet-dir)
            PARQUET_DIR="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --pinning-mode)
            PINNING_MODE="$2"
            shift 2
            ;;
        --operators)
            shift
            OPERATORS=()
            while [[ $# -gt 0 && "$1" != "--" && ! "$1" =~ ^[0-9]+$ ]]; do
                OPERATORS+=("$1")
                shift
            done
            if [[ "${1:-}" == "--" ]]; then
                shift
            fi
            if [[ ${#OPERATORS[@]} -eq 0 ]]; then
                echo "ERROR: --operators requires at least one operator name" >&2
                exit 2
            fi
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "ERROR: unknown option: $1" >&2
            usage
            exit 2
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -lt 2 ]]; then
    usage
    exit 2
fi

SF="$1"
shift
QUERIES=("$@")

if ! is_integer "$ITERATIONS" || [[ "$ITERATIONS" -lt 1 ]]; then
    echo "ERROR: --iterations must be a positive integer" >&2
    exit 2
fi

if [[ ! -x "$PROJECT_DIR/build/release/duckdb" ]]; then
    echo "ERROR: build/release/duckdb not found or not executable. Run: pixi run make" >&2
    exit 2
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Sirius config not found: $CONFIG_FILE" >&2
    exit 2
fi

timestamp="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_DIR="$OUTPUT_ROOT/${timestamp}_sf${SF}"
RAW_DIR="$RUN_DIR/raw"
ANALYSIS_DIR="$RUN_DIR/analysis"
mkdir -p "$RAW_DIR" "$ANALYSIS_DIR"

{
    echo "small-data overhead sweep"
    echo "timestamp=$timestamp"
    echo "scale_factor=$SF"
    echo "queries=${QUERIES[*]}"
    echo "iterations=$ITERATIONS"
    echo "timeout=$TIMEOUT"
    echo "config=$CONFIG_FILE"
    echo "parquet_dir=${PARQUET_DIR:-<runner default>}"
    echo "pinning_mode=$PINNING_MODE"
    echo "operators=${OPERATORS[*]}"
} > "$RUN_DIR/run_info.txt"

echo "Running Sirius queries with trace logging..."
echo "  run dir: $RUN_DIR"
echo "  queries: ${QUERIES[*]}"
echo "  target operators: ${OPERATORS[*]}"

runner_args=(--iterations "$ITERATIONS" --timeout "$TIMEOUT" --pinning-mode "$PINNING_MODE")
if [[ -n "$PARQUET_DIR" ]]; then
    runner_args=(--parquet-dir "$PARQUET_DIR" "${runner_args[@]}")
fi
runner_args+=(sirius "$SF" "${QUERIES[@]}")

(
    cd "$PROJECT_DIR"
    SIRIUS_CONFIG_FILE="$CONFIG_FILE" \
    SIRIUS_LOG_LEVEL=trace \
    OUTPUT_DIR="$RAW_DIR" \
    bash "$TPCH_RUNNER" "${runner_args[@]}"
) | tee "$RUN_DIR/runner_stdout.log"

echo ""
echo "Parsing logs and summarizing overhead..."

for q in "${QUERIES[@]}"; do
    query_log="$RAW_DIR/q${q}/sirius.log"
    query_out="$ANALYSIS_DIR/q${q}"

    if [[ ! -f "$query_log" ]]; then
        echo "WARNING: missing log for Q${q}: $query_log" >&2
        continue
    fi

    echo "  Q${q}: parsing $query_log"
    python3 "$PARSE_LOGS" "$query_log" --out "$query_out"

    echo "  Q${q}: summarizing target overhead"
    python3 "$SUMMARIZER" "$query_out" --operators "${OPERATORS[@]}" --out "$query_out" \
        > "$query_out/summarizer_stdout.log"
done

echo ""
echo "Combining per-query summaries..."
python3 - "$RUN_DIR" "${QUERIES[@]}" <<'PY'
import csv
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
queries = sys.argv[2:]
analysis_dir = run_dir / "analysis"

summary_rows = []
detail_rows = []
for q in queries:
    qdir = analysis_dir / f"q{q}"
    summary_path = qdir / "small_data_overhead.csv"
    detail_path = qdir / "small_data_overhead_details.csv"

    if summary_path.exists():
        with summary_path.open(newline="") as f:
            for row in csv.DictReader(f):
                row = {"query": f"Q{q}", **row}
                summary_rows.append(row)

    if detail_path.exists():
        with detail_path.open(newline="") as f:
            for row in csv.DictReader(f):
                row = {"query": f"Q{q}", **row}
                detail_rows.append(row)

def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

write_csv(run_dir / "combined_small_data_overhead.csv", summary_rows)
write_csv(run_dir / "combined_small_data_overhead_details.csv", detail_rows)

def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

lines = [
    "# Small-Data Overhead Sweep Summary",
    "",
    f"- Run directory: `{run_dir}`",
    f"- Queries: `{', '.join(f'Q{q}' for q in queries)}`",
    "",
    "## Query Results",
    "",
    "| Query | Iteration Folder | Duration | Span+Gap | % Span+Gap | Exec Sum | % Exec Sum | Decision |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
]

for row in summary_rows:
    duration = as_float(row.get("duration_ms"))
    span_gap = as_float(row.get("target_span_plus_gap_ms"))
    pct_span_gap = as_float(row.get("pct_span_plus_gap"))
    exec_sum = as_float(row.get("target_execution_sum_ms"))
    pct_exec_sum = as_float(row.get("pct_execution_sum"))
    lines.append(
        "| "
        + " | ".join(
            [
                row.get("query", ""),
                f"`{row.get('folder', '')}`",
                "n/a" if duration is None else f"{duration:.2f} ms",
                "n/a" if span_gap is None else f"{span_gap:.2f} ms",
                "n/a" if pct_span_gap is None else f"{pct_span_gap:.2f}%",
                "n/a" if exec_sum is None else f"{exec_sum:.2f} ms",
                "n/a" if pct_exec_sum is None else f"{pct_exec_sum:.2f}%",
                row.get("decision", ""),
            ]
        )
        + " |"
    )

lines.extend(
    [
        "",
        "## Outputs",
        "",
        "- `combined_small_data_overhead.csv`",
        "- `combined_small_data_overhead_details.csv`",
        "- Per-query parsed artifacts under `analysis/q<N>/`",
        "",
    ]
)

(run_dir / "SMALL_DATA_OVERHEAD_SUMMARY.md").write_text("\n".join(lines))
PY

echo ""
echo "Done."
echo "Summary: $RUN_DIR/SMALL_DATA_OVERHEAD_SUMMARY.md"
echo "Combined CSV: $RUN_DIR/combined_small_data_overhead.csv"
echo "Details CSV: $RUN_DIR/combined_small_data_overhead_details.csv"

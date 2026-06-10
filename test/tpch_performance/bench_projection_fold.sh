#!/usr/bin/env bash
# bench_projection_fold.sh
#
# A/B benchmark for the projection-folding optimization, on a SINGLE binary.
#
# Runs only the TPC-H queries identified as projection-fold candidates
# (the ones whose physical plan contains stacked PROJECTION -> PROJECTION
# operators that folding collapses) twice with the Sirius engine:
#
#   nofold  -> SIRIUS_DISABLE_PROJECTION_FOLDING=1  (folding turned off)
#   fold    -> folding active (default)
#
# The toggle is read by src/planner/sirius_plan_projection_utils.cpp, so no
# rebuild / branch swap is needed between the two runs. Both runs share the
# same warm cache behavior (1 cold + N-1 warm iterations per query), and the
# script prints a per-query warm-time comparison with speedup.
#
# Prerequisites:
#   pixi run make                                  # build with the toggle compiled in
#   export SIRIUS_CONFIG_FILE=$(pwd)/test/cpp/integration/integration.yaml
#
# Usage:
#   ./test/tpch_performance/bench_projection_fold.sh <scale_factor>
#   ./test/tpch_performance/bench_projection_fold.sh --iterations 5 100
#   ./test/tpch_performance/bench_projection_fold.sh --parquet-dir /data/tpch 100
#   ./test/tpch_performance/bench_projection_fold.sh --queries "1 9 17" 100

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run_tpch_parquet.sh"

# TPC-H queries identified as projection-fold candidates (from the candidate
# scan: Q1 Q2 Q7 Q8 Q9 Q11 Q13 Q16 Q17 Q19 Q20 Q22).
CANDIDATE_QUERIES="1 2 7 8 9 11 13 16 17 19 20 22"

ITERATIONS=3
PARQUET_DIR=""
QUERIES="$CANDIDATE_QUERIES"

while [ $# -gt 1 ]; do
    case "$1" in
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --parquet-dir) PARQUET_DIR="$2"; shift 2 ;;
        --queries) QUERIES="$2"; shift 2 ;;
        *) break ;;
    esac
done

if [ $# -ne 1 ]; then
    echo "Usage: $0 [--iterations N] [--parquet-dir <path>] [--queries \"1 9 17\"] <scale_factor>"
    exit 1
fi
SF="$1"

if [ -z "${SIRIUS_CONFIG_FILE:-}" ]; then
    echo "ERROR: SIRIUS_CONFIG_FILE must be set (the sirius engine needs it)."
    echo "  export SIRIUS_CONFIG_FILE=\$(pwd)/test/cpp/integration/integration.yaml"
    exit 1
fi

if [ ! -x "$PROJECT_DIR/build/release/duckdb" ]; then
    echo "ERROR: build/release/duckdb not found. Build first: pixi run make"
    exit 1
fi

RUN_DIR="$PROJECT_DIR/test/tpch_performance/runs/foldbench_$(date +%Y-%m-%d_%H-%M-%S)_sf${SF}"
mkdir -p "$RUN_DIR"

EXTRA_ARGS=(--iterations "$ITERATIONS")
[ -n "$PARQUET_DIR" ] && EXTRA_ARGS+=(--parquet-dir "$PARQUET_DIR")

echo "=========================================="
echo "Projection-fold A/B benchmark"
echo "  SF:         $SF"
echo "  Iterations: $ITERATIONS (1 cold + $((ITERATIONS - 1)) warm)"
echo "  Queries:    $QUERIES"
echo "  Run dir:    $RUN_DIR"
echo "=========================================="

# mode -> the value of SIRIUS_DISABLE_PROJECTION_FOLDING for that run.
for mode in nofold fold; do
    OUT="$RUN_DIR/$mode"
    mkdir -p "$OUT"
    echo ""
    echo "=== Running '$mode' ==="
    if [ "$mode" = "nofold" ]; then
        export SIRIUS_DISABLE_PROJECTION_FOLDING=1
    else
        unset SIRIUS_DISABLE_PROJECTION_FOLDING
    fi
    OUTPUT_DIR="$OUT" "$RUN_SCRIPT" "${EXTRA_ARGS[@]}" sirius "$SF" $QUERIES \
        2>&1 | tee "$OUT/run.log"
done
unset SIRIUS_DISABLE_PROJECTION_FOLDING

# ---------- Comparison ----------
# Warm time per query = min runtime over iterations 2..N (cold = iter_1 dropped).
warm_time() {
    local f="$1"
    [ -f "$f" ] || { echo "N/A"; return; }
    awk -F',' '
        $1 ~ /^iter_/ && $1 != "iter_1" && $2 != "" && $2 != "N/A" {
            v = $2 + 0; if (min == "" || v < min) min = v
        }
        END { print (min == "" ? "N/A" : min) }
    ' "$f"
}

COMPARISON_CSV="$RUN_DIR/comparison.csv"
echo "query,nofold_warm_s,fold_warm_s,speedup,delta_s" > "$COMPARISON_CSV"

{
echo ""
echo "============================================================"
printf "  Projection-fold warm-time comparison  (SF%s)\n" "$SF"
echo "============================================================"
printf "%-7s | %12s | %12s | %9s | %10s\n" \
    "Query" "NoFold (s)" "Fold (s)" "Speedup" "Delta (s)"
printf "%-7s-+-%12s-+-%12s-+-%9s-+-%10s\n" \
    "-------" "------------" "------------" "---------" "----------"

tot_nf=0; tot_f=0; have=0
for q in $QUERIES; do
    nf=$(warm_time "$RUN_DIR/nofold/q${q}/timings.csv")
    f=$(warm_time "$RUN_DIR/fold/q${q}/timings.csv")
    sp="N/A"; delta="N/A"
    if [ "$nf" != "N/A" ] && [ "$f" != "N/A" ]; then
        sp=$(echo "scale=3; $nf / $f" | bc 2>/dev/null)
        [ -n "$sp" ] && sp="${sp}x"
        delta=$(echo "scale=4; $nf - $f" | bc 2>/dev/null)
        tot_nf=$(echo "$tot_nf + $nf" | bc); tot_f=$(echo "$tot_f + $f" | bc); have=1
    fi
    printf "%-7s | %12s | %12s | %9s | %10s\n" "Q${q}" "$nf" "$f" "$sp" "$delta"
    echo "Q${q},${nf},${f},${sp},${delta}" >> "$COMPARISON_CSV"
done

printf "%-7s-+-%12s-+-%12s-+-%9s-+-%10s\n" \
    "-------" "------------" "------------" "---------" "----------"
if [ "$have" -eq 1 ]; then
    tsp=$(echo "scale=3; $tot_nf / $tot_f" | bc 2>/dev/null)
    tdelta=$(echo "scale=4; $tot_nf - $tot_f" | bc 2>/dev/null)
    printf "%-7s | %12s | %12s | %9s | %10s\n" \
        "TOTAL" "$tot_nf" "$tot_f" "${tsp}x" "$tdelta"
    echo "TOTAL,${tot_nf},${tot_f},${tsp}x,${tdelta}" >> "$COMPARISON_CSV"
fi
echo "============================================================"
echo "Speedup > 1.0x means folding is faster. Per-query results in $RUN_DIR"
} | tee "$RUN_DIR/comparison.txt"

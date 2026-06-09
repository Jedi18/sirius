#!/usr/bin/env bash
# Scan Sirius e2e logs for projection-fold candidates (adjacent PROJECTION pairs).
#
# Workflow:
#   1. On dev: run e2e with info-level logging, then scan logs → candidates TSV
#   2. On your feature branch: re-run the same e2e, scan again with --compare
#
# Usage:
#   # Step 1 — after running e2e on dev (see examples below)
#   ./scripts/scan_projection_fold_candidates.sh LOG_DIR_OR_FILE
#
#   # Step 2 — after re-running on your feature branch
#   ./scripts/scan_projection_fold_candidates.sh NEW_LOG_DIR --compare projection_fold_candidates.tsv
#
# Examples — capture logs while running e2e:
#
#   # Integration tests (~100 gpu_execution queries)
#   export SIRIUS_LOG_LEVEL=info
#   export SIRIUS_LOG_DIR=/tmp/sirius_logs_dev
#   rm -rf "$SIRIUS_LOG_DIR" && mkdir -p "$SIRIUS_LOG_DIR"
#   pixi run build/release/extension/sirius/test/cpp/sirius_unittest "[integration][gpu_execution]"
#   ./scripts/scan_projection_fold_candidates.sh "$SIRIUS_LOG_DIR"
#
#   # TPC-H benchmark (per-query logs under runs/<ts>/sirius/q*/sirius.log)
#   export SIRIUS_CONFIG_FILE=test/cpp/integration/integration.yaml
#   ./test/tpch_performance/benchmark_and_validate.sh 1
#   RUN_DIR=$(ls -td runs/*/ | head -1)
#   ./scripts/scan_projection_fold_candidates.sh "$RUN_DIR/sirius"
#
# Output:
#   projection_fold_scan_report.tsv   — full report
#   projection_fold_candidates.tsv    — queries with adjacent PROJECTION pairs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SCAN_PY="$SCRIPT_DIR/scan_projection_fold_from_logs.py"

if [ $# -lt 1 ]; then
  sed -n '2,32p' "$0"
  exit 1
fi

cd "$PROJECT_DIR"
python3 "$SCAN_PY" "$@"

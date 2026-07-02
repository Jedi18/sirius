# Cardinality-estimate accuracy analysis (issue #990)

Measures how accurate DuckDB's per-operator cardinality estimates are, to decide whether
the [#990](https://github.com/sirius-db/sirius/issues/990) "collapse operators when the
estimated data size is small" optimization is safe.

## Why this matters

#990 collapses operator chains (e.g. `grouped_aggregate -> partition -> merge_aggregate`
into a single `grouped_aggregate`; similar for ungrouped aggregates, joins, order-by) when
DuckDB estimates the data is small enough to fit in one task. That decision is made at
plan-construction time from DuckDB's estimate, which Sirius reads as
`op.estimated_cardinality` ([sirius_physical_plan_generator.cpp:117](../../src/planner/sirius_physical_plan_generator.cpp#L117)).

The two failure directions are **not** symmetric:

| Estimate vs reality | #990 consequence | Severity |
|---|---|---|
| **Under**-estimate (est ≤ threshold, actual ≫ threshold) | collapse, then the single task overflows → GPU OOM | **dangerous** |
| **Over**-estimate (est > threshold, actual ≤ threshold) | keep the multi-op pipeline unnecessarily | safe (missed perf) |

So the headline numbers are the **under-estimate rate** and the **worst actual/estimated
ratio** for the operator types #990 would collapse — not just an average error.

## How it works

DuckDB's JSON profiling emits estimated and actual rows for every operator in one run:

```json
{
  "operator_type": "HASH_GROUP_BY",
  "operator_cardinality": 4,                    // ACTUAL output rows
  "extra_info": { "__estimated_cardinality__": "6" },   // ESTIMATED (== op.estimated_cardinality)
  "children": [ ... ]
}
```

We run each query with `gpu_execution=false` so DuckDB executes normally and the full
per-operator tree is available (with `gpu_execution=true` Sirius collapses execution into
one opaque operator and the tree is lost). The *estimate* is DuckDB's optimizer output and
is identical either way, so this faithfully captures what #990 will act on.

Schema verified against this repo's `duckdb/` submodule — see the docstring in
[`cardinality_accuracy.py`](cardinality_accuracy.py) for exact file:line citations.

## Usage

```bash
# From this directory. Requires a release build: `pixi run make`.

# 1. Populate a TPC-H database (uses the tpch extension's dbgen; needs network for INSTALL,
#    or point `run` at a .duckdb produced by ../tpch_performance/generate_tpch_data.sh).
pixi run python3 cardinality_accuracy.py prepare --db tpch_sf1.duckdb --tpch-sf 1

# 2. Collect the tidy per-operator table + a quick summary.
pixi run python3 cardinality_accuracy.py run \
    --db tpch_sf1.duckdb \
    --queries-dir ../tpch_performance/tpch_queries/orig \
    --workload tpch --scale-factor 1 \
    --out results/tpch_sf1.csv --summary
```

For **TPC-DS**, generate data via `../../test/tpcds_performance/generate_tpcds_data.sh`,
then point `--queries-dir` at the TPC-DS queries with `--workload tpcds`.

For **parquet** inputs, add `--engine sirius` so the Sirius parquet footer-count callback
([sirius_extension.cpp:201](../../src/sirius_extension.cpp#L201)) provides exact base-table
cardinalities, matching production.

### Engines

- `--engine duckdb` (default): plain DuckDB over native tables. Estimates are identical to
  what Sirius sees, so this is the simplest faithful setup.
- `--engine sirius`: `LOAD`s the extension (`-unsigned`) and sets `gpu_execution=false`.
  Use for parquet workloads.

## Output

`results/*.csv`, one row per operator:

| column | meaning |
|---|---|
| `workload`, `scale_factor`, `engine`, `query` | run labels |
| `op_id` | pre-order index of the operator within the query plan |
| `depth` | depth in the plan tree (0 = root); a proxy for "how many operators upstream" |
| `op_type`, `op_name` | DuckDB physical operator type / name |
| `est_rows` | estimated output rows (`__estimated_cardinality__`) |
| `act_rows` | actual output rows (`operator_cardinality`) |
| `qerror` | `max(est,act)/min(est,act)`, floored at 1; **1.0 = perfect** |
| `under_estimate` | `True` when `act_rows > est_rows` (the collapse-unsafe direction) |
| `rows_scanned` | base-scan estimate for table scans |

The `--summary` flag prints per-operator-type q-error percentiles, under-estimate rate, and
the worst actual/estimated ratio, flagging the operator types #990 would collapse (`*`).

## What this harness does NOT do (next steps)

- **Bytes.** Operator output row-widths are not in the profiling JSON, and the real #990
  threshold is a *byte* budget. Derive `est_bytes`/`act_bytes` in the analysis step by
  joining these rows against per-query output schemas (from `EXPLAIN`/logical types and the
  fixed-width sizing Sirius already uses in
  [sirius_physical_table_scan.cpp:39](../../src/op/sirius_physical_table_scan.cpp#L39)).
- **Threshold sweep + confusion matrix.** The core #990 deliverable — for a candidate
  threshold `T`, the {correctly-collapse, correctly-keep, wrongly-collapse, wrongly-keep}
  matrix and the overrun-ratio tail — is computed from this CSV in the analysis step.
- **Join Order Benchmark (JOB).** TPC-H is unusually easy for estimators (independent,
  uniform columns). Add TPC-DS (skew) and ideally JOB (correlated real data) before drawing
  a safety conclusion.

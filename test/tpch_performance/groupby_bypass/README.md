# Memory-aware group-by bypass benchmark

Controlled off/on comparison for point 2 of [issue #1746](https://github.com/sirius-db/sirius/issues/1746).
See the [design](../../../docs/super-sirius/group-by-bypass.md) for eligibility and memory accounting.
The feature remains disabled by default.

## Run

From the repository root, using an existing partitioned TPC-H Parquet dataset:

```bash
pixi run python test/tpch_performance/groupby_bypass/run_bypass_sql.py \
  --duckdb build/release/duckdb \
  --data-dir test_datasets/tpch_parquet_sf1000 \
  --config /path/to/sirius.yaml \
  --output test/tpch_performance/output/bypass-sf1000 \
  --queries bypass_mid,bypass_twokey,bypass_small,reject_topn_wide,reject_sum \
  --warmups 2 --repeats 5

pixi run python test/tpch_performance/groupby_bypass/analyze_bypass_sql.py \
  test/tpch_performance/output/bypass-sf1000
```

Use a fresh output directory for each run; the runner appends records. Both arms use the same
binary and configuration, enable internal test options, and disable GPU-to-CPU fallback.
The runner captures an untimed CPU reference and compares integral outputs by sorted digest;
AVG comparisons use relative tolerance 1e-9 and absolute tolerance 1e-10.

CLI timings include result formatting even when output is directed to `/dev/null`. For large
outputs, also examine the interval between the engine log's `executing query` and
`query completed` messages. The first two executions are warmups, the next five are measured,
and the final execution captures results. Do not include that final capture in timing summaries.
The runner's integral-result sort holds the output in host memory; allow sufficient host RAM
for large scales or use a bounded external sort for validation.

The optional [128 MiB configuration](sirius-bypass-128m.yaml) makes smaller datasets exercise
multiple automatic partitions. It was **not** used for the GB300 results below; those used
normal operator defaults.

## GB300 results

Measured revision: `2a64af22cd9b1f133763fee9bd252f5d7bf3fd83`, on the original stacked prototype. The standalone port onto dev needs separate validation;
these historical numbers are not a performance rerun of the port.
One GB300, driver 595.91.07, 256703 MiB device memory; 80% GPU usage limit, 64 GiB host cache,
100 GiB spill limit. SF1000 had 5,999,989,709 lineitem rows. Each arm used a fresh process,
two warmups and five measured executions. OS caches were not flushed.

| SF1000 query | Partitions off → on | Engine median off | Engine median on | Speedup |
|---|---|---:|---:|---:|
| Part key, COUNT/MIN/MAX | 7 → 1 | 9.312 s | 8.075 s | 1.153× |
| Supplier + line number, COUNT/MAX | 6 → 1 | 7.411 s | 5.473 s | 1.354× |

These are engine execution times, excluding subsequent CLI formatting. CLI medians were
86.636 → 85.629 s and 34.437 → 32.958 s respectively. Large output formatting hides much of
the execution benefit. Small differences in sequential arm blocks are not statistical evidence.

All 54 pairs passed CPU validation: five targeted queries and all 22 TPC-H queries at each of
SF100 and SF1000. All 108 GPU arms had eight completed executions, five timing samples, and
no recorded reservation-floor shortfall. The fallback log audit was empty. For these runs,
the helper used `SIRIUS_DISABLE=1` for CPU references and bounded external sorting; TPC-H
numeric comparisons used relative tolerance 1e-9 and absolute tolerance 1e-10.

The bypass did not activate in TPC-H at either scale. Sum of the 22 warm engine medians:
SF100, 19.550 → 19.556 s; SF1000, 169.270 → 170.004 s. These totals are effectively unchanged
in this experiment and are not official TPC-H scores. SF1000 Q10 and one Q18 aggregate were
rejected for unsupported states; other candidates already used one partition.

## Focused unit tests

```bash
pixi run build/release/extension/sirius/test/cpp/sirius_unittest '[group_by_bypass]'
```

Tests cover supported selection, rejected candidates, budget and overflow boundaries, metadata,
configuration bounds, and retaining the memory floor when execution history exists.

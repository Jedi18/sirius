# Memory-aware group-by bypass

An opt-in implementation of point 2 of [issue #1746](https://github.com/sirius-db/sirius/issues/1746).
It is independent of projected partition sizing (#1765). With runtime size estimation disabled,
the existing full-input barrier is preserved. If estimation fixes the partition count before the
producer finishes, bypass is declined and that count remains fixed.

## Behavior

The merge first computes the normal partition count. If that count exceeds one, the bypass
may choose one instead, reusing the existing path that forwards batches without redistributing
rows. Otherwise the normal count remains unchanged.

The bypass requires complete input resident in one GPU memory space, one admitted GPU, and
known physical column metadata. Keys and partial aggregate states must be fixed-width integers
or Booleans; accepted aggregate operations are SUM, COUNT, MIN and MAX. AVG, COUNT(DISTINCT),
floating/decimal states and variable-width columns are excluded. Later operations may only be
unary projection, filter or limit steps ending at a result collector. Sorting, joins and other
plan shapes retain normal partitioning.

## Memory accounting

The model charges additional allocations for concatenation, the grouping hash table, a row-index
map, intermediate aggregates and final output. Output is sized for the worst case of one group
per partial row. Null masks and a possible downstream copy are included. The model rejects
arithmetic overflow and inputs beyond cuDF's row-index limit.

The budget is the memory space's **reservation cap minus charged bytes**, clamped at zero.
Charged bytes already include resident input and outstanding reservations, so they are not
subtracted again or added to the model. Physical free VRAM is not the reservation budget.

Selection requires the model plus a configurable margin to fit. A selected merge reuses its
additional-allocation estimate, before that margin, through `no_history_peak_memory_estimate()`.
The cold-start request is at least the existing 2× input estimate. The executor adds input
materialization costs and records the request through its existing reservation telemetry.

Memory history belongs to the current query's pipeline. The partition count is chosen before
merge tasks run, and a bypassed single partition produces one merge task; its first attempt has
no prior partitioned-merge history. After an OOM, the existing history and task-local retry floor
size subsequent requests, including resumes after the merge in a fused pipeline. No additional
executor-wide floor or separate history is introduced.

The partition reads the enable flag and the merge reads headroom from the existing immutable
query policy snapshot. A transient input summary carries only the observed rows, column layout,
residency, target device, and reservation-budget snapshot. It is not a repository cache: batch
placement can change after each read handle is released. No persistent row metrics are needed
for this complete-input decision.

This remains an estimate, not a guaranteed bound for the pinned cuDF implementation. Selected
GB300 allocation traces fit within the model, but do not establish a bound for every input or
plan. Selection does not reserve memory, and the executor can grant a partial
reservation. Existing executor diagnostics report reservation shortfalls; automatic
repartitioning after OOM is not implemented. The partition count stays fixed once chosen.

## Settings and validation

`enable_group_by_memory_aware_bypass` is **false by default**. Internal settings require
`SIRIUS_ENABLE_TEST_OPTIONS=1`. `group_by_bypass_headroom_fraction` defaults to 0.25 and accepts
finite values from 0 to 4. Decisions log the reason, automatic/chosen counts, estimated bytes and
budget; the required bytes include the configured margin.

Run focused tests with:

```bash
pixi run build/release/extension/sirius/test/cpp/sirius_unittest '[group_by_bypass]'
```

## GB300 measurements

The standalone engine at `273a1293c67f8fae7df5a40230befa38418f0a44` was tested on one GB300
(driver 595.91.07, 256703 MiB device memory), with an 80% GPU limit, 64 GiB host cache,
100 GiB spill limit, normal operator defaults, and the default 0.25 headroom fraction.
These are custom queries over TPC-H SF1000 data, not standard TPC-H queries or scores.

| Query shape | Partitions off → on | Engine median off | Engine median on | Speedup |
|---|---:|---:|---:|---:|
| Lineitem by part, COUNT/MIN/MAX | 7 → 1 | 10.030 s | 8.294 s | 1.209× |
| Lineitem by supplier + line number, COUNT/MAX | 6 → 1 | 7.476 s | 6.011 s | 1.244× |
| Orders by customer, COUNT/MIN/MAX order key | 2 → 1 | 3.543 s | 3.076 s | 1.152× |
| Orders by customer, COUNT/MIN/MAX ship priority | 2 → 1 | 3.323 s | 2.658 s | 1.250× |
| Lineitem by part, COUNT only | 3 → 1 | 6.425 s | 5.411 s | 1.187× |
| Lineitem by supplier + line number, COUNT only | 3 → 1 | 4.871 s | 4.516 s | 1.079× |

Each arm used a fresh process and two warmups, followed by five measured executions for the
first two rows and seven for the remaining four. Medians use the engine-log interval from
`executing query` to `query completed`, excluding subsequent CLI formatting. OS caches were
not flushed. Arms ran in sequential blocks, not randomized independent trials; the table lists
selected wins after screening more shapes. Ship priority is constant in this dataset. These
results demonstrate eligible cases, not an expected workload-wide speedup.

Both GPU arms of all six queries matched separate CPU references by sorted digest. Additional
correctness runs covered all 15 harness shapes at SF100 and 12 of those shapes at SF1000,
including empty/null inputs and rejected AVG, DISTINCT, string and sorted plans. At SF100 the
normal count was already one, so those runs tested compatibility, not bypass activation.
No GPU-to-CPU fallback was observed in the reported runs. Separate allocation traces for the
first two SF1000 queries showed model-to-peak ratios of about 1.42× and 1.34×, respectively,
without recorded reservation shortfalls. These checks are not injected-OOM or concurrency tests.

The reproducible harness is kept off this feature branch on
`groupby-bypass-gb300-experiments` (archive commit `ab3d0753`). Its directory is
`test/tpch_performance/groupby_bypass/`. Raw logs remain under `test/tpch_performance/output/`
in the GB300 worktree `groupby-bypass-reuse-estimation-273a1293`:

- `bypass-sf1000-273a1293-postfix-20260922`: first two speedups and controls.
- `bypass-sf1000-273a1293-trace-fix-20260922`: allocation traces.
- `bypass-all-sf100-273a1293-postfix-v2-20260922`: 15-shape correctness sweep.
- `bypass-broad-sf1000-273a1293-postfix-20260922`: additional SF1000 correctness cases.
- `bypass-more-repeated-sf1000-273a1293-20260922`: four additional speedups.
- `bypass-more-correctness-sf1000-273a1293-20260922`: their CPU comparisons.

Historical standard TPC-H runs at `2a64af22` never activated bypass; they are not a validation
rerun of this standalone revision. Cleanup after the measurements fixes a test input's lifetime
and rejects non-finite configuration values; it does not change the model or merge algorithm.

## Before default-on

Before default-on, verify the model, secure memory before committing to bypass, and add bounded
OOM recovery that repartitions preserved input without duplicate output. Validate recovery
under injected allocation failures and concurrent memory pressure as a separate change.

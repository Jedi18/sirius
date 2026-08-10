# Runtime Data Size Estimation

**Files:** `src/include/pipeline/data_size_estimator.hpp`, `src/pipeline/data_size_estimator.cpp`

An API that projects how many bytes will *ultimately* arrive at an operator's input port, by
chaining upstream pipelines' measured input→output ratios back to the first pipeline that has
finished (or to a source that knows its own total).

Implements [issue #1283](https://github.com/sirius-db/sirius/issues/1283).

## The API

```cpp
std::optional<data_size_estimate> estimate_port_total_input_bytes(
    op::sirius_physical_operator& op, std::string_view port_id, size_estimate_options = {});

std::optional<data_size_estimate> estimate_pipeline_total_output_bytes(
    pipeline::sirius_pipeline& p, size_estimate_options = {});
```

The number is a **total for the whole query**, not a so-far figure: it answers *"how many bytes
will this port have received once its producer is done?"*.

`estimate_pipeline_total_output_bytes` resolves in four cases, in the order tried:

| # | condition | result |
|---|-----------|--------|
| 1 | pipeline finished | its recorded output total, `exact = true` |
| 2 | several input ports (fan-in) | follow the source's nominated primary port, scaled by `output total / consumed primary bytes`; `nullopt` if it nominates none |
| 3 | source has no input ports (a leaf) | `total_source_input_bytes × ratio`, or `total_source_output_bytes` unscaled |
| 4 | exactly one input port | recurse into the producer, then apply this pipeline's ratio |

`estimate_port_total_input_bytes` resolves the port's `src_pipeline` and delegates. It returns
`nullopt` for a missing port, a dependency-only port (null repo), or a port with no producer.

### The result

```cpp
struct data_size_estimate {
  std::size_t bytes;          // projected total
  bool        exact;          // measured, not projected — anchored on a finished pipeline
  std::size_t hops;           // pipelines traversed; ratio error compounds per hop
  std::size_t ratio_samples;  // completed tasks behind the weakest ratio in the chain
};
```

`exact` and `ratio_samples` are the confidence signals to gate on when a decision is expensive to
reverse — a projection built from a handful of completed tasks is far weaker than one built from
hundreds.

**Everything is measurement-derived.** There is deliberately no DuckDB-cardinality fallback, and
any unknown link yields `nullopt` rather than a guess. The corollary is that this API **cannot
answer before the query has started running**; it has no pre-execution mode.

## Where the numbers come from

**The pipeline ratio.** Every completed GPU task already records `{input_basis, peak_memory,
output_bytes}` into its pipeline's history. `history_totals` accumulates alongside the 64-entry
ring buffer and is never evicted, so the aggregate ratio stays accurate on pipelines that run more
tasks than the ring holds. Tasks that OOM'd record no output and are excluded — they consumed
input and produced nothing, so counting them would drag the ratio toward zero.

**Leaf source totals.** Two virtuals on `sirius_physical_operator`, both defaulting to `nullopt`
(the correct answer for `STREAMING_SOURCE`, whose total is genuinely unknowable):

| operator | `total_source_input_bytes` | `total_source_output_bytes` |
|----------|---------------------------|-----------------------------|
| `GPU_SCAN` | Σ split bytes, once split discovery closes | `estimated_cardinality × bytes/row` |
| `GPU_VALUES` | exact, known at plan time | — |

Both exist because the quantities live in different coordinate systems.
`scan_info::estimated_bytes()` is **pre**-filter, and the pipeline ratio's denominator is that same
pre-filter number — so the ratio already encodes filter selectivity. `estimated_cardinality` is
**post**-filter, so scaling it by the ratio would count selectivity twice. Hence
`total_source_output_bytes` is used unscaled.

For `GPU_SCAN` the total is tallied in `split_connector::push_split` — the choke point every split
passes through — and `is_discovery_complete()` reports when the tally is final. That is distinct
from the pre-existing `is_closed()`, which means *closed and drained*.

## Fan-in

A `HASH_JOIN` heads its own pipeline with `"build"` and `"default"` ports. The estimator follows
only the volume-driving side, which the operator nominates:

```cpp
virtual std::optional<std::string_view> primary_input_port() const;        // "default" on a join
virtual std::optional<std::size_t>      consumed_primary_input_bytes() const;
```

The recorded `input_basis` cannot serve here: a STANDARD join pairs each probe batch with every
build batch and *borrows* rather than pops, so the same bytes enter `input_basis` once per pairing
and its sum is a cross product, not an input volume. The join therefore counts probe bytes itself,
**once per distinct batch, at the point each first enters a task** — not when it lands in the port,
which would measure arrival rather than consumption.

`consumed_primary_input_bytes()` returns `nullopt` until the build side is complete: output per
probe byte climbs while build batches arrive, so a ratio sampled during that window reads low, and
a consumer that latches an early estimate never corrects it.

### Sample floors

`size_estimate_options` carries two floors, below which a ratio is treated as absent (so
`assume_unit_ratio` still applies):

- `min_ratio_samples` (4) — single-input. That ratio accrues both terms on task completion and is
  unbiased at any count; the floor only rules out one unrepresentative batch.
- `min_fan_in_ratio_samples` (16) — fan-in. That ratio divides a completion-accrued numerator by a
  denominator advancing at task *start*, so it reads low by roughly
  `in_flight / (samples + in_flight)`. More samples shrink a systematic bias, which is why this
  floor is far higher.

There is deliberately no task-count correction for in-flight tasks: `consumed` does not advance
once per task, so the completed fraction of tasks does not map onto the consumed fraction of bytes.

## Coverage

| category | operators |
|----------|-----------|
| anchors | `GPU_SCAN`; `GPU_VALUES` (also covers `COLUMN_DATA_SCAN`, `DUMMY_SCAN`, `EMPTY_RESULT`, rewritten to it at plan generation); any finished pipeline |
| pass-through (recurse) | any single-ingress pipeline — `FILTER`, `PROJECTION`, `LIMIT`, sorts, aggregates, `CONCAT`, `PARTITION` |
| fan-in | `HASH_JOIN` |
| dead ends (`nullopt`) | `STREAMING_SOURCE` (by design); `TABLE_SCAN`; `NESTED_LOOP_JOIN`, delim joins, `CTE` (no nominated primary) |

Because the estimator works at pipeline granularity, single-input operators need no per-operator
model: a pipeline's ratio is measured end-to-end, so whatever a projection or filter does to byte
volume is captured automatically. The cost is attribution — a bad ratio cannot be traced to one
operator.

`NESTED_LOOP_JOIN` uses the same port names and could take the identical fan-in treatment; leaving
it unnominated preserves fall-back-to-waiting behaviour.

---

## The problem

### How a GROUP BY runs

A `GROUP BY` over a large table is split across three pipelines:

```
Pipeline 1                        Pipeline 2       Pipeline 3
[scan → … → HASH_GROUP_BY]   →    [PARTITION]  →   [MERGE_GROUP_BY]
partial aggregate, per batch      bucket by key    combine, per bucket
```

Pipeline 1 aggregates each batch **independently**, so the same key appears in many partial
results. Pipeline 2 hash-partitions those partials so every row with a given key lands in one
bucket. Pipeline 3 combines each bucket independently and in parallel — correct only because a key
never spans two buckets.

### Why the partition count is frozen

A row goes to bucket `murmur3(key) % N` (`cudf::hash_partition`, `HASH_MURMUR3`). The bucket is a
function of N.

If N changed mid-stream, a key could land in slot 3 under the old N and slot 1 under the new one.
`MERGE_GROUP_BY` combines each slot independently and emits its result, so the query would return
**two output rows for that key** — no crash, no warning, a duplicate row in a `GROUP BY` result.

So N is chosen once, on the first sizing decision, and never revised.

### Why that forced a FULL barrier

N is chosen by size: `natural_num_partitions()` computes
`ceil(total_bytes / hash_partition_bytes)`, floored to the GPU count once the input clears the
small-table threshold. Historically `total_bytes` came from
`sirius_physical_partition::compute_total_bytes()` — the bytes sitting on the input port at that
moment.

For that to equal the total, everything must already have arrived, which is why the edge into
`PARTITION` was a `FULL` barrier. The consequence: the GPU performs all aggregation, *then* all
partitioning, *then* all merging — three sequential phases where the first two could overlap.

---

## The consumer: grouped-aggregation PARTITION

**File:** `src/op/sirius_physical_partition.cpp`

Three coordinated changes.

**Barrier.** `sirius_pipeline_converter::resolve_barrier` returns `PARTIAL` for an aggregate-fanout
partition's ingress — a `PARTITION` whose tree parent is `MERGE_GROUP_BY` **and** which has
estimation enabled (`is_size_estimation_enabled()`). Only that edge changes;
`PARTITION → MERGE_GROUP_BY` remains `FULL`, since the merge needs every bucket. Branch formation
is unaffected — `query_index` only cuts branches at multiport consumers, and this pipeline has one
ingress.

Gating the edge on the operator's own flag rather than flipping it unconditionally matters twice.
It keeps the feature-off configuration on the *original* code path instead of an emulation of it,
so a future refactor of the gate cannot change behaviour for users who never turned this on. And
it excludes a second operator that the type test alone cannot distinguish: the delim-join
`DISTINCT` partition (`sirius_physical_plan_generator`'s delim path) sits under a
`grouped_aggregate_merge`, which is *also* typed `MERGE_GROUP_BY`, yet is never constructed with
estimation enabled. `test_gpu_execution_size_estimation.cpp` pins that shape under both settings.

**Gate.** With a `PARTIAL` port the base hint would return `READY` on the first batch, sizing N from
a fraction of the data. `get_next_task_hint()` therefore becomes the authority:

```cpp
if (no sibling && N not yet fixed && no projection available) {
    if (producing pipeline not finished)
        return WAITING_FOR_INPUT_DATA;   // reproduce FULL semantics
}
```

The barrier is advisory; the operator decides per-poll whether to behave as `FULL`. The gate is
itself guarded on `enable_runtime_size_estimation`, because it is only the authority when the edge
was relaxed for it — with the feature off the port is still `FULL` and the base hint enforces the
wait directly.

**Sizing.** `get_next_task_input_data()` feeds the projection into `get_partition_strategy()`
instead of `compute_total_bytes()`, floored at `max(projection, bytes_already_arrived)` so an
undershooting projection can never request fewer partitions than the data on hand justifies, and
scaled by `size_estimate_safety_factor` (projections only — a measured total is used as-is).

---

## Observability

Every fallback in this design is safe: same results, same tests, same output. The consequence is
that **a working feature and an inert one are indistinguishable from the outside** — there is no
error, no warning, and only a performance difference, which is noisy.

The feature therefore ships with an explicit liveness signal rather than only a correctness one.

| line | level | reports |
|------|-------|---------|
| `… projected N bytes on its input port (exact=, hops=, samples=)` | DEBUG | a projection was latched |
| `… sized N partitions from X bytes (measured \| upstream-complete \| projected)` | DEBUG | **which path chose N** |
| `… size estimate: sized from BASIS X bytes (exact=, hops=, samples=), actual … error …%` | INFO | accuracy at operator finalize, against the bytes that **actually** sized the partition — not the raw projection, which differs once the safety factor or the already-arrived floor applies |

Interpreting them:

| observation | meaning |
|-------------|---------|
| `projected` / `exact=false` | a learned ratio was applied while the producer was still running — **the feature is doing its job** |
| `upstream-complete` / `exact=true` | the estimator read an already-finished producer's exact total; correct, but too late to relax anything |
| `measured` | no estimate; sized from the drained port, i.e. pre-feature behaviour |
| `error +0.0%` | suspicious — a genuine projection is essentially never exact. Indicates the total was read after the fact |

`hops` counts the pipelines traversed; ratio error compounds per hop, so a long chain is less
trustworthy than a short one.

The same three bases are also published as counters on `SiriusContext`, so the question can be
answered programmatically rather than by grepping a log at the right level:

```cpp
struct size_estimation_stats {
  uint64_t partitions_sized_from_measured;           // pre-feature behaviour
  uint64_t partitions_sized_from_projection;         // the feature engaged
  uint64_t partitions_sized_from_upstream_complete;  // estimator answered, but too late to relax
};
```

One increment per PARTITION that fixed its count, recorded at the decision. Only the
grouped-aggregation path records: join-side sizing never consults the estimator, so counting it
would dilute the signal. `test/cpp/integration/test_gpu_execution_size_estimation.cpp` asserts on
these — the counters are what let a runtime test distinguish "engaged and correct" from "silently
inert and correct", which every other observable (results, absence of errors, wall clock) cannot.

---

## Where it might still pay

Relaxing the barrier only converts into wall clock when some resource is otherwise **idle** during
the wait. That predicts:

- **Untested: IO-bound scans** (S3, cold cache on slow storage). The GPU genuinely stalls on reads,
  so there is headroom to fill. The SF500 cold runs approximated this but the storage was fast
  enough that the noise dominated.
- **Untested: memory pressure.** A working set exceeding GPU memory trips the downgrade path;
  those stalls are idle capacity. SF1000 on 4×GB200 did *not* reach this regime — zero downgrade
  activity in the logs.
- **Untested: slower GPUs.** On GB200-class hardware TPC-H queries finish in 1–3 s at SF500, so
  the partition phase is milliseconds and the ceiling is correspondingly tiny.
- **Unlikely to pay: single GPU, warm local storage.** Aggregation and partitioning are both GPU
  work, so overlapping them time-slices one saturated device rather than filling a gap.

The ceiling is also bounded by query shape, and scaling does not move it: a group-by that collapses
600M rows to 20M distinct leaves the partition handling a small fraction of what the producer
handles, and doubling the scale factor doubles both.

### How to measure it

Wall-clock A/B is a weak instrument at this effect size. Establish the floor first, then prefer
direct observation of the mechanism:

- **A/A control.** Same config both arms, interleaved. Whatever delta appears is your detection
  threshold. Do not interpret an A/B whose effect is smaller.
- **NVTX pipeline ranges.** Each pipeline emits a process-wide NVTX range labelled
  `Pipeline N: SOURCE -> SINK`. Under `nsys`, `HASH_GROUP_BY` and `PARTITION` are butt-joined when
  the feature is off and overlapping when it is on — a direct reading of the mechanism that needs
  no statistical power.
- **Quent batch lifecycle.** Batches move `registered → queued → packaged → processing →
  consumed`. Dwell time in `queued` at the partition's input *is* the latency the barrier imposes,
  and it should collapse when the feature is active. Label runs with
  `CALL sirius_set_query_label('…')` to tell them apart.

---

## Applicability to GPU resource allocation

Per-query GPU allocation has two layers, and this estimator serves only one of them.

**Layer 1, admission ("how many GPUs does this query warrant?") — no.** Not a gap to fill: every
number here is measurement-derived, and at admission no task has run, so no pipeline has a ratio
and every case returns `nullopt`. `assume_unit_ratio` does not rescue it, since the leaf anchor
`total_source_input_bytes()` is itself `nullopt` until split discovery closes — which happens
after the GPU set is chosen. Admission needs a **plan-time** cost model over
`estimated_cardinality` and plan shape. The two are complementary, not substitutes.

**Layer 2, dynamic tapering ("can we release a GPU after this stage?") — yes.** That question is
`estimate_port_total_input_bytes` at a pipeline boundary, and by then upstream ratios exist. Gate
releases on `exact` / `ratio_samples`: a release is expensive to undo, so it warrants a stricter
`size_estimate_options::min_ratio_samples` than partition sizing uses — which is why the floors
are per-call rather than global config.

Two interactions to keep in mind. The `PARTIAL` ingress overlaps `HASH_GROUP_BY` with `PARTITION`,
so per-query *peak* memory rises slightly where it previously came in two sequential phases — the
wrong direction under a per-query budget, though [the measured overlap](#what-has-been-measured)
is milliseconds. And the partition count is floored to `num_gpus` at sizing time and then frozen,
so a query that later releases GPUs keeps an `N` reflecting the larger set. That is benign — extra
partitions queue — but Layer 2 should not assume `N` tracks the current GPU count.

---

## What has been measured

The mechanism works: projections land before the count is fixed on every group-by partition
across TPC-H, with error around -0.5%.

**It buys no measurable wall clock, because the phase it accelerates is a rounding error.**
Partition work as a share of total task time, feature off, 1x L4:

| query | SF30 | SF100 |
|-------|------|-------|
| Q1 | 0.00% | 0.00% |
| Q3 / Q5 / Q10 | 0.00% | 0.00-0.07% |
| Q9 | 0.22% | 0.28% |
| Q13 | 0.86% | 0.72% |
| synthetic 45M-group GROUP BY | 22.3% | 12.9% |

Scans are 96-100% of task time in every real query. A `GROUP BY` collapses its input, so the
partition handles a small fraction of what the aggregation handles -- Q1 turns 284M rows into 4
groups, leaving the partition ~12 KB against the aggregation's 13.7 GB.

Two things worth noting before re-running this. Scaling **up** makes the case worse, not better:
the synthetic best case fell from 22.3% to 12.9% between SF30 and SF100, because scan work grows
faster than partition work. And an earlier reading of these same runs put join partitioning at
13-29%; that measured pipeline *windows*, which for a streaming `PARTIAL` pipeline overlap the
scan feeding them and are not additive cost. Measure `sum_execution_time_ms`, not the window.

The flag is therefore off by default. Turning it on is expected to be neutral on this workload;
it exists so the path stays exercised and can be evaluated on a shape where partitioning is a
real cost.

---

## Configuration

Both settings are accepted as DuckDB `SET` variables and in YAML under `sirius.operator_params`.

| Variable | Default | Description |
|----------|---------|-------------|
| `enable_runtime_size_estimation` | `false` | Master switch. **Off by default**: the mechanism is verified but no wall-clock benefit has been measured (see [What has been measured](#what-has-been-measured)). On restores nothing that is missing — it enables the projection path. |
| `size_estimate_safety_factor` | `1.0` | Multiplier applied to a *projected* total before sizing partitions; raise above 1.0 to bias toward more (smaller) partitions when projections undershoot. Measured totals are not scaled. |

See [Configuration](configuration.md#runtime-data-size-estimation) for the full reference.

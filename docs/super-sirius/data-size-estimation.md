# Runtime Data Size Estimation

**Files:** `src/include/pipeline/data_size_estimator.hpp`, `src/pipeline/data_size_estimator.cpp`

An API that projects how many bytes will *ultimately* arrive at an operator's input port, by
chaining upstream pipelines' measured input→output ratios back to the first pipeline that has
finished (or to a source that knows its own total). Its one consumer today is the grouped
aggregation's `PARTITION`, which uses it to fix its partition count from the first batches instead
of waiting for its whole input — turning that hard barrier into a partial one.

Implements [issue #1283](https://github.com/sirius-db/sirius/issues/1283).

## Contents

- [The problem](#the-problem)
- [The idea](#the-idea)
- [The estimator](#the-estimator)
- [Where the numbers come from](#where-the-numbers-come-from)
- [The consumer: grouped-aggregation PARTITION](#the-consumer-grouped-aggregation-partition)
- [Observability](#observability)
- [Coverage](#coverage)
- [What has been measured](#what-has-been-measured)
- [Where it might still pay](#where-it-might-still-pay)
- [Applicability to GPU resource allocation](#applicability-to-gpu-resource-allocation)
- [Configuration](#configuration)

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

## The idea

Rather than measuring the arrived bytes, project the eventual total. Two equivalent formulations:

```
extrapolate by progress:   total = observed / (scanned / scan_total)
scale the source:          total = scan_total × (observed / scanned)
                                                 └── the pipeline's output/input ratio
```

They are algebraically identical. The second is implemented because:

- **It composes.** `total = scan_total × r₁ × r₂ × r₃`. Each pipeline records its own ratio
  locally, with no knowledge of the rest of the query. The progress form needs a single global
  "how far along is this query" quantity, which does not exist.
- **It has a natural stopping point.** Walking upstream, a pipeline that has already *finished*
  needs no ratio — its exact total output is known. Stop there.

### Why not adapt N instead?

Consistent hashing — or the cheaper restriction of N to powers of two, where `h % 2N` is either
`h % N` or `h % N + N` — would bound the rework of changing N mid-stream. It is not used because:

1. **It addresses the smaller cost.** The expense is not the volume of data moved but the
   lifecycle: batches are already deposited into per-slot repositories downstream, so changing N
   means tracking which batches used which N, relocating them inside the merge's repository, and
   proving no merge task has already consumed a slot being modified.
2. **Being wrong is soft.** N too small yields larger merge tasks and possible spilling; N too
   large adds per-bucket overhead. Both are slower, neither is incorrect. Adaptive repartitioning
   is worth its machinery when error is catastrophic, not when it is merely suboptimal.

---

## The estimator

Free functions over the pipeline graph — no engine state, so they unit-test against a synthetic
DAG (`test/cpp/pipeline/test_data_size_estimator.cpp`).

```cpp
std::optional<data_size_estimate> estimate_port_total_input_bytes(
    op::sirius_physical_operator& op, std::string_view port_id, size_estimate_options = {});

std::optional<data_size_estimate> estimate_pipeline_total_output_bytes(
    pipeline::sirius_pipeline& p, size_estimate_options = {});
```

`estimate_pipeline_total_output_bytes` resolves in four cases:

| # | condition | result |
|---|-----------|--------|
| 1 | pipeline finished | its recorded output total, `exact = true` |
| 2 | source has no input ports (a leaf) | `total_source_input_bytes × ratio`, or `total_source_output_bytes` unscaled |
| 3 | exactly one input port | recurse into the producer, then apply this pipeline's ratio |
| 4 | more than one input port | follow the source's nominated primary port, scaled by `output total / consumed primary bytes`; `nullopt` if it nominates none |

`estimate_port_total_input_bytes` resolves `port->src_pipeline` and delegates. It returns `nullopt`
for a missing port, a dependency-only port (null repo), or a port with no producer.

### The result

```cpp
struct data_size_estimate {
  std::size_t bytes;          // projected total
  bool        exact;          // measured, not projected: anchored on a finished pipeline
                              // or a source total, with no ratio applied
  std::size_t hops;           // pipelines traversed — error compounds per hop
  std::size_t ratio_samples;  // completed tasks behind the weakest ratio in the chain;
                              // 0 when no ratio was applied
};
```

`size_estimate_options` offers `assume_unit_ratio` (fall back to 1:1 for a pipeline with no
completed task — the flag issue #1283 asks for), `max_hops` (recursion guard), and two sample
floors below which a ratio is treated as absent: `min_ratio_samples` (4) for single-input
pipelines and `min_fan_in_ratio_samples` (16) for fan-in. They differ by an order of magnitude
because they suppress different errors — see [Fan-in](#fan-in).

### One non-obvious rule

**A finished pipeline with zero records reports `nullopt`, not `0`.**
`pipeline_memory_history::record()` drops tasks whose `input_basis` is 0, and
`scan_info::estimated_bytes()` returns 0 by default for formats with no a-priori estimate. So zero
records means *no evidence*, not *produced nothing*; reporting 0 would size a large input onto a
single partition. Callers fall back to waiting, which is also correct when the pipeline genuinely
emitted nothing.

Case 4 (fan-in) has enough subtlety of its own to warrant [its own section](#fan-in). There is
deliberately no DuckDB-cardinality fallback anywhere: every number in the chain stays
measurement-derived.

---

## Where the numbers come from

### The pipeline ratio

**File:** `src/include/pipeline/pipeline_memory_history.hpp`

Every completed GPU task already records `{input_basis, peak_memory_bytes, output_bytes}` into its
pipeline's history — machinery built for memory-reservation sizing. The estimator reads the
input/output relationship in the same record.

Two properties were added for this use:

- **Monotonic totals.** The records live in a 64-entry ring buffer, so a pipeline running hundreds
  of tasks loses most of them. `history_totals` accumulates alongside the ring and is never
  evicted, giving `total_output_bytes()` and `output_to_input_ratio()`.
- **Failure exclusion.** A task that OOM'd records `output_bytes = nullopt` — it consumed input and
  produced nothing. Counting it would drag the ratio toward zero.

The ratio is an **aggregate** (Σout / Σin), deliberately not the proximity-weighted average
`estimate_peak_memory()` uses: that weighting answers "size this one task", whereas a total needs
every byte counted once.

The history is owned by `sirius_pipeline` rather than by
`sirius_pipeline_task_global_state`, so the estimator can reach a producer's ratio through
`port::src_pipeline` without going via `task_creator`. There is exactly one global state per
pipeline, so the global state simply delegates.

### Leaf source totals

Two virtuals on `sirius_physical_operator`, both defaulting to `nullopt` — the correct answer for
`STREAMING_SOURCE`, whose total is genuinely unknowable:

```cpp
virtual std::optional<std::size_t> total_source_input_bytes()  const;  // scaled by the ratio
virtual std::optional<std::size_t> total_source_output_bytes() const;  // NOT scaled
```

| operator | `total_source_input_bytes` | `total_source_output_bytes` |
|----------|---------------------------|-----------------------------|
| `GPU_SCAN` | Σ split bytes, once split discovery closes | `estimated_cardinality × bytes/row` from per-batch counters |
| `GPU_VALUES` | exact, known at plan time | — |
| everything else | `nullopt` | `nullopt` |

For `GPU_SCAN`, the scan total is tallied in `split_connector::push_split` — the single choke point
every split passes through — and `is_discovery_complete()` reports when the tally is final. This is
distinct from the pre-existing `is_closed()`, which means *closed and drained*.

There is deliberately no partial-discovery extrapolation. `split_provider::run` claims every
metadata unit up front in a tight loop and parses footers asynchronously, so a claim-based progress
fraction saturates almost immediately while the byte tally is still near zero — it would
extrapolate a total of roughly zero.

### The units trap

The two source hooks exist because the quantities live in different coordinate systems:

| quantity | filtered? |
|----------|-----------|
| `scan_info::estimated_bytes()` | **pre**-filter — decoded bytes off disk |
| the pipeline ratio | denominator is that pre-filter number, so it **already encodes** filter selectivity |
| DuckDB's `estimated_cardinality` | **post**-filter |

Multiplying the cardinality-derived total by the ratio would count filter selectivity twice — a
`WHERE` clause keeping 10% would yield `0.1 × 0.1 = 1%`. Hence `total_source_output_bytes` is
returned unscaled.

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

## Coverage

**What the walk handles:**

| category | operators |
|----------|-----------|
| anchors (terminate with a number) | `GPU_SCAN`; `GPU_VALUES` (also covers `COLUMN_DATA_SCAN`, `DUMMY_SCAN`, `EMPTY_RESULT`, which are rewritten to it at plan generation); any finished pipeline |
| pass-through (single ingress, recurse) | `FILTER`, `PROJECTION`, `LIMIT` (folded invisibly into their pipeline's ratio); pipelines headed by `ORDER_BY`, `SORT_SAMPLE`/`SORT_PARTITION`, `MERGE_SORT`, `TOP_N`/`TOP_N_MERGE`, `UNGROUPED_AGGREGATE` and its merge, `HASH_GROUP_BY`, `MERGE_GROUP_BY`, `CONCAT`, `PARTITION` |
| fan-in (follow the nominated primary side) | `HASH_JOIN` — nominates its probe port and counts probe bytes once per batch |
| dead ends (`nullopt`) | `STREAMING_SOURCE` (by design); `TABLE_SCAN` (generic DuckDB table-function path); `NESTED_LOOP_JOIN`, delim joins, `CTE` (multiple ingress ports, no nominated primary) |

Because the estimator works at pipeline granularity, single-input operators need no per-operator
model: a pipeline's ratio is measured end-to-end, so whatever a projection or filter does to the
byte volume is captured automatically. The cost is attribution — a bad ratio cannot be traced to a
specific operator.

**What consumes the estimate:** only the grouped-aggregation `PARTITION`. The flag is threaded in
from a single call site, `wrap_hash_group_by` in
`src/planner/sirius_physical_plan_generator.cpp`. Join partitions receive the default `false` and
size exactly as before.

**Net effect:** group-bys reached through scans, single-input operators, and hash joins can be
projected. On TPC-H that covers the common `aggregate above joins` shape — Q3, Q5, Q9, Q10, Q13 —
in addition to the scan-only Q1. Plans routed through a nested-loop join, a delim join, a CTE, or a
generic `TABLE_SCAN` still fall back to waiting.

### Fan-in

A `HASH_JOIN` heads its own pipeline with two ports, `"build"` and `"default"`. The estimator
follows only the volume-driving one, which the operator nominates by overriding:

```cpp
virtual std::optional<std::string_view> primary_input_port() const;        // "default" on a join
virtual std::optional<std::size_t>      consumed_primary_input_bytes() const;
```

**Why not just use `input_basis`?** A STANDARD join schedules a cross product: every task carries
a `(probe, build)` pair, and batches are *borrowed* (`get`, not `pop`) and re-paired. The same
bytes therefore enter `input_basis` once per pairing, making `Σ input_basis` a cross-product sum
rather than an input volume. So the join counts probe bytes itself and the estimator forms
`pipeline output total / consumed primary bytes`. Build bytes are never counted — the build is
consumed once into a hash table and does not scale the join's output.

The contract is *bytes **processed**, once per distinct batch*, and both halves are load-bearing:

- **Once per batch**, or the cross product multiplies the same bytes.
- **Processed, not arrived** — count on first entry into a task, not on landing in the port.
  Measured: an early version counted at registration in `refresh_cross_schedule`, which fires on
  arrival, and under-projected by **82%**.

For `HASH_JOIN` that gives three accumulation points: the first pairing on the STANDARD/MIXED
path (`probe_paired_count == 1`), and the two probe pops on BUILD_PROBE.

#### Two remaining downward biases

**The build side is still arriving.** A probe batch joined against a partial hash table emits
proportionally less than it eventually will, so output-per-probe-byte *climbs* while the build
streams in — and the consumer latches the first estimate it gets, so an early sample is never
corrected. `consumed_primary_input_bytes()` therefore returns `nullopt` until the build port's
producing pipeline has finished: no answer at all during the window in which any answer is wrong.

**Tasks are in flight.** The numerator accrues on task completion, the denominator on task start,
so the ratio reads low by roughly `in_flight / (samples + in_flight)`. Measured: on a 4-thread
pipeline, sampling at 16 completed tasks under-projected by **21.7%**, which `16 / (16 + 4.4)`
predicts almost exactly.

An earlier version corrected this by scaling the denominator by `tasks_completed / tasks_created`.
**That was removed as unsound, not merely imprecise:** it assumes every task consumes an equal
share of `consumed`, but `consumed` advances on a probe batch's *first pairing*, so with `B` build
batches only one task in `B` moves it at all. The correction moved the estimate by an amount
unrelated to the error it was meant to cancel.

What is left is bounded rather than corrected, by the two sample floors in
`size_estimate_options`. They differ by an order of magnitude because they suppress different
errors: `min_fan_in_ratio_samples` (16) shrinks the systematic bias above, while
`min_ratio_samples` (4) only needs to rule out one unrepresentative batch on the single-input
path, whose numerator and denominator both accrue on completion and which is unbiased at any
count. Below either floor the ratio is treated as "no ratio yet", so `assume_unit_ratio` still
applies. `size_estimate_safety_factor` biases against whatever remains.

`data_size_estimate::ratio_samples` reports the count behind the *weakest* ratio in the chain and
is logged alongside `hops` — it separates "we sampled too early" from "the model is wrong".

`NESTED_LOOP_JOIN` uses the same port names and could take identical treatment; leaving it
unnominated preserves the previous fall-back-to-waiting behaviour.

Join partitions still size themselves from their own measured build side
(`enable_size_estimation` is `false` for them), so `BUILD_PROBE` and broadcast admission are
untouched by any of this.

---

## What has been measured

**The mechanism works.** Across TPC-H at SF500 and SF1000, on 1, 2 and 4 GPUs, cold and hot cache,
every group-by partition reports `sized from projected` — a real prediction, made while the
producing pipeline was still running. Projection error was −0.5% on a scan-only chain (Q1) and,
after two accounting bugs were fixed, near zero across a join.

**No wall-clock benefit has been demonstrated, and the hardware could not resolve one.** An A/A
control — the *identical* configuration in both arms, so no variable at all — fabricated per-query
deltas of up to **4.7%**, with a consistent sign across all five queries. Any real effect is
expected to be a few percent, i.e. below that threshold. Campaigns run:

| campaign | result |
|---|---|
| SF100, 1 GPU, hot | +0.02% across 21 queries; feature barely engaged (mostly `measured`) |
| SF500, 4×GB200, cold | nothing above a ±10.6% control floor |
| SF500, 4×GB200, hot | targets −1.3% to −4.4%, controls +0.5% to +1.2% — **did not replicate** |
| SF500, 4×GB200, hot, confirmation | effect gone with alternating arm order and 2× samples |
| SF1000, 4×GB200, hot | control moved +22.5%; comparison confounded |
| **A/A control, 2 GPUs** | **±4.7% from no variable — the binding constraint** |

The correct reading is *not* "no benefit" but "**not measurable in this environment**". A machine
quiet enough for an A/A floor under ~1% is a prerequisite for answering the question.

Two methodological notes for whoever picks this up:

- **Always run A/A first.** The apparent 4/4 sign pattern in the third campaign looked significant
  (p ≈ 0.008) but assumed queries were independent samples. They are not: a shared per-round
  ordering effect moves every query together, which the A/A run reproduces from nothing.
- **Drop iteration 0.** `drop_os_cache()` runs at the start of every invocation regardless of
  `--mode`, so the first iteration is always cold.

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

## Configuration

Both settings are accepted as DuckDB `SET` variables and in YAML under `sirius.operator_params`.

| Variable | Default | Description |
|----------|---------|-------------|
| `enable_runtime_size_estimation` | `false` | Master switch. **Off by default**: the mechanism is verified but no wall-clock benefit has been measured (see [What has been measured](#what-has-been-measured)). On restores nothing that is missing — it enables the projection path. |
| `size_estimate_safety_factor` | `1.0` | Multiplier applied to a *projected* total before sizing partitions; raise above 1.0 to bias toward more (smaller) partitions when projections undershoot. Measured totals are not scaled. |

See [Configuration](configuration.md#runtime-data-size-estimation) for the full reference.

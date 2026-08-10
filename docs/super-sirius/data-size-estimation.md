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

## Consumers

None in-tree yet. The API is exercised by `test/cpp/pipeline/test_data_size_estimator.cpp` against
a synthetic pipeline DAG, which covers each terminating case, the sample floors, overflow, and the
fan-in rules above.

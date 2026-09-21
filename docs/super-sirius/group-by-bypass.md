# Memory-aware group-by partition bypass — design

Prototype for point 2 of [sirius-db/sirius#1746](https://github.com/sirius-db/sirius/issues/1746).
Default off. Single admitted GPU only. Targets the grouped-aggregation `PARTITION` →
`MERGE_GROUP_BY` pair and nothing else.

Standalone base: upstream `dev` @ `b2961f8f`. This change does not include or require PR #1765
(projected partition sizing). cuCascade `e9929fff`, DuckDB `3ff87f1e`. Earlier benchmark results
were collected on the stacked prototype at `2a64af22`; they are historical measurements, not
a performance rerun of this standalone revision.

## 1. What the prototype does

When the automatic policy picks `AUTO > 1` partitions for a grouped aggregation, the prototype asks
one question: *would merging the whole partial input in a single unpartitioned pass fit in the
memory this GPU can still hand out?* If yes, and only if every eligibility gate below also passes,
it selects `P = 1`. Hash partitioning is then skipped (`sirius_physical_partition::execute` already
forwards the batch untouched when `_num_partitions < 2`), and the merge does the global
concatenate + groupby in one task.

If any gate fails it returns the automatic count unchanged, with a named reason.

### Decision order

Evaluated in `sirius_physical_grouped_aggregate_merge::get_partition_strategy`, which is the only
place a group-by partition count is decided. The order is fixed and each step returns:

1. **Compute `AUTO` first**, via the untouched `natural_num_partitions(total_bytes,
   hash_partition_bytes, num_gpus)`. Its arithmetic is not modified, and the repository
   pre-sizing uses the final chosen count.
2. **Forced-count override** — *not present on this baseline.* The investigation's
   `force_num_partitions` control lives only on the separate rerun branch. The policy input carries
   an `std::optional<int> forced_num_partitions` that is always `nullopt` here, so the precedence is
   defined and unit-testable: a nonzero forced count wins outright, the decision reason is
   `forced_override`, no memory model is computed, and the result is never labelled
   `bypass_selected`. If that control is ever merged, it must populate this field rather than
   short-circuit elsewhere.
3. **Setting off** → return `AUTO` immediately. The `PARTITION` does not build the metadata block at
   all (it is gated on the same flag), so the hot path does no extra repository walk, no batch
   inspection and no memory-policy work. Reason `disabled`, not logged.
4. **More than one admitted GPU** → return `AUTO`, reason `multi_gpu`. Read from the consumer's
   `_num_gpus` (the admitted count the pipeline converter already stamps) *and* cross-checked
   against the number of distinct memory spaces the input batches actually live in. Physical device
   0 is never assumed: the target space is whichever space the resident input is in, read from the
   batches themselves.
5. **`AUTO` already 1** → return 1, reason `already_one`. No activation is claimed, no memory model
   is computed, and no reservation floor is applied — this is the pre-existing choice, unchanged.
6. **`AUTO > 1`** → run the eligibility gates and the memory check. Select `P = 1` only if all pass.
7. **The count is then frozen.** The prototype writes the count through the same
   `partition_strategy` return value the existing code uses, so the existing latch (`_num_partitions`
   set once, under the partition's lock) and its barrier behaviour are untouched. Nothing
   re-evaluates the decision later.

The rule lives in the group-by consumer. `natural_num_partitions()` is shared with joins and is not
touched; the multi-GPU floor, GPU admission, and merge/aggregate semantics are not touched.

## 2. Eligibility envelope (v1)

All of these must hold. Each failure has its own reason string so rejections are diagnosable.

| Gate | v1 rule | Reason on failure |
|---|---|---|
| Upstream complete | The partition's input port's source pipeline reports `is_pipeline_finished()`. This is the honest test for *"the real batches are all here"*, rather than relying on a projected byte count. | `projected_input` |
| Metadata readable | Every batch yields a `cudf::table_view` with a consistent column count and schema. | `unknown_metadata` |
| Residency | Every batch is `Tier::GPU` and in the *same* memory space. No HOST/DISK/compressed/unknown carriers. | `unsupported_residency` |
| Group keys | Every key column is fixed-width integral: `BOOL8`, `INT8/16/32/64`, `UINT8/16/32/64`. | `unsupported_state` |
| Aggregate partial states | `has_avg` and `has_count_distinct` are both false, every `cudf_aggregates[i]` is `SUM`, `MIN`, `MAX`, `COUNT_ALL` or `COUNT_VALID`, and every physical aggregate column is fixed-width integral. | `unsupported_state` |
| Downstream | The parent chain from the merge to the first sink is unary and non-expanding — only `PROJECTION`, `FILTER`, `LIMIT`, `STREAMING_LIMIT` — and terminates at a `RESULT_COLLECTOR`. | `unsupported_downstream` |
| Row limit | Total partial rows ≤ `cudf::size_type` max (2,147,483,647). | `size_overflow` |
| Arithmetic | Every model term computes without saturating. | `size_overflow` |
| Budget | `modelled_additional × (1 + headroom) ≤ admissible additional budget`. | `insufficient_budget` |

Deliberately excluded in v1, all returning `AUTO` with SQL semantics unchanged: `STRING`, `LIST`,
`STRUCT`, `DICTIONARY32`, decimal and floating aggregate states, `AVG`, `COUNT(DISTINCT)`, fused
`TOP_N` / `ORDER BY` / joins / a second `GROUP BY` downstream, multi-GPU, and any projected
(incomplete) input.

The whitelist is checked against the **physical** partial-state columns read from the live
`cudf::table_view`, not against the SQL output types. That matters: a logical `COUNT` arrives at the
merge as a `COUNT_ALL`/`COUNT_VALID` partial state that is re-merged with `SUM`, and a
`COUNT(DISTINCT)` arrives as a `LIST` column — the two look identical in the SQL schema and are
classified oppositely here.

Nulls are supported, and are *not* a free pass: a nullable column is charged an explicit validity
mask in every term of the model below. Unknown metadata is represented as an absent optional, never
as a zero, so "no nulls" and "null count unknown" cannot be confused.

## 3. Memory model

### Two separate operations

- **Candidate selection** (this prototype): *will the unpartitioned merge fit?* An estimate.
- **Task admission** (unchanged): `gpu_pipeline_executor` acquires a real reservation from the
  memory space immediately before the task runs.

A free-memory snapshot is not a reservation, and this prototype does not turn one into one. It does
not hold a speculative reservation, does not reserve-and-release to "prove" capacity, does not block
on memory while holding the partition or merge lock, and adds no new reservation lifecycle. Actual reservation, OOM handling, downgrade and
retry all keep running exactly as before, even when the estimate said the candidate fits.

### Budget semantics (verified against the pinned cuCascade)

From `cucascade@e9929fff`. A GPU `memory_space` carries **two different limits**, and confusing
them is the easiest way to over-admit:

| Quantity | Value | Meaning |
|---|---|---|
| `memory_space::get_max_memory()` | `_memory_limit` = `memory_capacity × reservation_limit_fraction` | the **reservation** cap |
| `memory_space::get_available_memory()` | `_capacity - _total_allocated_bytes` | headroom against the larger **allocation** capacity |

`_total_allocated_bytes` is incremented by **both** direct allocations (`do_allocate_unmanaged`,
line 464) **and** reservations (`do_reserve` / `do_reserve_upto`, lines 527 and 537). Allocations
served from inside an already-attached reservation arena pass a `tracking_size` of zero
(`check_reservation_and_handle_overflow`), so they are not charged twice.

A reservation succeeds only while it fits under the **reservation** cap:

```
impl_type::reserve(bytes)  ->  do_reserve(bytes, _memory_limit)
                           ->  _total_allocated_bytes.try_add(bytes, _memory_limit)
```

So the budget this policy must compare against is

```
admissible_additional_budget = get_max_memory() - total_allocated_bytes      (clamped at 0)
```

read via the space's `reservation_aware_resource_adaptor`. Because live allocations and
outstanding reservation arenas share `_total_allocated_bytes`, this subtracts each exactly once —
there is no double-counting to avoid, and `get_total_reserved_memory()` must **not** be subtracted
on top.

**`get_available_memory()` is the wrong quantity** and an earlier draft of this design used it.
It measures headroom against `_capacity`, which is larger than `_memory_limit` whenever
`reservation_limit_fraction < 1`, so it over-states what a reservation can obtain by the difference between the allocation
capacity and reservation cap — and can exceed `get_max_memory()` outright. A unit test asserting
`budget <= get_max_memory()` caught this on a space configured with capacity 512 MiB and
reservation limit 384 MiB, where it reported 512 MiB of "available" memory.

`cudaMemGetInfo`, pool reservation size, and `get_max_memory()` alone are all insufficient and are
not used.

When the space does not expose the GPU reservation adaptor, the budget is left **absent** and the
policy rejects on `unknown_metadata` rather than assuming one.

### Terms

For `R` total partial rows across `N` batches, `G` fixed-width key columns of widths `wg_j`, `A`
fixed-width aggregate-state columns of widths `wa_j`, and `align(x) = rmm::align_up(x, 256)`
(`rmm::CUDA_ALLOCATION_ALIGNMENT`):

```
mask(R)         = align( align_up(ceil(R/8), 64) )        # per nullable column; cudf pads bitmasks to 64B
col(R, w)       = align(R * w)

concat_bytes    = Σ_keys col(R, wg_j) + Σ_aggs col(R, wa_j)
                + mask(R) * (number of columns that are nullable in any input)

output_bytes    = concat_bytes                             # worst case: every row is its own group

hash_set_bytes  = align(2 * R * 4)                         # cuco open-addressing set of size_type,
                                                           # 0.5 load factor
gather_map_bytes= align(R * 4)
sparse_agg_bytes= Σ_aggs col(R, wa_j) + mask(R) * nullable_agg_columns
```

```
additional_needed = input_materialization        # 0 in v1: residency is required, nothing to clone
                  + concat_bytes
                  + hash_set_bytes
                  + gather_map_bytes
                  + sparse_agg_bytes
                  + output_bytes                 # dense keys + dense aggregates, retained downstream
                  + downstream_copy_bytes        # output_bytes again iff a PROJECTION/FILTER sits
                                                 # between the merge and the collector

candidate_fits  = additional_needed + ceil(additional_needed * headroom)
                  <= admissible_additional_budget      # see 'Budget semantics' above
```

Every term uses `sirius::memory::saturating_add` / `saturating_mul`; any saturation rejects the
candidate as `size_overflow` rather than wrapping to a small number.

### Why each term

`cudf::concatenate` allocates one new fixed-width buffer per column over all `R` rows, plus a
concatenated validity mask for any column that is nullable in any input. The concatenated table
stays live for the whole groupby, so it is charged alongside, not instead of, the groupby's own
allocations. The **input batches are already-live allocations already charged to
`_total_allocated_bytes`**, so they appear nowhere in `additional_needed` — they are resident bytes
already subtracted from the budget. This is the "resident bytes already charged" versus "additional
bytes to reserve" split the model keeps strictly separate.

`cudf::groupby::groupby(keys, null_policy::INCLUDE).aggregate(...)` with only hash-supported
aggregations takes cuDF's hash path: a `cuco` set over the key rows, sparse per-row results, then a
gather down to the dense group set and the unique keys. **Worst-case group cardinality is taken as
`R`**, the full partial row count — the model never assumes the result is small, which is the
failure mode that would make a "reduction" query look safe and then OOM on near-unique keys.

### Stated gap — this is an estimate, not a proof

The cuco capacity (load factor 0.5, 4-byte slots) and the absence of additional transient scratch
inside cuDF's `compute_groupby` **have not been verified against the pinned cuDF 26.08.01
implementation**. The structure above is an estimate of that path, not a line-by-line
accounting of the pinned implementation. The single `headroom` parameter is the declared
empirical margin covering it.

Before this prototype is considered for anything beyond an experiment, the terms must be checked
against `cudf/src/groupby/hash/compute_groupby.cu` at the pinned version and the margin re-derived.
Until then the model is labelled an estimate everywhere it is reported. No "4 × input bytes" rule,
no physical-VRAM budget, and no byte threshold derived from the rerun's measured 4 GiB range is used
anywhere in this design.

### The one tuning parameter

`group_by_bypass_headroom_fraction`, default `0.25`, validated to `[0.0, 4.0]`. It is the only
empirical knob introduced, it is reported in every decision record, and it is the number to move if
measurement shows the model is optimistic. It deliberately folds the model margin and the
free-capacity headroom into a single documented value rather than a family of thresholds.

## 4. Admission integration

The modelled additional allocation requirement (before the headroom margin) is pushed into the
reservation request as a **floor**, not as a replacement:

- A new virtual `sirius_physical_operator::mandatory_peak_memory_floor(const input_stats&)` returns
  0 by default.
- `gpu_pipeline_task::get_estimated_reservation_size_info` takes the max of that floor over the
  pipeline's operators and applies it to `peak_memory_estimate` **in both branches** — with and
  without history.

That last point is the whole reason a new hook exists. `no_history_peak_memory_estimate` is only
consulted when `estimate_peak_memory()` returns `nullopt`; once a warm pipeline has history, the
history value is used verbatim and can be far smaller than the candidate needs. A floor that only
lives in the cold-start branch would silently disappear on the second query.

Ordering is preserved: `reservation_size = max(peak_estimate_with_floor + bytes_to_materialize,
retry_reservation_floor)`. A larger history estimate still wins, an OOM-derived retry floor still
wins, and no other operator's estimate is weakened. `MERGE_GROUP_BY` returns a nonzero floor only
when the prototype actually selected the bypass for it.

**Materialization is counted exactly once.** v1 requires the input to be resident in the target
space, so the model's `input_materialization` term is 0 and `bytes_to_materialize_input` remains the
sole source of that cost. If residency changes between decision and execution, the existing
preparation/admission path handles the clone and charges it there — the model does not also add it.

### The honest limitation in the current admission path

`gpu_pipeline_executor` clamps the request to `get_max_memory()`, and after a failed downgrade it
logs `"proceeding with partial reservation"` and runs the task anyway
(`src/pipeline/gpu_pipeline_executor.cpp`). So a raised floor makes the executor *request* the
modelled bytes; it does not guarantee they are granted.

v1 does not redesign that. Instead:

- Selection is conservative enough that the gap should not normally be reached (the budget read is
  the true admissible additional budget, plus headroom).
- `reservation_size_info` carries the `mandatory_floor`, and a bypass-selected merge that is admitted
  with `granted < floor` is logged at WARN and recorded in the decision/admission artifacts. A
  truncated reservation is never treated as evidence that the modelled work fits.
- If it OOMs, the existing OOM → reschedule → retry-floor path handles it. v1 adds **no**
  OOM-triggered repartitioning and **no** whole-query retry, and does not assume either exists. `P`
  is never mutated after routing begins.

## 5. Independence from projected partition sizing

This change keeps the existing full-input barrier and measures bytes from the completed input.
It does not add runtime size-estimation settings, partial barriers, or projection-history state.
Those are separate work in PR #1765 (point 3 of the issue).

The policy still rejects incomplete input with `projected_input`. This is a defensive gate for
callers that bypass normal scheduling and for possible future integration with early sizing;
it does not require that feature to exist. Tests exercise both metadata collection and the
unchanged full-input barrier without a projected-size fixture.

## 6. Observability

One decision record per group-by operator, at INFO, only when the setting is on, emitted once per
operator outside any per-row or per-batch loop. It carries: query and operator id, target device id,
`AUTO` and chosen `P`, reason, input bytes / rows / batches, completion provenance
(`upstream-complete` vs `projected`), each modelled term and the total, the budget and headroom, the
model version, and the type-eligibility verdict. Unknown values print as `unknown`, never as `0`.
No plans and no tables are serialized.

Reasons: `already_one`, `multi_gpu`, `projected_input`, `unsupported_state`,
`unsupported_downstream`, `unknown_metadata`, `unsupported_residency`, `size_overflow`,
`insufficient_budget`, `forced_override`, `bypass_selected`, `disabled`.

A candidate *decision* and a successful execution *admission* are reported as two different things,
and the report never presents the first as the second.

## 7. Validation and follow-up

See the [benchmark guide](../../test/tpch_performance/groupby_bypass/README.md) for GB300
SF100/SF1000 results and reproduction commands. The option remains off by default.

Before considering default-on, add admission that retains the automatic partition choice when
the required memory cannot be secured, and bounded recovery that repartitions preserved partial
batches after a bypass merge runs out of memory. Recovery must disable bypass for that retry,
release failed-attempt allocations, and prevent duplicate output. Validate this with injected
allocation failures, competing queries, and constrained-memory runs; also verify the memory model
against the pinned cuDF implementation. These changes are outside this prototype.

# Fuzzing Sirius: investigation and proposal (2026-09-22)

Status: implemented in [`test/fuzz/`](../test/fuzz/README.md) on 2026-09-22 (phase 0 and the
generator, oracle, variants, reduction and report of phase 1; CI wiring and the nightly are open).
Facts about Sirius were verified against `dev` at `a0ee56ed`; facts about external tools were
verified by reading their source on 2026-09-22 unless marked *(unverified)*.

**Correction to the first draft:** strict GPU mode already exists. With
`SET enable_duckdb_fallback = false`, `SiriusContext::OnFinalizePrepare` rethrows a plan-time
`NotImplementedException` to the client as `GPU plan generation failed: <reason>`
([sirius_context.cpp](../src/sirius_context.cpp), `rethrow_gpu_error_no_fallback`), and the
transparent operator surfaces runtime failures as `Sirius GPU execution failed: <message>`. The
harness classifies on those prefixes, so no new setting or SQL-visible counter was needed.

## Recommendation

Build a small **Sirius-specific generative differential fuzzer** (Python, `test/fuzz/`) whose
grammar is driven by an explicit feature config mirroring Sirius's supported surface, and whose
oracle is DuckDB CPU in the *same process* (`SET gpu_execution = false`). Layer a second, cheaper
oracle on top: run the same query under randomized Sirius settings (expression strategy, partition
sizes, GPU memory cap, GPU count) and require identical results. This is the "differential query
plans" idea from the literature, and Sirius's memory-tiering and partitioning code is exactly where
it pays off.

Reuse open source where it is a clean fit rather than adopting a framework: DuckDB's own `sqlsmith`
extension for **query reduction** (`reduce_sql_statement`), and the `tpch`/`tpcds` extensions for
realistic schemas. Neither SQLancer nor SQLsmith fits as the generator: SQLsmith's grammar has no
`GROUP BY`, `ORDER BY` or set operations and no toggles; SQLancer's DuckDB generator has no
subqueries, CTEs, set operations, window functions or DECIMAL, its toggles are per expression kind
rather than per operator, and it brings a Java toolchain for metamorphic oracles we mostly don't
need with a trusted reference in-process. SQLancer's TLP aggregate oracle is a reasonable later
add-on if we want a self-consistency check independent of DuckDB.

Two small Sirius changes make any fuzzer far more effective: a strict mode that turns plan-time
fallback into an error carrying the reason, and (optionally) SQL-visible execution counters.

## Why differential testing is unusually easy here

| Fact | Consequence |
|------|-------------|
| Sirius intercepts plain SQL in a post-optimizer hook; `SET gpu_execution` is session scoped ([sirius_extension.cpp](../src/sirius_extension.cpp)). | Run each query twice on one connection: identical parse/bind/optimize, only execution differs. No dialect translation, no schema sync. The classic differential-testing false positives (dialect and semantic drift between engines) mostly vanish. |
| Unsupported plan shapes throw `NotImplementedException` in the planner and fall back to CPU while `enable_duckdb_fallback` is on (the default); so do runtime GPU failures. With it off, both surface as errors with distinct prefixes (`GPU plan generation failed: ` / `Sirius GPU execution failed: `). | A "pass" under the default is a possible silent CPU run. The harness runs with `enable_duckdb_fallback = false` and classifies on the prefix: ran-on-GPU / plan-fallback (with reason) / runtime error. |
| Fallback counters exist only as `SiriusContext::transparent_execution_stats` (`successful_rebinds`, `fallbacks`, `executions`, `runtime_fallbacks`) in C++; the Catch2 fixture reads them via `registered_state` ([gpu_execution_fixture.hpp](../test/cpp/utils/gpu_execution_fixture.hpp)). | Not needed once fallback is disabled: success then implies a GPU execution. The harness still proves interception once per worker with the TEST ONLY `sirius_test_inject_transparent_gpu_error` option (a query that fails with the injected text went through the GPU operator). |
| GPU native scan requires a single-file block manager; in-memory tables never reach the GPU. Supported scans: `seq_scan`, `read_parquet`, `sirius_read_parquet`, `iceberg_scan` ([sirius_plan_get.cpp](../src/planner/sirius_plan_get.cpp)). | Generated tables must live in an attached `.duckdb` file or parquet, otherwise the harness silently compares CPU with CPU. |
| The build reports DuckDB `v1.5.5` (`OVERRIDE_GIT_DESCRIBE` in [cmake/CMakePresets.json](../cmake/CMakePresets.json)). Prebuilt `sqlsmith`, `tpch`, `tpcds` extensions exist for v1.5.5 on linux_amd64 and linux_arm64 (checked). Only `core_functions`, `parquet`, `sirius` are statically linked. | `INSTALL sqlsmith` works against the Sirius-built shell/Python. Vendor the extension files so CI stays offline, like the TPC-DS fixture DB. |
| `duckdb-python` is built from the submodule in CI ([check.yml](../.github/workflows/check.yml)); the `duckdb` shell is a build target. | A Python harness has a supported runtime. |
| Existing comparator: order-insensitive stringified rows, ordered variant, per-column relative tolerance with absolute floor, non-finite values exact only. | Reuse these rules; they encode the lessons from the TPC-H and null suites. |
| The #1095 null-data suites found six real GPU/CPU divergences by hand (Kleene `OR`, ungrouped `AVG` denominator, all-NULL column sentinel, `concat` vs `\|\|`, null-safe join, mixed null-safe keys). | Exactly the bug class a null-aware generator finds automatically. |

## Sirius's supported surface (what the toggles must mirror)

From [sirius_physical_plan_generator.cpp](../src/planner/sirius_physical_plan_generator.cpp),
[function_id.hpp](../src/expression/function_id.hpp), [aggregate_id.hpp](../src/expression/aggregate_id.hpp),
[from_duckdb.cpp](../src/expression/from_duckdb.cpp), [logical_type.hpp](../src/helper/logical_type.hpp):

| Area | On GPU today | Falls back today (toggle default **off**; flip when implemented) |
|------|--------------|------------------------------------------------------------------|
| Operators | scan, projection, filter, hash/ungrouped aggregate, comparison join, delim join (correlated subqueries), order by, top-N, limit/offset, VALUES, materialized CTE, UNION ALL, empty result | WINDOW, UNNEST, SAMPLE, CROSS PRODUCT, POSITIONAL/ASOF/ANY join, `SELECT DISTINCT` / `DISTINCT ON` (LogicalDistinct), RECURSIVE CTE, PIVOT, UNION (distinct), EXCEPT/INTERSECT, `LIMIT ... PERCENT`, GROUPING SETS/ROLLUP/CUBE, DML/DDL |
| Join types | INNER, LEFT, RIGHT, SEMI, ANTI, RIGHT_SEMI, RIGHT_ANTI, MARK, SINGLE; equality (hash) and inequality (nested loop); `IS NOT DISTINCT FROM` keys | MARK join mixing null-safe and plain keys |
| Aggregates | `sum`, `sum_no_overflow`, `count`, `count(*)`, `min`, `max`, `avg`, `first`; grouped `COUNT(DISTINCT)` | ungrouped `COUNT(DISTINCT)` (runtime fallback, #1218), aggregate `FILTER`/`ORDER BY` modifiers, every other aggregate |
| Scalar functions | `+ - * / // %`, `substring`, `like`/`not_like`, `contains`, `prefix`, `suffix`, `strlen`, `length`, `regexp_replace`, `concat`, `\|\|`, `year`..`microsecond`, `date_trunc`, `row`/`struct_pack`, `error` | every other function |
| Expressions | comparisons incl. `IS [NOT] DISTINCT FROM`, AND/OR (Kleene), NOT, IS [NOT] NULL, TRY, BETWEEN, CASE, CAST, COALESCE, IN-list | temporal<->numeric CAST (declined in `translate_cast`) |
| Types | BOOLEAN, TINYINT..BIGINT (+HUGEINT narrowed), unsigned variants, FLOAT, DOUBLE, DECIMAL(32/64/128), DATE, TIMESTAMP_S/MS/US/NS, VARCHAR, STRUCT, LIST, ARRAY | SQLNULL-typed columns |

Ship two profiles from one schema: **strict** (left column only; any fallback is a finding, either
a generator bug or an undocumented gap) and **frontier** (right column enabled too; asserts fallback
stays *clean*: correct result, no crash, no hang).

## Options considered

| Option | What you get | What's missing for Sirius | Verdict |
|--------|--------------|---------------------------|---------|
| **SQLancer** (Java, [DuckDB provider](https://github.com/sqlancer/sqlancer/blob/main/src/sqlancer/duckdb/DuckDBProvider.java)) | Oracles for DuckDB: NoREC, TLP (WHERE/HAVING/GROUP BY/AGGREGATE/DISTINCT), query partitioning. Toggles in [DuckDBOptions.java](https://github.com/sqlancer/sqlancer/blob/main/src/sqlancer/duckdb/DuckDBOptions.java): `--test-functions`, `--test-casts`, `--test-between`, `--test-in`, `--test-case`, `--test-binary-logicals`, `--test-binary-comparisons`, per-type constant gates, `--max-expression-depth`, `--use-reducer`. Actively developed (last commit 2026-09-20, no release since v2.0.0). | Generates no window functions, CTEs, subqueries, EXISTS or set operations for DuckDB; joins limited to INNER/NATURAL/LEFT/RIGHT; no DECIMAL; NULL literal probability hard-coded. JDBC pinned at 1.3.0.0 (1.5.5.1 exists). No connection-init hook: `LOAD` + `SET gpu_execution` is a code change. A GPU-vs-CPU oracle is ~150 LOC following `MySQLDQPOracle`, but `ComparatorHelper` compares only the first column as a set and floats as strings. | Not the primary tool. Candidate for a later TLP-aggregate self-consistency layer. |
| **SQLsmith** / DuckDB's [`sqlsmith` extension](https://github.com/duckdb/duckdb-sqlsmith) | Random query stream in-process: `sqlsmith(seed, max_queries, max_query_length, exclude_catalog, dump_all_queries, dump_all_graphs, verbose_output, complete_log, log)`, `fuzzyduck`, `fuzz_all_functions`, `reduce_sql_statement(sql)`. Nightly-proven by [duckdb-fuzzer](https://github.com/duckdb/duckdb-fuzzer) (crash-only oracle, dedup on error title, auto-files and auto-closes issues via `run_fuzzer.py`). | Grammar is `select [distinct] ... from ... where ... [limit]`: **no GROUP BY, HAVING, ORDER BY, or set ops**; aggregates appear only inside window functions; `LIMIT` without `ORDER BY`; ~12% DML. No toggles (hard-coded dice), no null control. `dump_all_queries` writes to stdout; the usable feed is `complete_log='file.sql'`. It opens its own internal connection, so a session-scoped `SET gpu_execution` on ours does not reach it. | Use `reduce_sql_statement` for shrinking; optionally replay its log as a low-priority crash stream. Not the generator. |
| **CockroachDB's Go sqlsmith** ([sqlsmith.go](https://github.com/cockroachdb/cockroach/blob/master/pkg/internal/sqlsmith/sqlsmith.go)) | The best-known toggle design: 47 options (`DisableWindowFuncs`, `DisableAggregateFuncs`, `DisableWith`, `DisableJoins`, `DisableCrossJoins`, `DisableLimits`, `DisableNondeterministicFns`, `DisableDecimals`, `IgnoreFNs(regex)`, `SimpleDatums`, `FavorCommonData`, `UnlikelyRandomNulls`, `SetComplexity`, `SetScalarComplexity`, `OutputSort`, ...), and bundles (`CompareMode`, `PostgresMode` = "only queries whose semantics agree between two engines"). Nightly `costfuzz`, `unoptimized-query-oracle` and TLP roachtests with a 43-pass AST reducer. | Go `internal` package: cannot be imported; emits CockroachDB dialect and needs `crdb_internal` catalog tables; NULL rate is a two-position switch (1/6 vs 1/20 in expressions, 1/10 in data, hard-coded). | Copy the design, not the code. |
| **Engine fuzzers**: [Velox](https://velox-lib.io/docs/develop/testing/fuzzer) expression/aggregation/join/window fuzzers | Closest precedent for an accelerator under a reference engine: `--null_ratio` (0.1), `--only`/skip lists with issue links per entry, `--special_forms`, `--assign_function_tickets` (per-function weights), `--enable_spill`, `--retry_with_try`, `--seed`, `--repro_persist_path`, automatic minimal-subexpression finding. Two-layer oracle: DuckDB/Presto as reference (unsupported = counted, not failed) plus self-differential across plan variants (single vs partial/final aggregation, with/without spill, join side flip). Multiset compare; float epsilon only when non-float columns form unique keys; per-function data restriction (`DataSpec{includeNaN, includeInfinity}`); custom verifiers for approximate/nondeterministic aggregates. | Tied to Velox's plan builder; nothing to link against. DataFusion Comet **deleted** its standalone `fuzz-testing` module in May 2026: it never ran in CI, rarely got run by hand, and covered few expressions ([#3350](https://github.com/apache/datafusion-comet/issues/3350)). What survives there is fixed-seed in-process fuzz suites on every PR plus SQL-file tests with per-query annotations (`query expect_fallback(reason)`, `query tolerance=`, `query ignore(<issue link>)`). spark-rapids tooling *(unverified this pass)*. | Copy the design. |
| **Custom generator** (Python, `test/fuzz/`) | Exactly the toggles requested, null control, high GPU hit rate, config randomization, repro artifacts shaped like our Catch2 tests. Prior art to crib from: Impala's open-source [random query generator](https://github.com/apache/impala/tree/master/tests/comparison) (Python; weighted `query_profile.py`, PostgreSQL reference), which Databricks forked into SparkFuzz. | About 1-2 weeks to a useful v1; grammar upkeep as features land | **Primary tool.** |

## Proposed design

```
test/fuzz/
  config/strict.toml, frontier.toml     # feature/null/data/oracle settings (schema below)
  known_issues.toml                     # pattern -> issue #; matched findings are counted, not failed
  siriusfuzz/
    schema_gen.py     # random tables -> attached .duckdb file; null ratio, shared value pool, edge values
    sqlast.py         # typed SQL AST with generic traversal (rendering + reduction)
    query_gen.py      # typed AST generator gated by the feature config; emits DuckDB SQL
    session.py        # connection: LOAD sirius, strict settings, attached datasets, timeouts, canary
    classify.py       # verdicts from error prefixes / text
    runner.py         # worker processes: CPU run, GPU run, variants, ambiguity filter; orchestrator
    compare.py        # multiset / ordered compare, float tolerance, NULL, decimals
    reduce.py         # AST shrinking, then reduce_sql_statement() when sqlsmith is available
    report.py         # dedup by signature; repro dir (query, reduced, dataset.sql, config, Catch2 snippet)
    cli.py            # pixi run -e duckdb-python fuzz run --seed 42 --duration 30m --workers 2
  tests/              # unittest suite, runs without a GPU
```

**Feature config.** One TOML file (stdlib `tomllib`, so no new pixi dependency; the sketch
below is the original YAML and maps 1:1 onto `config/strict.toml`); every default matches
today's GPU surface, so enabling window functions later is a one-line change.

```yaml
features:
  window_functions: false
  distinct: false                 # SELECT DISTINCT -> LogicalDistinct falls back today
  grouping_sets: false
  set_ops: {union_all: true, union: false, except: false, intersect: false}
  joins:
    types: [inner, left, right, semi, anti]      # full/cross off
    inequality: true
    null_safe_keys: true          # IS NOT DISTINCT FROM
    max_tables: 4
  subqueries: {exists: true, in: true, scalar: true, correlated: true, max_depth: 2}
  cte: {materialized: true, recursive: false}
  aggregates: [sum, count, count_star, min, max, avg]   # first excluded: nondeterministic
  aggregate_modifiers: {distinct: grouped_only, filter: false, order_by: false}
  scalar_functions: [add, sub, mul, div, int_div, mod, substring, like, contains, prefix, suffix,
                     strlen, length, regexp_replace, concat, concat_operator, year, month, day,
                     hour, minute, second, millisecond, microsecond, date_trunc]
  function_weights: {like: 3, regexp_replace: 1}        # Velox-style tickets, optional
  expressions: {case: true, coalesce: true, in_list: true, between: true, is_null: true, try: false}
  casts: {enabled: true, temporal_numeric: false, targets: [BIGINT, UBIGINT, DOUBLE, DECIMAL, VARCHAR]}
  order_by: {enabled: true, nulls_first_last: true}
  limit: {enabled: true, offset: true, percent: false, require_total_order: true}
  types: [BOOLEAN, TINYINT, SMALLINT, INTEGER, BIGINT, UTINYINT, USMALLINT, UINTEGER, UBIGINT,
          FLOAT, DOUBLE, DECIMAL, DATE, TIMESTAMP, TIMESTAMP_S, TIMESTAMP_MS, TIMESTAMP_NS, VARCHAR]
  complexity: {query: 0.2, scalar: 0.2}   # recursion probabilities (CockroachDB's two knobs)
nulls:
  column_null_ratio: 0.15         # per column; 0 disables NULLs in data
  all_null_column_probability: 0.05
  null_literals_in_queries: true  # NULL in CASE/COALESCE/IN/comparisons
data:
  tables: 3..6, rows: 100..50000
  common_value_ratio: 0.4         # draw from a small shared pool so joins and predicates hit
  numeric_range: small            # avoid overflow: DuckDB raises, cuDF wraps
  strings: {charset: ascii|unicode, empty_string_ratio: 0.05, max_len: 40}
  edge_values: 0.1                # INT min/max, epoch edges, -0.0; NaN/inf off by default
oracle:
  float_rel_tol: 1e-9
  ordered_compare_only_when_total_order: true
  ambiguity_filter: true          # re-run with permuted base-table row order before reporting
  on_plan_fallback: skip|count|fail       # strict: fail
  known_issues: test/fuzz/known_issues.yaml   # pattern -> issue #; skipped and counted
variants:                         # DQP-style: same query, different Sirius settings, identical results required
  expression_evaluator_strategy: [materialize, ast_interpret, ast_jit]
  hash_partition_bytes: [1MB, 8MB, 100MB]
  max_build_hash_table_bytes: [1MB, 90MB]
  max_sort_partition_bytes: [0, 1MB]
```

**Generator.** Type-directed and scope-aware: pick a result type, then build an expression of that
type from enabled functions over in-scope columns (SQLsmith's typed AST and CockroachDB's smither
both work this way; ours uses the closed function list from `function_id.hpp`). Query shape is
`SELECT ... FROM ... [JOIN] [WHERE] [GROUP BY ... HAVING] [ORDER BY] [LIMIT]`, optionally wrapped
in CTEs, `UNION ALL` or subqueries per config. Every query is emitted with its seed and config hash.

**Runner.** A worker child process per N queries: fresh file-backed DB, generated data,
`LOAD sirius`, `SET enable_duckdb_fallback = false`. Per query: GPU run, read counters, CPU run,
compare; then a few variant runs with randomized `SET`s from `variants`, each compared with the
GPU baseline. Child processes give crash isolation (a GPU fault kills one worker) and hang detection
(wall-clock timeout, then kill). YAML-level Sirius knobs (GPU `usage_limit_fraction` to force
downgrade/spill, `num_gpus` 1 vs 2) vary per worker process.

**Classification per query** (this is what makes the numbers meaningful):

| CPU result | GPU result | Verdict |
|-----------|-----------|---------|
| rows | rows, counters show 1 execution / 0 fallbacks | compare; mismatch that survives the ambiguity filter = **logic bug** |
| rows | plan-time fallback | strict: **coverage gap** (record the reason); frontier: compare, mismatch = bug in the fallback path |
| rows | error (runtime, fallback disabled) | **GPU error**, dedup by message |
| error | any | skip the query (CockroachDB's rule: an error in the reference run is not interesting), unless the GPU result is a crash |
| rows | error containing an internal/CUDA error | **GPU internal error** |
| any | timeout or crash | **hang / crash**, highest priority |

**Comparison rules** (from the existing fixture plus Velox and CockroachDB practice): compare as a
multiset unless the query's `ORDER BY` is a total order over the projected columns; exact except
FLOAT/DOUBLE columns, which get a relative tolerance with absolute floor (GPU reductions reorder
`SUM`/`AVG`; CockroachDB uses 1e-14 relative / 1e-15 margin, Velox 1e-5 absolute then relative);
apply the float tolerance only when the non-float columns form unique keys, otherwise exact; NaN/inf
exact only; NULL equals only NULL; DECIMAL compared as scaled integers; never generate `first()`,
`random()`, `now()`; `LIMIT` requires a total order or compares counts only. Before reporting a
mismatch, re-run with the base tables' row order permuted: if the verdict changes, the query is
ambiguous, not buggy (the DQP paper's filter, which eliminated its false alarms). A SQL-level
alternative to string comparison is the symmetric `EXCEPT ALL` through the `gpu_execution()` table
function, as the older suites did; it uses DuckDB's own type-aware equality but cannot do float
tolerance, so reserve it for float-free queries.

Semantic differences the fuzzer will hit early and that need a bug-or-documented decision: integer
overflow (DuckDB raises, cuDF wraps), division by zero (DuckDB returns NULL), `%` and `//` on
negatives, `SUBSTRING` with non-positive positions, `LIKE` escapes, `date_trunc` parts, `-0.0`
in ordering and grouping (Comet ships an `--exclude-negative-zero` data flag for exactly this). Each
confirmed divergence goes either into a fix or into `known_issues.yaml` with its issue number, the
way `[!shouldfail]` quarantines work in the Catch2 suites; Velox's rule that every skip carries an
issue link is worth adopting verbatim.

**Reduction and repro.** On a finding, call `reduce_sql_statement(query)` from the `sqlsmith`
extension (returns candidate simplifications), re-run the oracle on each, recurse on the smallest
that still fails, then write `repro/<signature>/{schema.sql,data.parquet,query.sql,config.yaml,seed}`
and a ready-to-paste `compare_gpu_vs_cpu("...")` Catch2 snippet. Dedup on a normalized signature
(operator kinds in the reduced query + error class) so one bug produces one report per night.

## Sirius changes that make this work (small, independent PRs)

1. **Strict GPU mode**: *not needed.* `SET enable_duckdb_fallback = false` already returns a
   plan-time `NotImplementedException` to the client as `GPU plan generation failed: <reason>`
   (see the correction at the top). The harness builds the *reason histogram* ("Window not
   supported": 312, "Unsupported expression in aggregate: ...": 41) from those messages; it is in
   `summary.txt` of every run and doubles as a coverage report and roadmap signal.
2. **SQL-visible counters**: *not needed* for classification (see above); still a nice-to-have for
   asserting "exactly one GPU execution" from SQL.
3. **Vendor** `sqlsmith.duckdb_extension` for v1.5.5 (linux_amd64/arm64) next to the TPC-DS fixture
   so `reduce_sql_statement` works offline; or add
   `duckdb_extension_load(sqlsmith DONT_LINK GIT_URL https://github.com/duckdb/duckdb-sqlsmith GIT_TAG <pin>)`
   to [extension_config.cmake](../extension_config.cmake) (DuckDB v1.5.5's own pin is in
   `duckdb/.github/config/extensions/sqlsmith.cmake`). The harness currently tries `INSTALL
   sqlsmith` at runtime and falls back to AST-level reduction alone when that fails.

## Phased plan

| Phase | Scope | Effort |
|-------|-------|--------|
| 0. Enablement | ~~Strict-mode PR~~ (exists as `enable_duckdb_fallback = false`); vendored sqlsmith extension (open; runtime `INSTALL` for now); `test/fuzz/` runner + comparator (**done**); `fuzz replay-file` classifies an external query stream such as a `sqlsmith(complete_log=...)` log (**done**). Also try the Photon trick: run DuckDB's own sqllogictest corpus (`duckdb/test/sql/**`) through the Sirius-linked `unittest` runner with `--force-storage` and `gpu_execution = true`, so the host engine's thousands of recorded answers become a free oracle; needs an hour's feasibility check (extension loading in the runner, file-backed storage) | 2-3 days |
| 1. Generator v1 | Feature config + typed generator over the strict surface; null/data controls; setting variants; reduction; repro artifacts; `pixi run -e duckdb-python fuzz` task (**all done**, with a unit suite that gates SQL validity and feature coverage on CPU); a 2-3 minute fixed-seed smoke run wired into the PR test workflow (**open**: the GPU runner needs the submodule-built Python module, see the README); run against dev and triage the first batch (**in progress**) | 1-2 weeks |
| 2. Nightly | Scheduled workflow on `gpu-2xt4` (no scheduled workflow exists today), 45-min budget, date-derived seed, artifacts uploaded, `known_issues.yaml`; weekly run under `compute-sanitizer --tool memcheck` with a smaller budget; a run with GPU memory capped low to exercise downgrade/spill | 3-4 days |
| 3. Later | Frontier profile as features land; TPC-H/TPC-DS schemas with null injection as a second data source; automatic issue filing like duckdb-fuzzer; SQLancer TLP-aggregate as a DuckDB-independent oracle if wanted | as needed |

## What the literature and other engines say (short)

- **Oracles.** Differential testing against a trusted reference is the primary oracle when one is
  available (RAGS, [Slutz 1998](https://www.vldb.org/conf/1998/p618.pdf)). Of the metamorphic
  oracles, **DQP** (same query, different plans/settings; [SIGMOD 2024](https://bajinsheng.github.io/assets/pdf/dqp_sigmod24.pdf))
  is the best add-on: under 100 lines per DBMS, and 81% of its logic bugs were missed by NoREC and
  TLP combined. **TLP** ([OOPSLA 2020](https://www.manuelrigger.at/preprints/TLP.pdf)) aggregate
  partitioning directly stresses partial-aggregate merging, a GPU risk area. **NoREC** is subsumed by
  a reference engine. **PQS** ([OSDI 2020](https://www.usenix.org/system/files/osdi20-rigger.pdf))
  is high effort per engine and validates one row; skip. EET (OSDI 2024) and CODDTest (SIGMOD 2025,
  12 DuckDB bugs) are moderate-value plan perturbations. LLM-generated oracles without a formal
  equivalence prover produced 100% false positives in Argus (2025); do not go there.
- **False positives.** DQP's row-permutation ambiguity filter removed its false alarms; CODDTest
  and CockroachDB both exclude nondeterministic functions and extreme floats; CockroachDB skips any
  query whose reference run errors and fails the second run only on "internal error".
- **Photon** ([SIGMOD 2022](https://cs.stanford.edu/~matei/papers/2022/sigmod_photon.pdf), section 5.6) is
  the closest architectural precedent: a native engine under a host engine with per-operator
  fallback, where "the same query expression can run in either Photon or Spark ... and the results
  must be consistent". Its testing is three-tier (expression unit tests over all null/no-null
  specializations, end-to-end Spark-vs-Photon comparisons, random data+query fuzzers), it runs the
  **entire Spark SQL test suite with Photon forced on** (which found memory-corruption bugs), and it
  keeps a "behavior whitelist" for deliberate, performance-motivated divergences (decimal). SparkFuzz
  ([DBTest 2020](https://ir.cwi.nl/pub/30222/3395032.3395327.pdf), never open-sourced) found 40+
  bugs a year with tiny data (5 tables x 5 columns x 5 rows) and reports that operator coverage
  saturates after about 100 queries, which is why a short fixed-seed run per PR is worthwhile.
- **Comet's post-mortem** is the most useful precedent for an accelerator project: the standalone
  tool died because it was not in CI; the replacement is fixed-seed, small (about 1000 rows),
  in-process suites that finish inside the PR budget, a strict-testing ratchet that fails a weak
  assertion once the plan runs fully native, an accelerator-ran check that walks the plan tree
  against an allow-list rather than string-matching, NaN payload canonicalization before compare,
  and a guard that the generator still emits every type it claims to (otherwise a suite passes by
  testing nothing).
- **DuckDB and SQLancer.** DuckDB ran SQLancer in CI from 2020 to about 2023 (75 bugs, the
  second-most of any SQLancer target; regression tests live in `test/issues/rigger/`), then dropped
  it for the in-house sqlsmith/fuzzyduck suite plus OSS-Fuzz. SQLancer++ (ASPLOS 2026, Java) is an
  adaptive variant that found 10 DuckDB bugs. No Python SQLancer-like library exists, which is one
  more reason to write the generator ourselves.
- **Production practice.** DuckDB runs sqlsmith/fuzzyduck nightly with crash-only detection and
  auto-filed issues; CockroachDB runs `costfuzz`, `unoptimized-query-oracle` and TLP nightly with
  a Smither option bundle (`DisableMutations`, `DisableNondeterministicFns`, `UnlikelyRandomNulls`,
  `FavorCommonData`, `DisableCrossJoins`, `DisableDecimals`, `SetComplexity(.3)`,
  `SetScalarComplexity(.1)`) that is a good starting point for our strict profile; Velox runs its
  fuzzers on a schedule with DuckDB and Presto as references and a plan-variant self-check.

## References

- SQLancer: [DuckDBOracleFactory.java](https://github.com/sqlancer/sqlancer/blob/main/src/sqlancer/duckdb/DuckDBOracleFactory.java), [MySQLDQPOracle.java](https://github.com/sqlancer/sqlancer/blob/main/src/sqlancer/mysql/oracle/MySQLDQPOracle.java) (template for a GPU/CPU oracle), [ComparatorHelper.java](https://github.com/sqlancer/sqlancer/blob/main/src/sqlancer/ComparatorHelper.java)
- DuckDB: [duckdb-sqlsmith](https://github.com/duckdb/duckdb-sqlsmith) (`src/sqlsmith_extension.cpp`, `scripts/run_fuzzer.py`, `reduce_sql.py`), [duckdb-fuzzer](https://github.com/duckdb/duckdb-fuzzer), upstream [anse1/sqlsmith](https://github.com/anse1/sqlsmith)
- CockroachDB: [sqlsmith.go](https://github.com/cockroachdb/cockroach/blob/master/pkg/internal/sqlsmith/sqlsmith.go), [query_comparison_util.go](https://github.com/cockroachdb/cockroach/blob/master/pkg/cmd/roachtest/tests/query_comparison_util.go), [costfuzz.go](https://github.com/cockroachdb/cockroach/blob/master/pkg/cmd/roachtest/tests/costfuzz.go), [unoptimized_query_oracle.go](https://github.com/cockroachdb/cockroach/blob/master/pkg/cmd/roachtest/tests/unoptimized_query_oracle.go), [floatcmp.go](https://github.com/cockroachdb/cockroach/blob/master/pkg/testutils/floatcmp/floatcmp.go), [reduce](https://github.com/cockroachdb/cockroach/blob/master/pkg/cmd/reduce/main.go), [blog: SQLsmith randomized SQL testing](https://www.cockroachlabs.com/blog/sqlsmith-randomized-sql-testing/)
- Velox: [fuzzer docs](https://velox-lib.io/docs/develop/testing/fuzzer), [exec/fuzzer](https://github.com/facebookincubator/velox/tree/main/velox/exec/fuzzer), [PrestoSkippedFunctions.cpp](https://github.com/facebookincubator/velox/blob/main/velox/expression/fuzzer/PrestoSkippedFunctions.cpp), [QueryAssertions.cpp](https://github.com/facebookincubator/velox/blob/main/velox/exec/tests/utils/QueryAssertions.cpp)
- Comet: [removal PR #4085](https://github.com/apache/datafusion-comet/pull/4085), [issue #3350](https://github.com/apache/datafusion-comet/issues/3350), [SQL file tests](https://github.com/apache/datafusion-comet/blob/main/docs/source/contributor-guide/sql-file-tests.md), [FuzzDataGenerator.scala](https://github.com/apache/datafusion-comet/blob/main/spark/src/main/scala/org/apache/comet/testing/FuzzDataGenerator.scala), [CometPlanChecker.scala](https://github.com/apache/datafusion-comet/blob/main/spark/src/test/scala/org/apache/spark/sql/comet/CometPlanChecker.scala)
- DuckDB and SQLancer history: [PR #3818](https://github.com/duckdb/duckdb/pull/3818), [issue #5031](https://github.com/duckdb/duckdb/issues/5031), [sqlancer/bugs](https://github.com/sqlancer/bugs), [SQLancer++](https://github.com/suyZhong/SQLancerPlusPlus) ([arXiv:2503.21424](https://arxiv.org/abs/2503.21424))
- Databricks: [Photon (SIGMOD 2022)](https://cs.stanford.edu/~matei/papers/2022/sigmod_photon.pdf), [SparkFuzz (DBTest 2020)](https://ir.cwi.nl/pub/30222/3395032.3395327.pdf), [Impala RQG](https://github.com/apache/impala/tree/master/tests/comparison)
- Papers: [RAGS](https://www.vldb.org/conf/1998/p618.pdf), [PQS](https://www.usenix.org/system/files/osdi20-rigger.pdf), [NoREC](https://arxiv.org/abs/2007.08292), [TLP](https://www.manuelrigger.at/preprints/TLP.pdf), [DQP](https://bajinsheng.github.io/assets/pdf/dqp_sigmod24.pdf), [EET](https://www.usenix.org/system/files/osdi24-jiang.pdf), [CODDTest](https://arxiv.org/html/2501.11252v1), [Argus](https://arxiv.org/html/2510.06663v1)

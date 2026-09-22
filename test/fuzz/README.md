# siriusfuzz: differential fuzzing for Sirius

A generative fuzzer that runs random SQL on the GPU and on DuckDB CPU in the same process and
reports every difference. It implements the proposal in
[docs/fuzzing-investigation-2026-09-22.md](../../docs/fuzzing-investigation-2026-09-22.md).

## How it works

1. **Schema and data** (`schema_gen.py`): 3 to 6 tables with a shared integer key column and
   random columns over the configured types; per-column NULL ratios, all-NULL columns, a shared
   value pool so joins and predicates hit, optional edge values.
2. **Queries** (`query_gen.py`): a typed, scope-aware generator over the GPU-supported surface
   (joins incl. null-safe and inequality keys, grouped and ungrouped aggregates, correlated
   `EXISTS` / `IN` / scalar subqueries, materialized CTEs, `UNION ALL`, `ORDER BY ... NULLS
   FIRST|LAST`, `LIMIT/OFFSET`, `CASE`, `COALESCE`, `IN`, `BETWEEN`, `LIKE`, casts, the closed
   scalar-function list). Every construct is gated by the TOML feature config and counted, so a
   profile that claims a feature but never emits it is flagged in the summary.
3. **Oracle** (`session.py`, `runner.py`): each query runs on CPU (`SET gpu_execution = false`),
   then on GPU with `SET enable_duckdb_fallback = false`. In that mode Sirius surfaces a plan-time
   rejection as `GPU plan generation failed: <reason>` and a runtime failure as
   `Sirius GPU execution failed: <message>`, which is what the classifier keys on. After a match,
   the query re-runs under randomized Sirius settings (`variants`) and must match the GPU
   baseline. A mismatch is re-checked on CPU over the same rows inserted in a different order; if
   the CPU answer changes, the query is ambiguous (`LIMIT` without a total order, ...) and not a
   bug.
4. **Comparison** (`compare.py`): multiset unless `ORDER BY` covers every output column; exact
   except FLOAT/DOUBLE columns (relative tolerance with absolute floor); NULL only equals NULL;
   NaN/inf exact only; DECIMAL exact.
5. **Reduction** (`reduce.py`): greedy shrinking on the generator's AST (drop clauses, join
   sides, select items, set-op arms; hoist or replace subexpressions), then optionally
   `reduce_sql_statement()` from DuckDB's `sqlsmith` extension when it can be installed.
6. **Report** (`report.py`): findings deduplicated by signature (verdict plus normalized
   reason, or the reduced query's operator shape), one directory per finding with the query,
   the reduced query, the dataset as SQL, the config, and a Catch2 snippet; a summary with
   verdict counts and the plan-time fallback reason histogram.

Generated tables live in an ATTACHed file-backed database and are CHECKPOINTed, because
in-memory tables never reach the GPU native scan. Each worker is a separate process, so a GPU
fault or hang costs one worker (the orchestrator records the in-flight query and respawns).

## Verdicts

| Verdict | Meaning | Finding? |
|---------|---------|----------|
| `ok` | GPU rows match CPU rows, and every setting variant matched the GPU baseline | no |
| `cpu_error` / `cpu_timeout` | the reference run failed; the query is skipped and counted | no |
| `ambiguous` | mismatch that changes under permuted row order (nondeterministic query) | no |
| `mismatch` | GPU rows differ from CPU rows | **yes** |
| `variant_mismatch` | same query, one Sirius setting changed, different rows | **yes** |
| `plan_fallback` | Sirius declined the plan; the reason is recorded | strict: yes; frontier: counted |
| `fallback_mismatch` | frontier only: the CPU fallback path returned wrong rows | **yes** |
| `gpu_error` / `gpu_internal_error` / `gpu_oom` | the GPU run raised | **yes** |
| `timeout` | the GPU run exceeded `oracle.query_timeout_seconds` | **yes** (hang candidate) |
| `crash` | the worker process died during the query | **yes** |
| `known_issue` | a finding matching `known_issues.toml`; counted, not failed | no |

## Running

The harness needs the Python `duckdb` module built from this repo's submodule (the stock wheel
is not ABI-safe for loading the Sirius extension) and a built extension:

```bash
pixi run make                                   # build/release/extension/sirius/sirius.duckdb_extension
pixi run -e duckdb-python build-duckdb-python   # once; builds the Python module from ./duckdb

pixi run -e duckdb-python fuzz run --seed 42 --duration 30m --workers 2
pixi run -e duckdb-python fuzz run --config test/fuzz/config/frontier.toml --queries 2000
pixi run -e duckdb-python fuzz run --set features.window_functions=true --set oracle.on_plan_fallback=count
pixi run -e duckdb-python fuzz replay test/fuzz/out/run-.../findings/000-mismatch-abcd1234
pixi run -e duckdb-python fuzz replay-file sqlsmith.log --dataset-seed 3   # classify an external query stream
pixi run -e duckdb-python fuzz show-config --config test/fuzz/config/strict.toml
pixi run -e duckdb-python fuzz-test                                        # harness unit tests, no GPU
pixi run -e duckdb-python fuzz selftest                                    # CPU-vs-CPU pipeline check, no GPU
```

`--sirius-config` selects the Sirius YAML (default `test/cpp/integration/integration.yaml`);
repeat it to round-robin worker processes across configs, e.g. a low GPU memory cap to force
downgrade and spill. `--cpu-only` runs without the extension and compares CPU with CPU, which is
how the generator and comparator are validated on machines without a GPU.

Outputs go to `test/fuzz/out/run-<timestamp>-seed<seed>/`:

```
summary.txt / summary.json      verdict counts, unique findings, fallback reason histogram, feature stats
queries.jsonl                   one record per query
datasets/w<worker>-d<n>.sql     every generated dataset (CREATE + INSERT + CHECKPOINT)
logs/w<worker>-s<spawn>.stderr  each worker process's stderr (Sirius backtraces land here)
findings/<n>-<verdict>-<hash>/  query.sql, reduced.sql, dataset.sql, detail.txt, meta.json,
                                config.toml, repro_catch2.cpp, query-<k>.sql for further
                                reproducers of the same signature
```

A crash is attributed to the in-flight query; its reason is the `std::terminate` message or
the signal plus the first extension frames below Sirius's signal handler (symbolized with
`addr2line` when available). Workers are respawned after crashes and hangs up to
`--max-respawns` (default 200).

A finding directory replays on its own: `fuzz replay <dir>` loads `dataset.sql`, runs
`reduced.sql` (or `query.sql` with `--original`) and prints the verdict.

## Profiles

`config/strict.toml` enables only what runs on the GPU today; it is identical to the built-in
defaults (a unit test enforces this) and treats any plan-time fallback as a finding.
`config/frontier.toml` also enables window functions, `DISTINCT`, grouping sets, `FULL`/`CROSS`
joins, `UNION`/`EXCEPT`/`INTERSECT`, uncorrelated subqueries, `TRY`, temporal-numeric and
to-VARCHAR casts; fallbacks are counted and the fallback path is checked for correct results.
When a feature lands on the GPU, flip its key in `strict.toml`.

`known_issues.toml` quarantines confirmed divergences by regex; every entry carries an issue
link. Divergences the fuzzer is expected to surface early and that need a fix-or-document
decision: integer `//` and `%` by zero (DuckDB returns NULL), `SUBSTRING` with non-positive
positions, `LIKE` escapes, `date_trunc` parts, `-0.0` in grouping and ordering, number
formatting in casts to VARCHAR (off by default).

## Proposed CI step

A fixed-seed smoke run belongs in the GPU test workflow from day one. The test runner currently
receives only `build/release/`, so the Python module has to be built there or shipped as an
artifact; that plumbing is the open item. The step itself:

```yaml
- name: Fuzz smoke run (fixed seed)
  run: |
    set -o pipefail
    pixi run -e duckdb-python fuzz run --seed ${{ github.run_number }} --duration 3m --workers 2 \
      --fail-on-findings 2>&1 | tee /tmp/fuzz.log
- uses: actions/upload-artifact@...
  if: always()
  with:
    name: fuzz-run-${{ github.run_id }}
    path: test/fuzz/out/
```

## Layout

```
test/fuzz/
  config/strict.toml, frontier.toml   feature/null/data/oracle/variant settings
  known_issues.toml                   quarantined findings with issue links
  siriusfuzz/
    config.py      TOML loading, validation, --set overrides
    sqltypes.py    type model and literal rendering
    schema_gen.py  random tables and rows
    sqlast.py      typed SQL AST with generic traversal
    query_gen.py   feature-gated generator
    compare.py     result comparison rules
    classify.py    verdicts from error text
    session.py     connection, strict settings, attached datasets, timeouts, interception canary
    reduce.py      AST and sqlsmith reduction
    report.py      dedup, artifacts, summary
    runner.py      worker processes and orchestrator
    cli.py         run / replay / replay-file / show-config / selftest
  tests/           unittest suite (runs without a GPU)
```

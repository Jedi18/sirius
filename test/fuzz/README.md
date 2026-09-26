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

## Developer quick start

Use a supported Linux GPU host (x86-64 or aarch64 with the repository's CUDA environment).
The `duckdb-python` Pixi environment does not support macOS. A stock DuckDB wheel is suitable
for CPU-only harness tests, but is not a supported runtime for loading Sirius.

Build the extension and Python module from the same checkout and initialized submodules:

```bash
git submodule update --init --recursive
pixi run make
pixi run -e duckdb-python build-duckdb-python
pixi run -e duckdb-python fuzz doctor
```

`doctor` validates the TOML profile and output location, imports DuckDB, loads Sirius,
creates a tiny file-backed dataset, proves interception with the canary, and checks a GPU
query's answer. It runs in a disposable subprocess with a 120-second hard deadline, including
initialization and cleanup. It leaves a diagnostic directory with logs, configuration, runtime
information and `outcome.json`; temporary database files are cleaned up on normal exit.
It does not install dependencies, rebuild the engine, switch branches or download extensions.

A successful probe demonstrates that this runtime can execute a small GPU query. It does not
certify that arbitrary binaries have compatible ABIs or that the engine is free of bugs.

```bash
# Choose an existing build, configuration and output location explicitly.
pixi run -e duckdb-python fuzz doctor \
  --extension /absolute/path/sirius.duckdb_extension \
  --sirius-config /absolute/path/sirius.yaml --out /absolute/path/fuzz-results

# A short, bounded exploration: one worker by default.
pixi run -e duckdb-python fuzz run --seed 42 --queries 100 --duration 3m --no-reduce

# A longer campaign; check shared GPU availability before increasing workers.
pixi run -e duckdb-python fuzz run --seed 42 --duration 30m --workers 2

# Targeted generation; other enabled expressions/operators can still appear.
pixi run -e duckdb-python fuzz run --queries 100 \
  --set 'features.scalar_functions.enabled=["substring","like"]'

pixi run -e duckdb-python fuzz show-config
pixi run -e duckdb-python fuzz-test
pixi run -e duckdb-python fuzz selftest --queries 100
```

Use absolute paths for input/output outside the repository: the Pixi `fuzz` task runs from
`test/fuzz`. Profile-relative engine paths are resolved by the existing configuration loader.
`--sirius-config` overrides the profile's YAML; the harness sets `SIRIUS_CONFIG_FILE` from that
selection. Repeat the flag to alternate configurations across worker processes.

A run defaults to one worker and 500 queries. If both `--queries` and `--duration` are supplied,
the first reached budget stops the campaign. Workers already executing a query may produce a
small query-count overshoot. The duration includes worker startup and execution, but excludes
initial provenance collection. In-flight work is stopped at the duration boundary and is not
reported as an engine hang merely because the campaign budget expired.

Ctrl-C stops workers, preserves completed findings and writes a summary marked `cancelled`.
Worker startup failures, failed interception checks and exhausted restart budgets produce an
`incomplete` summary. SIGKILL or machine failure cannot guarantee a final summary; completed
finding bundles and the flushed query log remain on disk. Worker database files from forcibly
terminated processes can remain under the run directory and can be removed after inspection.

| Exit code | Meaning |
|-----------|---------|
| `0` | Run completed (findings may exist); doctor passed; replay matched |
| `1` | Run findings with `--fail-on-findings`, or replay discrepancy/crash/timeout/inconclusive CPU outcome |
| `2` | Invalid setup/input, incomplete campaign, or failed doctor |
| `130` | Cancelled with Ctrl-C |

### Build metadata

`build-duckdb-python` passes the same `OVERRIDE_GIT_DESCRIBE=v1.5.5` as the extension preset.
The harness no longer automatically bypasses DuckDB's version check. For an independently
verified build from matching sources that still has different version metadata, explicitly pass
`--allow-metadata-mismatch`. This only bypasses the metadata check; it does not fix an ABI mismatch.
The override and whether it was actually used are recorded. Bundle replay restores a recorded
allowance and announces it; `--no-allow-metadata-mismatch` disables it for that attempt.

### Replay a finding or your own SQL

```bash
# Copy the entire finding directory, then replay it from any compatible Sirius checkout.
pixi run -e duckdb-python fuzz replay /absolute/path/finding
pixi run -e duckdb-python fuzz replay /absolute/path/finding --original

# Deliberately test another binary/config; the new attempt records the overrides.
pixi run -e duckdb-python fuzz replay /absolute/path/finding \
  --extension /absolute/path/new/sirius.duckdb_extension \
  --sirius-config /absolute/path/local.yaml --timeout 180

# A custom SELECT/WITH query and its CREATE/INSERT dataset.
pixi run -e duckdb-python fuzz replay /absolute/path/query.sql \
  --dataset /absolute/path/dataset.sql

# A stream of SELECT statements, each in its own supervised process.
pixi run -e duckdb-python fuzz replay-file /absolute/path/queries.sql \
  --dataset /absolute/path/dataset.sql
```

Replay restores the saved TOML, YAML, captured baseline session settings, comparison mode and
exact failing setting variant. It never chooses a new random variant. It uses `reduced.sql`
when available, or `query.sql` with `--original`. Missing required inputs fail explicitly;
replay never silently invents a replacement dataset for a finding. For custom SQL only,
`--dataset-seed N` explicitly requests a generated dataset instead of `--dataset`.

Each replay runs under a supervising process with a hard deadline (`--timeout`, 180 seconds
by default), so a native crash or an uninterruptible GPU call cannot kill or indefinitely block
the CLI. The supervisor records the last query, CPU/GPU phase, settings, signal/exit code and
logs. CPU-only replay is available with `--cpu-only` and does not establish GPU correctness.
Custom SQL defaults to multiset comparison; `--ordered` requires deterministic output ordering.
The generated-AST ambiguity filter is not rerun for SQL-only replays; replay is evidence for
manual investigation, not automatic bug confirmation.

Replay writes a new attempt directory and never modifies the source bundle. Each attempt records
its source, inputs, effective settings, environment and outcome. Changed extension fingerprints
are announced. Machine-specific YAML paths (for example spill directories) may need an explicit
`--sirius-config` override on another host. CUDA device visibility is recorded and uses the current
host's environment. Reproducibility across different builds, GPUs or runtimes is not guaranteed.
Legacy bundles without saved YAML require an explicit `--sirius-config`; bundles without integrity
metadata emit a warning. Additional query files from older runs may not have their own dataset;
recover the association from that run's `queries.jsonl` before attempting reproduction.

### Saved evidence

Outputs default to unique directories below `test/fuzz/out/`. Use `--out` to choose another root.

```text
run-<timestamp>-seed<seed>-<unique>/
  config.toml, sirius-<n>.yaml       effective configuration snapshots
  environment.json, invocation.json source revisions, binary hashes, GPU/runtime, CLI arguments
  summary.json, summary.txt         verdict counts, completion status, coverage histogram
  queries.jsonl                    flushed per-query records
  datasets/w<worker>-s<spawn>-d<n>.sql
  logs/                            native stderr, active operation, runtime per worker
  findings/<n>-<verdict>-<hash>/
    query.sql, dataset.sql          immutable original inputs
    config.toml, sirius.yaml        configuration for this particular worker
    meta.json                      original outcome, variant, comparison, timing and sample evidence
    environment.json, runtime.json binary/source provenance and session settings
    worker.stderr                  log snapshot at observation
    bundle.json                    SHA-256 integrity manifest for original inputs
    reduced.sql, reduction.json    optional additional reduction evidence
    REPLAY.md, repro.sql            CLI and standalone shell reproduction instructions
    repro_catch2.cpp               regression-test starting point, requiring developer review
    additional/<n>/               up to five additional complete bundles in the same signature group
```

Datasets include the worker incarnation in their filenames: a respawn cannot overwrite earlier
data. Every retained additional reproducer carries its own dataset and settings. All observations
are still in `queries.jsonl`; signatures are grouping heuristics, not confirmed root causes.

The original observation is saved before reduction starts. A reducer failure is recorded separately
with the active candidate and `stage=reduction`; it does not replace the original mismatch.
Reduction output remains additional evidence and must be replayed. `bundle.json` detects missing
or edited original inputs. To test an edited query, use SQL-file replay with an explicit dataset.

Source revisions describe the checkout at run time. The extension fingerprint identifies the
actual binary; source revision alone is not proof of how that binary was built. Findings include
small, clearly labelled result samples and comparator differences rather than unbounded result dumps.
The standalone `repro.sql` prints CPU/GPU/variant results; the CLI performs the comparison.

The optional sqlsmith reducer uses an already installed extension. It no longer runs `INSTALL`
automatically. AST reduction remains available without sqlsmith.

## Triage and local issue drafts

Triage validates saved findings and writes copy-ready Markdown issue drafts for eligible
candidates. Human review is optional. Nothing is created on GitHub. Start with a small selection;
each replay initializes a fresh GPU process.

```bash
# Accepts one or more finding directories or campaign run directories.
pixi run -e duckdb-python fuzz triage /absolute/path/run-or-finding \
  --out /absolute/path/triage --attempts 3 --timeout 90 --duration 30m

# Reuse the same command and output directory to resume after interruption.
# A changed binary, DuckDB runtime, host environment, harness, or triage configuration
# starts a separate evidence batch.
# Force new attempts even with identical inputs using --rerun.

# Optionally compare an explicitly selected second compatible binary.
pixi run -e duckdb-python fuzz triage /absolute/path/finding \
  --out /absolute/path/triage-comparison --no-reduce \
  --extension /absolute/path/current/sirius.duckdb_extension \
  --compare-extension /absolute/path/other/sirius.duckdb_extension

# Refresh reports and eligible drafts from saved evidence without executing queries.
pixi run -e duckdb-python fuzz triage-report /absolute/path/triage

# Export all automatically eligible candidates, or one candidate with an optional title.
pixi run -e duckdb-python fuzz issue-draft /absolute/path/triage --all
pixi run -e duckdb-python fuzz issue-draft /absolute/path/triage F-CANDIDATE_ID
```

Open `REPORT.md` for the queue, then follow each candidate link for SQL, data, attempt outcomes,
CPU fingerprints, comparator differences, logs and binary provenance. `report.json` contains
the same queue for other tools. Original inputs are copied and fingerprinted under stable
candidate IDs. All attempts and reduction trials remain available under each candidate's
`batches/` directory. Reports distinguish automatic observations, automated validation and a person's disposition.
`DRAFTS.md` links current eligible files in `drafts/`; `drafts.json` contains validation checks
and reasons a candidate needs investigation. Drafts include query SQL, expected and observed
results, replay commands, environment fingerprints and limitations. Large datasets are separate
`.sql` attachments linked beside their drafts in `DRAFTS.md`; copy the draft and attach its
dataset when you choose. Files from earlier batches remain historical; only current links in `DRAFTS.md`
indicate eligibility. Regeneration preserves edits to existing drafts and reports the conflict.

The default is three independent replays, followed by a replay with rows reversed within each
supported `INSERT ... VALUES` batch. Exact CPU result fingerprints detect changes across runs
and insertion orders. This is a conservative ambiguity check: floating-point accumulation can
also change fingerprints, and a single permutation cannot prove determinism. A missing CPU
reference or failed interception check prevents an automatically reproduced classification.
Timeouts include their deadline and CPU/GPU phase; a timed-out query is a hang candidate.
Plan rejections remain coverage gaps until a reviewer determines whether support was promised.

Automatic results include `automatically_reproduced`, `intermittent`, `not_reproduced`,
`changed_failure`, `unstable_reference` and `needs_investigation`. Reproduction alone does not
qualify a candidate for a draft. Suggested groups use failure signatures and are hypotheses, not
proof of shared root cause. Query structure refines suggested groups so a generic mismatch
reason does not group every query together. Every candidate retains its own review state.

Triage optionally shrinks SQL and data with fresh supervised processes, including native crash
and timeout candidates. Defaults are 20 proposed edits and 120 seconds per candidate; use
`--reduce-steps`, `--reduce-seconds` or `--no-reduce` to control this. A reduction must preserve
the failure signature twice, have a stable CPU reference, and pass the available row-order check.
The reducer removes selected top-level clauses/projections and INSERT batches/rows. It preserves
quoted literals, declines unsupported escape/dollar quoting, and leaves ordered-query SQL
unchanged. It is deliberately limited: the retained input is the best observed reduction under
the budget, not a guaranteed minimum. A new unrelated crash never substitutes for a mismatch.

All query work is bounded by supervised replay deadlines. Provenance collection and report I/O
add overhead to `--duration`; the budget does not bypass cleanup. Ctrl-C preserves completed
attempts and exits 130. Invalid/incomplete work or an exhausted overall budget exits 2; a
completed investigation exits 0 even if discrepancies remain. Resume with the same command.
The workspace lock prevents concurrent writers to the same report.

A new attempt starts only when enough budget remains for its full `--timeout`. Budget exhaustion
never shortens a replay timeout or creates a hang observation. Choose `--reduce-seconds` greater
than `--timeout` to allow reduction work. Completed attempts are reused only after checking their
saved evidence; edited or missing inputs, outputs, or accepted reductions stop reuse and export.
The runtime recipe records the Python and DuckDB binary hashes, GPU/driver identity, host and
selected execution environment variables. Changes to unrecorded system libraries are outside
this check; use `--rerun` after such changes. Avoid changing binaries during an investigation.
Candidate reports link the accepted reduced replay bundle and include commands with the tested
extension, YAML override, timeout and metadata-mismatch policy.

### Automated validation policy

The `saved-evidence-v1` policy recomputes observations from fingerprinted evidence and requires:

- At least three distinct original replays with the same failure, a stable CPU fingerprint and
  successful interception probes; mismatch evidence must include a completed GPU operation.
- Matching extension and DuckDB fingerprints, effective configuration and baseline session settings.
  Runs that bypassed extension version metadata need compatibility investigation.
- A mismatch or GPU-phase crash using multiset comparison. Timeouts, setup errors, memory exhaustion,
  plan rejections, intermittent results and unstable references need investigation.
- Conservative SQL screening: unknown functions, volatile functions, windows, limits, sampling and
  ordered-result semantics need investigation. This screening does not prove SQL semantics.
- The available insertion-order check must preserve both failure and CPU fingerprint. Only a
  recognized empty or one-row literal dataset can omit that replay.
- A completed bounded reduction pass. The selected reduction must preserve the failure in two
  distinct replays, with a stable CPU fingerprint and the applicable insertion-order check.

Eligible drafts say **automatically validated** and never claim a human reviewed them. Any
recorded human objection or uncertain disposition blocks automatic drafting, even if its batch
is stale. Grouping does not transfer eligibility. A current explicit human verification can
still support a per-candidate draft when the conservative automatic policy declines it.

### Optional human verification

For each candidate, inspect the original and best reduced inputs, CPU/GPU differences, settings,
binary hashes and logs. Check expected SQL semantics, NULLs, types, ordering and floating-point
tolerance. Run a separate replay yourself using the original bundle or a retained reduction
attempt bundle, which carries its exact query, dataset and settings. Inspect the new replay
evidence before recording a disposition. Automated reproduction does not perform this step.

```bash
pixi run -e duckdb-python fuzz replay /absolute/path/candidate/source --original

# Record uncertainty, expected behavior, a harness problem, or another disposition.
pixi run -e duckdb-python fuzz review /absolute/path/triage F-CANDIDATE_ID \
  --disposition needs_investigation --reviewer 'Your name' \
  --notes 'Explain what you checked and what remains unresolved.'

# Only a person who has actually performed the verification should run this command.
pixi run -e duckdb-python fuzz review /absolute/path/triage F-CANDIDATE_ID \
  --disposition manually_verified --reviewer 'Your name' \
  --evidence /absolute/path/new-replay-attempt \
  --expected 'Explain the correct CPU result and SQL semantics.' \
  --actual 'Explain the observed GPU discrepancy.' \
  --notes 'Describe the checks you personally performed.' \
  --acknowledge-manual-verification

# Export a local draft incorporating the recorded human verification.
pixi run -e duckdb-python fuzz issue-draft /absolute/path/triage F-CANDIDATE_ID
```

Review dispositions also include `intermittent`, `not_reproduced`, `harness_problem`,
`expected_behavior` and `duplicate` (requires `--duplicate-of F-OTHER_ID`). Manual verification
requires a completed triage batch and saved matching replay evidence with the same extension,
DuckDB runtime, configuration, CPU result fingerprint and original or accepted reduced inputs.
Review history and an evidence copy are
retained. New evidence batches make prior reviews stale; edited/missing evidence prevents export.
The attestation records the reviewer's statement; software cannot prove that a person inspected
the results. Agents must not supply it on the user's behalf.

Use `fuzz group /absolute/path/triage F-ID1 F-ID2 --name 'Possible cause' --reviewer 'Your name'
--notes 'Why these appear related'` to assign a group; reassign selected candidates to split it.
Grouping never verifies other candidates or erases their evidence. A human-reviewed draft is
written under `drafts/` with reproduction inputs, expected/actual behavior, environment and review
provenance. Publishing it is a separate manual action. This workflow has no GitHub integration or CI.

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
    artifacts.py   input integrity and build/runtime provenance
    isolation.py   supervised doctor and replay processes
    triage.py      repeated replay, resumable reduction and intermediate reports
    triage_reduce.py conservative query/data reduction candidates
    triage_review.py human dispositions and evidence checks
    triage_drafts.py automatic validation and local issue drafts
    cli.py         doctor / run / replay / triage / review and other commands
  tests/           unittest suite (runs without a GPU)
```

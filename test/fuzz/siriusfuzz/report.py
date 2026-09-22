# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""Finding dedup, repro artifacts and the run summary."""

from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import re
import shutil
import time
import tomllib
from dataclasses import asdict, dataclass, field
from typing import Any

from .classify import SEVERITY, Verdict, normalize_reason
from .config import FuzzConfig, FUZZ_DIR


@dataclass
class QueryRecord:
    worker: int
    dataset: str  # dataset file stem, e.g. "w0-d3"
    seed: int
    sql: str
    verdict: str
    reason: str = ""  # short classification text (error reason / setting name)
    detail: str = ""  # comparator detail or full error
    labels: list[str] = field(default_factory=list)
    reduced_sql: str | None = None
    reduced_labels: list[str] = field(default_factory=list)
    elapsed_cpu: float = 0.0
    elapsed_gpu: float = 0.0
    variant: dict[str, Any] | None = None
    diffs: list[str] = field(default_factory=list)
    reduction_steps: int = 0


@dataclass
class KnownIssue:
    pattern: str
    issue: str
    note: str = ""
    verdicts: list[str] = field(default_factory=list)

    def matches(self, rec: QueryRecord) -> bool:
        if self.verdicts and rec.verdict not in self.verdicts:
            return False
        hay = "\n".join([rec.reason, rec.detail, rec.reduced_sql or "", rec.sql])
        return re.search(self.pattern, hay, re.IGNORECASE | re.MULTILINE) is not None


def load_known_issues(path: pathlib.Path | None) -> list[KnownIssue]:
    if path is None or not path.exists():
        return []
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    out = []
    for item in data.get("issue", []):
        if "pattern" not in item or "issue" not in item:
            raise ValueError(f"known issue entry needs pattern and issue: {item}")
        out.append(
            KnownIssue(
                item["pattern"],
                item["issue"],
                item.get("note", ""),
                list(item.get("verdicts", [])),
            )
        )
    return out


def signature(rec: QueryRecord) -> str:
    """Dedup key: verdict plus normalized reason, or the reduced query's operator shape."""
    v = rec.verdict
    if v in (
        Verdict.PLAN_FALLBACK,
        Verdict.GPU_ERROR,
        Verdict.GPU_INTERNAL_ERROR,
        Verdict.GPU_OOM,
    ):
        key = f"{v}|{normalize_reason(rec.reason)}"
    elif v == Verdict.VARIANT_MISMATCH:
        setting = next(iter(rec.variant or {}), "")
        key = f"{v}|{setting}|{','.join(rec.reduced_labels or rec.labels)}"
    else:
        key = f"{v}|{','.join(rec.reduced_labels or rec.labels)}"
    return key


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:8]


class Report:
    def __init__(
        self,
        run_dir: pathlib.Path,
        config: FuzzConfig,
        known_issues: list[KnownIssue],
        seed: int,
    ):
        self.run_dir = run_dir
        self.cfg = config
        self.known = known_issues
        self.seed = seed
        self.counts: collections.Counter[str] = collections.Counter()
        self.fallback_reasons: collections.Counter[str] = collections.Counter()
        self.cpu_error_reasons: collections.Counter[str] = collections.Counter()
        self.findings: dict[str, dict[str, Any]] = {}
        self.feature_stats: collections.Counter[str] = collections.Counter()
        self.started = time.time()
        self.queries = 0
        self.datasets = 0
        (run_dir / "findings").mkdir(parents=True, exist_ok=True)
        (run_dir / "datasets").mkdir(parents=True, exist_ok=True)
        self.log_path = run_dir / "queries.jsonl"
        self._log = open(self.log_path, "a", encoding="utf-8")

    # -- ingestion -------------------------------------------------------------

    def add(self, rec: QueryRecord) -> str | None:
        """Record a query; returns the finding directory name when a new finding was written."""
        self.queries += 1
        verdict = Verdict(rec.verdict)
        for ki in self.known:
            if verdict.is_finding and ki.matches(rec):
                rec.reason = f"{ki.issue}: {rec.reason}"
                rec.verdict = Verdict.KNOWN_ISSUE.value
                verdict = Verdict.KNOWN_ISSUE
                break
        self.counts[rec.verdict] += 1
        if verdict == Verdict.PLAN_FALLBACK or (
            verdict == Verdict.KNOWN_ISSUE and "plan" in rec.detail[:40].lower()
        ):
            self.fallback_reasons[normalize_reason(rec.reason)] += 1
        if verdict == Verdict.CPU_ERROR:
            self.cpu_error_reasons[normalize_reason(rec.reason)] += 1
        self._log.write(json.dumps(asdict(rec), default=str) + "\n")
        self._log.flush()
        if not verdict.is_finding and verdict != Verdict.KNOWN_ISSUE:
            return None
        sig = signature(rec)
        if sig in self.findings:
            self.findings[sig]["count"] += 1
            return None
        name = f"{len(self.findings):03d}-{rec.verdict}-{short_hash(sig)}"
        self.findings[sig] = {
            "name": name,
            "count": 1,
            "verdict": rec.verdict,
            "reason": rec.reason,
            "worker": rec.worker,
        }
        self._write_finding(name, rec, sig)
        return name

    def add_feature_stats(self, stats: dict[str, int]) -> None:
        self.feature_stats.update(stats)

    def dataset_path(self, stem: str) -> pathlib.Path:
        return self.run_dir / "datasets" / f"{stem}.sql"

    # -- artifacts -------------------------------------------------------------

    def _write_finding(self, name: str, rec: QueryRecord, sig: str) -> None:
        d = self.run_dir / "findings" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "query.sql").write_text(rec.sql.rstrip() + ";\n")
        repro_sql = rec.reduced_sql or rec.sql
        if rec.reduced_sql:
            (d / "reduced.sql").write_text(rec.reduced_sql.rstrip() + ";\n")
        ds_src = self.dataset_path(rec.dataset)
        if ds_src.exists():
            shutil.copy(ds_src, d / "dataset.sql")
        (d / "config.toml").write_text(self.cfg.to_toml())
        meta = asdict(rec)
        meta.update(
            {
                "signature": sig,
                "run_seed": self.seed,
                "config_hash": self.cfg.config_hash(),
            }
        )
        (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        detail = [
            f"verdict: {rec.verdict}",
            f"reason: {rec.reason}",
            f"detail: {rec.detail}",
        ]
        if rec.variant:
            detail.append(f"variant: {rec.variant}")
        detail += rec.diffs
        detail.append("")
        detail.append("Reproduce with the DuckDB shell built by this repo:")
        detail.append(
            f"  ./build/release/duckdb -unsigned -c \"LOAD 'build/release/extension/sirius/sirius.duckdb_extension'; ATTACH 'x.duckdb' AS x; USE x;\" ..."
        )
        detail.append(f"  python -m siriusfuzz replay {d}")
        (d / "detail.txt").write_text("\n".join(detail) + "\n")
        (d / "repro_catch2.cpp").write_text(
            catch2_snippet(name, rec, ds_src if ds_src.exists() else None, repro_sql)
        )

    def finish(self) -> dict[str, Any]:
        self._log.close()
        elapsed = time.time() - self.started
        ordered = sorted(
            self.findings.values(),
            key=lambda f: (
                (
                    SEVERITY.index(Verdict(f["verdict"]))
                    if Verdict(f["verdict"]) in SEVERITY
                    else 99
                ),
                -f["count"],
            ),
        )
        summary = {
            "seed": self.seed,
            "config_hash": self.cfg.config_hash(),
            "profile": self.cfg.profile,
            "elapsed_seconds": round(elapsed, 1),
            "queries": self.queries,
            "datasets": self.datasets,
            "counts": dict(self.counts),
            "findings": ordered,
            "plan_fallback_reasons": self.fallback_reasons.most_common(),
            "cpu_error_reasons": self.cpu_error_reasons.most_common(20),
            "feature_stats": dict(sorted(self.feature_stats.items())),
        }
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        text = self.render_summary(summary)
        (self.run_dir / "summary.txt").write_text(text)
        return summary

    def render_summary(self, summary: dict[str, Any]) -> str:
        lines = [
            f"siriusfuzz run: {self.run_dir}",
            f"profile={summary['profile']} seed={summary['seed']} config={summary['config_hash']} "
            f"queries={summary['queries']} datasets={summary['datasets']} elapsed={summary['elapsed_seconds']}s",
            "",
            "verdict counts:",
        ]
        for k, v in sorted(summary["counts"].items(), key=lambda kv: -kv[1]):
            lines.append(f"  {k:20s} {v}")
        lines.append("")
        lines.append(f"findings ({len(summary['findings'])} unique):")
        for f in summary["findings"]:
            lines.append(
                f"  [{f['verdict']}] x{f['count']:<4d} {f['name']}  {f['reason'][:100]}"
            )
        if summary["plan_fallback_reasons"]:
            lines.append("")
            lines.append("plan-time fallback reasons:")
            for reason, n in summary["plan_fallback_reasons"]:
                lines.append(f"  {n:5d}  {reason}")
        if summary["cpu_error_reasons"]:
            lines.append("")
            lines.append("skipped (CPU error) reasons, top 20:")
            for reason, n in summary["cpu_error_reasons"]:
                lines.append(f"  {n:5d}  {reason}")
        never = [
            f for f in _CLAIMED_FEATURES(self.cfg) if self.feature_stats.get(f, 0) == 0
        ]
        lines.append("")
        lines.append(f"features emitted: {len(summary['feature_stats'])} distinct")
        if never and summary["queries"] >= 100:
            lines.append("WARNING enabled features never emitted: " + ", ".join(never))
        return "\n".join(lines) + "\n"


def _CLAIMED_FEATURES(cfg: FuzzConfig) -> list[str]:
    """Feature counters that must be non-zero for a run of any length, given the config."""
    f = cfg.features
    claimed = ["where"]
    if f.aggregates.functions:
        claimed += ["group_by", "ungrouped_aggregate"]
    claimed += [f"join:{jt}" for jt in f.joins.types]
    if f.order_by.enabled:
        claimed.append("order_by")
    if f.limit.enabled:
        claimed.append("limit")
    if f.cte.materialized:
        claimed.append("cte")
    if f.set_ops.union_all:
        claimed.append("setop:UNION ALL")
    if f.subqueries.exists:
        claimed.append("subquery:exists")
    if f.subqueries.in_:
        claimed.append("subquery:in")
    if f.subqueries.scalar:
        claimed.append("subquery:scalar")
    if f.window_functions:
        claimed.append("window")
    if f.distinct:
        claimed.append("distinct")
    return claimed


def catch2_snippet(
    name: str, rec: QueryRecord, dataset_path: pathlib.Path | None, query_sql: str
) -> str:
    schema_hint = (
        f"// Data: see dataset.sql next to this file ({dataset_path.name}); paste its CREATE/INSERT\n"
        "// statements into run_ok() calls, or load it with the shell before running the query.\n"
        if dataset_path
        else "// Data: dataset SQL was not captured.\n"
    )
    return (
        "// Generated by siriusfuzz; drop into test/cpp/integration/test_gpu_execution_fuzz.cpp\n"
        f"{schema_hint}"
        "TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,\n"
        f'                 "fuzz repro {name}",\n'
        '                 "[integration][gpu_execution][fuzz]")\n'
        "{\n"
        '  run_ok(R"SQL(\n'
        "    -- schema + data from dataset.sql\n"
        '  )SQL");\n'
        '  run_ok("CHECKPOINT;");\n'
        f"  // verdict: {rec.verdict}; reason: {rec.reason[:100]}\n"
        '  compare_gpu_vs_cpu(R"SQL(\n'
        f"{query_sql}\n"
        '  )SQL");\n'
        "}\n"
    )


def default_known_issues_path(cfg: FuzzConfig) -> pathlib.Path | None:
    if not cfg.oracle.known_issues:
        return None
    p = pathlib.Path(cfg.oracle.known_issues)
    if p.is_absolute():
        return p
    if cfg.source_path:
        candidate = pathlib.Path(cfg.source_path).resolve().parent / p
        if candidate.exists():
            return candidate
    return FUZZ_DIR / p

# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""Command line entry point: ``python -m siriusfuzz <command>``."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time
from dataclasses import asdict

from . import __version__
from .classify import Verdict
from .config import FUZZ_DIR, REPO_ROOT, FuzzConfig, load_config, resolve_repo_path
from .report import Report, default_known_issues_path, load_known_issues
from .runner import Evaluator, Orchestrator, OrchestratorOptions
from .schema_gen import DataGenerator
from .session import Session

DEFAULT_PROFILE = FUZZ_DIR / "config" / "strict.toml"


def parse_duration(text: str | None) -> float | None:
    if text is None:
        return None
    t = text.strip().lower()
    mult = 1.0
    if t.endswith("ms"):
        return float(t[:-2]) / 1000
    if t[-1] in "smh":
        mult = {"s": 1, "m": 60, "h": 3600}[t[-1]]
        t = t[:-1]
    return float(t) * mult


def _common_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--config",
        default=str(DEFAULT_PROFILE),
        help="TOML profile (default: config/strict.toml)",
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config key, e.g. features.window_functions=true",
    )


def _common_engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--extension", help="path to sirius.duckdb_extension (default: from config)"
    )
    p.add_argument(
        "--sirius-config",
        action="append",
        default=None,
        help="Sirius YAML config; repeat to round-robin across workers",
    )
    p.add_argument(
        "--cpu-only",
        action="store_true",
        help="do not load Sirius; compare CPU against CPU (harness self-check)",
    )


def _load(args: argparse.Namespace) -> FuzzConfig:
    path = args.config if args.config and args.config != "-" else None
    return load_config(path, args.set)


def _engine(args: argparse.Namespace, cfg: FuzzConfig) -> tuple[str | None, list[str]]:
    if args.cpu_only:
        return None, []
    ext = (
        pathlib.Path(args.extension)
        if args.extension
        else resolve_repo_path(cfg.sirius.extension)
    )
    if not ext.exists():
        sys.exit(
            f"extension not found: {ext} (build Sirius first, or pass --extension / --cpu-only)"
        )
    configs = args.sirius_config or cfg.sirius.configs
    resolved = []
    for c in configs:
        p = resolve_repo_path(c)
        if not p.exists():
            sys.exit(f"Sirius config not found: {p}")
        resolved.append(str(p))
    return str(ext), resolved


def _make_run_dir(out: str | None, seed: int) -> pathlib.Path:
    base = pathlib.Path(out) if out else FUZZ_DIR / "out"
    run_dir = base / f"run-{time.strftime('%Y%m%d-%H%M%S')}-seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _load(args)
    extension, sirius_configs = _engine(args, cfg)
    seed = (
        args.seed
        if args.seed is not None
        else random.SystemRandom().randrange(1, 2**31)
    )
    run_dir = _make_run_dir(args.out, seed)
    (run_dir / "config.toml").write_text(cfg.to_toml())
    known = load_known_issues(default_known_issues_path(cfg))
    report = Report(run_dir, cfg, known, seed)
    opts = OrchestratorOptions(
        workers=args.workers,
        duration=parse_duration(args.duration),
        max_queries=args.queries,
        extension=extension,
        sirius_configs=sirius_configs,
        reduce=not args.no_reduce,
        quiet=args.quiet,
    )
    if opts.duration is None and opts.max_queries is None:
        opts.max_queries = 500
    print(
        f"siriusfuzz {__version__}: profile={cfg.profile} seed={seed} workers={opts.workers} "
        f"{'cpu-only' if extension is None else extension} -> {run_dir}",
        file=sys.stderr,
    )
    summary = Orchestrator(cfg, report, seed, opts).run()
    print(report.render_summary(summary))
    findings = [
        f for f in summary["findings"] if f["verdict"] != Verdict.KNOWN_ISSUE.value
    ]
    if args.fail_on_findings and findings:
        return 1
    return 0


def _open_session(
    args: argparse.Namespace, cfg: FuzzConfig, work: pathlib.Path
) -> Session:
    extension, sirius_configs = _engine(args, cfg)
    s = Session(
        extension, sirius_configs[0] if sirius_configs else None, work / "db", 0
    )
    s.open()
    return s


def _dataset_from_args(
    args: argparse.Namespace,
    cfg: FuzzConfig,
    session: Session,
    finding_dir: pathlib.Path | None,
):
    """Attach the dataset for a replay: dataset.sql from a finding dir, --dataset, or a generated one."""
    sql_path = None
    if args.dataset:
        sql_path = pathlib.Path(args.dataset)
    elif finding_dir is not None and (finding_dir / "dataset.sql").exists():
        sql_path = finding_dir / "dataset.sql"
    if sql_path is not None:
        session.db_dir.mkdir(parents=True, exist_ok=True)
        path = session.db_dir / "replay.duckdb"
        for p in (path, pathlib.Path(str(path) + ".wal")):
            if p.exists():
                p.unlink()
        session.con.execute(f"ATTACH '{path}' AS replay")
        session.con.execute("USE replay")
        for stmt in sql_path.read_text().split(";\n"):
            if stmt.strip() and not stmt.strip().startswith("--"):
                session.con.execute(stmt)
        session.con.execute("CHECKPOINT")
        session.current_alias = "replay"
        return None
    seed = args.dataset_seed if args.dataset_seed is not None else 1
    ds = DataGenerator(cfg, random.Random(seed)).generate(seed)
    session.load_dataset(ds, "replay")
    return ds


def cmd_replay(args: argparse.Namespace) -> int:
    cfg = _load(args)
    target = pathlib.Path(args.target)
    finding_dir = target if target.is_dir() else None
    if finding_dir is not None:
        sql_file = finding_dir / ("query.sql" if args.original else "reduced.sql")
        if not sql_file.exists():
            sql_file = finding_dir / "query.sql"
        sql = sql_file.read_text()
    else:
        sql = target.read_text()
    sql = sql.strip().rstrip(";")
    work = _make_run_dir(args.out, 0)
    session = _open_session(args, cfg, work)
    try:
        ds = _dataset_from_args(args, cfg, session, finding_dir)
        if session.gpu_available:
            ok, why = session.check_interception()
            print(
                f"interception canary: {'ok' if ok else 'FAILED'} ({why})",
                file=sys.stderr,
            )
        ev = Evaluator(cfg, session, lambda m: print(m, file=sys.stderr))
        if ds is not None:
            ev.set_dataset(ds, "replay")
        rec = ev.evaluate(None, sql, 0, "replay", 0)
        print(json.dumps(asdict(rec), indent=2, default=str))
        return (
            0
            if rec.verdict
            in (Verdict.OK.value, Verdict.AMBIGUOUS.value, Verdict.CPU_ERROR.value)
            else 1
        )
    finally:
        session.close()


def cmd_replay_file(args: argparse.Namespace) -> int:
    """Classify every statement of a SQL file (e.g. a sqlsmith complete_log) against a dataset."""
    cfg = _load(args)
    statements = [
        s.strip() for s in pathlib.Path(args.file).read_text().split(";") if s.strip()
    ]
    work = _make_run_dir(args.out, args.dataset_seed or 1)
    known = load_known_issues(default_known_issues_path(cfg))
    report = Report(work, cfg, known, args.dataset_seed or 1)
    session = _open_session(args, cfg, work)
    try:
        ds = _dataset_from_args(args, cfg, session, None)
        ev = Evaluator(cfg, session, lambda m: print(m, file=sys.stderr))
        if ds is not None:
            ev.set_dataset(ds, "replay")
        for i, sql in enumerate(statements):
            if not sql.lower().startswith(("select", "with", "(")):
                continue
            rec = ev.evaluate(None, sql, 0, "replay", 0)
            report.add(rec)
            if args.verbose:
                print(f"[{i}] {rec.verdict}: {rec.reason[:100]}")
        summary = report.finish()
        print(report.render_summary(summary))
        return 0
    finally:
        session.close()


def cmd_show_config(args: argparse.Namespace) -> int:
    cfg = _load(args)
    print(cfg.to_toml())
    print(f"# config hash: {cfg.config_hash()}")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """CPU-vs-CPU run: proves the generator, comparator, reducer and report work without a GPU."""
    args.cpu_only = True
    args.extension = None
    args.sirius_config = None
    args.no_reduce = False
    args.quiet = True
    args.fail_on_findings = True
    args.duration = None
    if args.queries is None:
        args.queries = 150
    if args.seed is None:
        args.seed = 20260922
    rc = cmd_run(args)
    print("selftest " + ("PASSED" if rc == 0 else "FAILED"))
    return rc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="siriusfuzz", description="Generative differential fuzzer for Sirius"
    )
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="generate, run and compare queries")
    _common_config_args(run)
    _common_engine_args(run)
    run.add_argument("--seed", type=int)
    run.add_argument("--duration", help="time budget, e.g. 30m, 2h, 90s")
    run.add_argument(
        "--queries", type=int, help="total query budget (default 500 when no duration)"
    )
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--out", help=f"output root (default {FUZZ_DIR / 'out'})")
    run.add_argument("--no-reduce", action="store_true")
    run.add_argument("--quiet", action="store_true")
    run.add_argument(
        "--fail-on-findings",
        action="store_true",
        help="exit 1 when any non-known finding was recorded",
    )
    run.set_defaults(func=cmd_run)

    rp = sub.add_parser("replay", help="re-run one finding directory or .sql file")
    _common_config_args(rp)
    _common_engine_args(rp)
    rp.add_argument("target")
    rp.add_argument(
        "--dataset", help="dataset.sql to load (default: the finding's dataset.sql)"
    )
    rp.add_argument("--dataset-seed", type=int)
    rp.add_argument(
        "--original",
        action="store_true",
        help="replay query.sql instead of reduced.sql",
    )
    rp.add_argument("--out")
    rp.set_defaults(func=cmd_replay)

    rf = sub.add_parser(
        "replay-file", help="classify every SELECT in a SQL file (e.g. a sqlsmith log)"
    )
    _common_config_args(rf)
    _common_engine_args(rf)
    rf.add_argument("file")
    rf.add_argument("--dataset")
    rf.add_argument("--dataset-seed", type=int)
    rf.add_argument("--out")
    rf.add_argument("--verbose", action="store_true")
    rf.set_defaults(func=cmd_replay_file)

    sc = sub.add_parser("show-config", help="print the effective configuration")
    _common_config_args(sc)
    sc.set_defaults(func=cmd_show_config)

    stp = sub.add_parser("selftest", help="CPU-only harness check (no GPU needed)")
    _common_config_args(stp)
    stp.add_argument("--seed", type=int)
    stp.add_argument("--queries", type=int)
    stp.add_argument("--workers", type=int, default=1)
    stp.add_argument("--out")
    stp.set_defaults(func=cmd_selftest)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)

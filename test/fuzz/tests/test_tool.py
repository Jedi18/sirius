"""Developer-facing contracts: portable evidence, exact replay and process isolation."""

import argparse
import json
import os
import pathlib
import signal
import tempfile
import time
import unittest
from dataclasses import asdict
from unittest.mock import patch

from . import conftest_path  # noqa: F401
from .test_evaluator import FakeSession, gpu_variant_wrong
from siriusfuzz.artifacts import seal, verify, write_json
from siriusfuzz.cli import build_parser, cmd_replay, validate_limits
from siriusfuzz.config import load_config
from siriusfuzz.isolation import supervise
from siriusfuzz.report import QueryRecord, Report
from siriusfuzz.runner import Evaluator, Mailbox, Orchestrator, OrchestratorOptions
from siriusfuzz.session import RunResult, Session, SessionError


def crash_child(payload, work):
    os.kill(os.getpid(), signal.SIGKILL)


def hang_child(payload, work):
    time.sleep(30)


def ok_child(payload, work):
    write_json(pathlib.Path(work) / "result.json", {"status": "ok"})


def restarting_worker(args, cfg, out, stop):
    """Two incarnations produce the same signature against different datasets."""
    stem = f"w0-s{args.spawn_id}-d0"
    dataset = pathlib.Path(args.run_dir) / "datasets" / f"{stem}.sql"
    dataset.write_text(
        f"CREATE TABLE t(k INTEGER); INSERT INTO t VALUES ({args.spawn_id});"
    )

    def send(message):
        out.put({**message, "worker": args.worker_id, "spawn": args.spawn_id})

    record = QueryRecord(0, stem, args.seed, "SELECT k FROM t", "mismatch")
    send({"type": "begin", "sql": record.sql, "dataset": stem})
    send({"type": "result", "record": asdict(record)})
    send({"type": "idle"})
    if args.spawn_id == 1:
        send({"type": "begin", "sql": "SELECT k + 1 FROM t", "dataset": stem})
        os._exit(9)
    send({"type": "done", "reason": "finished"})


def startup_dead_worker(args, cfg, out, stop):
    os._exit(9)


class ToolTests(unittest.TestCase):
    def test_interrupt_during_process_start_records_cancellation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("siriusfuzz.isolation.mp.get_context") as context:
                process = context.return_value.Process.return_value
                process.start.side_effect = KeyboardInterrupt
                process.pid = None
                result = supervise({}, pathlib.Path(tmp), 5)
                self.assertEqual(result["status"], "cancelled")
                self.assertEqual(
                    json.loads((pathlib.Path(tmp) / "outcome.json").read_text())[
                        "status"
                    ],
                    "cancelled",
                )
                process.kill.assert_not_called()

    def test_startup_crash_does_not_exhaust_query_restart_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(None)
            report = Report(pathlib.Path(tmp), cfg, [], 1)
            runner = Orchestrator(
                cfg,
                report,
                1,
                OrchestratorOptions(max_queries=1, max_respawns=200, quiet=True),
            )
            with patch("siriusfuzz.runner.worker_main", startup_dead_worker):
                summary = runner.run()
            self.assertEqual(summary["status"], "incomplete")
            self.assertEqual(runner.spawn_count, 1)
            self.assertEqual(summary["queries"], 0)

    def test_worker_restart_preserves_both_datasets_and_completed_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            cfg = load_config(None)
            report = Report(root, cfg, [], 1)
            runner = Orchestrator(
                cfg,
                report,
                1,
                OrchestratorOptions(max_queries=3, max_respawns=1, quiet=True),
            )
            with patch("siriusfuzz.runner.worker_main", restarting_worker):
                summary = runner.run()
            self.assertEqual(summary["status"], "complete", summary)
            self.assertEqual(summary["counts"], {"mismatch": 2, "crash": 1})
            bundles = [
                p
                for p in (root / "findings").rglob("meta.json")
                if json.loads(p.read_text())["verdict"] == "mismatch"
            ]
            self.assertEqual(len(bundles), 2)
            self.assertNotEqual(
                (bundles[0].parent / "dataset.sql").read_text(),
                (bundles[1].parent / "dataset.sql").read_text(),
            )

    def test_reduction_does_not_accept_an_unrelated_runtime_error(self):
        cfg = load_config(None)
        session = FakeSession(
            lambda *args: RunResult(
                "error", error="Sirius GPU execution failed: unrelated operator failure"
            )
        )
        ev = Evaluator(cfg, session, lambda message: None)
        record = QueryRecord(
            0,
            "dataset",
            0,
            "SELECT 1",
            "gpu_error",
            reason="original expression failure",
        )
        predicate = ev._make_still_fails(record)
        self.assertIsNotNone(predicate)
        self.assertFalse(predicate("SELECT 2"))

    def test_mailbox_ignores_interrupted_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            box = Mailbox(pathlib.Path(tmp))
            (pathlib.Path(tmp) / "partial.json.tmp").write_text('{"unfinished":')
            box.put({"type": "result", "record": "complete"})
            self.assertEqual(box.get_nowait()["record"], "complete")

    def test_cancellation_preserves_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(None)
            report = Report(pathlib.Path(tmp), cfg, [], 1)
            report.add(QueryRecord(0, "missing", 1, "SELECT 1", "mismatch"))
            runner = Orchestrator(cfg, report, 1, OrchestratorOptions())
            with patch.object(runner, "_run", side_effect=KeyboardInterrupt):
                summary = runner.run()
            self.assertEqual(summary["status"], "cancelled")
            self.assertEqual(summary["counts"]["mismatch"], 1)
            self.assertTrue((pathlib.Path(tmp) / "summary.json").is_file())

    def test_replay_parses_comments_and_quoted_semicolons(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "dataset.sql").write_text(
                "-- header\nCREATE TABLE t(k VARCHAR);\nINSERT INTO t VALUES ('a;\nb'), ('--literal');\nCHECKPOINT;\n"
            )
            (root / "query.sql").write_text(
                "-- query comment\nSELECT k FROM t ORDER BY k;"
            )
            (root / "config.toml").write_text(load_config(None).to_toml())
            outcome = supervise(
                {
                    "operation": "replay",
                    "query": str(root / "query.sql"),
                    "dataset": str(root / "dataset.sql"),
                    "config": str(root / "config.toml"),
                    "comparison": "ordered",
                },
                root,
                30,
            )
            self.assertEqual(outcome["status"], "ok", outcome)
            self.assertEqual(outcome["record"]["verdict"], "ok")
            self.assertEqual(outcome["record"]["evidence"]["cpu"]["row_count"], 2)

    def test_supervision_handles_native_death_and_deadline(self):
        for target, expected, timeout in (
            (ok_child, "ok", 10),
            (crash_child, "crash", 10),
            (hang_child, "timeout", 0.5),
        ):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as tmp:
                result = supervise({}, pathlib.Path(tmp), timeout, target)
                self.assertEqual(result["status"], expected)
                self.assertTrue((pathlib.Path(tmp) / "outcome.json").exists())
                self.assertLess(result["elapsed_seconds"], timeout + 6)

    def test_repeated_signature_keeps_each_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            report = Report(root, load_config(None), [], 4)
            for i in range(2):
                (root / "datasets" / f"d{i}.sql").write_text(
                    f"CREATE TABLE t(k INT); INSERT INTO t VALUES ({i});"
                )
                report.add(
                    QueryRecord(
                        0,
                        f"d{i}",
                        i,
                        "SELECT k FROM t",
                        "mismatch",
                        reason="different rows",
                    )
                )
            report.finish()
            bundles = sorted((root / "findings").rglob("dataset.sql"))
            self.assertEqual(len(bundles), 2)
            self.assertNotEqual(bundles[0].read_text(), bundles[1].read_text())
            for data in bundles:
                verify(data.parent)
                self.assertTrue((data.parent / "meta.json").is_file())

    def test_integrity_rejects_missing_and_modified_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            query = root / "query.sql"
            query.write_text("SELECT 1")
            seal(root)
            verify(root)
            query.write_text("SELECT 2")
            with self.assertRaisesRegex(ValueError, "modified"):
                verify(root)
            query.unlink()
            with self.assertRaises(ValueError):
                verify(root)

    def test_replay_restores_bundle_and_does_not_modify_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            bundle = root / "copied-bundle"
            bundle.mkdir()
            (bundle / "query.sql").write_text("SELECT k FROM t;")
            (bundle / "dataset.sql").write_text(
                "CREATE TABLE t(k INT); INSERT INTO t VALUES (1);"
            )
            cfg = load_config(None, ["oracle.float64_rel_tol=0.0007"])
            (bundle / "config.toml").write_text(cfg.to_toml())
            write_json(
                bundle / "meta.json",
                {
                    "comparison": "ordered",
                    "variant": {"hash_partition_bytes": 123},
                    "execution": {"cpu_only": True},
                },
            )
            seal(bundle)
            before = {p.name: p.read_bytes() for p in bundle.iterdir()}
            args = build_parser().parse_args(
                ["replay", str(bundle), "--out", str(root / "out")]
            )
            with patch(
                "siriusfuzz.cli.supervise",
                return_value={"status": "ok", "record": {"verdict": "ok"}},
            ) as probe, patch("siriusfuzz.cli.provenance", return_value={}):
                self.assertEqual(cmd_replay(args), 0)
            payload = probe.call_args.args[0]
            self.assertEqual(payload["variant"], {"hash_partition_bytes": 123})
            self.assertEqual(payload["comparison"], "ordered")
            self.assertEqual(
                load_config(payload["config"]).oracle.float64_rel_tol, 0.0007
            )
            self.assertEqual(before, {p.name: p.read_bytes() for p in bundle.iterdir()})

    def test_replay_does_not_invent_missing_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = pathlib.Path(tmp)
            (bundle / "query.sql").write_text("SELECT k FROM t")
            args = build_parser().parse_args(["replay", str(bundle), "--cpu-only"])
            with self.assertRaisesRegex(ValueError, "missing dataset.sql"):
                cmd_replay(args)

    def test_forced_variant_ignores_random_variant_budget(self):
        cfg = load_config(
            None, ["variants.per_query=0", "oracle.ambiguity_filter=false"]
        )
        session = FakeSession(gpu_variant_wrong)
        ev = Evaluator(cfg, session, lambda message: None)
        ev.forced_variant = {"hash_partition_bytes": 1024}
        ev.variants = {}
        record = ev.evaluate(None, "SELECT 1", 0, "dataset", 0)
        self.assertEqual(record.verdict, "variant_mismatch")
        self.assertEqual(record.variant, ev.forced_variant)
        self.assertEqual(session.settings, {})

    def test_metadata_override_is_explicit(self):
        class Connection:
            calls = []

            def execute(self, sql):
                self.calls.append(sql)
                if len(self.calls) == 1:
                    raise RuntimeError("built specifically for DuckDB version abc")

        s = Session("extension", None, pathlib.Path("unused"), 0)
        s.con = Connection()
        with self.assertRaises(SessionError):
            s._load_extension()
        self.assertEqual(len(s.con.calls), 1)

    def test_invalid_limits(self):
        for value in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                validate_limits(argparse.Namespace(timeout=value))


if __name__ == "__main__":
    unittest.main()

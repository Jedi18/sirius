"""Evaluator + Report with a fake session: findings, dedup, known issues, artifacts."""

import pathlib
import random
import tempfile
import unittest

from . import conftest_path  # noqa: F401
from siriusfuzz import sqltypes as st
from siriusfuzz.classify import Verdict
from siriusfuzz.compare import ColumnInfo, ResultSet
from siriusfuzz.config import load_config
from siriusfuzz.report import KnownIssue, QueryRecord, Report, signature
from siriusfuzz.runner import Evaluator
from siriusfuzz.session import RunResult


class FakeSession:
    """Scripted stand-in for Session: `gpu_behaviour` decides what the GPU run returns."""

    def __init__(self, gpu_behaviour):
        self.gpu_available = True
        self.gpu_behaviour = gpu_behaviour
        self.settings = {}
        self.current_alias = "main"
        self.con = self
        self.calls = []

    # Session API used by Evaluator
    def setting_supported(self, name):
        return name == "hash_partition_bytes"

    def set(self, name, value):
        self.settings[name] = value

    def restore(self, name):
        self.settings.pop(name, None)

    def execute(self, sql, *a):
        self.calls.append(sql)
        return self

    def fetchall(self):
        return []

    def load_dataset(self, ds, alias, permutation_seed=None):
        pass

    def use(self, alias):
        self.current_alias = alias

    def load_sqlsmith(self):
        return False

    def describe(self, sql):
        return [ColumnInfo("c0", st.INTEGER), ColumnInfo("c1", st.DOUBLE)]

    def run(self, sql, gpu, timeout):
        cols = [ColumnInfo("c0", st.INTEGER), ColumnInfo("c1", st.DOUBLE)]
        cpu_rows = [(1, 1.5), (2, 2.5)]
        if not gpu:
            return RunResult("ok", ResultSet(cols, list(cpu_rows)))
        return self.gpu_behaviour(sql, cols, cpu_rows, self)


def gpu_ok(sql, cols, rows, s):
    return RunResult("ok", ResultSet(cols, list(rows)))


def gpu_float_noise(sql, cols, rows, s):
    return RunResult("ok", ResultSet(cols, [(1, 1.5 + 1e-13), (2, 2.5)]))


def gpu_wrong(sql, cols, rows, s):
    return RunResult("ok", ResultSet(cols, [(1, 1.5), (3, 2.5)]))


def gpu_plan_fallback(sql, cols, rows, s):
    return RunResult(
        "error",
        error="Not implemented Error: GPU plan generation failed: Window not supported",
    )


def gpu_variant_wrong(sql, cols, rows, s):
    if s.settings.get("hash_partition_bytes") is not None:
        return RunResult("ok", ResultSet(cols, [(1, 1.5)] * 2))
    return RunResult("ok", ResultSet(cols, list(rows)))


def gpu_count_distinct_error(sql, cols, rows, s):
    return RunResult(
        "error",
        error="Sirius GPU execution failed: count(DISTINCT x) not supported ungrouped",
    )


def evaluate(behaviour, sql="SELECT 1 AS c0, 1.5 AS c1", **over):
    cfg = load_config(
        None, ["oracle.ambiguity_filter=false", *[f"{k}={v}" for k, v in over.items()]]
    )
    session = FakeSession(behaviour)
    ev = Evaluator(cfg, session, lambda m: None)
    ev.rng = random.Random(1)
    return ev.evaluate(None, sql, 0, "w0-d0", 0), ev


class EvaluatorTests(unittest.TestCase):
    def test_ok_and_float_noise(self):
        rec, _ = evaluate(gpu_ok)
        self.assertEqual(rec.verdict, Verdict.OK.value)
        rec, _ = evaluate(gpu_float_noise)
        self.assertEqual(rec.verdict, Verdict.OK.value)

    def test_mismatch(self):
        rec, _ = evaluate(gpu_wrong)
        self.assertEqual(rec.verdict, Verdict.MISMATCH.value)
        self.assertTrue(rec.diffs)

    def test_plan_fallback_strict(self):
        rec, _ = evaluate(gpu_plan_fallback)
        self.assertEqual(rec.verdict, Verdict.PLAN_FALLBACK.value)
        self.assertEqual(rec.reason, "Window not supported")

    def test_variant_mismatch(self):
        rec, _ = evaluate(gpu_variant_wrong, **{"variants.per_query": "1"})
        self.assertEqual(rec.verdict, Verdict.VARIANT_MISMATCH.value)
        self.assertEqual(list(rec.variant), ["hash_partition_bytes"])


class ReportTests(unittest.TestCase):
    def test_dedup_known_issue_and_artifacts(self):
        cfg = load_config(None)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp)
            (run_dir / "datasets").mkdir()
            (run_dir / "datasets" / "w0-d0.sql").write_text(
                "CREATE TABLE t(k INTEGER);\n"
            )
            known = [
                KnownIssue(
                    ".",
                    "sirius-db/sirius#1218",
                    verdicts=["gpu_error"],
                    sql_pattern="count\\(DISTINCT",
                )
            ]
            report = Report(run_dir, cfg, known, seed=1)
            rec1, _ = evaluate(gpu_wrong)
            rec1.labels = ["Select", "Compare(=)"]
            rec2, _ = evaluate(gpu_wrong)
            rec2.labels = ["Select", "Compare(=)"]
            name1 = report.add(rec1)
            name2 = report.add(rec2)
            self.assertIsNotNone(name1)
            self.assertIsNone(name2, "same signature must dedup")
            self.assertEqual(signature(rec1), signature(rec2))
            d = run_dir / "findings" / name1
            for f in (
                "query.sql",
                "dataset.sql",
                "config.toml",
                "meta.json",
                "detail.txt",
                "repro_catch2.cpp",
            ):
                self.assertTrue((d / f).exists(), f)
            rec3, _ = evaluate(
                gpu_count_distinct_error, sql="SELECT count(DISTINCT k) AS c0 FROM t"
            )
            report.add(rec3)
            self.assertEqual(rec3.verdict, Verdict.KNOWN_ISSUE.value)
            self.assertIn("#1218", rec3.reason)
            summary = report.finish()
            self.assertEqual(summary["counts"]["mismatch"], 2)
            self.assertEqual(summary["counts"]["known_issue"], 1)
            self.assertTrue((run_dir / "summary.txt").exists())
            self.assertIn("mismatch", report.render_summary(summary))


if __name__ == "__main__":
    unittest.main()

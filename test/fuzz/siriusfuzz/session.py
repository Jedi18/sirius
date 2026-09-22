# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""One DuckDB connection with Sirius loaded, in strict mode.

Generated tables live in an ATTACHed file-backed database (in-memory tables
never reach the GPU native scan) and are CHECKPOINTed so the scan sees them on
disk. Queries run with ``enable_duckdb_fallback = false`` so plan-time and
runtime fallbacks surface as errors instead of silent CPU runs.
"""

from __future__ import annotations

import os
import pathlib
import threading
import time
from dataclasses import dataclass
from typing import Any

from . import sqltypes as st
from .compare import ColumnInfo, ResultSet
from .schema_gen import Dataset

CANARY_SETTING = "sirius_test_inject_transparent_gpu_error"


class SessionError(RuntimeError):
    pass


@dataclass
class RunResult:
    status: str  # ok | error | timeout
    result: ResultSet | None = None
    error: str = ""
    elapsed: float = 0.0


class Session:
    def __init__(
        self,
        extension: str | None,
        sirius_config: str | None,
        db_dir: pathlib.Path,
        worker_id: int,
        strict: bool = True,
    ):
        self.extension = extension
        self.sirius_config = sirius_config
        self.db_dir = db_dir
        self.worker_id = worker_id
        self.strict = strict
        self.con: Any = None
        self.gpu_available = extension is not None
        self.current_alias: str | None = None
        self._setting_defaults: dict[str, Any] = {}
        self._canary_ok: bool | None = None
        self.sqlsmith_loaded = False
        self.version_mismatch_bypassed = False

    # -- lifecycle -----------------------------------------------------------

    def open(self) -> None:
        import duckdb

        if self.sirius_config:
            os.environ["SIRIUS_CONFIG_FILE"] = str(self.sirius_config)
        # Registers the TEST ONLY fault-injection option used as the interception canary.
        os.environ.setdefault("SIRIUS_ENABLE_TEST_OPTIONS", "1")
        self.con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
        self.con.execute("SET enable_progress_bar = false")
        if self.extension:
            self._load_extension()
            if self.strict:
                self.con.execute("SET enable_duckdb_fallback = false")
        self.set_gpu(False)

    def _load_extension(self) -> None:
        try:
            self.con.execute(f"LOAD '{self.extension}'")
        except Exception as e:  # noqa: BLE001
            # The submodule-built Python module reports its git hash as the DuckDB version
            # while the extension carries OVERRIDE_GIT_DESCRIBE's v1.5.5; same source tree,
            # same ABI, so only the metadata check is skipped.
            if "built specifically for DuckDB version" not in str(e):
                raise
            self.con.execute("SET allow_extensions_metadata_mismatch = true")
            self.con.execute(f"LOAD '{self.extension}'")
            self.version_mismatch_bypassed = True

    def close(self) -> None:
        if self.con is not None:
            try:
                self.drop_dataset()
            except Exception:
                pass
            try:
                self.con.close()
            except Exception:
                pass
            self.con = None

    def set_gpu(self, enabled: bool) -> None:
        if self.gpu_available:
            self.con.execute(f"SET gpu_execution = {'true' if enabled else 'false'}")

    # -- datasets -------------------------------------------------------------

    def load_dataset(
        self, ds: Dataset, alias: str, permutation_seed: int | None = None
    ) -> None:
        self.db_dir.mkdir(parents=True, exist_ok=True)
        path = self.db_dir / f"{alias}.duckdb"
        for p in (path, pathlib.Path(str(path) + ".wal")):
            if p.exists():
                p.unlink()
        self.set_gpu(False)
        self.con.execute(f"ATTACH '{path}' AS {alias}")
        self.con.execute(f"USE {alias}")
        self.con.execute(ds.schema_sql())
        for stmt in ds.data_sql(permutation_seed).split(";\n"):
            if stmt.strip():
                self.con.execute(stmt)
        self.con.execute("CHECKPOINT")
        self.current_alias = alias

    def use(self, alias: str) -> None:
        self.con.execute(f"USE {alias}")
        self.current_alias = alias

    def drop_dataset(self, alias: str | None = None) -> None:
        aliases = [alias] if alias else list(self._attached())
        for a in aliases:
            try:
                self.con.execute("USE memory")
                self.con.execute(f"DETACH {a}")
            except Exception:
                pass
            for suffix in ("", ".wal"):
                p = self.db_dir / f"{a}.duckdb{suffix}"
                if p.exists():
                    p.unlink()
        self.current_alias = None

    def _attached(self) -> list[str]:
        rows = self.con.execute(
            "SELECT database_name FROM duckdb_databases() WHERE NOT internal AND database_name NOT IN ('memory','system','temp')"
        ).fetchall()
        return [r[0] for r in rows]

    # -- queries --------------------------------------------------------------

    def run(self, sql: str, gpu: bool, timeout: float) -> RunResult:
        import duckdb

        self.set_gpu(gpu)
        timer = threading.Timer(timeout, self._interrupt) if timeout > 0 else None
        start = time.monotonic()
        interrupted = False
        try:
            if timer:
                timer.start()
            cur = self.con.execute(sql)
            rows = cur.fetchall()
            desc = cur.description or []
        except duckdb.InterruptException:
            interrupted = True
            return RunResult(
                "timeout", elapsed=time.monotonic() - start, error="interrupted"
            )
        except (
            Exception
        ) as e:  # noqa: BLE001 - every DuckDB error type is a query error here
            msg = str(e)
            if "INTERRUPT" in msg.upper() or "interrupted" in msg.lower():
                return RunResult("timeout", elapsed=time.monotonic() - start, error=msg)
            return RunResult("error", error=msg, elapsed=time.monotonic() - start)
        finally:
            if timer:
                timer.cancel()
            if not interrupted:
                self.set_gpu(False)
        cols = [ColumnInfo(d[0], st.parse_duckdb_type(str(d[1]))) for d in desc]
        return RunResult("ok", ResultSet(cols, rows), elapsed=time.monotonic() - start)

    def _interrupt(self) -> None:
        try:
            self.con.interrupt()
        except Exception:
            pass

    def describe(self, sql: str) -> list[ColumnInfo] | None:
        """Exact output types (CPU side); None when DESCRIBE itself fails."""
        self.set_gpu(False)
        try:
            rows = self.con.execute(f"DESCRIBE {sql}").fetchall()
        except Exception:
            return None
        return [ColumnInfo(r[0], st.parse_duckdb_type(str(r[1]))) for r in rows]

    # -- settings ---------------------------------------------------------------

    def setting_supported(self, name: str) -> bool:
        row = self.con.execute(
            "SELECT value FROM duckdb_settings() WHERE name = ?", [name]
        ).fetchone()
        if row is None:
            return False
        self._setting_defaults.setdefault(name, row[0])
        return True

    def set(self, name: str, value: Any) -> None:
        if isinstance(value, str):
            self.con.execute(f"SET {name} = '{value}'")
        else:
            self.con.execute(f"SET {name} = {value}")

    def restore(self, name: str) -> None:
        default = self._setting_defaults.get(name)
        if default is None:
            return
        # Settings are stored as text in duckdb_settings(); numeric defaults round-trip as-is.
        try:
            self.con.execute(f"SET {name} = {int(default)}")
        except Exception:
            self.con.execute(f"SET {name} = '{default}'")

    def load_sqlsmith(self) -> bool:
        if self.sqlsmith_loaded:
            return True
        try:
            self.con.execute("LOAD sqlsmith")
        except Exception:
            try:
                self.con.execute("INSTALL sqlsmith")
                self.con.execute("LOAD sqlsmith")
            except Exception:
                return False
        self.sqlsmith_loaded = True
        return True

    # -- canary -----------------------------------------------------------------

    def check_interception(self) -> tuple[bool, str]:
        """Prove Sirius intercepts plain SQL on this connection.

        Injects a runtime GPU error via the TEST ONLY option; a query that then fails
        with the injected text went through the GPU operator. Falls back to a probe
        query that Sirius rejects at plan time when the option is unavailable.
        """
        if not self.gpu_available:
            return False, "no extension loaded (cpu-only mode)"
        if self.current_alias is None:
            return False, "no dataset attached"
        table = self.con.execute(
            "SELECT table_name FROM duckdb_tables() WHERE database_name = ? LIMIT 1",
            [self.current_alias],
        ).fetchone()
        if table is None:
            return False, "dataset has no tables"
        probe = f'SELECT count(*) FROM "{table[0]}"'
        if self.setting_supported(CANARY_SETTING):
            try:
                self.set(CANARY_SETTING, "fuzz-canary")
                res = self.run(probe, gpu=True, timeout=30)
            finally:
                self.set(CANARY_SETTING, "")
            if res.status == "error" and "fuzz-canary" in res.error:
                return True, "canary injected error observed"
            return (
                False,
                f"canary not observed (status={res.status}: {res.error[:120]})",
            )
        res = self.run(f'SELECT DISTINCT "k" FROM "{table[0]}"', gpu=True, timeout=30)
        if res.status == "error" and "GPU plan generation failed" in res.error:
            return True, "plan-time rejection observed"
        return (
            False,
            f"probe did not reach Sirius (status={res.status}: {res.error[:120]})",
        )

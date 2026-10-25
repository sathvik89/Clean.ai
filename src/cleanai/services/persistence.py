"""Durable run history, stored per user in SQLite."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class RunStore:
    def __init__(self, db_path: Path | str):
        self.db_path = str(db_path)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    dataset_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    user_context TEXT,
                    status TEXT NOT NULL,
                    error TEXT,
                    rows_before INTEGER, rows_after INTEGER,
                    cols_before INTEGER, cols_after INTEGER,
                    nulls_before INTEGER, nulls_after INTEGER,
                    duplicates_before INTEGER, duplicates_after INTEGER,
                    score_before INTEGER, score_after INTEGER,
                    steps_json TEXT, ledger_json TEXT, logs_text TEXT,
                    analysis_report TEXT,
                    cleaned_csv TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_runs_user ON runs(user_id, created_at DESC);
                """
            )

    def save_run(self, user_id: int, payload: Dict[str, Any]) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO runs (
                    user_id, dataset_name, created_at, user_context, status, error,
                    rows_before, rows_after, cols_before, cols_after,
                    nulls_before, nulls_after, duplicates_before, duplicates_after,
                    score_before, score_after, steps_json, ledger_json, logs_text,
                    analysis_report, cleaned_csv
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    payload["dataset_name"],
                    datetime.now(timezone.utc).isoformat(),
                    payload.get("user_context", ""),
                    payload["status"],
                    payload.get("error", ""),
                    payload.get("rows_before"), payload.get("rows_after"),
                    payload.get("cols_before"), payload.get("cols_after"),
                    payload.get("nulls_before"), payload.get("nulls_after"),
                    payload.get("duplicates_before"), payload.get("duplicates_after"),
                    payload.get("score_before"), payload.get("score_after"),
                    json.dumps(payload.get("steps", []), default=str),
                    json.dumps(payload.get("ledger", []), default=str),
                    payload.get("logs_text", ""),
                    payload.get("analysis_report", ""),
                    payload.get("cleaned_csv", ""),
                ),
            )
            return int(cursor.lastrowid)

    def list_runs(self, user_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, dataset_name, created_at, user_context, status, error,
                       rows_before, rows_after, score_before, score_after,
                       nulls_before, nulls_after, steps_json
                FROM runs WHERE user_id = ? ORDER BY created_at DESC LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        results = []
        for row in rows:
            entry = dict(row)
            entry["n_steps"] = len(json.loads(entry.pop("steps_json") or "[]"))
            results.append(entry)
        return results

    def get_run(self, run_id: int, user_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE id = ? AND user_id = ?", (run_id, user_id)
            ).fetchone()
        if row is None:
            return None
        entry = dict(row)
        entry["steps"] = json.loads(entry.pop("steps_json") or "[]")
        entry["ledger"] = json.loads(entry.pop("ledger_json") or "[]")
        return entry

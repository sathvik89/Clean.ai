"""Memory layers for the cleaning agent.

Three layers, cheapest first:
- ActionLedger: episodic memory of what each tool call did this run.
- Column cards (see profiling.py): semantic memory about each column.
- CaseMemory: cross-run memory in SQLite. Finished runs leave behind a
  fingerprint of the dataset plus which strategies worked, and future
  runs on similar-looking data get those strategies back as hints.

Everything renders to compact text because the LLM reads these, not us.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from cleanai.core.profiling import ColumnCard


@dataclass
class ActionRecord:
    seq: int
    tool: str
    args: Dict[str, Any]
    status: str                      # ok | error | noop
    delta: str                       # rendered metrics delta
    error: str = ""
    df_signature: str = ""

    def render(self) -> str:
        args_txt = json.dumps(self.args, default=str)
        if len(args_txt) > 110:
            args_txt = args_txt[:110] + "...}"
        line = f"#{self.seq} {self.tool}({args_txt}) -> {self.status}"
        if self.status == "error":
            return f"{line}: {self.error.splitlines()[0][:120]}"
        if self.delta:
            return f"{line} | {self.delta}"
        return line

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def render_ledger(records: List[Dict[str, Any]], max_entries: int = 15) -> str:
    """Render action records (stored as dicts in graph state) for the prompt."""
    if not records:
        return "no actions taken yet"
    actions = [ActionRecord(**r) for r in records]
    lines = []
    hidden = len(actions) - max_entries
    if hidden > 0:
        ok = sum(1 for a in actions[:hidden] if a.status == "ok")
        err = sum(1 for a in actions[:hidden] if a.status == "error")
        lines.append(f"(earlier: {hidden} actions collapsed, {ok} ok / {err} errors)")
    for action in actions[-max_entries:]:
        lines.append(action.render())
    return "\n".join(lines)


def render_cards(cards: List[ColumnCard], max_cards: int = 18, only_dirty: bool = False) -> str:
    chosen = [c for c in cards if c.issues_open or c.status != "clean"] if only_dirty else cards
    if only_dirty and not chosen:
        return "all columns are clean"
    lines = [c.render() for c in chosen[:max_cards]]
    hidden = len(chosen) - max_cards
    if hidden > 0:
        lines.append(f"(+{hidden} more columns)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# cross-run case memory
# ---------------------------------------------------------------------------

_NORM_PATTERN = re.compile(r"[\d_]+")


def _norm_name(name: str) -> str:
    return _NORM_PATTERN.sub("", name.lower()).rstrip("s")


def dataset_fingerprint(cards: List[ColumnCard]) -> List[str]:
    """Typed token set describing the dataset's structure and problems."""
    tokens = set()
    for card in cards:
        tokens.add(f"role:{card.role}")
        for issue in card.issues_open:
            tokens.add(f"issue:{issue.kind}")
            tokens.add(f"col:{_norm_name(card.name)}|{issue.kind}")
    n = len(cards)
    bucket = "1-5" if n <= 5 else "6-15" if n <= 15 else "16-50" if n <= 50 else "50+"
    tokens.add(f"ncols:{bucket}")
    return sorted(tokens)


def _weighted_jaccard(a: List[str], b: List[str]) -> float:
    def weight(token: str) -> float:
        return 2.0 if token.startswith("col:") else 1.0

    set_a, set_b = set(a), set(b)
    union = set_a | set_b
    if not union:
        return 0.0
    inter_weight = sum(weight(t) for t in set_a & set_b)
    union_weight = sum(weight(t) for t in union)
    return inter_weight / union_weight


class CaseMemory:
    """SQLite-backed memory of past cleaning runs and what worked."""

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
                CREATE TABLE IF NOT EXISTS cases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    fingerprint_json TEXT NOT NULL,
                    outcome_score REAL NOT NULL,
                    n_actions INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS case_strategies (
                    case_id INTEGER REFERENCES cases(id),
                    issue_kind TEXT NOT NULL,
                    tool TEXT NOT NULL,
                    succeeded INTEGER NOT NULL,
                    detail TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_strategies
                    ON case_strategies(issue_kind, tool);
                """
            )

    def store_case(
        self,
        fingerprint: List[str],
        outcome_score: float,
        strategies: List[Dict[str, Any]],
    ) -> int:
        """strategies: [{issue_kind, tool, succeeded, detail}]"""
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO cases (created_at, fingerprint_json, outcome_score, n_actions) VALUES (?, ?, ?, ?)",
                (
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(fingerprint),
                    outcome_score,
                    len(strategies),
                ),
            )
            case_id = cursor.lastrowid
            for strategy in strategies:
                conn.execute(
                    "INSERT INTO case_strategies (case_id, issue_kind, tool, succeeded, detail) VALUES (?, ?, ?, ?, ?)",
                    (
                        case_id,
                        strategy["issue_kind"],
                        strategy["tool"],
                        1 if strategy.get("succeeded") else 0,
                        strategy.get("detail", ""),
                    ),
                )
        return int(case_id)

    def hints_for(self, fingerprint: List[str], min_similarity: float = 0.3, top_k: int = 3) -> str:
        """Aggregated advice from the most similar successful past runs."""
        with self._connect() as conn:
            cases = conn.execute(
                "SELECT id, fingerprint_json, outcome_score FROM cases WHERE outcome_score >= 0.7"
            ).fetchall()
            scored = []
            for case in cases:
                similarity = _weighted_jaccard(fingerprint, json.loads(case["fingerprint_json"]))
                if similarity >= min_similarity:
                    scored.append((similarity, case["id"]))
            scored.sort(reverse=True)
            matched_ids = [case_id for _, case_id in scored[:top_k]]
            if not matched_ids:
                return ""

            placeholders = ",".join("?" * len(matched_ids))
            rows = conn.execute(
                f"SELECT issue_kind, tool, succeeded FROM case_strategies WHERE case_id IN ({placeholders})",
                matched_ids,
            ).fetchall()

        tally: Dict[tuple, Dict[str, int]] = {}
        for row in rows:
            key = (row["issue_kind"], row["tool"])
            entry = tally.setdefault(key, {"won": 0, "lost": 0})
            entry["won" if row["succeeded"] else "lost"] += 1

        lines = [f"hints from {len(matched_ids)} similar past datasets (advisory only):"]
        for (issue_kind, tool), counts in sorted(tally.items()):
            total = counts["won"] + counts["lost"]
            if counts["won"] > counts["lost"]:
                lines.append(f"- {issue_kind}: {tool} succeeded {counts['won']}/{total} times")
            elif counts["lost"] >= 2:
                lines.append(f"- {issue_kind}: {tool} FAILED {counts['lost']}/{total} times, avoid it")
        return "\n".join(lines) if len(lines) > 1 else ""

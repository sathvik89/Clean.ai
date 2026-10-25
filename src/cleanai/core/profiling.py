"""Deterministic data profiling.

This module is the agent's eyes. It turns a DataFrame into compact,
structured facts: column cards, detected issues, quality scores and
cheap snapshots used to measure what each tool call actually changed.
No LLM is involved anywhere in this file.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@dataclass
class Issue:
    kind: str          # e.g. "missing_values", "duplicate_rows", "invalid_email"
    column: str        # "" for table-level issues
    severity: str      # "high" | "medium" | "low"
    evidence: str      # short human-readable proof, <= 90 chars

    def describe(self) -> str:
        where = f"[{self.column}] " if self.column else ""
        return f"{where}{self.kind}: {self.evidence}"


@dataclass
class ColumnCard:
    name: str
    role: str                      # id | email | numeric | datetime | categorical | text
    dtype: str
    semantic_hint: str = ""        # age | non_negative_numeric | email | past_datetime
    is_key: bool = False
    null_count: int = 0
    null_pct: float = 0.0
    n_unique: int = 0
    sample_values: List[str] = field(default_factory=list)
    issues_open: List[Issue] = field(default_factory=list)
    fixes_applied: List[str] = field(default_factory=list)
    status: str = "untouched"      # untouched | in_progress | clean | blocked

    def render(self) -> str:
        parts = [f"{self.name} [{self.role}/{self.dtype}] {self.status}"]
        parts.append(f"nulls={self.null_count} ({self.null_pct:.1f}%)")
        if self.is_key:
            parts.append("KEY")
        if self.issues_open:
            open_txt = "; ".join(i.describe() for i in self.issues_open[:3])
            parts.append(f"open: {open_txt}")
        if self.fixes_applied:
            parts.append(f"fixed: {', '.join(self.fixes_applied[-3:])}")
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# role and hint inference
# ---------------------------------------------------------------------------

def infer_semantic_hint(column_name: str) -> str:
    name = column_name.lower()
    if name == "age" or name.endswith("_age") or name.startswith("age_"):
        return "age"
    if any(k in name for k in ("salary", "wage", "income", "amount", "price", "cost")):
        return "non_negative_numeric"
    if "email" in name:
        return "email"
    if any(k in name for k in ("join", "created", "start_date", "date")):
        return "past_datetime"
    return ""


def is_key_column(series: pd.Series, column_name: str) -> bool:
    name = column_name.lower()
    non_null = series.dropna()
    if non_null.empty:
        return False
    if "email" in name:
        return True
    if "manager" in name:
        return False
    if name == "id" or name.endswith("_id") or name.endswith("id"):
        return non_null.nunique() / len(non_null) >= 0.85
    return False


def infer_column_role(series: pd.Series, column_name: str) -> str:
    name = column_name.lower()
    values = series.dropna()
    if values.empty:
        return "text"
    text_values = values.astype(str).str.strip()
    if "email" in name:
        return "email"
    if any(k in name for k in ("date", "time", "joined", "created")):
        return "datetime"
    numeric_ratio = pd.to_numeric(text_values, errors="coerce").notna().mean()
    if numeric_ratio >= 0.75:
        return "numeric"
    datetime_ratio = pd.to_datetime(
        text_values, errors="coerce", format="mixed"
    ).notna().mean()
    if datetime_ratio >= 0.75:
        return "datetime"
    unique_count = text_values.nunique()
    if unique_count <= min(20, max(5, int(len(text_values) * 0.3))):
        return "categorical"
    return "text"


# ---------------------------------------------------------------------------
# issue detection
# ---------------------------------------------------------------------------

def detect_table_issues(df: pd.DataFrame) -> List[Issue]:
    issues: List[Issue] = []
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        issues.append(Issue("duplicate_rows", "", "high", f"{dup_count} exact duplicate rows"))
    return issues


def detect_column_issues(df: pd.DataFrame, card: ColumnCard) -> List[Issue]:
    issues: List[Issue] = []
    series = df[card.name]
    non_null = series.dropna()

    if card.null_count > 0:
        severity = "high" if card.null_pct > 20 else "medium"
        issues.append(Issue(
            "missing_values", card.name, severity,
            f"{card.null_count} nulls ({card.null_pct:.1f}%)",
        ))

    if non_null.empty:
        return issues

    if card.is_key:
        dup_values = int(non_null.duplicated().sum())
        if dup_values > 0:
            issues.append(Issue(
                "duplicate_key_values", card.name, "high",
                f"{dup_values} duplicated key values",
            ))

    text_values = non_null.astype(str)
    if card.role == "numeric" and not is_numeric_dtype(series):
        invalid = int(pd.to_numeric(non_null, errors="coerce").isna().sum())
        if invalid > 0:
            issues.append(Issue(
                "non_numeric_values", card.name, "high",
                f"{invalid} values cannot parse as numbers",
            ))
        else:
            issues.append(Issue(
                "wrong_dtype", card.name, "medium",
                f"numeric data stored as {card.dtype}",
            ))
    elif card.role == "datetime" and not is_datetime64_any_dtype(series):
        invalid = int(pd.to_datetime(text_values, errors="coerce", format="mixed").isna().sum())
        if invalid > 0:
            issues.append(Issue(
                "invalid_dates", card.name, "high",
                f"{invalid} values cannot parse as dates",
            ))
        else:
            issues.append(Issue(
                "wrong_dtype", card.name, "medium",
                f"date data stored as {card.dtype}",
            ))
    elif card.role == "email":
        invalid = int((~text_values.str.match(EMAIL_PATTERN)).sum())
        if invalid > 0:
            issues.append(Issue(
                "invalid_email", card.name, "high",
                f"{invalid} malformed email values",
            ))

    numeric_values = pd.to_numeric(non_null, errors="coerce")
    if card.semantic_hint == "age":
        bad = int(((numeric_values < 0) | (numeric_values > 120)).fillna(False).sum())
        if bad > 0:
            issues.append(Issue(
                "out_of_range", card.name, "high", f"{bad} ages outside 0-120",
            ))
    elif card.semantic_hint == "non_negative_numeric":
        bad = int((numeric_values < 0).fillna(False).sum())
        if bad > 0:
            issues.append(Issue(
                "negative_values", card.name, "high", f"{bad} negative values",
            ))
    elif card.semantic_hint == "past_datetime":
        parsed = pd.to_datetime(text_values, errors="coerce", format="mixed")
        future = int((parsed > pd.Timestamp.now()).fillna(False).sum())
        if future > 0:
            issues.append(Issue(
                "future_dates", card.name, "medium", f"{future} dates in the future",
            ))

    if series.dtype == object:
        type_names = {type(v).__name__ for v in non_null.head(50)}
        if len(type_names) > 1:
            issues.append(Issue(
                "mixed_types", card.name, "medium",
                f"mixed python types: {'|'.join(sorted(type_names))}",
            ))

    return issues


# ---------------------------------------------------------------------------
# cards, snapshots, deltas
# ---------------------------------------------------------------------------

def build_cards(df: pd.DataFrame) -> List[ColumnCard]:
    cards = []
    row_count = max(len(df), 1)
    for col in df.columns:
        series = df[col]
        null_count = int(series.isnull().sum())
        card = ColumnCard(
            name=str(col),
            role=infer_column_role(series, str(col)),
            dtype=str(series.dtype),
            semantic_hint=infer_semantic_hint(str(col)),
            is_key=is_key_column(series, str(col)),
            null_count=null_count,
            null_pct=100.0 * null_count / row_count,
            n_unique=int(series.nunique(dropna=True)),
            sample_values=[str(v)[:40] for v in series.dropna().head(3)],
        )
        card.issues_open = detect_column_issues(df, card)
        if not card.issues_open:
            card.status = "clean"
        cards.append(card)
    return cards


def refresh_cards(df: pd.DataFrame, cards: List[ColumnCard]) -> List[ColumnCard]:
    """Re-check every card against the current data, keeping fix history."""
    refreshed = []
    row_count = max(len(df), 1)
    existing = {c.name: c for c in cards}
    for col in df.columns:
        name = str(col)
        old = existing.get(name)
        series = df[col]
        null_count = int(series.isnull().sum())
        card = ColumnCard(
            name=name,
            role=old.role if old else infer_column_role(series, name),
            dtype=str(series.dtype),
            semantic_hint=old.semantic_hint if old else infer_semantic_hint(name),
            is_key=old.is_key if old else is_key_column(series, name),
            null_count=null_count,
            null_pct=100.0 * null_count / row_count,
            n_unique=int(series.nunique(dropna=True)),
            sample_values=[str(v)[:40] for v in series.dropna().head(3)],
            fixes_applied=list(old.fixes_applied) if old else [],
        )
        card.issues_open = detect_column_issues(df, card)
        if card.issues_open:
            card.status = "in_progress" if (old and old.fixes_applied) else (old.status if old and old.status == "blocked" else "untouched")
            if old and old.status == "blocked":
                card.status = "blocked"
        else:
            card.status = "clean"
        refreshed.append(card)
    return refreshed


def all_open_issues(df: pd.DataFrame, cards: List[ColumnCard]) -> List[Issue]:
    issues = detect_table_issues(df)
    for card in cards:
        issues.extend(card.issues_open)
    return issues


def snapshot(df: pd.DataFrame) -> Dict[str, Any]:
    """Cheap state snapshot used to compute per-action metric deltas."""
    return {
        "rows": len(df),
        "cols": len(df.columns),
        "nulls": {str(c): int(df[c].isnull().sum()) for c in df.columns},
        "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
        "duplicates": int(df.duplicated().sum()),
    }


def df_signature(df: pd.DataFrame) -> str:
    """Short hash of shape + dtypes + null counts; used for loop detection."""
    snap = snapshot(df)
    raw = f"{snap['rows']}|{snap['cols']}|{sorted(snap['nulls'].items())}|{sorted(snap['dtypes'].items())}|{snap['duplicates']}"
    return hashlib.sha1(raw.encode()).hexdigest()[:10]


def metrics_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    delta: Dict[str, Any] = {}
    if before["rows"] != after["rows"]:
        delta["rows"] = f"{before['rows']}->{after['rows']}"
    if before["cols"] != after["cols"]:
        delta["cols"] = f"{before['cols']}->{after['cols']}"
    if before["duplicates"] != after["duplicates"]:
        delta["duplicates"] = f"{before['duplicates']}->{after['duplicates']}"
    for col, nulls_after in after["nulls"].items():
        nulls_before = before["nulls"].get(col)
        if nulls_before is not None and nulls_before != nulls_after:
            delta[f"nulls.{col}"] = f"{nulls_before}->{nulls_after}"
    for col, dtype_after in after["dtypes"].items():
        dtype_before = before["dtypes"].get(col)
        if dtype_before is not None and dtype_before != dtype_after:
            delta[f"dtype.{col}"] = f"{dtype_before}->{dtype_after}"
    return delta


def render_delta(delta: Dict[str, Any]) -> str:
    if not delta:
        return "no measurable change"
    parts = [f"{k} {v}" for k, v in list(delta.items())[:8]]
    extra = len(delta) - 8
    if extra > 0:
        parts.append(f"(+{extra} more changes)")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# summaries and scoring
# ---------------------------------------------------------------------------

def summarize_for_llm(df: pd.DataFrame, cards: Optional[List[ColumnCard]] = None) -> str:
    cards = cards or build_cards(df)
    lines = [f"dataset: {len(df)} rows x {len(df.columns)} columns"]
    table_issues = detect_table_issues(df)
    for issue in table_issues:
        lines.append(f"table issue: {issue.describe()}")
    lines.append("columns:")
    for card in cards[:40]:
        lines.append(f"- {card.render()}")
    if len(cards) > 40:
        lines.append(f"(+{len(cards) - 40} more columns not shown)")
    return "\n".join(lines)


def quality_score(df: pd.DataFrame, cards: Optional[List[ColumnCard]] = None) -> int:
    """0-100 score: starts at 100 and loses points per open issue."""
    if df.empty or len(df.columns) == 0:
        return 0
    cards = cards or build_cards(df)
    issues = all_open_issues(df, cards)
    penalty = 0.0
    weights = {"high": 8.0, "medium": 4.0, "low": 1.5}
    for issue in issues:
        penalty += weights.get(issue.severity, 2.0)
    null_pct = 100.0 * df.isnull().sum().sum() / max(df.size, 1)
    penalty += null_pct * 0.5
    return max(0, min(100, round(100 - penalty)))

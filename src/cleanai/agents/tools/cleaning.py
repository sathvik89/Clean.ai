"""The actual pandas transformations the agent can apply.

Each function takes a DataFrame and returns a new one; none of them
mutate their input. The registry wires these to LLM-callable tools.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

SUPPORTED_MISSING_STRATEGIES = {"drop", "constant", "mean", "median", "mode", "interpolate"}

STRATEGY_ALIASES = {
    "fill_with_constant": "constant",
    "fill_with_value": "constant",
    "fill_with_median": "median",
    "fill_with_mean": "mean",
    "fill_with_mode": "mode",
    "most_frequent": "mode",
    "ffill": "interpolate",
    "bfill": "interpolate",
}


def resolve_missing_strategy(series: pd.Series, strategy: str) -> str:
    """Map loose model phrasing onto a supported strategy."""
    raw = (strategy or "").strip().lower()
    raw = STRATEGY_ALIASES.get(raw, raw)
    if raw in SUPPORTED_MISSING_STRATEGIES:
        return raw

    # unknown or "auto" style input: pick something sensible for the column
    numeric_ratio = pd.to_numeric(series, errors="coerce").notna().mean()
    if is_numeric_dtype(series) or numeric_ratio >= 0.6:
        return "median"
    return "mode"


def handle_missing(
    df: pd.DataFrame,
    strategy: str,
    columns: Optional[List[str]] = None,
    fill_value: Any = None,
) -> pd.DataFrame:
    out = df.copy()
    target_columns = columns if columns else list(out.columns)
    for col in target_columns:
        if col not in out.columns:
            continue
        resolved = resolve_missing_strategy(out[col], strategy)
        if resolved == "drop":
            out = out.dropna(subset=[col])
        elif resolved == "constant":
            value = fill_value
            if value is None:
                value = 0 if is_numeric_dtype(out[col]) else "Unknown"
            out[col] = out[col].fillna(value)
        elif resolved == "interpolate":
            if is_datetime64_any_dtype(out[col]):
                out[col] = out[col].ffill().bfill()
            else:
                numeric = pd.to_numeric(out[col], errors="coerce")
                filled = numeric.interpolate(limit_direction="both")
                out[col] = out[col].fillna(filled)
        elif resolved in ("mean", "median"):
            numeric = pd.to_numeric(out[col], errors="coerce")
            value = numeric.mean() if resolved == "mean" else numeric.median()
            if is_numeric_dtype(out[col]):
                out[col] = out[col].fillna(value)
            else:
                # column still holds text; coerce it to numeric as we fill
                out[col] = numeric.fillna(value)
        elif resolved == "mode":
            modes = out[col].mode()
            if not modes.empty:
                out[col] = out[col].fillna(modes[0])
    return out


def remove_duplicates(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    valid = [c for c in columns if c in df.columns] if columns else None
    return df.drop_duplicates(subset=valid or None).copy()


def convert_types(df: pd.DataFrame, column_type_map: Dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for col, target in column_type_map.items():
        if col not in out.columns:
            continue
        target = target.strip().lower()
        try:
            if target in ("datetime", "date", "timestamp"):
                out[col] = pd.to_datetime(out[col], errors="coerce", format="mixed")
            elif target in ("int", "integer"):
                out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
            elif target in ("float", "number", "numeric"):
                out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
            elif target in ("str", "string", "text"):
                out[col] = out[col].astype("string")
            elif target in ("bool", "boolean"):
                out[col] = out[col].astype("boolean")
            elif target in ("category", "categorical"):
                out[col] = out[col].astype("category")
            else:
                out[col] = out[col].astype(target)
        except (ValueError, TypeError):
            continue
    return out


def clean_text(df: pd.DataFrame, columns: List[str], actions: List[str]) -> pd.DataFrame:
    out = df.copy()
    normalized_actions = {a.strip().lower() for a in actions}
    for col in columns:
        if col not in out.columns:
            continue
        mask = out[col].notna()
        series = out.loc[mask, col].astype(str)
        if "lowercase" in normalized_actions:
            series = series.str.lower()
        if "uppercase" in normalized_actions:
            series = series.str.upper()
        if "titlecase" in normalized_actions:
            series = series.str.title()
        if "strip" in normalized_actions:
            series = series.str.strip()
        if "collapse_spaces" in normalized_actions:
            series = series.str.replace(r"\s+", " ", regex=True)
        if "remove_special" in normalized_actions:
            series = series.str.replace(r"[^\w\s@.\-]", "", regex=True)
        out.loc[mask, col] = series
    return out


def encode_categorical(df: pd.DataFrame, columns: List[str], method: str = "label") -> pd.DataFrame:
    out = df.copy()
    valid = [c for c in columns if c in out.columns]
    if method == "onehot":
        return pd.get_dummies(out, columns=valid)
    for col in valid:
        out[col] = out[col].astype("category").cat.codes
    return out


def outlier_report(df: pd.DataFrame, columns: List[str]) -> str:
    lines = []
    for col in columns:
        if col not in df.columns:
            lines.append(f"{col}: column not found")
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((numeric < low) | (numeric > high)).sum())
        lines.append(f"{col}: {count} outliers outside [{low:.2f}, {high:.2f}]")
    return "; ".join(lines) if lines else "no columns given"


def clip_outliers(df: pd.DataFrame, columns: List[str], method: str = "clip") -> pd.DataFrame:
    """Clip to IQR fences, or set outliers to NaN with method='nullify'."""
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        numeric = pd.to_numeric(out[col], errors="coerce")
        q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        if method == "nullify":
            mask = (numeric < low) | (numeric > high)
            out.loc[mask.fillna(False), col] = pd.NA
        else:
            out[col] = numeric.clip(lower=low, upper=high)
    return out


def rename_columns(df: pd.DataFrame, rename_map: Dict[str, str]) -> pd.DataFrame:
    valid = {old: new for old, new in rename_map.items() if old in df.columns}
    return df.rename(columns=valid).copy()


def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    valid = [c for c in columns if c in df.columns]
    return df.drop(columns=valid).copy()

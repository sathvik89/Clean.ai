import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import re
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

# optimized enterprise-grade tools for high token efficiency

def _detect_type_hints(series: pd.Series, sample_limit: int = 30) -> str:
    non_null_values = series.dropna().head(sample_limit)
    if non_null_values.empty:
        return "empty"

    type_names = []
    for value in non_null_values:
        name = type(value).__name__
        if name not in type_names:
            type_names.append(name)
        if len(type_names) >= 3:
            break
    return "|".join(type_names)

def get_data_summary(df: pd.DataFrame) -> str:
    """returns a compact, token-efficient summary of the data."""
    lines = [f"dataset: rows={df.shape[0]}, columns={df.shape[1]}"]
    lines.append("schema overview:")

    max_columns = 40
    visible_columns = list(df.columns[:max_columns])
    for col in visible_columns:
        null_count = df[col].isnull().sum()
        dtype = str(df[col].dtype)
        type_hints = _detect_type_hints(df[col])
        lines.append(f"- {col}: dtype={dtype}, nulls={null_count}, sample_types={type_hints}")

    hidden_columns = len(df.columns) - len(visible_columns)
    if hidden_columns > 0:
        lines.append(f"additional_columns_not_listed={hidden_columns}")

    null_columns = df.isnull().sum()
    remaining_nulls = null_columns[null_columns > 0].sort_values(ascending=False)
    if not remaining_nulls.empty:
        top_nulls = [f"{col}={int(count)}" for col, count in remaining_nulls.head(12).items()]
        lines.append(f"null_hotspots: {', '.join(top_nulls)}")
    else:
        lines.append("null_hotspots: none")

    duplicate_count = int(df.duplicated().sum())
    lines.append(f"duplicate_rows={duplicate_count}")
    key_duplicate_counts = get_key_duplicate_counts(df)
    if key_duplicate_counts:
        key_parts = [f"{col}={count}" for col, count in key_duplicate_counts.items()]
        lines.append(f"key_duplicates: {', '.join(key_parts)}")
    return "\n".join(lines)

def audit_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """performs a concise audit focusing only on remaining issues."""
    nulls = df.isnull().sum()
    issues = {
        "remaining_nulls": nulls[nulls > 0].to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "key_duplicates": get_key_duplicate_counts(df),
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "schema_preview": {
            col: str(dtype)
            for col, dtype in list(df.dtypes.astype(str).items())[:40]
        },
    }
    return issues

def infer_validation_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """builds a lightweight validation profile from the original dataset."""
    profile = {}
    for col in df.columns:
        profile[col] = {
            "role": infer_column_role(df[col], col),
            "original_dtype": str(df[col].dtype),
            "is_key": is_key_column(df[col], col),
            "semantic_hint": infer_semantic_hint(col),
        }
    return profile

def infer_semantic_hint(column_name: str) -> str:
    """infers lightweight semantic constraints from column names."""
    name = column_name.lower()
    if name == "age" or name.endswith("_age") or name.startswith("age_"):
        return "age"
    if any(keyword in name for keyword in ["salary", "wage", "income", "amount", "price", "cost"]):
        return "non_negative_numeric"
    if "email" in name:
        return "email"
    if any(keyword in name for keyword in ["join", "created", "start_date", "date"]):
        return "past_datetime"
    return ""

def is_key_column(series: pd.Series, column_name: str) -> bool:
    """checks whether a column behaves like a unique business key."""
    name = column_name.lower()
    non_null = series.dropna()
    if non_null.empty:
        return False

    if "email" in name:
        return True

    if "manager" in name:
        return False

    if name == "id" or name.endswith("_id") or name.endswith("id"):
        uniqueness_ratio = non_null.nunique() / len(non_null)
        return uniqueness_ratio >= 0.85

    return False

def get_key_duplicate_counts(df: pd.DataFrame) -> Dict[str, int]:
    """counts duplicate values for key-like columns."""
    counts = {}
    for col in df.columns:
        if not is_key_column(df[col], col):
            continue
        duplicate_values = int(df[col].dropna().duplicated().sum())
        if duplicate_values > 0:
            counts[col] = duplicate_values
    return counts

def infer_column_role(series: pd.Series, column_name: str) -> str:
    """infers the logical role for a column using name and parse ratios."""
    name = column_name.lower()
    values = series.dropna()
    if values.empty:
        return "text"

    string_values = values.astype(str).str.strip()
    if "email" in name:
        return "email"
    if any(keyword in name for keyword in ["date", "time", "joined", "created"]):
        return "datetime"

    numeric_ratio = pd.to_numeric(string_values, errors="coerce").notna().mean()
    datetime_ratio = pd.to_datetime(string_values, errors="coerce", format="mixed").notna().mean()
    unique_count = string_values.nunique()

    if numeric_ratio >= 0.75:
        return "numeric"
    if datetime_ratio >= 0.75:
        return "datetime"
    if unique_count <= min(20, max(5, int(len(string_values) * 0.3))):
        return "categorical"
    return "text"

def get_validation_issues(df: pd.DataFrame) -> List[str]:
    """returns deterministic validation issues from the current dataframe."""
    issues = []

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        issues.append(f"{duplicate_count} duplicate rows still remain")

    key_duplicate_counts = get_key_duplicate_counts(df)
    for col, count in key_duplicate_counts.items():
        issues.append(f"key column {col} still has {count} duplicate values")

    null_counts = df.isnull().sum()
    remaining_nulls = null_counts[null_counts > 0].sort_values(ascending=False)
    if not remaining_nulls.empty:
        top_nulls = [f"{col}={int(count)}" for col, count in remaining_nulls.head(12).items()]
        issues.append("missing values remain in: " + ", ".join(top_nulls))

    mixed_columns = []
    for col in df.columns[:40]:
        type_hints = _detect_type_hints(df[col])
        if "|" in type_hints:
            mixed_columns.append(f"{col}({type_hints})")
    if mixed_columns:
        issues.append("mixed value types still detected in: " + ", ".join(mixed_columns[:12]))

    return issues

def get_profile_validation_issues(df: pd.DataFrame, validation_profile: Dict[str, Any]) -> List[str]:
    """validates current data against the inferred column roles."""
    issues = []
    email_pattern = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

    for col, config in validation_profile.items():
        if col not in df.columns:
            issues.append(f"expected column missing after cleaning: {col}")
            continue

        role = config.get("role", "text")
        is_key = config.get("is_key", False)
        semantic_hint = config.get("semantic_hint", "")
        series = df[col]
        non_null = series.dropna()
        if non_null.empty:
            continue

        if is_key:
            duplicate_values = int(non_null.duplicated().sum())
            if duplicate_values > 0:
                issues.append(f"key column {col} still has {duplicate_values} duplicate values")

        if role == "numeric":
            invalid_mask = pd.to_numeric(non_null, errors="coerce").isna()
            invalid_count = int(invalid_mask.sum())
            if invalid_count > 0:
                issues.append(f"numeric column {col} still has {invalid_count} invalid values")
        elif role == "datetime":
            invalid_mask = pd.to_datetime(non_null.astype(str), errors="coerce", format="mixed").isna()
            invalid_count = int(invalid_mask.sum())
            if invalid_count > 0:
                issues.append(f"datetime column {col} still has {invalid_count} invalid values")
        elif role == "email":
            invalid_count = int((~non_null.astype(str).str.match(email_pattern)).sum())
            if invalid_count > 0:
                issues.append(f"email column {col} still has {invalid_count} invalid values")

        numeric_values = pd.to_numeric(non_null, errors="coerce")
        if semantic_hint == "age":
            invalid_count = int(((numeric_values < 0) | (numeric_values > 120)).fillna(False).sum())
            if invalid_count > 0:
                issues.append(f"age column {col} still has {invalid_count} out-of-range values")
        elif semantic_hint == "non_negative_numeric":
            invalid_count = int((numeric_values < 0).fillna(False).sum())
            if invalid_count > 0:
                issues.append(f"numeric column {col} still has {invalid_count} negative values")
        elif semantic_hint == "past_datetime":
            parsed_dates = pd.to_datetime(non_null.astype(str), errors="coerce", format="mixed")
            invalid_count = int((parsed_dates > pd.Timestamp.now()).fillna(False).sum())
            if invalid_count > 0:
                issues.append(f"datetime column {col} still has {invalid_count} future values")

    return issues

def format_log_details(log_text: str) -> str:
    """normalizes log text for simple log rendering."""
    return log_text.replace(" | ", "\n")

def choose_missing_strategy(series: pd.Series, strategy: str) -> str:
    """maps loose llm strategy names to supported strategies."""
    raw = (strategy or "").strip().lower()
    alias_map = {
        "fill_with_constant": "constant",
        "fill_with_value": "constant",
        "fill_with_median": "median",
        "fill_with_mean": "mean",
        "fill_with_mode": "mode",
        "most_frequent": "mode",
    }
    if raw in alias_map:
        return alias_map[raw]
    if raw in {"drop", "constant", "mean", "median", "mode", "interpolate"}:
        return raw

    if raw in {"median/mean/mode", "auto", ""}:
        if is_numeric_dtype(series) or pd.to_numeric(series, errors="coerce").notna().mean() >= 0.6:
            return "median"
        if is_datetime64_any_dtype(series) or pd.to_datetime(series.astype(str), errors="coerce", format="mixed").notna().mean() >= 0.6:
            return "mode"
        return "mode"

    return raw

def smart_handle_missing(df: pd.DataFrame, strategy: str, columns: Optional[List[str]] = None, fill_value: Any = None) -> pd.DataFrame:
    df_copy = df.copy()
    cols = columns if columns else df_copy.columns
    for col in cols:
        if col not in df_copy.columns: continue
        resolved_strategy = choose_missing_strategy(df_copy[col], strategy)
        if resolved_strategy == 'drop':
            df_copy = df_copy.dropna(subset=[col])
        elif resolved_strategy == 'constant':
            value = fill_value
            if value is None:
                value = "Unknown" if not is_numeric_dtype(df_copy[col]) else 0
            df_copy[col] = df_copy[col].fillna(value)
        elif resolved_strategy == 'interpolate':
            if is_datetime64_any_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].ffill().bfill()
            else:
                numeric_series = pd.to_numeric(df_copy[col], errors='coerce')
                filled = numeric_series.interpolate(limit_direction='both')
                df_copy[col] = df_copy[col].fillna(filled)
        else:
            if resolved_strategy in ['mean', 'median']:
                temp = pd.to_numeric(df_copy[col], errors='coerce')
                val = temp.mean() if resolved_strategy == 'mean' else temp.median()
                df_copy[col] = df_copy[col].fillna(val)
            elif resolved_strategy == 'mode':
                if not df_copy[col].mode().empty:
                    df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
    return df_copy

def rigorous_remove_duplicates(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    return df.drop_duplicates(subset=columns).copy()

def smart_type_conversion(df: pd.DataFrame, column_type_map: Dict[str, str]) -> pd.DataFrame:
    df_copy = df.copy()
    for col, dtype in column_type_map.items():
        if col not in df_copy.columns: continue
        try:
            if dtype == 'datetime':
                df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce', format='mixed')
            elif dtype in ['int', 'float']:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                if dtype == 'int':
                    df_copy[col] = df_copy[col].astype('Int64')
                else:
                    df_copy[col] = df_copy[col].astype(float)
            elif dtype in ['str', 'string']:
                df_copy[col] = df_copy[col].astype('string')
            else:
                df_copy[col] = df_copy[col].astype(dtype)
        except: pass
    return df_copy

def detect_outliers_report(df: pd.DataFrame, columns: List[str]) -> str:
    report = ""
    for col in columns:
        if col not in df.columns: continue
        num = pd.to_numeric(df[col], errors='coerce')
        q1, q3 = num.quantile(0.25), num.quantile(0.75)
        iqr = q3 - q1
        count = ((num < (q1 - 1.5 * iqr)) | (num > (q3 + 1.5 * iqr))).sum()
        report += f"{col}: {count} outliers; "
    return report

def perform_text_cleaning(df: pd.DataFrame, columns: List[str], actions: List[str]) -> pd.DataFrame:
    df_copy = df.copy()
    for col in columns:
        if col not in df_copy.columns: continue
        s = df_copy[col].astype(str)
        if 'lowercase' in actions: s = s.str.lower()
        if 'strip' in actions: s = s.str.strip()
        if 'remove_special' in actions: s = s.str.replace(r'[^\w\s]', '', regex=True)
        df_copy[col] = s
    return df_copy

def handle_categorical(df: pd.DataFrame, columns: List[str], method: str = 'label') -> pd.DataFrame:
    df_copy = df.copy()
    if method == 'label':
        for col in columns:
            df_copy[col] = df_copy[col].astype('category').cat.codes
    elif method == 'onehot':
        df_copy = pd.get_dummies(df_copy, columns=columns)
    return df_copy

def execute_custom_pandas(df: pd.DataFrame, code: str) -> pd.DataFrame:
    local_vars = {'df': df.copy(), 'pd': pd, 'np': np}
    try:
        exec(code, {}, local_vars)
        return local_vars['df']
    except Exception as e:
        raise ValueError(f"custom code failed: {e}")

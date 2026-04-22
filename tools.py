import pandas as pd
import numpy as np
import io
from typing import Optional, List, Dict, Any, Union

# optimized enterprise-grade tools for high token efficiency

def get_data_summary(df: pd.DataFrame) -> str:
    """returns a compact, token-efficient summary of the data."""
    summary = f"### data snapshot: {df.shape[0]} rows, {df.shape[1]} columns\n"
    summary += "columns (nulls | type):\n"
    for col in df.columns:
        null_count = df[col].isnull().sum()
        dtype = str(df[col].dtype)
        # identify mixed types simply
        types = df[col].apply(type).unique()
        type_info = f"{dtype}"
        if len(types) > 1:
            type_info += f" (mixed: {[t.__name__ for t in types]})"
        summary += f"- {col}: {null_count} | {type_info}\n"
    
    # use csv format for sample data to save tokens vs markdown
    summary += f"\nsample (first 3 rows):\n{df.head(3).to_csv(index=False)}"
    return summary

def audit_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """performs a concise audit focusing only on remaining issues."""
    nulls = df.isnull().sum()
    issues = {
        "remaining_nulls": nulls[nulls > 0].to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "total_rows": len(df)
    }
    return issues

def smart_handle_missing(df: pd.DataFrame, strategy: str, columns: Optional[List[str]] = None, fill_value: Any = None) -> pd.DataFrame:
    df_copy = df.copy()
    cols = columns if columns else df_copy.columns
    for col in cols:
        if col not in df_copy.columns: continue
        if strategy == 'drop':
            df_copy = df_copy.dropna(subset=[col])
        elif strategy == 'constant':
            df_copy[col] = df_copy[col].fillna(fill_value)
        else:
            if strategy in ['mean', 'median']:
                temp = pd.to_numeric(df_copy[col], errors='coerce')
                val = temp.mean() if strategy == 'mean' else temp.median()
                df_copy[col] = df_copy[col].fillna(val)
            elif strategy == 'mode':
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
                    df_copy[col] = df_copy[col].fillna(0).astype(int)
                else:
                    df_copy[col] = df_copy[col].astype(float)
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

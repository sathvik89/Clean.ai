import pandas as pd
import pytest

from cleanai.agents.tools.sandbox import UnsafeCodeError, run_snippet, validate_code


@pytest.fixture
def df():
    return pd.DataFrame({"a": [1, 2, 3]})


@pytest.mark.parametrize("code", [
    "import os",
    "from pathlib import Path",
    "open('/etc/passwd')",
    "__import__('os')",
    "df.__class__",
    "getattr(df, 'to_csv')",
    "df.to_csv('out.csv')",
    "eval('1+1')",
    "globals()",
])
def test_dangerous_code_is_rejected(code):
    with pytest.raises(UnsafeCodeError):
        validate_code(code)


def test_valid_snippet_runs(df):
    result = run_snippet(df, "df = df[df['a'] > 1]")
    assert len(result) == 2


def test_original_df_not_mutated(df):
    run_snippet(df, "df['a'] = 0")
    assert df["a"].tolist() == [1, 2, 3]


def test_snippet_must_keep_df(df):
    with pytest.raises(ValueError, match="DataFrame named 'df'"):
        run_snippet(df, "df = 42")


def test_runtime_errors_are_wrapped(df):
    with pytest.raises(ValueError, match="failed while running"):
        run_snippet(df, "df = df['missing_column_xyz']")


def test_safe_builtins_available(df):
    result = run_snippet(df, "df = df.head(len(df) - 1)")
    assert len(result) == 2

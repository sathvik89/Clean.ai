"""A guarded executor for LLM-written pandas snippets.

The old version ran `exec()` on whatever the model produced. This one
walks the AST first and rejects anything outside a small, safe subset:
no imports, no dunder access, no file/network/system calls, no exec
tricks. The snippet only ever sees `df`, `pd` and `np`.
"""

from __future__ import annotations

import ast
from typing import Set

import numpy as np
import pandas as pd

BLOCKED_CALLS: Set[str] = {
    "open", "eval", "exec", "compile", "__import__", "input", "breakpoint",
    "getattr", "setattr", "delattr", "globals", "locals", "vars", "dir",
    "memoryview", "exit", "quit", "help",
}

BLOCKED_ATTRIBUTES: Set[str] = {
    "to_pickle", "to_sql", "to_hdf", "to_parquet", "to_feather",
    "to_csv", "to_excel", "to_json", "to_clipboard", "read_pickle",
    "read_sql", "read_csv", "read_json", "read_html", "eval", "query",
}

SAFE_BUILTINS = {
    "len": len, "range": range, "min": min, "max": max, "sum": sum,
    "abs": abs, "round": round, "sorted": sorted, "enumerate": enumerate,
    "zip": zip, "list": list, "dict": dict, "set": set, "tuple": tuple,
    "str": str, "int": int, "float": float, "bool": bool, "any": any,
    "all": all, "isinstance": isinstance, "print": print, "map": map,
    "filter": filter, "reversed": reversed,
}


class UnsafeCodeError(ValueError):
    pass


def validate_code(code: str) -> None:
    """Raise UnsafeCodeError if the snippet steps outside the safe subset."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise UnsafeCodeError(f"snippet has a syntax error: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeCodeError("imports are not allowed in snippets")
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise UnsafeCodeError("global/nonlocal statements are not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise UnsafeCodeError(f"access to '{node.id}' is not allowed")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise UnsafeCodeError(f"dunder attribute '{node.attr}' is not allowed")
            if node.attr in BLOCKED_ATTRIBUTES:
                raise UnsafeCodeError(f"method '{node.attr}' is not allowed in snippets")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in BLOCKED_CALLS:
                raise UnsafeCodeError(f"call to '{func.id}' is not allowed")


def run_snippet(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """Validate and run a snippet; it must leave its result in `df`."""
    validate_code(code)
    scope = {"df": df.copy(), "pd": pd, "np": np}
    try:
        exec(compile(code, "<snippet>", "exec"), {"__builtins__": SAFE_BUILTINS}, scope)
    except UnsafeCodeError:
        raise
    except Exception as exc:
        raise ValueError(f"snippet failed while running: {exc}") from exc

    result = scope.get("df")
    if not isinstance(result, pd.DataFrame):
        raise ValueError("snippet must keep the result in a DataFrame named 'df'")
    return result

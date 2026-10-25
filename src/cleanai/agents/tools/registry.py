"""Single source of truth for agent tools.

Every tool is declared once: name, description, argument schema and
handler live together. The registry produces the LangChain tool objects
the LLM binds to, and also executes the calls, so the schema the model
sees can never drift from the code that runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type

import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError

from cleanai.agents.tools import cleaning, sandbox
from cleanai.core import profiling


@dataclass
class ToolResult:
    message: str
    df: Optional[pd.DataFrame] = None   # None means the tool was read-only
    status: str = "ok"                  # ok | error | noop | blocked

    @property
    def mutated(self) -> bool:
        return self.df is not None


@dataclass
class ToolSpec:
    name: str
    description: str
    args_model: Type[BaseModel]
    handler: Callable[..., ToolResult]
    mutates: bool


REGISTRY: Dict[str, ToolSpec] = {}


def register(name: str, description: str, args_model: Type[BaseModel], mutates: bool):
    def decorator(handler: Callable[..., ToolResult]):
        REGISTRY[name] = ToolSpec(name, description, args_model, handler, mutates)
        return handler
    return decorator


# ---------------------------------------------------------------------------
# argument schemas
# ---------------------------------------------------------------------------

class NoArgs(BaseModel):
    pass


class HandleMissingArgs(BaseModel):
    strategy: str = Field(description="one of: drop, constant, mean, median, mode, interpolate")
    columns: Optional[List[str]] = Field(default=None, description="target columns; omit for all columns with nulls")
    fill_value: Optional[Any] = Field(default=None, description="value to use with the constant strategy")


class RemoveDuplicatesArgs(BaseModel):
    columns: Optional[List[str]] = Field(default=None, description="key columns that define a duplicate; omit for full-row duplicates")


class ConvertTypesArgs(BaseModel):
    column_type_map: Dict[str, str] = Field(description="column name -> target type (int, float, datetime, string, category, bool)")


class CleanTextArgs(BaseModel):
    columns: List[str] = Field(description="text columns to clean")
    actions: List[str] = Field(description="any of: lowercase, uppercase, titlecase, strip, collapse_spaces, remove_special")


class EncodeCategoricalArgs(BaseModel):
    columns: List[str] = Field(description="categorical columns to encode")
    method: str = Field(default="label", description="label or onehot")


class ColumnsArgs(BaseModel):
    columns: List[str] = Field(description="columns to inspect")


class FixOutliersArgs(BaseModel):
    columns: List[str] = Field(description="numeric columns to treat")
    method: str = Field(default="clip", description="clip (cap at IQR fences) or nullify (set outliers to missing)")


class RenameColumnsArgs(BaseModel):
    rename_map: Dict[str, str] = Field(description="old column name -> new column name")


class DropColumnsArgs(BaseModel):
    columns: List[str] = Field(description="columns to drop entirely")


class RunPandasArgs(BaseModel):
    code: str = Field(description="short pandas snippet; operate on the DataFrame named df and keep the result in df. imports and file/network access are rejected")


class ReportBlockedArgs(BaseModel):
    reason: str = Field(description="why the current step cannot be completed")


# ---------------------------------------------------------------------------
# tool handlers
# ---------------------------------------------------------------------------

@register("get_summary", "Get a fresh summary of the current data state: shape, per-column issues, duplicates.", NoArgs, mutates=False)
def _get_summary(df: pd.DataFrame) -> ToolResult:
    return ToolResult(message=profiling.summarize_for_llm(df))


@register("normalize_nulls", "Convert disguised null text like 'N/A', '-', 'null' into real missing values. Run this before filling missing values.", RemoveDuplicatesArgs, mutates=True)
def _normalize_nulls(df: pd.DataFrame, columns=None) -> ToolResult:
    result = cleaning.normalize_nulls(df, columns)
    return ToolResult(message="disguised nulls converted to real missing values", df=result)


@register("handle_missing", "Fill or drop missing values in one or more columns.", HandleMissingArgs, mutates=True)
def _handle_missing(df: pd.DataFrame, strategy: str, columns=None, fill_value=None) -> ToolResult:
    result = cleaning.handle_missing(df, strategy, columns, fill_value)
    return ToolResult(message="missing values handled", df=result)


@register("remove_duplicates", "Remove duplicate rows, optionally keyed on specific columns like ids or emails.", RemoveDuplicatesArgs, mutates=True)
def _remove_duplicates(df: pd.DataFrame, columns=None) -> ToolResult:
    result = cleaning.remove_duplicates(df, columns)
    removed = len(df) - len(result)
    return ToolResult(message=f"removed {removed} duplicate rows", df=result)


@register("convert_types", "Convert columns to their proper types; unparseable values become missing.", ConvertTypesArgs, mutates=True)
def _convert_types(df: pd.DataFrame, column_type_map: Dict[str, str]) -> ToolResult:
    result = cleaning.convert_types(df, column_type_map)
    return ToolResult(message="type conversion applied", df=result)


@register("clean_text", "Standardize text columns: casing, whitespace, special characters.", CleanTextArgs, mutates=True)
def _clean_text(df: pd.DataFrame, columns: List[str], actions: List[str]) -> ToolResult:
    result = cleaning.clean_text(df, columns, actions)
    return ToolResult(message=f"text cleaned on {columns}", df=result)


@register("encode_categorical", "Encode categorical columns with label or onehot encoding (for ML-ready output).", EncodeCategoricalArgs, mutates=True)
def _encode_categorical(df: pd.DataFrame, columns: List[str], method: str = "label") -> ToolResult:
    result = cleaning.encode_categorical(df, columns, method)
    return ToolResult(message=f"encoded {columns} using {method}", df=result)


@register("detect_outliers", "Report IQR outlier counts for numeric columns without changing anything.", ColumnsArgs, mutates=False)
def _detect_outliers(df: pd.DataFrame, columns: List[str]) -> ToolResult:
    return ToolResult(message=cleaning.outlier_report(df, columns))


@register("fix_outliers", "Treat numeric outliers by clipping to IQR fences or nullifying them.", FixOutliersArgs, mutates=True)
def _fix_outliers(df: pd.DataFrame, columns: List[str], method: str = "clip") -> ToolResult:
    result = cleaning.clip_outliers(df, columns, method)
    return ToolResult(message=f"outliers treated on {columns} via {method}", df=result)


@register("rename_columns", "Rename columns, e.g. to fix inconsistent naming.", RenameColumnsArgs, mutates=True)
def _rename_columns(df: pd.DataFrame, rename_map: Dict[str, str]) -> ToolResult:
    result = cleaning.rename_columns(df, rename_map)
    return ToolResult(message=f"renamed {list(rename_map.keys())}", df=result)


@register("drop_columns", "Drop columns that are useless or beyond repair.", DropColumnsArgs, mutates=True)
def _drop_columns(df: pd.DataFrame, columns: List[str]) -> ToolResult:
    result = cleaning.drop_columns(df, columns)
    return ToolResult(message=f"dropped {columns}", df=result)


@register("run_pandas", "Run a short custom pandas snippet for logic no standard tool covers. Use sparingly.", RunPandasArgs, mutates=True)
def _run_pandas(df: pd.DataFrame, code: str) -> ToolResult:
    result = sandbox.run_snippet(df, code)
    return ToolResult(message="custom snippet applied", df=result)


@register("report_blocked", "Declare that the current step cannot be completed and why; moves the run forward.", ReportBlockedArgs, mutates=False)
def _report_blocked(df: pd.DataFrame, reason: str) -> ToolResult:
    return ToolResult(message=f"step marked blocked: {reason}", status="blocked")


# ---------------------------------------------------------------------------
# registry API
# ---------------------------------------------------------------------------

def langchain_tools() -> List[StructuredTool]:
    tools = []
    for spec in REGISTRY.values():
        tools.append(
            StructuredTool.from_function(
                func=lambda **_: None,   # dispatch happens in execute(), not here
                name=spec.name,
                description=spec.description,
                args_schema=spec.args_model,
            )
        )
    return tools


def tool_signatures() -> str:
    """Compact tool list for prompts."""
    lines = []
    for spec in REGISTRY.values():
        fields = ", ".join(spec.args_model.model_fields.keys()) or "no args"
        lines.append(f"- {spec.name}({fields}): {spec.description}")
    return "\n".join(lines)


def normalize_tool_name(name: str) -> str:
    cleaned = name.strip().lower()
    if cleaned.startswith("tool_"):
        cleaned = cleaned[5:]
    aliases = {
        "tool_get_summary": "get_summary",
        "handle_categorical": "encode_categorical",
        "perform_text_cleaning": "clean_text",
        "execute_custom_pandas": "run_pandas",
        "type_conversion": "convert_types",
    }
    return aliases.get(cleaned, cleaned)


def execute(name: str, df: pd.DataFrame, args: Dict[str, Any]) -> ToolResult:
    """Validate and run one tool call; never raises."""
    tool_name = normalize_tool_name(name)
    spec = REGISTRY.get(tool_name)
    if spec is None:
        valid = ", ".join(REGISTRY.keys())
        return ToolResult(
            message=f"error: unknown tool '{name}'. valid tools are: {valid}",
            status="error",
        )
    try:
        validated = spec.args_model(**(args or {}))
    except ValidationError as exc:
        first_error = exc.errors()[0]
        return ToolResult(
            message=f"error: bad arguments for {tool_name}: {first_error['loc']}: {first_error['msg']}",
            status="error",
        )
    try:
        return spec.handler(df, **validated.model_dump())
    except Exception as exc:
        return ToolResult(message=f"error: {tool_name} failed: {exc}", status="error")

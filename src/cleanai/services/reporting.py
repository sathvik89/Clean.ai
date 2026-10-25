"""Markdown run reports for download and audit trails."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict


def build_run_report(summary: Dict[str, Any]) -> str:
    lift = (summary.get("score_after") or 0) - (summary.get("score_before") or 0)
    lines = [
        f"# Clean.ai run report - {summary['dataset_name']}",
        "",
        f"- generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- goal: {summary.get('user_context', 'general cleaning')}",
        f"- status: **{summary['status']}**" + (f" ({summary['error']})" if summary.get("error") else ""),
        "",
        "## Outcome",
        "",
        "| metric | before | after |",
        "|---|---|---|",
        f"| quality score | {summary.get('score_before')} | {summary.get('score_after')} ({lift:+d}) |",
        f"| rows | {summary.get('rows_before')} | {summary.get('rows_after')} |",
        f"| columns | {summary.get('cols_before')} | {summary.get('cols_after')} |",
        f"| missing cells | {summary.get('nulls_before')} | {summary.get('nulls_after')} |",
        f"| duplicate rows | {summary.get('duplicates_before')} | {summary.get('duplicates_after')} |",
        "",
        "## Plan",
        "",
    ]
    for step in summary.get("steps", []):
        status = step.get("status", "pending")
        result = f" - {step.get('result')}" if step.get("result") else ""
        lines.append(f"- `{step['id']}` **{step['tool']}** on {step.get('columns') or 'table'} [{status}]{result}")
    lines += ["", "## Actions taken", ""]
    for record in summary.get("ledger", []):
        lines.append(f"- #{record['seq']} `{record['tool']}` -> {record['status']}"
                     + (f" | {record['delta']}" if record.get("delta") else "")
                     + (f" | {record['error']}" if record.get("error") else ""))
    lines += ["", "## Analysis report", "", summary.get("analysis_report", "")]
    return "\n".join(lines)

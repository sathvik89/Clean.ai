"""Live end-to-end smoke test against the real Groq API.

Runs a full cleaning session on a deliberately messy dataset, auto-approves
the plan, and prints the outcome. Needs GROQ_API_KEY in .env.

Usage: python scripts/smoke_test.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cleanai.services.pipeline import CleaningSession, simulate_plan
from cleanai.services.recipe import build_recipe_script


def make_messy_dataset() -> pd.DataFrame:
    return pd.DataFrame({
        "employee_id": [101, 102, 102, 104, 105, 106, 107, 108],
        "name": ["  Ravi Kumar", "ANITA desai", "ANITA desai", "joe  ", "Meera N", "sunil", "Priya S", "amit"],
        "age": ["29", "34", "34", "n/a", "151", "27", None, "41"],
        "salary": ["55000", "72000", "72000", "61000", "-5000", "48000", "66000", "not disclosed"],
        "email": ["ravi@corp.com", "anita@corp.com", "anita@corp.com", "joe[at]corp.com",
                  "meera@corp.com", "sunil@corp.com", "priya@corp.com", "amit@corp.com"],
        "joined": ["2021-04-01", "15/06/2020", "15/06/2020", "2022-01-10",
                   "2030-01-01", "2019-11-23", "2021-08-30", "2020-02-14"],
    })


def main() -> int:
    df = make_messy_dataset()
    print(f"input: {len(df)} rows, issues seeded on purpose\n")

    session = CleaningSession(df, "smoke_test.csv", "ML modeling on employee data")
    session.run_until_review(on_event=lambda node, out: print(f"  [{node}] done"))

    plan = session.get_plan()
    print(f"\nplan ({len(plan)} steps):")
    for step in plan:
        print(f"  {step['id']}: {step['tool']} on {step['columns'] or 'table'} - {step['rationale']}")

    print("\ndry-run impact:")
    for impact in simulate_plan(df, plan):
        print(f"  {impact['id']} {impact['tool']}: {impact['effect']}")

    print("\nexecuting...")
    session.resume(on_event=lambda node, out: print(f"  [{node}] done"))

    summary = session.result_summary()
    print(f"\nstatus: {summary['status']}")
    print(f"quality: {summary['score_before']} -> {summary['score_after']}")
    print(f"rows: {summary['rows_before']} -> {summary['rows_after']}")
    print(f"nulls: {summary['nulls_before']} -> {summary['nulls_after']}")
    print(f"duplicates: {summary['duplicates_before']} -> {summary['duplicates_after']}")
    print(f"tokens used: {summary['tokens_used']}")
    if summary["error"]:
        print(f"stop reason: {summary['error']}")

    session.learn_from_run()
    print("\ncase memory updated")

    recipe = build_recipe_script(summary)
    compile(recipe, "recipe.py", "exec")
    print(f"recipe script generated and compiles ({len(recipe.splitlines())} lines)")

    ok = summary["score_after"] >= summary["score_before"] and summary["status"] in ("clean", "partial", "stopped")
    print("\nSMOKE TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

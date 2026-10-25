"""Clean.ai command line runner.

Usage: python main.py path/to/data.csv --goal "ML modeling"
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cleanai.services.pipeline import CleaningSession, simulate_plan
from cleanai.services.recipe import build_recipe_script
from cleanai.services.reporting import build_run_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean a messy CSV with the Clean.ai agent")
    parser.add_argument("csv_path", help="path to the input CSV")
    parser.add_argument("--goal", default="general cleaning and formatting",
                        help="what the data will be used for")
    parser.add_argument("--yes", action="store_true", help="auto-approve the plan")
    parser.add_argument("--out", default="cleaned_data.csv", help="output CSV path")
    args = parser.parse_args()

    path = Path(args.csv_path)
    if not path.exists():
        print(f"error: file not found: {path}")
        return 1
    df = pd.read_csv(path)
    print(f"loaded {len(df)} rows x {len(df.columns)} columns from {path.name}\n")

    session = CleaningSession(df, path.name, args.goal)
    session.run_until_review()

    plan = session.get_plan()
    print("proposed plan:")
    for step in plan:
        print(f"  {step['id']}: {step['tool']} on {step['columns'] or 'table'} - {step['rationale']}")
    print("\ndry-run impact:")
    for impact in simulate_plan(df, plan):
        print(f"  {impact['id']} {impact['tool']}: {impact['effect']}")

    if not args.yes:
        answer = input("\napprove this plan? [y/N] ").strip().lower()
        if answer != "y":
            print("aborted.")
            return 0

    print("\nexecuting...")
    session.resume()
    summary = session.result_summary()
    session.learn_from_run()

    Path(args.out).write_text(summary["cleaned_csv"])
    Path("run_report.md").write_text(build_run_report(summary))
    Path("clean_recipe.py").write_text(build_recipe_script(summary))

    print(f"\nstatus: {summary['status']}")
    print(f"quality: {summary['score_before']} -> {summary['score_after']}")
    print(f"saved: {args.out}, run_report.md, clean_recipe.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

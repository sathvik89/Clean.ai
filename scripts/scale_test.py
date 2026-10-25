"""Prove dataset size doesn't inflate LLM cost: the model only ever sees
compact profiles, so tokens stay flat as rows grow.

Usage: python scripts/scale_test.py [rows]
"""

import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cleanai.services.pipeline import CleaningSession


def make_big_messy(rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    random.seed(11)
    ages = rng.integers(18, 65, rows).astype(object)
    bad = rng.choice(rows, rows // 20, replace=False)
    for i in bad[: len(bad) // 2]:
        ages[i] = "n/a"
    for i in bad[len(bad) // 2:]:
        ages[i] = int(rng.integers(130, 220))
    salaries = rng.integers(20000, 150000, rows).astype(object)
    for i in rng.choice(rows, rows // 30, replace=False):
        salaries[i] = -int(rng.integers(100, 5000))
    df = pd.DataFrame({
        "user_id": np.arange(1, rows + 1),
        "age": ages,
        "salary": salaries,
        "city": rng.choice([" Pune", "pune", "Delhi", "delhi ", "Mumbai", "N/A", "-"], rows),
        "score": rng.normal(50, 12, rows).round(2),
        "signup": rng.choice(["2023-01-15", "03/02/2023", "2024-06-09", None], rows),
    })
    dupes = df.sample(rows // 50, random_state=1)
    return pd.concat([df, dupes], ignore_index=True)


def main() -> int:
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 150_000
    df = make_big_messy(rows)
    print(f"dataset: {len(df):,} rows x {len(df.columns)} cols "
          f"({df.memory_usage(deep=True).sum() / 1e6:.0f} MB in memory)")

    started = time.time()
    session = CleaningSession(df, "scale_test.csv", "ML modeling")
    session.run_until_review()
    session.resume()
    summary = session.result_summary()

    print(f"status: {summary['status']}")
    print(f"quality: {summary['score_before']} -> {summary['score_after']}")
    print(f"rows: {summary['rows_before']:,} -> {summary['rows_after']:,}")
    print(f"nulls: {summary['nulls_before']:,} -> {summary['nulls_after']:,}")
    print(f"LLM tokens: {summary['tokens_used']:,} (same order as an 8-row run)")
    print(f"wall time: {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

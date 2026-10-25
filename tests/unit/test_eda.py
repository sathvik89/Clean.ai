import numpy as np
import pandas as pd

from cleanai.core import eda


def make_df(rows=200):
    rng = np.random.default_rng(5)
    return pd.DataFrame({
        "a": rng.normal(10, 2, rows),
        "b": rng.normal(10, 2, rows) * 2 + 1,       # correlated with a? no - independent draw
        "city": rng.choice(["Pune", "Delhi", None], rows),
        "joined": rng.choice(["2023-01-02", "2024-05-06"], rows),
    })


def test_bundle_has_all_sections():
    df = make_df()
    df.loc[0:9, "a"] = None
    bundle = eda.compute_bundle(df)
    assert bundle["rows"] == 200
    assert not bundle["sampled"]
    assert any(m["column"] == "a" for m in bundle["missing"])
    assert any(n["column"] == "a" for n in bundle["numeric"])
    hist = bundle["numeric"][0]["hist"]
    assert len(hist["edges"]) == len(hist["counts"]) + 1
    assert any(c["column"] == "city" for c in bundle["categorical"])
    assert any(d["column"] == "joined" for d in bundle["datetime"])


def test_correlations_detected():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 500)
    df = pd.DataFrame({"x": base, "y": base * 3 + rng.normal(0, 0.1, 500), "z": rng.normal(0, 1, 500)})
    bundle = eda.compute_bundle(df)
    assert bundle["corr_matrix"] is not None
    assert any({p["a"], p["b"]} == {"x", "y"} and p["r"] > 0.9 for p in bundle["correlations"])


def test_large_frames_are_sampled():
    df = pd.DataFrame({"v": np.arange(70_000, dtype=float)})
    df.loc[:99, "v"] = None
    bundle = eda.compute_bundle(df)
    assert bundle["sampled"]
    assert bundle["sample_rows"] == eda.SAMPLE_SIZE
    # missing counts stay exact even when sampled
    assert bundle["missing"][0]["count"] == 100


def test_render_for_llm_is_compact():
    text = eda.render_for_llm(eda.compute_bundle(make_df()))
    assert "shape: 200 rows" in text
    assert len(text) < 4000

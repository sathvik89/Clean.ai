import pandas as pd

from cleanai.agents.tools import cleaning


def test_median_fill_coerces_text_numbers():
    df = pd.DataFrame({"age": ["25", "35", None]})
    out = cleaning.handle_missing(df, "median", ["age"])
    assert out["age"].isnull().sum() == 0
    assert out["age"].tolist() == [25.0, 35.0, 30.0]


def test_strategy_aliases_resolve():
    series = pd.Series([1.0, 2.0])
    assert cleaning.resolve_missing_strategy(series, "fill_with_median") == "median"
    assert cleaning.resolve_missing_strategy(series, "most_frequent") == "mode"
    assert cleaning.resolve_missing_strategy(series, "auto") == "median"


def test_auto_strategy_prefers_mode_for_text():
    series = pd.Series(["a", "b", "a"])
    assert cleaning.resolve_missing_strategy(series, "") == "mode"


def test_constant_fill_defaults_by_type():
    df = pd.DataFrame({"n": pd.array([1, None], dtype="Int64"), "t": ["x", None]})
    out = cleaning.handle_missing(df, "constant")
    assert out["n"].tolist() == [1, 0]
    assert out["t"].tolist() == ["x", "Unknown"]


def test_normalize_nulls_only_touches_tokens():
    df = pd.DataFrame({"c": ["ok", "N/A", " - ", "none", "real value"]})
    out = cleaning.normalize_nulls(df)
    assert out["c"].isnull().sum() == 3
    assert out["c"].iloc[0] == "ok"


def test_convert_types_coerces_bad_values_to_nan():
    df = pd.DataFrame({"age": ["25", "unknown", "31"]})
    out = cleaning.convert_types(df, {"age": "int"})
    assert str(out["age"].dtype) == "Int64"
    assert out["age"].isnull().sum() == 1


def test_duplicates_removed_on_key():
    df = pd.DataFrame({"email": ["a@x.com", "a@x.com"], "v": [1, 2]})
    out = cleaning.remove_duplicates(df, ["email"])
    assert len(out) == 1


def test_clean_text_preserves_nulls():
    df = pd.DataFrame({"name": ["  Bob ", None]})
    out = cleaning.clean_text(df, ["name"], ["strip", "lowercase"])
    assert out["name"].iloc[0] == "bob"
    assert pd.isna(out["name"].iloc[1])


def test_clip_outliers_nullify():
    df = pd.DataFrame({"v": [1, 2, 3, 2, 1000]})
    out = cleaning.clip_outliers(df, ["v"], method="nullify")
    assert out["v"].isnull().sum() == 1


def test_missing_columns_are_ignored():
    df = pd.DataFrame({"a": [1]})
    out = cleaning.drop_columns(df, ["ghost"])
    assert list(out.columns) == ["a"]

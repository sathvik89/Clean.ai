import pandas as pd

from cleanai.core import profiling


def test_cards_detect_core_issues(messy_df):
    cards = {c.name: c for c in profiling.build_cards(messy_df)}
    age_kinds = {i.kind for i in cards["age"].issues_open}
    assert "missing_values" in age_kinds
    assert "out_of_range" in age_kinds
    email_kinds = {i.kind for i in cards["email"].issues_open}
    assert "invalid_email" in email_kinds
    assert cards["email"].is_key


def test_disguised_nulls_detected():
    df = pd.DataFrame({"city": ["Delhi", "N/A", "-", "Pune"]})
    cards = profiling.build_cards(df)
    kinds = {i.kind for i in cards[0].issues_open}
    assert "disguised_nulls" in kinds


def test_pii_samples_are_masked(messy_df):
    cards = {c.name: c for c in profiling.build_cards(messy_df)}
    assert all("@" not in s or "***" in s for s in cards["email"].sample_values)


def test_quality_score_improves_when_cleaned(messy_df):
    dirty_score = profiling.quality_score(messy_df)
    clean_df = pd.DataFrame({"id": [1, 2, 3], "age": [25.0, 30.0, 31.0]})
    assert profiling.quality_score(clean_df) > dirty_score


def test_dimension_scores_bounded(messy_df):
    scores = profiling.dimension_scores(messy_df)
    assert set(scores) == {"completeness", "validity", "uniqueness", "consistency"}
    assert all(0 <= v <= 100 for v in scores.values())


def test_metrics_delta_tracks_changes(messy_df):
    before = profiling.snapshot(messy_df)
    after_df = messy_df.drop_duplicates()
    delta = profiling.metrics_delta(before, profiling.snapshot(after_df))
    assert "rows" in delta


def test_signature_changes_with_data(messy_df):
    sig_before = profiling.df_signature(messy_df)
    sig_after = profiling.df_signature(messy_df.drop_duplicates())
    assert sig_before != sig_after
    assert sig_before == profiling.df_signature(messy_df.copy())


def test_refresh_cards_keeps_fix_history(messy_df):
    cards = profiling.build_cards(messy_df)
    cards[0].fixes_applied.append("#1 convert_types")
    refreshed = profiling.refresh_cards(messy_df, cards)
    assert refreshed[0].fixes_applied == ["#1 convert_types"]

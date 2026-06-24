import pandas as pd

from src.features import build_escalation_target, create_conflict_lag_features, create_leakage_safe_external_lags


def test_escalation_target_uses_next_year_within_country():
    df = pd.DataFrame(
        {
            "country": ["A", "A", "A", "B", "B"],
            "year": [2000, 2001, 2002, 2000, 2001],
            "intensity_level": [0, 2, 1, 1, 1],
        }
    )
    out = build_escalation_target(df)
    assert out.loc[(out["country"] == "A") & (out["year"] == 2000), "escalation_target"].item() == 1
    assert out.loc[(out["country"] == "A") & (out["year"] == 2001), "escalation_target"].item() == 0
    assert pd.isna(out.loc[(out["country"] == "B") & (out["year"] == 2001), "escalation_target"].item())


def test_lag_features_are_country_specific():
    df = pd.DataFrame({"country": ["A", "A", "B", "B"], "year": [2000, 2001, 2000, 2001], "intensity_level": [1, 3, 2, 4]})
    out = create_conflict_lag_features(df, columns=["intensity_level"], lags=(1,))
    assert out.loc[(out["country"] == "B") & (out["year"] == 2001), "intensity_level_lag1"].item() == 2


def test_external_lags_drop_same_year_by_default():
    df = pd.DataFrame({"country": ["A", "A"], "year": [2000, 2001], "gdp": [10, 20]})
    out = create_leakage_safe_external_lags(df, ["gdp"], lags=(1,))
    assert "gdp" not in out.columns
    assert out.loc[out["year"] == 2001, "gdp_lag1"].item() == 10

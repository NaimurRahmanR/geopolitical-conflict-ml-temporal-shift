"""Feature engineering with explicit temporal leakage safeguards."""

import pandas as pd


def _sort_panel(df, country_col="country", year_col="year"):
    return df.sort_values([country_col, year_col]).reset_index(drop=True)


def build_escalation_target(
    df,
    intensity_col="intensity_level",
    country_col="country",
    year_col="year",
    target_col="escalation_target",
):
    """Mark cases where next-year conflict intensity exceeds current intensity."""
    out = _sort_panel(df, country_col, year_col).copy()
    out["_next_intensity"] = out.groupby(country_col)[intensity_col].shift(-1)
    out[target_col] = (out["_next_intensity"] > out[intensity_col]).astype(int)
    out.loc[out["_next_intensity"].isna(), target_col] = pd.NA
    return out.drop(columns=["_next_intensity"])


def create_conflict_lag_features(
    df,
    columns=None,
    lags=(1, 2, 3),
    country_col="country",
    year_col="year",
):
    """Create country-level lagged conflict-history variables."""
    out = _sort_panel(df, country_col, year_col).copy()
    columns = columns or [c for c in ["intensity_level", "deaths", "battle_deaths"] if c in out.columns]
    for col in columns:
        for lag in lags:
            out[f"{col}_lag{lag}"] = out.groupby(country_col)[col].shift(lag)
    return out


def create_leakage_safe_external_lags(
    df,
    columns,
    lags=(1,),
    country_col="country",
    year_col="year",
    drop_current=True,
):
    """Lag external indicators so same-year information cannot enter forecasts."""
    out = _sort_panel(df, country_col, year_col).copy()
    for col in columns:
        for lag in lags:
            out[f"{col}_lag{lag}"] = out.groupby(country_col)[col].shift(lag)
    if drop_current:
        out = out.drop(columns=[c for c in columns if c in out.columns])
    return out


def create_regional_lag_features(
    df,
    value_col="intensity_level",
    region_col="region",
    country_col="country",
    year_col="year",
):
    """Create prior-year regional conflict context, excluding same-year leakage."""
    out = _sort_panel(df, country_col, year_col).copy()
    region_year = (
        out.groupby([region_col, year_col])[value_col]
        .mean()
        .rename(f"regional_{value_col}_mean")
        .reset_index()
        .sort_values([region_col, year_col])
    )
    region_year[f"regional_{value_col}_mean_lag1"] = region_year.groupby(region_col)[
        f"regional_{value_col}_mean"
    ].shift(1)
    out = out.merge(region_year[[region_col, year_col, f"regional_{value_col}_mean_lag1"]], on=[region_col, year_col], how="left")
    return out


def create_country_lag_features(df, columns, lags=(1,), country_col="country", year_col="year"):
    """General-purpose country-year lag builder."""
    out = _sort_panel(df, country_col, year_col).copy()
    for col in columns:
        for lag in lags:
            out[f"{col}_country_lag{lag}"] = out.groupby(country_col)[col].shift(lag)
    return out

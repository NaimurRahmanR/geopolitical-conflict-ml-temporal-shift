"""Dataset cleaning and leakage-aware merging utilities."""

import re

import pandas as pd

COUNTRY_ALIASES = {
    "Cote d'Ivoire": "Ivory Coast",
    "Congo, Dem. Rep.": "Democratic Republic of the Congo",
    "Congo, Rep.": "Republic of the Congo",
    "Russian Federation": "Russia",
    "Syrian Arab Republic": "Syria",
    "Viet Nam": "Vietnam",
    "Yemen, Rep.": "Yemen",
}


def _normalise_column_name(name):
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower())).strip("_")


def _standardise_country_year(df, country_candidates=None, year_candidates=None):
    df = df.copy()
    df.columns = [_normalise_column_name(c) for c in df.columns]
    country_candidates = country_candidates or ["country", "country_name", "location", "state", "name"]
    year_candidates = year_candidates or ["year", "time"]
    rename = {}
    for col in country_candidates:
        if col in df.columns:
            rename[col] = "country"
            break
    for col in year_candidates:
        if col in df.columns:
            rename[col] = "year"
            break
    df = df.rename(columns=rename)
    if "country" in df.columns:
        df["country"] = harmonise_country_names(df["country"])
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    return df


def clean_world_bank_wide_to_long(df, indicators=None):
    """Reshape WDI wide year columns to country-year indicator columns.

    Expected raw files often contain rows like country, indicator, 1960, 1961, ...
    The returned frame has one row per country-year and one column per indicator.
    """
    df = df.copy()
    df.columns = [_normalise_column_name(c) for c in df.columns]
    country_col = next((c for c in ["country_name", "country"] if c in df.columns), None)
    indicator_col = next((c for c in ["indicator_code", "series_code", "indicator_name"] if c in df.columns), None)
    if country_col is None or indicator_col is None:
        raise ValueError("World Bank data must contain country and indicator columns.")
    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]
    if not year_cols:
        raise ValueError("World Bank data must contain wide year columns such as 2000.")
    keep = [country_col, indicator_col] + year_cols
    long = df[keep].melt(id_vars=[country_col, indicator_col], var_name="year", value_name="value")
    long["year"] = long["year"].astype(int)
    long["country"] = harmonise_country_names(long[country_col])
    long["indicator"] = long[indicator_col].astype(str)
    if indicators is not None:
        long = long[long["indicator"].isin(indicators)]
    wide = long.pivot_table(index=["country", "year"], columns="indicator", values="value", aggfunc="first")
    wide = wide.reset_index()
    wide.columns.name = None
    return wide


def select_vdem_features(df, features=None):
    """Return a country-year V-Dem frame with selected governance variables."""
    df = _standardise_country_year(df, ["country_name", "country_text_id", "country_name_text", "country"])
    default_features = [
        "v2x_polyarchy",
        "v2x_libdem",
        "v2x_corr",
        "v2x_rule",
        "v2x_cspart",
        "v2xcl_rol",
    ]
    requested = features or default_features
    cols = ["country", "year"] + [c for c in requested if c in df.columns]
    return df.loc[:, list(dict.fromkeys(cols))]


def harmonise_country_names(values):
    """Apply conservative country-name harmonisation."""
    series = pd.Series(values, copy=False).astype("string").str.strip()
    return series.replace(COUNTRY_ALIASES)


def merge_datasets(ucdp, world_bank=None, vdem=None, how="left"):
    """Merge country-year datasets after standardising keys."""
    merged = _standardise_country_year(ucdp)
    if world_bank is not None:
        wb = _standardise_country_year(world_bank)
        merged = merged.merge(wb, on=["country", "year"], how=how, suffixes=("", "_wb"))
    if vdem is not None:
        vd = _standardise_country_year(vdem)
        merged = merged.merge(vd, on=["country", "year"], how=how, suffixes=("", "_vdem"))
    return merged.sort_values(["country", "year"]).reset_index(drop=True)

"""Temporal distribution shift diagnostics."""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def compute_psi(expected, actual, bins=10, epsilon=1e-6):
    """Population Stability Index between reference and shifted samples."""
    expected = pd.Series(expected).dropna().astype(float)
    actual = pd.Series(actual).dropna().astype(float)
    if expected.empty or actual.empty:
        return np.nan
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)
    exp_pct = np.maximum(exp_counts / max(exp_counts.sum(), 1), epsilon)
    act_pct = np.maximum(act_counts / max(act_counts.sum(), 1), epsilon)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def compute_ks_drift(expected, actual):
    expected = pd.Series(expected).dropna().astype(float)
    actual = pd.Series(actual).dropna().astype(float)
    if expected.empty or actual.empty:
        return {"ks_statistic": np.nan, "ks_pvalue": np.nan}
    stat, pvalue = ks_2samp(expected, actual)
    return {"ks_statistic": float(stat), "ks_pvalue": float(pvalue)}


def run_temporal_drift_analysis(df, split_year, features):
    train = df[df["year"] < split_year]
    test = df[df["year"] >= split_year]
    rows = []
    for feature in features:
        ks = compute_ks_drift(train[feature], test[feature])
        rows.append(
            {
                "split_year": split_year,
                "feature": feature,
                "psi": compute_psi(train[feature], test[feature]),
                **ks,
            }
        )
    return pd.DataFrame(rows)

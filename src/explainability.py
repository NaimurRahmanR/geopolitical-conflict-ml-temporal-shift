"""Explanation stability summaries across temporal splits."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance


def coefficient_stability_across_splits(fitted_models, feature_names):
    rows = []
    for key, model in fitted_models.items():
        model_name, split_year = key
        estimator = model.named_steps.get("model", model) if hasattr(model, "named_steps") else model
        coef = getattr(estimator, "coef_", None)
        if coef is None:
            continue
        for feature, value in zip(feature_names, coef.ravel(), strict=False):
            rows.append({"model": model_name, "split_year": split_year, "feature": feature, "coefficient": value})
    return pd.DataFrame(rows)


def permutation_importance_analysis(model, x, y, feature_names=None, n_repeats=20, random_state=42):
    result = permutation_importance(model, x, y, n_repeats=n_repeats, random_state=random_state, scoring="average_precision")
    feature_names = feature_names or list(getattr(x, "columns", range(x.shape[1])))
    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)


def rank_correlation_stability(importance_frames, feature_col="feature", score_col="importance_mean"):
    rows = []
    keys = list(importance_frames)
    for i, left_key in enumerate(keys):
        for right_key in keys[i + 1 :]:
            left = importance_frames[left_key][[feature_col, score_col]].rename(columns={score_col: "left_score"})
            right = importance_frames[right_key][[feature_col, score_col]].rename(columns={score_col: "right_score"})
            merged = left.merge(right, on=feature_col, how="inner")
            corr = np.nan
            if len(merged) >= 2:
                corr = spearmanr(merged["left_score"].rank(), merged["right_score"].rank()).correlation
            rows.append({"left": left_key, "right": right_key, "spearman_rank_correlation": corr, "n_features": len(merged)})
    return pd.DataFrame(rows)

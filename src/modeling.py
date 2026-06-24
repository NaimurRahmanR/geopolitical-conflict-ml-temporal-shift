"""Temporal modeling experiments for conflict escalation forecasting."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import compute_classification_metrics, tune_threshold_by_f1


def get_feature_sets(df=None):
    """Define full and ablated feature families from available columns."""
    if df is None:
        return {
            "full": [],
            "conflict_history": [],
            "governance": [],
            "socioeconomic": [],
            "regional": [],
        }
    excluded = {"country", "year", "escalation_target", "target", "region"}
    numeric = [c for c in df.select_dtypes(include="number").columns if c not in excluded]
    return {
        "full": numeric,
        "conflict_history": [c for c in numeric if "intensity" in c or "death" in c or "conflict" in c],
        "governance": [c for c in numeric if c.startswith("v2") or "dem" in c or "rule" in c],
        "socioeconomic": [c for c in numeric if c not in {"year"} and any(k in c.lower() for k in ["gdp", "pop", "school", "life", "unemployment"])],
        "regional": [c for c in numeric if "regional" in c or "region" in c],
    }


def make_models(random_state=42):
    """Return reproducible sklearn pipelines."""
    return {
        "logistic_l2": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def _predict_scores(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.decision_function(x)


def run_temporal_split(df, split_year, features, target_col="escalation_target", model=None):
    """Train before split_year and test from split_year onward.

    This is the core safeguard against random-split optimism under temporal shift.
    """
    data = df.dropna(subset=[target_col]).copy()
    train = data[data["year"] < split_year]
    test = data[data["year"] >= split_year]
    if train.empty or test.empty:
        raise ValueError(f"Split year {split_year} produced empty train or test data.")
    model = model or make_models()["logistic_l2"]
    model.fit(train[features], train[target_col].astype(int))
    train_score = _predict_scores(model, train[features])
    threshold, validation_f1 = tune_threshold_by_f1(train[target_col].astype(int), train_score)
    test_score = _predict_scores(model, test[features])
    metrics = compute_classification_metrics(test[target_col].astype(int), test_score, threshold=threshold)
    metrics.update({"split_year": split_year, "n_train": len(train), "n_test": len(test), "validation_f1": validation_f1})
    predictions = test[["country", "year", target_col]].copy()
    predictions["score"] = test_score
    predictions["threshold"] = threshold
    predictions["prediction"] = (predictions["score"] >= threshold).astype(int)
    return metrics, predictions, model


def run_ablation_study(df, split_year, feature_sets=None, target_col="escalation_target", model_name="logistic_l2"):
    feature_sets = feature_sets or get_feature_sets(df)
    rows = []
    for name, features in feature_sets.items():
        if not features:
            continue
        metrics, _, _ = run_temporal_split(df, split_year, features, target_col, make_models()[model_name])
        metrics["feature_set"] = name
        rows.append(metrics)
    return pd.DataFrame(rows)


def run_temporal_robustness(df, split_years=(2000, 2005, 2010, 2015), features=None, target_col="escalation_target"):
    features = features or get_feature_sets(df)["full"]
    rows = []
    prediction_frames = []
    fitted = {}
    for split_year in split_years:
        for model_name, model in make_models().items():
            metrics, predictions, fitted_model = run_temporal_split(df, split_year, features, target_col, model)
            metrics["model"] = model_name
            predictions["model"] = model_name
            rows.append(metrics)
            prediction_frames.append(predictions.assign(split_year=split_year))
            fitted[(model_name, split_year)] = fitted_model
    return pd.DataFrame(rows), pd.concat(prediction_frames, ignore_index=True), fitted

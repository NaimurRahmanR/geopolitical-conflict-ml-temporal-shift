"""Metrics, threshold tuning, confidence intervals, and calibration error."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    metrics = {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_score),
        "ece": expected_calibration_error(y_true, y_score),
    }
    metrics["pr_auc"] = average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan
    metrics["roc_auc"] = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan
    try:
        metrics["log_loss"] = log_loss(y_true, y_score, labels=[0, 1])
    except ValueError:
        metrics["log_loss"] = np.nan
    return metrics


def tune_threshold_by_f1(y_true, y_score, thresholds=None):
    """Tune on validation/training data only and report the chosen threshold."""
    thresholds = np.asarray(thresholds if thresholds is not None else np.linspace(0.05, 0.95, 19))
    scores = [f1_score(y_true, np.asarray(y_score) >= t, zero_division=0) for t in thresholds]
    idx = int(np.argmax(scores))
    return float(thresholds[idx]), float(scores[idx])


def bootstrap_confidence_intervals(y_true, y_score, metric_fn=None, n_bootstrap=1000, random_state=42, alpha=0.05):
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    metric_fn = metric_fn or (lambda yt, ys: average_precision_score(yt, ys) if len(np.unique(yt)) > 1 else np.nan)
    estimates = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), len(y_true))
        estimates.append(metric_fn(y_true[idx], y_score[idx]))
    estimates = np.asarray(estimates, dtype=float)
    return {
        "estimate": metric_fn(y_true, y_score),
        "ci_lower": np.nanquantile(estimates, alpha / 2),
        "ci_upper": np.nanquantile(estimates, 1 - alpha / 2),
        "n_bootstrap": n_bootstrap,
    }


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        mask = (y_prob >= left) & (y_prob <= right if i == n_bins - 1 else y_prob < right)
        if not np.any(mask):
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)

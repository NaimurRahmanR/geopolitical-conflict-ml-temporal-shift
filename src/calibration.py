"""Calibration and reliability analysis."""

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss

from .evaluation import expected_calibration_error


def run_raw_vs_platt_vs_isotonic_calibration(base_model, x_train, y_train, x_test):
    """Fit raw, Platt sigmoid, and isotonic calibrated probabilities."""
    base_model.fit(x_train, y_train)
    raw = base_model.predict_proba(x_test)[:, 1]
    platt = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
    platt.fit(x_train, y_train)
    isotonic = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    isotonic.fit(x_train, y_train)
    return {
        "raw": raw,
        "platt": platt.predict_proba(x_test)[:, 1],
        "isotonic": isotonic.predict_proba(x_test)[:, 1],
    }


def compare_brier_ece_logloss(y_true, probability_dict):
    rows = []
    for method, probs in probability_dict.items():
        rows.append(
            {
                "method": method,
                "brier": brier_score_loss(y_true, probs),
                "ece": expected_calibration_error(y_true, probs),
                "log_loss": log_loss(y_true, probs, labels=[0, 1]),
            }
        )
    return pd.DataFrame(rows)

import numpy as np
import pandas as pd

from src.drift import compute_psi
from src.evaluation import expected_calibration_error
from src.modeling import run_temporal_split


def test_expected_calibration_error_for_two_bins():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    assert round(expected_calibration_error(y_true, y_prob, n_bins=2), 3) == 0.15


def test_psi_is_zero_for_identical_distribution():
    values = np.array([1, 2, 3, 4, 5])
    assert compute_psi(values, values, bins=3) == 0.0


def test_temporal_split_trains_before_split_and_tests_after():
    df = pd.DataFrame(
        {
            "country": ["A"] * 8,
            "year": [1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005],
            "x": [0, 0, 1, 1, 0, 1, 0, 1],
            "escalation_target": [0, 0, 1, 1, 0, 1, 0, 1],
        }
    )
    metrics, predictions, _ = run_temporal_split(df, 2002, ["x"])
    assert metrics["n_train"] == 4
    assert metrics["n_test"] == 4
    assert predictions["year"].min() >= 2002

"""Run the full journal reproducibility pipeline."""

import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ruff: noqa: E402
from src import config
from src.data_loading import load_ucdp, load_vdem, load_world_bank
from src.drift import run_temporal_drift_analysis
from src.evaluation import bootstrap_confidence_intervals
from src.explainability import (
    coefficient_stability_across_splits,
    permutation_importance_analysis,
    rank_correlation_stability,
)
from src.features import (
    build_escalation_target,
    create_conflict_lag_features,
    create_leakage_safe_external_lags,
    create_regional_lag_features,
)
from src.modeling import get_feature_sets, run_ablation_study, run_temporal_robustness
from src.plotting import plot_pr_auc_robustness, plot_roc_auc_robustness, plot_temporal_split_performance
from src.preprocessing import clean_world_bank_wide_to_long, merge_datasets, select_vdem_features


def _require_raw_file(name):
    path = config.RAW_DATA_DIR / config.EXPECTED_RAW_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. See data/README.md for download instructions.")
    return path


def main():
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    ucdp = load_ucdp(_require_raw_file("ucdp"))
    wb = clean_world_bank_wide_to_long(load_world_bank(_require_raw_file("world_bank")))
    vdem = select_vdem_features(load_vdem(_require_raw_file("vdem")))

    merged = merge_datasets(ucdp, wb, vdem)
    if "intensity_level" not in merged.columns:
        raise ValueError("Merged data must contain an intensity_level column to build escalation targets.")
    panel = build_escalation_target(merged)
    panel = create_conflict_lag_features(panel)

    external_cols = [
        c
        for c in panel.select_dtypes(include="number").columns
        if c not in {"year", "intensity_level", "escalation_target"} and not c.endswith(("lag1", "lag2", "lag3"))
    ]
    panel = create_leakage_safe_external_lags(panel, external_cols, lags=(1,), drop_current=False)
    if "region" in panel.columns:
        panel = create_regional_lag_features(panel)
    panel.to_csv(config.PROCESSED_DATA_DIR / "analysis_panel.csv", index=False)

    features = get_feature_sets(panel)["full"]
    robustness, predictions, fitted_models = run_temporal_robustness(panel, config.DEFAULT_SPLIT_YEARS, features)
    robustness.to_csv(config.TABLE_DIR / "peer_review_temporal_robustness_results.csv", index=False)
    predictions.to_csv(config.TABLE_DIR / "model_predictions_by_split.csv", index=False)
    joblib.dump(fitted_models, config.MODEL_DIR / "temporal_models.joblib")

    best = robustness.sort_values("pr_auc", ascending=False).iloc[0]
    best_predictions = predictions[(predictions["model"] == best["model"]) & (predictions["split_year"] == best["split_year"])]
    ci = bootstrap_confidence_intervals(best_predictions["escalation_target"], best_predictions["score"], n_bootstrap=1000)
    pd.DataFrame([{**ci, "model": best["model"], "split_year": best["split_year"], "metric": "pr_auc"}]).to_csv(
        config.TABLE_DIR / "bootstrap_confidence_intervals.csv", index=False
    )

    ablation = run_ablation_study(panel, int(best["split_year"]))
    ablation.to_csv(config.TABLE_DIR / "ablation_study_results.csv", index=False)

    drift = pd.concat([run_temporal_drift_analysis(panel, year, features) for year in config.DEFAULT_SPLIT_YEARS], ignore_index=True)
    drift.to_csv(config.TABLE_DIR / "temporal_drift_analysis.csv", index=False)

    coefs = coefficient_stability_across_splits(fitted_models, features)
    coefs.to_csv(config.TABLE_DIR / "explainability_coefficients_by_split.csv", index=False)

    importance_frames = {}
    for split_year, group in predictions.groupby("split_year"):
        model = fitted_models[(group["model"].iloc[0], split_year)]
        test = panel[(panel["year"] >= split_year) & panel["escalation_target"].notna()]
        importance_frames[str(split_year)] = permutation_importance_analysis(model, test[features], test["escalation_target"].astype(int), features)
    pd.concat(importance_frames, names=["split_year"]).reset_index(level=0).to_csv(config.TABLE_DIR / "permutation_importance.csv", index=False)
    rank_correlation_stability(importance_frames).to_csv(config.TABLE_DIR / "explanation_stability_rank_correlations.csv", index=False)

    plot_pr_auc_robustness(robustness, config.FIGURE_DIR / "figure_1_robustness_pr_auc.png")
    plot_roc_auc_robustness(robustness, config.FIGURE_DIR / "figure_2_robustness_roc_auc.png")
    plot_temporal_split_performance(robustness, "pr_auc", config.FIGURE_DIR / "figure_6_pr_auc_across_splits.png")
    plot_temporal_split_performance(robustness, "roc_auc", config.FIGURE_DIR / "figure_7_roc_auc_across_splits.png")


if __name__ == "__main__":
    main()

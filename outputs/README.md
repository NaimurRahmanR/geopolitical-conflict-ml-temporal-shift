# Outputs

This folder contains all model outputs, evaluation results, and figures generated from the research pipeline.

## Output Files

### Model Performance Results

| File | Description |
|------|-------------|
| `improved_model_auc_pr_results (2).csv` | AUC and PR-AUC results for the best improved escalation model |
| `improved_model_threshold_results (2).csv` | Threshold tuning results for the best model (precision, recall, F1 at each threshold) |
| `improved_model_predictions (2).csv` | Final model predictions on test set (year, actual label, predicted probability, predicted label) |
| `peer_review_temporal_robustness_results.csv` | Full temporal robustness evaluation across multiple train/test split years (2000, 2005, 2010, 2015) and feature sets |
| `peer_review_summary_table.csv` | Summary table of key results suitable for peer review / paper submission |

### Calibration & Reliability

| File | Description |
|------|-------------|
| `calibration_table_best_model.csv` | Calibration table for the best model (predicted probability bins vs. actual positive rate) |
| `bootstrap_confidence_intervals.csv` | Bootstrap 95% confidence intervals for ROC-AUC and PR-AUC of the best model |
| `temporal_drift_analysis.csv` | Temporal drift analysis: model performance degradation across test periods |

### Error Analysis

| File | Description |
|------|-------------|
| `error_analysis_best_model.csv` | Detailed error analysis per test instance (false positives, false negatives, true positives, true negatives) |
| `false_alarms.csv` | False positive cases (predicted escalation that did not occur) with conflict context |
| `missed_escalations.csv` | False negative cases (missed escalations) with conflict context |

### Explainability

| File | Description |
|------|-------------|
| `explainability_coefficients_by_split.csv` | Feature coefficients / importances by temporal split year (SHAP or tree importance) |
| `explanation_stability_rank_correlations.csv` | Rank correlation of feature importances across temporal splits (Spearman rho) |
| `figure_8_explainability_stability.png` | Figure: Explainability stability across temporal splits (line chart of top feature importances) |

## Google Drive Access

All output files are also available on Google Drive:
https://drive.google.com/drive/folders/1UDvdjrnlStdHAHjz0oVnAF9ZJEFALV24

## Notes
- Train/test split cutoff: **2010** (trained on ≤2010, tested on >2010)
- Additional split years evaluated: 2000, 2005, 2015
- Best model: Weighted XGBoost or Weighted Random Forest (see `improved_model_auc_pr_results.csv`)
- All results use leakage-safe lagged features for external variables (V-Dem, World Bank)

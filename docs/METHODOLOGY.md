# Methodology

## Target

The target is next-year conflict escalation: a country-year is positive when the following observed country-year has higher conflict intensity than the current year. The final observed year for a country has an undefined target and is excluded from supervised modeling.

## Main Evaluation Design

The main empirical claim is evaluated with temporal splits. For split years `2000`, `2005`, `2010`, and `2015`, models train on prior years and test on the split year and later. This design approximates historical deployment more closely than random splits.

## Scientific Safeguards

- Random split is not reported as the primary result.
- External indicators are lagged before modeling.
- Regional conflict features are lagged by one year.
- Decision thresholds are tuned on training-period predictions and saved with outputs.
- Robustness is evaluated across multiple split years to avoid overclaiming from one historical period.

## Models

The default package includes balanced logistic regression and balanced random forest pipelines with median imputation. Additional models can be added through `src/modeling.py` while preserving temporal split semantics.

## Robustness and Reliability

The analysis reports discrimination, threshold metrics, calibration metrics, bootstrap confidence intervals, temporal drift diagnostics, ablation studies, and explanation stability. These diagnostics separate accuracy from reliability: a model can rank cases well but still be poorly calibrated or unstable across periods.

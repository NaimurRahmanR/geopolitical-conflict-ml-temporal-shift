# Outputs Directory

This directory stores reviewer-facing tables, figures, and model artifacts.

## Tables

`outputs/tables/` contains CSV outputs such as temporal robustness results, summary tables, bootstrap confidence intervals, drift statistics, explanation-stability outputs, calibration tables, permutation importance, ablation results, and error-analysis files.

## Figures

`outputs/figures/` contains publication figures generated from the tables and predictions, including PR-AUC and ROC-AUC robustness, reliability, calibration, permutation importance, temporal split performance, and explanation stability figures.

## Models

`outputs/models/` stores regenerated model binaries. Model files are intentionally ignored by Git unless explicitly curated for release.

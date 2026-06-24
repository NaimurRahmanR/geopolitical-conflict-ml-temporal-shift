# Reproducibility Guide

## Data Acquisition

Download the raw datasets from official providers:

- UCDP/PRIO Armed Conflict Dataset v26.1: Uppsala Conflict Data Program / PRIO Armed Conflict Dataset release page.
- World Bank World Development Indicators: World Bank DataBank or API export.
- V-Dem country-year dataset: Varieties of Democracy project download portal.

Save files as:

```text
data/raw/ucdp_prio_acd_v26_1.csv
data/raw/world_bank_wdi.csv
data/raw/vdem_country_year.csv
```

## Execution Order

1. Install dependencies with `pip install -r requirements.txt` or `conda env create -f environment.yml`.
2. Place raw data in `data/raw/`.
3. Run `python scripts/run_all.py`.
4. Inspect outputs under `outputs/tables/` and `outputs/figures/`.
5. Run `pytest -q` to verify package-level safeguards with synthetic data.

## Expected Outputs

The full pipeline writes `data/processed/analysis_panel.csv`, tables for temporal robustness, ablations, calibration, confidence intervals, temporal drift, permutation importance, and explanation stability, plus paper-ready figures.

## Temporal Split Logic

For split year `s`, training observations satisfy `year < s` and test observations satisfy `year >= s`. The default split years are `2000`, `2005`, `2010`, and `2015`.

Random splits are not the primary result because they can leak period-specific distributional structure across train and test data. The study asks whether models trained on earlier historical periods remain reliable in later periods.

## Leakage-Safe Lag Construction

Conflict-history variables use country-specific lags. External WDI and V-Dem indicators are lagged before modeling so same-year governance or socioeconomic measurements cannot act as post-treatment or unavailable information. Regional conflict features are computed at region-year level and shifted by one year to avoid same-year regional conflict leakage.

## Metrics

Reported metrics include PR-AUC, ROC-AUC, F1, precision, recall, Brier score, log loss, expected calibration error, bootstrap confidence intervals, PSI, KS statistics, and rank correlations for explanation stability.

## Figures and Tables

Reproduce all tables and figures with:

```bash
python scripts/run_all.py
```

Use `scripts/make_figures.py` when only figure regeneration is needed from the same pipeline entry point.

# Reliability of Machine Learning for Conflict Escalation Forecasting under Temporal Distribution Shift

Lead researcher: Naimur Rahman

This repository is a journal-oriented reproducibility package for evaluating whether machine-learning models for armed conflict escalation remain reliable when trained on past country-years and evaluated on later historical periods.

## Research Question

How reliable are machine-learning models for conflict escalation forecasting when the evaluation setting respects temporal distribution shift rather than using random train/test splits?

## Motivation

Conflict forecasting is a high-stakes applied setting. A model can appear accurate when observations from the same historical period are mixed across train and test folds, but fail when deployed into later periods with different geopolitical conditions, governance distributions, conflict prevalence, or reporting regimes. This project therefore treats temporal robustness, calibration, drift, and explanation stability as core empirical claims rather than optional diagnostics.

## Dataset Summary

The analysis combines country-year information from UCDP/PRIO Armed Conflict Dataset v26.1, World Bank World Development Indicators, V-Dem country-year data, and engineered conflict-history, governance, socioeconomic, regional, and geopolitical features.

Raw data are not committed. Place the expected source files in `data/raw/` as described in [data/README.md](data/README.md).

## Pipeline

```text
data/raw/
  -> load UCDP/PRIO, WDI, V-Dem
  -> harmonise country-year keys
  -> merge country-year panel
  -> build next-year escalation target
  -> construct leakage-safe lags
  -> temporal splits: train years < split, test years >= split
  -> robustness, ablation, calibration, drift, explainability analyses
  -> outputs/tables/ and outputs/figures/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Conda users can instead run:

```bash
conda env create -f environment.yml
conda activate conflict-shift-repro
```

## Data Download

Download the source datasets from their official providers and save them with these filenames:

- `data/raw/ucdp_prio_acd_v26_1.csv`
- `data/raw/world_bank_wdi.csv`
- `data/raw/vdem_country_year.csv`

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for provider links, execution order, and expected outputs.

## Reproduction Command

```bash
python scripts/run_all.py
```

The full command preprocesses data, creates leakage-safe features, runs split years `2000`, `2005`, `2010`, and `2015`, estimates ablation and robustness results, computes confidence intervals, evaluates temporal drift, checks explanation stability, and generates final tables and figures.

## Key Outputs

Tables are written to `outputs/tables/`, including temporal robustness results, bootstrap confidence intervals, ablation results, drift statistics, calibration summaries, permutation importance, and error-analysis files where available.

Figures are written to `outputs/figures/`, including PR-AUC and ROC-AUC robustness plots, reliability curves, calibration-bin plots, permutation importance, temporal performance across split years, and explanation-stability visualisations.

## Methodological Summary

The primary design is temporal evaluation: models train only on years before each split year and evaluate on subsequent years. Random splits are not used as the main result because they can mix countries and time periods in ways that overstate deployment reliability under temporal shift.

External socioeconomic and governance variables are lagged before modeling. Regional conflict features are also lagged so the model cannot use same-year regional conflict information when forecasting escalation. Thresholds are tuned on training-period predictions and reported in the outputs.

## Limitations

The package supports reproducibility of the empirical workflow, but causal interpretation remains limited. Forecast performance depends on source-data coverage, country harmonisation, missingness, conflict coding changes, and the target definition. Temporal robustness across four split years is stronger than a single split, but it is still historical validation rather than proof of future deployment reliability.

## Citation

Use the metadata in [CITATION.cff](CITATION.cff) when citing this repository.

## Contact

Naimur Rahman. Open an issue or pull request on the repository for reproducibility questions.

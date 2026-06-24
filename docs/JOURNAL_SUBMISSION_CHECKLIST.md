# Journal Submission Checklist

## Reproducibility

- [x] Environment files supplied: `requirements.txt`, `environment.yml`, `pyproject.toml`.
- [x] One-command pipeline supplied: `python scripts/run_all.py`.
- [x] Raw data filenames and locations documented.
- [x] Lightweight tests run without restricted raw data.
- [x] CI runs linting and tests.

## Data Availability

- [x] Source datasets identified by provider and version.
- [x] Raw data excluded from Git with documented placement.
- [ ] Confirm provider-specific redistribution rules before archive release.
- [ ] Add DOI or archive links for exact data snapshots if journal policy requires them.

## Code Availability

- [x] Reusable modules are separated from notebooks.
- [x] Scripts regenerate tables and figures.
- [x] Existing generated outputs are organised under `outputs/tables/`.
- [ ] Tag a release after final paper acceptance.

## Limitations

- [x] README states that historical validation is not proof of future deployment reliability.
- [x] Methodology distinguishes predictive performance from causal claims.
- [x] Data coverage, harmonisation, and missingness risks are documented.

## Robustness

- [x] Multiple temporal split years are supported.
- [x] Ablation study entry point is included.
- [x] Bootstrap confidence intervals are included.
- [x] Temporal drift is measured with PSI and KS statistics.
- [x] Explanation stability is measured by coefficient and rank-correlation summaries.

## Reviewer Risk

- [x] Random-split optimism is explicitly avoided as the primary result.
- [x] Same-year external-indicator leakage is explicitly guarded against.
- [x] Same-year regional conflict leakage is explicitly guarded against.
- [x] Threshold tuning is reported.
- [x] Claims are framed around temporal robustness, not universal forecasting reliability.

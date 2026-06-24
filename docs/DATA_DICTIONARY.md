# Data Dictionary

This document records expected fields and engineered variables. Exact source columns may differ by provider export; preprocessing functions standardise common country and year names.

## Key Columns

| Column | Description |
| --- | --- |
| `country` | Harmonised country name. |
| `year` | Calendar year of the country-year observation. |
| `region` | Optional geopolitical region used for lagged regional conflict context. |
| `intensity_level` | UCDP/PRIO conflict intensity variable used to build escalation target. |
| `escalation_target` | Indicator for higher conflict intensity in the next country-year. |

## Engineered Features

| Pattern | Description |
| --- | --- |
| `*_lag1`, `*_lag2`, `*_lag3` | Country-specific lagged features. |
| `regional_*_lag1` | Prior-year regional aggregate conflict context. |
| `*_country_lag1` | Generic country lag features produced by `create_country_lag_features`. |

## Source Families

- Conflict history: UCDP/PRIO intensity, deaths, or conflict counts.
- Governance: selected V-Dem democracy, liberal democracy, corruption, rule-of-law, and participation indices.
- Socioeconomic: selected WDI indicators, reshaped from wide indicator-year format.
- Regional/geopolitical: region-year aggregates lagged to avoid same-year leakage.

"""Input readers for the raw research datasets."""

from pathlib import Path

import pandas as pd


def _read_tabular(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".tsv"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix in {".parquet"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format for {path}")


def load_ucdp(path):
    """Load UCDP/PRIO Armed Conflict data."""
    return _read_tabular(path)


def load_world_bank(path):
    """Load World Bank WDI data, usually in wide country-indicator-year form."""
    return _read_tabular(path)


def load_vdem(path):
    """Load V-Dem country-year data."""
    return _read_tabular(path)

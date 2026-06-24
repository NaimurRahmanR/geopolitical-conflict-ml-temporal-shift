"""Project paths and default experiment configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"

COUNTRY_COL = "country"
YEAR_COL = "year"
TARGET_COL = "escalation_target"
DEFAULT_SPLIT_YEARS = [2000, 2005, 2010, 2015]
RANDOM_STATE = 42

EXPECTED_RAW_FILES = {
    "ucdp": "ucdp_prio_acd_v26_1.csv",
    "world_bank": "world_bank_wdi.csv",
    "vdem": "vdem_country_year.csv",
}

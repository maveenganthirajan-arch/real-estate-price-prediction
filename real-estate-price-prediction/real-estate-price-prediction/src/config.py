"""Central paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
SQL_DIR = ROOT / "sql"

RAW_CSV = DATA_RAW / "listings_raw.csv"
AMENITIES_CSV = DATA_RAW / "amenities.csv"
CLEAN_CSV = DATA_PROCESSED / "listings_clean.csv"
CLEANING_REPORT = DATA_PROCESSED / "cleaning_report.json"
DB_PATH = DATA_PROCESSED / "real_estate.db"

MODEL_PATH = MODELS / "best_model.joblib"
METRICS_PATH = MODELS / "metrics.json"
TEST_PRED_PATH = MODELS / "test_predictions.csv"
IMPORTANCE_PATH = MODELS / "feature_importance.json"
COMPARISON_PATH = MODELS / "model_comparison.csv"

RANDOM_STATE = 42
PROPERTY_TYPES = ["Single Family", "Condo", "Townhouse", "Multi-Family"]

# Time-ordered split: oldest 70 % train, next 15 % validation, newest 15 % test.
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
INTERVAL_LEVEL = 0.90

"""
Centralized project configuration.

All paths, constants, and tunables live here so that individual modules
never hardcode their own values.  Import from ``src.config`` everywhere.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # …/ml-analytics-dashboard
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_CHURN_CSV = RAW_DATA_DIR / "Telco_Cusomer_Churn.csv"

MLRUNS_DIR = PROJECT_ROOT / "mlruns"

# ── MLflow ───────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME = "Churn_prediction"

# ── Model training ──────────────────────────────────────────────────
TEST_SIZE = 0.2
RANDOM_STATE = 42

LOGISTIC_REGRESSION_PARAMS = {
    "max_iter": 500,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "random_state": RANDOM_STATE,
}

# ── Feature engineering ─────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "InternetService",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

TENURE_BINS = [0, 12, 24, 48, 72]
TENURE_LABELS = ["0-1yr", "1-2yr", "2-4yr", "4-6yr"]

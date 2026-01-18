import mlflow
from pathlib import Path

EXPERIMENT_NAME = "Churn_prediction"
MLRUNS_PATH = Path("mlruns")

def setup_mlflow():
    """Set up MLflow experiment configuration."""
    mlflow.set_tracking_uri(f"file://{MLRUNS_PATH.absolute()}")
    mlflow.set_experiment(EXPERIMENT_NAME)
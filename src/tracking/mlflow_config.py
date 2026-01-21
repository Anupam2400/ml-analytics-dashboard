import mlflow
from pathlib import Path

EXPERIMENT_NAME = "Churn_prediction"
MLRUNS_PATH = Path("mlruns")

def setup_mlflow():
    """Set up MLflow experiment configuration."""
    # Prefer a proper file:// URI for local filesystem stores. Path.as_uri()
    # produces a well-formed URI (e.g. file:///C:/path/...). Some older
    # MLflow versions may still raise; in that case fall back to a plain
    # filesystem path.
    try:
        mlflow.set_tracking_uri(MLRUNS_PATH.absolute().as_uri())
    except Exception:
        mlflow.set_tracking_uri(str(MLRUNS_PATH.absolute()))

    mlflow.set_experiment(EXPERIMENT_NAME)
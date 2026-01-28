# import mlflow
# from mlflow.tracking import MlflowClient
# from pathlib import Path
# import os

# def load_best_model(experiment_name="churn_prediction"):
#     # ✅ Windows-safe local tracking URI
#     # tracking_path = Path("mlruns").absolute()
#     # mlflow.set_tracking_uri(str(tracking_path))
#     # tracking_path = Path(__file__).resolve().parents[2] / "mlruns"
#     # mlflow.set_tracking_uri(str(tracking_path))

#     # client = MlflowClient()
#     base_path = Path(__file__).resolve().parents[2] 
#     print(f"Base path for MLflow tracking: {base_path}")
#     mlruns_path = base_path / "mlruns"

#     # Use the file:// protocol for local tracking
#     mlflow.set_tracking_uri(f"file:///{mlruns_path.as_posix()}")
#     # mlflow.set_tracking_uri("http://127.0.0.1:5000")
#     client = MlflowClient()         
#     experiment = client.get_experiment_by_name(experiment_name)
#     print(f"Experiment details: {experiment}")
#     if experiment is None:
#         raise ValueError(
#             f"Experiment '{experiment_name}' not found. "
#             "Have you run training at least once?"
#         )

#     runs = client.search_runs(
#         experiment_ids=[experiment.experiment_id],
#         order_by=["metrics.roc_auc DESC"],
#         max_results=1
#     )

#     if not runs:
#         raise ValueError("No runs found for this experiment")

#     best_run = runs[0]
#     model_uri = f"runs:/{best_run.info.run_id}/model"

#     model = mlflow.sklearn.load_model(model_uri)
#     return model, best_run.data.metrics


import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

def load_best_model(experiment_name="Churn_prediction"):

    # Project root
    base_path = Path(__file__).resolve().parents[2]
    mlruns_path = base_path / "mlruns"

    # Set tracking URI
    mlflow.set_tracking_uri(f"file:///{mlruns_path.as_posix()}")

    client = MlflowClient()

    # Debug info
    print("MLflow URI:", mlflow.get_tracking_uri())
    print("MLruns path exists:", mlruns_path.exists())

    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(
            f"Experiment '{experiment_name}' not found.\n"
            f"Run training first:\n"
            f"python src/models/train.py\n"
            f"Tracking URI: {mlflow.get_tracking_uri()}"
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.roc_auc DESC"],
        max_results=1
    )

    if not runs:
        raise ValueError(
            f"No runs found in experiment '{experiment_name}'."
        )

    best_run = runs[0]
    model_uri = f"runs:/{best_run.info.run_id}/model"

    model = mlflow.sklearn.load_model(model_uri)

    return model, best_run.data.metrics

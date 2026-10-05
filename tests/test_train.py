"""Tests for the model training module."""

import os
import pandas as pd
import pytest

from src.models.train import train_model

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


class TestTrainModel:
    """Tests for :func:`train_model`."""

    @pytest.fixture
    def training_data(self, processed_churn_df: pd.DataFrame):
        """Prepare X/y splits from the fixture data."""
        df = processed_churn_df
        X = df.drop(columns=["Churn", "customerID"])
        y = df["Churn"].map({"Yes": 1, "No": 0})
        X = pd.get_dummies(X, drop_first=True)

        # Keep both target classes in the training split; the fixture is tiny.
        return X.iloc[[0, 1, 2]], X.iloc[[3, 4]], y.iloc[[0, 1, 2]], y.iloc[[3, 4]]

    def test_logistic_regression_returns_metrics(
        self, training_data, tmp_path, monkeypatch
    ) -> None:
        """train_model should return (accuracy, roc_auc) floats."""
        import mlflow

        # Use a temporary MLflow tracking dir so tests don't pollute the real one
        mlflow.set_tracking_uri(tmp_path.as_uri())
        mlflow.set_experiment("test_experiment")

        from sklearn.linear_model import LogisticRegression

        X_train, X_test, y_train, y_test = training_data
        acc, roc = train_model(
            LogisticRegression(max_iter=500),
            X_train, X_test, y_train, y_test,
            "TestLogistic",
        )

        assert isinstance(acc, float)
        assert isinstance(roc, float)
        assert 0.0 <= acc <= 1.0
        assert 0.0 <= roc <= 1.0

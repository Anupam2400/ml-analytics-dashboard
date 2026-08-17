"""Tests for the model training module."""

import pandas as pd
import pytest

from src.models.train import train_model


class TestTrainModel:
    """Tests for :func:`train_model`."""

    @pytest.fixture
    def training_data(self, processed_churn_df: pd.DataFrame):
        """Prepare X/y splits from the fixture data."""
        from sklearn.model_selection import train_test_split

        df = processed_churn_df
        X = df.drop(columns=["Churn", "customerID"])
        y = df["Churn"].map({"Yes": 1, "No": 0})
        X = pd.get_dummies(X, drop_first=True)

        # With only 5 rows, use 1 for test so the model can fit
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def test_logistic_regression_returns_metrics(
        self, training_data, tmp_path, monkeypatch
    ) -> None:
        """train_model should return (accuracy, roc_auc) floats."""
        import mlflow

        # Use a temporary MLflow tracking dir so tests don't pollute the real one
        mlflow.set_tracking_uri(f"file:///{tmp_path.as_posix()}")
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

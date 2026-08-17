"""Shared pytest fixtures for the ML Analytics Dashboard test suite."""

import pandas as pd
import pytest


@pytest.fixture
def sample_churn_df() -> pd.DataFrame:
    """Return a small but realistic Telco Churn DataFrame for testing.

    Contains the minimum required columns with a mix of churned and
    non-churned customers, edge cases (empty TotalCharges, zero tenure).
    """
    return pd.DataFrame({
        "customerID": ["C001", "C002", "C003", "C004", "C005"],
        "gender": ["Male", "Female", "Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0, 0, 1],
        "Partner": ["Yes", "No", "Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes", "No", "Yes"],
        "tenure": [1, 34, 0, 72, 12],
        "PhoneService": ["Yes", "Yes", "No", "Yes", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "No", "DSL", "Fiber optic"],
        "MonthlyCharges": [29.85, 56.95, 0.00, 89.10, 45.25],
        # TotalCharges sometimes comes as string in the real CSV
        "TotalCharges": ["29.85", "1889.5", " ", "6369.05", "542.4"],
        "Churn": ["No", "Yes", "No", "No", "Yes"],
    })


@pytest.fixture
def processed_churn_df(sample_churn_df: pd.DataFrame) -> pd.DataFrame:
    """Return feature-engineered DataFrame (ready for model training)."""
    from src.preprocessing.features import build_features

    return build_features(sample_churn_df)

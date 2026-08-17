"""Tests for the feature engineering module."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.features import build_features


class TestBuildFeatures:
    """Tests for :func:`build_features`."""

    def test_does_not_modify_input(self, sample_churn_df: pd.DataFrame) -> None:
        """build_features should not mutate the input DataFrame."""
        original = sample_churn_df.copy()
        build_features(sample_churn_df)
        pd.testing.assert_frame_equal(sample_churn_df, original)

    def test_total_charges_numeric(self, sample_churn_df: pd.DataFrame) -> None:
        """TotalCharges should be converted to float."""
        result = build_features(sample_churn_df)
        assert pd.api.types.is_float_dtype(result["TotalCharges"])

    def test_no_null_total_charges(self, sample_churn_df: pd.DataFrame) -> None:
        """Missing TotalCharges (spaces in CSV) should be filled."""
        result = build_features(sample_churn_df)
        assert result["TotalCharges"].isna().sum() == 0

    def test_avg_monthly_spend_created(self, sample_churn_df: pd.DataFrame) -> None:
        """An avg_monthly_spend feature should exist and be finite."""
        result = build_features(sample_churn_df)
        assert "avg_monthly_spend" in result.columns
        assert np.isfinite(result["avg_monthly_spend"]).all()

    def test_tenure_bucket_created(self, sample_churn_df: pd.DataFrame) -> None:
        """A tenure_bucket feature should exist as a categorical."""
        result = build_features(sample_churn_df)
        assert "tenure_bucket" in result.columns
        assert hasattr(result["tenure_bucket"], "cat")

    def test_zero_tenure_no_division_error(self) -> None:
        """tenure=0 should not cause division by zero (we add 1)."""
        df = pd.DataFrame({
            "customerID": ["X1"],
            "gender": ["Male"],
            "SeniorCitizen": [0],
            "Partner": ["No"],
            "Dependents": ["No"],
            "tenure": [0],
            "PhoneService": ["Yes"],
            "InternetService": ["DSL"],
            "MonthlyCharges": [20.0],
            "TotalCharges": ["20.0"],
            "Churn": ["No"],
        })
        result = build_features(df)
        assert np.isfinite(result["avg_monthly_spend"].iloc[0])

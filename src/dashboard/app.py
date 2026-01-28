import streamlit as st
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from src.ingestion.load_data import load_data
from src.preprocessing.features import build_features
from src.models.load_best_model import load_best_model

st.set_page_config(page_title="Churn Analytics Dashboard", layout="wide")

st.title("📊 Customer Churn Analytics Dashboard")

# Load data
df = build_features(load_data())

# Load best model
model, metrics = load_best_model()

# Prepare features
X = pd.get_dummies(
    df.drop(columns=["Churn", "customerID"]),
    drop_first=True
)

# Predictions
df["churn_probability"] = model.predict_proba(X)[:, 1]

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", len(df))
col2.metric("Churn Rate", f"{df['Churn'].value_counts(normalize=True)['Yes']*100:.2f}%")
col3.metric("Avg Churn Probability", f"{df['churn_probability'].mean():.2f}")

# Table
st.subheader("High Risk Customers")
st.dataframe(
    df.sort_values("churn_probability", ascending=False)[
        ["customerID", "churn_probability"]
    ].head(20)
)

st.subheader("Model Metrics")
st.write(metrics)

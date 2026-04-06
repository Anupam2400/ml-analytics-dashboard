import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from src.ingestion.load_data import load_data
from src.preprocessing.features import build_features
from src.models.load_best_model import load_best_model

tab1, tab2, tab3 = st.tabs([
    "📊 Overview",
    "📈 Risk Analysis",
    "🧠 Model Insights"
])

st.set_page_config(
    page_title="Churn Analytics Dashboard",
    layout="wide"
)

st.markdown("###")

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

st.subheader("Churn Distribution")

fig, ax = plt.subplots()
df["Churn"].value_counts().plot(kind="bar", ax=ax)
ax.set_xlabel("Churn")
ax.set_ylabel("Number of Customers")
st.pyplot(fig)

# Table
st.subheader("High Risk Customers")
st.dataframe(
    df.sort_values("churn_probability", ascending=False)[
        ["customerID", "churn_probability"]
    ].head(20)
)
st.subheader("Top Drivers of Churn")

if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    feature_names = X.columns

    fi = (
        pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        })
        .sort_values("importance", ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots()
    ax.barh(fi["feature"], fi["importance"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    st.pyplot(fig)
else:
    st.info("Feature importance not available for this model.")
    
with tab1:
    st.title("📊 Customer Churn Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric(
        "Churn Rate",
        f"{df['Churn'].value_counts(normalize=True)['Yes']*100:.2f}%"
    )
    col3.metric(
        "Avg Risk Score",
        f"{df['churn_probability'].mean():.2f}"
    )

    st.markdown("---")

    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    df["Churn"].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)
    
with tab2:
    st.title("📈 Customer Risk Analysis")

    threshold = st.slider(
        "Select churn risk threshold",
        0.0, 1.0, 0.5, 0.05
    )

    high_risk = df[df["churn_probability"] > threshold]

    st.metric("High Risk Customers", len(high_risk))

    st.subheader("Risk Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["churn_probability"], bins=20)
    st.pyplot(fig)

    st.subheader("Top High-Risk Customers")
    st.dataframe(
        high_risk.sort_values("churn_probability", ascending=False)[
            ["customerID", "churn_probability"]
        ].head(20)
    )
    
with tab3:
    st.title("🧠 Model Insights")

    st.subheader("Model Performance")
    st.write(metrics)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_names = X.columns

        fi = (
            pd.DataFrame({
                "feature": feature_names,
                "importance": importances
            })
            .sort_values("importance", ascending=False)
            .head(10)
        )

        fig, ax = plt.subplots()
        ax.barh(fi["feature"], fi["importance"])
        ax.invert_yaxis()
        st.pyplot(fig)

st.subheader("Model Metrics")
st.write(metrics)

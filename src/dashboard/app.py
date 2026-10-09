import streamlit as st
import pandas as pd
from pathlib import Path
import sys

st.set_page_config(
    page_title="Churn Analytics Dashboard",
    page_icon=":material/analytics:",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from src.ingestion.load_data import load_data
from src.preprocessing.features import build_features
from src.models.load_best_model import load_best_model

@st.cache_data(show_spinner=False)
def get_data():
    return build_features(load_data())

@st.cache_resource(show_spinner=False)
def get_model():
    return load_best_model()

st.title("Customer Churn Analytics", icon=":material/dashboard:")

# Load data and model
with st.spinner("Loading data and model..."):
    df = get_data()
    model, metrics = get_model()

# Prepare features and predictions
X = df.drop(columns=["Churn", "customerID"])
df["churn_probability"] = model.predict_proba(X)[:, 1]

tab1, tab2, tab3 = st.tabs([
    "Overview",
    "Risk Analysis",
    "Model Insights"
])

with tab1:
    st.subheader("Overview", icon=":material/bar_chart:")
    with st.container(horizontal=True):
        st.metric("Total Customers", f"{len(df):,}", border=True)
        st.metric("Churn Rate", f"{df['Churn'].value_counts(normalize=True).get('Yes', 0)*100:.2f}%", border=True)
        st.metric("Avg Risk Score", f"{df['churn_probability'].mean():.2f}", border=True)

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("Churn Distribution", icon=":material/pie_chart:")
            churn_counts = df["Churn"].value_counts().reset_index()
            churn_counts.columns = ["Churn", "Count"]
            st.bar_chart(churn_counts, x="Churn", y="Count")
            
    with col2:
        with st.container(border=True):
            st.subheader("Top Drivers of Churn", icon=":material/trending_up:")
            if hasattr(model, "named_steps") and "classifier" in model.named_steps:
                classifier = model.named_steps["classifier"]
                if hasattr(classifier, "feature_importances_"):
                    importances = classifier.feature_importances_
                    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
                    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
                    fi = fi.sort_values("importance", ascending=False).head(10)
                    st.bar_chart(fi, x="feature", y="importance")
            else:
                st.info("Feature importance not available for this model.")

with tab2:
    st.subheader("Risk Analysis", icon=":material/warning:")
    
    with st.container(border=True):
        threshold = st.slider("Select churn risk threshold", 0.0, 1.0, 0.5, 0.05)
        high_risk = df[df["churn_probability"] > threshold]
        st.metric("High Risk Customers", f"{len(high_risk):,}")

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("Risk Distribution", icon=":material/analytics:")
            risk_bins = pd.cut(df["churn_probability"], bins=20).value_counts().sort_index().reset_index()
            risk_bins.columns = ["Probability Range", "Count"]
            risk_bins["Probability Range"] = risk_bins["Probability Range"].astype(str)
            st.bar_chart(risk_bins, x="Probability Range", y="Count")

    with col2:
        with st.container(border=True):
            st.subheader("Top High-Risk Customers", icon=":material/group:")
            st.dataframe(
                high_risk.sort_values("churn_probability", ascending=False)[
                    ["customerID", "churn_probability"]
                ].head(20),
                hide_index=True
            )

with tab3:
    st.subheader("Model Insights", icon=":material/memory:")
    
    with st.container(border=True):
        st.subheader("Model Metrics")
        st.json(metrics)

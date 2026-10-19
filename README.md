# 📊 ML-Powered Customer Analytics Dashboard

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E.svg)](https://scikit-learn.org/)

An end-to-end customer churn analytics project that combines data preparation, model training, experiment tracking, and an interactive dashboard for business decision-making.

This project is designed to help teams understand churn patterns, identify at-risk customers, and monitor model performance in a production-like workflow.

---

## 🎯 Business Value

The dashboard helps teams answer questions like:
- Which customers are most likely to churn?
- What factors contribute most to churn risk?
- How much revenue is exposed to churn risk?
- Which model performs best on the validation set?

---

## 🚀 Features

- **Data Pipeline:** Automated ingestion, validation, and feature engineering.
- **Model Training:** Training multiple machine learning algorithms (XGBoost, LightGBM, Random Forest, etc.).
- **Experiment Tracking:** MLflow integration for model comparison and selection based on ROC-AUC.
- **Interactive Dashboard:** Streamlit application providing business insights for churn risk monitoring.

---

## 🗺️ Project Roadmap

Work through the milestones in order. The current focus is reliable, reproducible model evaluation; dashboard scores computed on the full dataset are useful for exploration but should not be treated as estimates of performance on unseen customers.

### Phase 1: Reliable evaluation (next)

- [x] Use a stratified train/validation/test split with a fixed random seed.
- [ ] Keep preprocessing inside the fitted model pipeline and evaluate only on held-out data.
- [ ] Log precision, recall, F1, PR-AUC, ROC-AUC, and a confusion matrix to MLflow.
- [ ] Add tests for split reproducibility, leakage prevention, and metric calculations.
- [ ] Show held-out metrics in the dashboard and clearly distinguish them from full-dataset descriptive insights.

**Done when:** training and evaluation are reproducible, and the dashboard reports model quality on data excluded from fitting and model selection.

### Phase 2: Useful customer-level insights

- [ ] Add threshold tradeoff charts based on validation data and select a threshold using an explicit business objective.
- [ ] Add customer-level explanations for individual churn predictions.
- [ ] Add churn segments by contract, tenure, service type, and monthly charges.
- [ ] Add a retention scenario calculator with visible assumptions for contact rate, intervention effectiveness, and cost.

**Done when:** a user can understand why a customer is flagged and compare transparent outreach scenarios without confusing correlation with causation.

### Phase 3: Reproducible data and model management

- [ ] Add data-quality checks for schema, missing values, and unexpected categories.
- [ ] Record dataset and training configuration details with every MLflow run.
- [ ] Generate a versioned evaluation report when a model is trained.
- [ ] Add a documented model promotion process so the dashboard loads an explicitly selected model version.

**Done when:** a past model can be tied to its data, configuration, evaluation results, and promotion decision.

### Phase 4: Delivery and monitoring

- [ ] Add CI to run tests and linting on each change.
- [ ] Containerize the dashboard and document local and hosted deployment.
- [ ] Add a prediction API if the dashboard needs to integrate with other systems.
- [ ] Monitor incoming data for drift and evaluate predictions against later-observed churn outcomes.

**Done when:** the project can be deployed repeatably and model health can be reviewed after deployment.

### Current status

- [x] Data ingestion, feature engineering, model comparison, and MLflow tracking.
- [x] Streamlit dashboard with risk filtering, monthly charges at risk, CSV export, and threshold reference metrics.
- [ ] Phase 1: reliable held-out evaluation.

---

## 📊 Dataset Summary

The project uses the **Telco Customer Churn** dataset.
- Total records: **7,043**
- Churn rate: **26.54%**
- Missing values: **0**
- Columns analyzed: **21**

---

## 🧠 Model Performance

The current best-performing model in the tracked MLflow experiment is a **Random Forest classifier**.
- **Best accuracy:** 80.77%
- **Best ROC-AUC:** 86.46%
- **Tracking:** MLflow
- **Selection Criteria:** Highest ROC-AUC

These values are based on the project’s saved MLflow runs in the current workspace.

---

## 🛠️ Tech Stack

- **Data Processing:** Python, Pandas
- **Machine Learning:** scikit-learn, XGBoost, LightGBM
- **Tracking & Deployment:** MLflow, Streamlit

---

## 📁 Project Structure

```text
ml-analytics-dashboard/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── dashboard/
│   ├── ingestion/
│   ├── models/
│   ├── preprocessing/
│   ├── tracking/
│   └── utils/
├── tests/
├── notebooks/
├── mlruns/
├── requirements.txt
├── pyproject.toml
├── README.md
└── PROJECT_EXPLANATION.md
```

---

## 💻 Setup

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Train the Model

To train the baseline and comparison models, run:
```bash
python src/models/train.py
```
This logs the results to MLflow under the `Churn_prediction` experiment.

---

## 📈 Launch the Dashboard

Start the interactive Streamlit application:
```bash
streamlit run src/dashboard/app.py
```
Open the local URL shown in your terminal, typically: `http://localhost:8501`

---

## 🖼️ Screenshots

<div align="center">
  <img src="assets/dashboard-overview.png" alt="Dashboard Overview" width="80%">
  <br>
  <em>Dashboard Overview</em>
</div>
<br>

<div align="center">
  <img src="assets/risk-insights.png" alt="Churn Risk Insights" width="80%">
  <br>
  <em>Churn Risk Insights</em>
</div>
<br>

<div align="center">
  <img src="assets/model-performance.png" alt="Model Performance" width="80%">
  <br>
  <em>Model Performance</em>
</div>

---

## 📝 Notes

- Model selection uses **ROC-AUC** as the primary optimization metric.
- **MLflow** provides experiment reproducibility and run comparison.
- The dashboard is built to be business-readable and suitable for presentation or stakeholder review.

---

## 📄 License

This project is intended for learning and portfolio/demo use.


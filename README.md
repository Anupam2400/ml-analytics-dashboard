# ML-Powered Customer Analytics Dashboard

An end-to-end customer churn analytics project that combines data preparation, model training, experiment tracking, and an interactive dashboard for business decision-making.

This project is designed to help teams understand churn patterns, identify at-risk customers, and monitor model performance in a production-like workflow.

---

## Overview

Customer churn is one of the biggest drivers of lost revenue in subscription and digital services businesses. This project addresses that problem by:

- loading and validating raw customer data
- transforming features for modeling
- training multiple machine learning algorithms
- comparing model performance with MLflow
- deploying a Streamlit dashboard for business-facing insights

---

## Business Value

The dashboard helps teams answer questions like:

- Which customers are most likely to churn?
- What factors contribute most to churn risk?
- How much revenue is exposed to churn risk?
- Which model performs best on the validation set?

---

## Features

- Data ingestion and validation
- Feature engineering and preprocessing
- Baseline and comparison models
- MLflow experiment tracking
- Model selection based on ROC-AUC
- Interactive Streamlit dashboard
- Business insights for churn risk monitoring

---

## Dataset Summary

The project uses the Telco Customer Churn dataset.

- Total records: 7,043
- Churn rate: 26.54%
- Missing values: 0
- Columns analyzed: 21

---

## Model Performance

The current best-performing model in the tracked MLflow experiment is a Random Forest classifier.

- Best accuracy: 80.77%
- Best ROC-AUC: 86.46%
- Experiment tracking: MLflow
- Best run selection: by highest ROC-AUC

These values are based on the project’s saved MLflow runs in the current workspace.

---

## Tech Stack

- Python
- Pandas
- scikit-learn
- XGBoost
- LightGBM
- MLflow
- Streamlit

---

## Project Structure

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

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Train the Model

```bash
python src/models/train.py
```

This trains several models and logs the results to MLflow under the `Churn_prediction` experiment.

---

## Launch the Dashboard

```bash
streamlit run src/dashboard/app.py
```

Then open the local URL shown by Streamlit in the terminal, typically:

```text
http://localhost:8501
```

---

## Screenshots

![Dashboard Overview](assets/dashboard-overview.png)
![Churn Risk Insights](assets/risk-insights.png)
![Model Performance](assets/model-performance.png)

---

## Notes

- Model selection uses ROC-AUC as the primary optimization metric.
- MLflow provides experiment reproducibility and run comparison.
- The dashboard is built to be business-readable and suitable for presentation or stakeholder review.

---

## License

This project is intended for learning and portfolio/demo use.


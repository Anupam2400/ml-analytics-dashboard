# ML-Powered Customer Analytics Dashboard

An end-to-end machine learning and analytics platform that predicts customer churn and visualizes actionable business insights through an interactive dashboard.

This project integrates:
- Data ingestion & validation  
- Feature engineering  
- ML model training & evaluation  
- MLflow experiment tracking  
- Business-oriented analytics dashboard  

It is designed to mimic real-world ML systems used by SaaS and e-commerce companies.

---

## 🚀 Problem Statement

Customer churn is one of the biggest revenue killers for subscription-based and e-commerce businesses.  
However, most analytics tools show **what happened**, not **what will happen**.

This project aims to:
- Predict which customers are likely to churn
- Explain why they are at risk
- Provide decision-makers with a clear, visual dashboard

---

## 🧠 Solution Overview

We build a machine learning pipeline that:
1. Loads and validates customer data  
2. Cleans and transforms features  
3. Trains multiple ML models  
4. Tracks experiments using MLflow  
5. Selects the best model  
6. Serves predictions via a dashboard  

Business users can explore churn risk, model accuracy, and customer insights in real time.

---

## 🏗️ System Architecture

ml-analytics-dashboard/
│
├── data/
│ ├── raw/
│ ├── processed/
│
├── src/
│ ├── ingestion/
│ ├── preprocessing/
│ ├── models/
│ ├── tracking/
│ ├── dashboard/
│ └── utils/
│
├── tests/
├── notebooks/
├── mlruns/
├── requirements.txt
└── README.md

---

## 🤖 Machine Learning Approach

We start with a baseline Logistic Regression model and improve it using:
- Random Forest
- Feature engineering
- Threshold tuning

Metrics tracked:
- Accuracy
- Precision
- Recall
- ROC-AUC

All experiments are logged in MLflow for reproducibility and comparison.

---

## 📊 Dashboard Features

The Streamlit dashboard provides:
- Overall churn rate  
- Revenue at risk  
- Model performance metrics  
- Feature importance  
- High-risk customer list  

This allows business users to make data-driven decisions.

---

## 🧪 How to Run

1. Install dependencies  
```bash

pip install -r requirements.txt

```
2. Train the model
   
   python src/model/train.py

4. start the dashboard
   
   streamlit run src/dashboard/app.py

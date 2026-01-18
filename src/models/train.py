import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from src.ingestion.load_data import load_data
from src.tracking.mlflow_config import setup_mlflow

def train():
    setup_mlflow()
    df = load_data()

    x = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"].map({"Yes": 1, "No": 0})

    x = pd.get_dummies(x, drop_first=True)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        model = LogisticRegression(max_iter=500)
        model.fit(x_train, y_train)

        preds = model.predict(x_test)
        acc = accuracy_score(y_test, preds)
        roc = roc_auc_score(y_test, preds)

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", roc)
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc:.4f}, ROC-AUC: {roc:.4f}")

if __name__ == "__main__":
    train()
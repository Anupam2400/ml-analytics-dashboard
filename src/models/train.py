# import sys
# from pathlib import Path

# import pandas as pd
# import mlflow
# import mlflow.sklearn
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, roc_auc_score

# Ensure project root is on sys.path so the `src` package is importable
# when executing this file directly (e.g. `python src\models\train.py`).
# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

# from src.ingestion.load_data import load_data
# from src.tracking.mlflow_config import setup_mlflow
# from src.preprocessing.features import build_features

# def train():
#     setup_mlflow()
#     df = build_features(load_data())

#     x = df.drop(columns=["Churn", "customerID"])
#     y = df["Churn"].map({"Yes": 1, "No": 0})

#     x = pd.get_dummies(x, drop_first=True)

#     x_train, x_test, y_train, y_test = train_test_split(
#         x, y, test_size=0.2, random_state=42
#     )

#     with mlflow.start_run():
#         model = LogisticRegression(max_iter=500)
#         model.fit(x_train, y_train)

#         preds = model.predict(x_test)
#         acc = accuracy_score(y_test, preds)
#         roc = roc_auc_score(y_test, preds)

#         mlflow.log_param("model", "LogisticRegression")
#         mlflow.log_metric("accuracy", acc)
#         mlflow.log_metric("roc_auc", roc)
#         mlflow.sklearn.log_model(model, "model")

#         print(f"Accuracy: {acc:.4f}, ROC-AUC: {roc:.4f}")

# if __name__ == "__main__":
#     train()

########################### model comparison version ###########################

import pandas as pd
import mlflow
import mlflow.sklearn
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.load_data import load_data
from src.preprocessing.features import build_features, create_preprocessor
from src.tracking.mlflow_config import setup_mlflow
from src.config import (
    LOGISTIC_REGRESSION_PARAMS,
    RANDOM_FOREST_PARAMS,
    XGBOOST_PARAMS,
    LIGHTGBM_PARAMS,
    TEST_SIZE,
    RANDOM_STATE,
)


def train_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    model_name,
    preprocessor=None,
):
    fitted_model = model
    if preprocessor is not None:
        fitted_model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model),
        ])

    with mlflow.start_run(run_name=model_name):
        fitted_model.fit(X_train, y_train)

        preds = fitted_model.predict(X_test)
        proba = fitted_model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        roc = roc_auc_score(y_test, proba)

        mlflow.log_param("model", model_name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", roc)
        mlflow.sklearn.log_model(fitted_model, "model")

        print(f"{model_name} → Accuracy: {acc:.4f}, ROC-AUC: {roc:.4f}")
        return float(acc), float(roc)


def train():
    setup_mlflow()
    df = build_features(load_data())

    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"].map({"Yes": 1, "No": 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 1️⃣ Logistic Regression (baseline)
    train_model(
        LogisticRegression(**LOGISTIC_REGRESSION_PARAMS),
        X_train, X_test, y_train, y_test,
        "LogisticRegression",
        create_preprocessor(),
    )

    # 2️⃣ Random Forest
    train_model(
        RandomForestClassifier(**RANDOM_FOREST_PARAMS),
        X_train, X_test, y_train, y_test,
        "RandomForest",
        create_preprocessor(),
    )

    # 3️⃣ XGBoost
    train_model(
        XGBClassifier(**XGBOOST_PARAMS),
        X_train, X_test, y_train, y_test,
        "XGBoost",
        create_preprocessor(),
    )

    # 4️⃣ LightGBM
    train_model(
        LGBMClassifier(**LIGHTGBM_PARAMS),
        X_train, X_test, y_train, y_test,
        "LightGBM",
        create_preprocessor(),
    )


if __name__ == "__main__":
    train()

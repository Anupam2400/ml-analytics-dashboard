import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/Telco_Cusomer_Churn.csv")

REQUIRED_COLUMNS = [

    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "InternetService",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the Telco Customer Churn dataset from a CSV file.

    Args:
        path (Path): Path to the CSV file containing the dataset.
    """
    df = pd.read_csv(path)

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    
    return df

if __name__ == "__main__":
    df = load_data()
    print(f"Dataset loaded: {df.shape}")


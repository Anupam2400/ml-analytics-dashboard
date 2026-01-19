import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw churn data into ML-ready features
    """
    df = df.copy()

    # Convert TotalCharges to numeric (it comes as string)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing values
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Create business feature: average monthly spend
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)

    # Create tenure bucket
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"]
    )

    print(df)
    return df

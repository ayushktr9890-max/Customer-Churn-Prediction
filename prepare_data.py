"""
prepare_data.py
---------------
Converts the IBM Telco Customer Churn dataset from the extended Excel/CSV format
(with extra columns like City, Zip Code, CLTV, Churn Reason, etc.)
to the standard format expected by this project.

Usage
-----
    python prepare_data.py --input data/your_file.xlsx
    python prepare_data.py --input data/your_file.csv

Output
------
    data/customer_churn.csv  — cleaned, standard-format CSV ready for training
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np


# ── Column mapping: standard column names expected by the project ────────────
STANDARD_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

# ── Possible source column name variants → standard name ─────────────────────
COLUMN_ALIASES = {
    # CustomerID variants
    "CustomerID": "customerID",
    "customer_id": "customerID",
    # Gender variants
    "Gender": "gender",
    # SeniorCitizen variants
    "Senior Citizen": "SeniorCitizen",
    "senior_citizen": "SeniorCitizen",
    # Churn label variants
    "Churn Label": "Churn",
    "churn_label": "Churn",
    "Churn Value": "_ChurnValue",   # keep temporarily for fallback
    # Tenure variants
    "Tenure Months": "tenure",
    "tenure_months": "tenure",
    # Other spacing variants
    "Phone Service": "PhoneService",
    "Multiple Lines": "MultipleLines",
    "Internet Service": "InternetService",
    "Online Security": "OnlineSecurity",
    "Online Backup": "OnlineBackup",
    "Device Protection": "DeviceProtection",
    "Tech Support": "TechSupport",
    "Streaming TV": "StreamingTV",
    "Streaming Movies": "StreamingMovies",
    "Paperless Billing": "PaperlessBilling",
    "Payment Method": "PaymentMethod",
    "Monthly Charges": "MonthlyCharges",
    "Total Charges": "TotalCharges",
}


def load_file(filepath: str) -> pd.DataFrame:
    """Load Excel or CSV file."""
    ext = os.path.splitext(filepath)[1].lower()
    print(f"Loading: {filepath}")
    if ext in (".xlsx", ".xls"):
        # Try reading the first sheet
        df = pd.read_excel(filepath, sheet_name=0)
    elif ext == ".csv":
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .xlsx, .xls, or .csv")
    print(f"  Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to standard names using the alias map."""
    df = df.rename(columns=COLUMN_ALIASES)
    return df


def fix_senior_citizen(df: pd.DataFrame) -> pd.DataFrame:
    """
    SeniorCitizen should be 0/1 integer.
    The extended dataset uses 'Yes'/'No' strings.
    """
    if "SeniorCitizen" in df.columns:
        col = df["SeniorCitizen"].astype(str).str.strip().str.lower()
        if col.isin(["yes", "no"]).any():
            print("  Converting SeniorCitizen from Yes/No to 1/0...")
            df["SeniorCitizen"] = col.map({"yes": 1, "no": 0}).fillna(0).astype(int)
        else:
            df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
    return df


def fix_churn_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure Churn column contains 'Yes'/'No' strings.
    Extended datasets sometimes have 'Churn Label' (Yes/No) or 'Churn Value' (1/0).
    """
    if "Churn" not in df.columns:
        # Try to derive from _ChurnValue
        if "_ChurnValue" in df.columns:
            print("  Deriving Churn from Churn Value column...")
            df["Churn"] = df["_ChurnValue"].map({1: "Yes", 0: "No"}).fillna("No")
        else:
            raise ValueError("Cannot find a Churn column. Expected 'Churn', 'Churn Label', or 'Churn Value'.")
    
    # Normalise to 'Yes'/'No'
    churn_vals = df["Churn"].astype(str).str.strip()
    # Handle numeric 1/0
    if churn_vals.isin(["1", "0"]).any():
        df["Churn"] = churn_vals.map({"1": "Yes", "0": "No"}).fillna("No")
    else:
        df["Churn"] = churn_vals.str.capitalize()

    return df


def select_standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the 21 standard columns, in order."""
    missing = [c for c in STANDARD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns after standardization: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )
    return df[STANDARD_COLUMNS].copy()


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure TotalCharges is numeric (whitespace rows → NaN, which will be dropped)."""
    df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_bad = df["TotalCharges"].isna().sum()
    if n_bad > 0:
        print(f"  Dropping {n_bad} rows with non-numeric TotalCharges (new customers with tenure=0)...")
        df = df.dropna(subset=["TotalCharges"])
    return df


def validate(df: pd.DataFrame) -> None:
    """Basic sanity checks on the prepared dataset."""
    print("\n  Validation checks:")
    print(f"    Rows: {len(df)}")
    print(f"    Churn distribution: {df['Churn'].value_counts().to_dict()}")
    print(f"    SeniorCitizen values: {sorted(df['SeniorCitizen'].unique())}")
    print(f"    NaN values: {df.isnull().sum().sum()}")
    assert df["Churn"].isin(["Yes", "No"]).all(), "Churn column must only contain 'Yes' or 'No'"
    assert df["SeniorCitizen"].isin([0, 1]).all(), "SeniorCitizen must only contain 0 or 1"
    print("    ✅  All checks passed!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert IBM Telco Churn dataset to standard format."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input file (.xlsx, .xls, or .csv)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "data", "customer_churn.csv"),
        help="Output path (default: data/customer_churn.csv)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    print("=" * 60)
    print("  IBM Telco Churn Dataset Converter")
    print("=" * 60)

    # Load
    df = load_file(args.input)
    print(f"\n  Original columns ({len(df.columns)}):")
    for col in df.columns:
        print(f"    - {col}")

    # Process
    print("\nProcessing...")
    df = standardize_columns(df)
    df = fix_senior_citizen(df)
    df = fix_churn_column(df)

    # Drop temporary helper columns
    if "_ChurnValue" in df.columns:
        df = df.drop(columns=["_ChurnValue"])

    df = select_standard_columns(df)
    df = clean_total_charges(df)

    # Validate
    validate(df)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\n✅  Saved standard dataset to: {args.output}")
    print(f"    Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. python src/train.py        — train all models")
    print("  2. python src/predict.py      — interactive prediction")
    print("  3. Open notebooks/            — run Jupyter notebooks")
    print("=" * 60)


if __name__ == "__main__":
    main()

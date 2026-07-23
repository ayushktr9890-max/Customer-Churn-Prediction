"""
preprocessing.py
----------------
All data cleaning, encoding, and feature engineering steps for the
Customer Churn Prediction project.

Each function is self-contained and can be called independently from
notebooks or from train.py / predict.py.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 – Basic data inspection
# ---------------------------------------------------------------------------

def inspect_dataset(df: pd.DataFrame) -> None:
    """
    Print a structured summary of the raw dataset including shape,
    dtypes, missing values, and duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe loaded from CSV.
    """
    print("\n" + "=" * 55)
    print("  DATASET OVERVIEW")
    print("=" * 55)
    print(f"  Shape         : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Duplicate rows: {df.duplicated().sum()}")

    print("\n  Column Summary:")
    print("-" * 55)
    col_info = pd.DataFrame({
        "dtype": df.dtypes,
        "non_null": df.notnull().sum(),
        "null_count": df.isnull().sum(),
        "null_%": (df.isnull().mean() * 100).round(2),
    })
    print(col_info.to_string())
    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# Step 2 – Data cleaning
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform all data-cleaning steps:
      1. Drop the 'customerID' column (not predictive).
      2. Convert 'TotalCharges' from object → float
         (handles whitespace strings by coercing to NaN).
      3. Drop rows where TotalCharges is NaN after conversion
         (these are new customers with 0 tenure — negligible count).
      4. Strip leading/trailing whitespace from string columns.
      5. Remove exact duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    df = df.copy()

    # 1. Drop customer ID — not a predictive feature
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)
        logger.info("Dropped 'customerID' column.")

    # 2. Convert TotalCharges to numeric (it can contain ' ' spaces)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_coerced = df["TotalCharges"].isna().sum()
    if n_coerced > 0:
        logger.info(
            "TotalCharges: coerced %d non-numeric value(s) to NaN.", n_coerced
        )

    # 3. Drop rows with NaN TotalCharges
    df.dropna(subset=["TotalCharges"], inplace=True)
    logger.info("Dropped %d row(s) with NaN TotalCharges.", n_coerced)

    # 4. Strip whitespace from object columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # 5. Remove exact duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    removed = before - len(df)
    if removed:
        logger.info("Removed %d duplicate row(s).", removed)

    logger.info("Cleaning complete — shape after cleaning: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Step 3 – Feature engineering helpers
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional derived features that improve model signal:
      - tenure_group : bucketed tenure (0-12, 13-24, …, 61-72 months)
      - monthly_charges_per_tenure : average monthly spend per month stayed

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with new columns appended.
    """
    df = df.copy()

    # Bucket tenure into 6-month bands
    bins = [0, 12, 24, 36, 48, 60, 72]
    labels = ["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=bins, labels=labels, include_lowest=True
    )

    # Ratio feature: spend efficiency per month of stay
    df["monthly_per_tenure"] = np.where(
        df["tenure"] > 0,
        df["MonthlyCharges"] / df["tenure"],
        df["MonthlyCharges"],
    )
    df["monthly_per_tenure"] = df["monthly_per_tenure"].round(4)

    logger.info("Feature engineering complete — new features: tenure_group, monthly_per_tenure")
    return df


# ---------------------------------------------------------------------------
# Step 4 – Encoding
# ---------------------------------------------------------------------------

# Columns that are binary Yes/No — label-encode directly
_BINARY_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn",
]

# Columns with more than 2 categories — one-hot encode
_OHE_COLS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
    "tenure_group",
]


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Encode all categorical columns:
      - Binary columns (Yes/No, Male/Female) → Label Encoding.
      - Multi-class columns → One-Hot Encoding (drop_first=True to avoid
        multicollinearity).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned (and optionally feature-engineered) dataframe.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        - encoded dataframe
        - encoding_info dict with label encoder objects keyed by column name
          (useful for inverse-transforming in the prediction script).
    """
    df = df.copy()
    encoding_info: dict = {"label_encoders": {}, "ohe_columns": []}

    # --- Label Encoding ---
    le = LabelEncoder()
    for col in _BINARY_COLS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
            encoding_info["label_encoders"][col] = le
            logger.info("Label-encoded: %s", col)

    # --- One-Hot Encoding ---
    ohe_present = [c for c in _OHE_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=ohe_present, drop_first=True)
    encoding_info["ohe_columns"] = [c for c in df.columns if c not in df.select_dtypes(include="number").columns]
    logger.info("One-hot encoded: %s", ohe_present)
    logger.info("Encoded shape: %s", df.shape)

    return df, encoding_info


# ---------------------------------------------------------------------------
# Step 5 – Train / test split + scaling
# ---------------------------------------------------------------------------

def split_and_scale(
    df: pd.DataFrame,
    target: str = "Churn",
    test_size: float = 0.20,
    random_state: int = 42,
    scale: bool = True,
) -> tuple:
    """
    Split the encoded dataframe into train/test sets and optionally
    apply StandardScaler to numeric features.

    Parameters
    ----------
    df : pd.DataFrame
        Fully encoded dataframe.
    target : str
        Name of the target column.
    test_size : float
        Fraction of data reserved for testing (default 0.20 → 80:20 split).
    random_state : int
        Reproducibility seed.
    scale : bool
        If True, apply StandardScaler to all feature columns.

    Returns
    -------
    tuple : (X_train, X_test, y_train, y_test, scaler, feature_names)
        scaler is None if scale=False.
        feature_names is the list of column names used as features.
    """
    X = df.drop(columns=[target])
    y = df[target]

    feature_names = list(X.columns)

    # Convert boolean columns produced by get_dummies to int
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        logger.info("Applied StandardScaler to features.")

    logger.info(
        "Train/test split — Train: %d rows | Test: %d rows (%.0f%% / %.0f%%)",
        len(y_train),
        len(y_test),
        (1 - test_size) * 100,
        test_size * 100,
    )

    return X_train, X_test, y_train, y_test, scaler, feature_names


# ---------------------------------------------------------------------------
# Full pipeline convenience wrapper
# ---------------------------------------------------------------------------

def run_preprocessing_pipeline(
    df: pd.DataFrame,
    target: str = "Churn",
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple:
    """
    Run the complete preprocessing pipeline end-to-end:
        clean → engineer features → encode → split & scale.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe loaded from CSV.
    target : str
    test_size : float
    random_state : int

    Returns
    -------
    tuple : (X_train, X_test, y_train, y_test, scaler, feature_names, encoding_info)
    """
    logger.info("Starting preprocessing pipeline...")
    df_clean = clean_data(df)
    df_feat = engineer_features(df_clean)
    df_enc, encoding_info = encode_features(df_feat)
    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale(
        df_enc, target=target, test_size=test_size, random_state=random_state
    )
    logger.info("Preprocessing pipeline complete.")
    return X_train, X_test, y_train, y_test, scaler, feature_names, encoding_info

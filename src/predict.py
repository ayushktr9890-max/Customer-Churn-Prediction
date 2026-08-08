"""
predict.py
----------
Interactive terminal-based prediction script for the Customer Churn model.

The script prompts the user for customer details one-by-one, preprocesses
the input the same way the training pipeline does, runs the saved model,
and prints:
  - Predicted Churn: Yes / No
  - Probability of Churn: e.g. 84%
  - Top reasons influencing the prediction
  - Suggested retention actions

Usage
-----
    python src/predict.py
    python src/predict.py --model model/churn_model.pkl   (custom path)
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd

# Ensure src/ is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_model, logger

# Optional SHAP for local explanations
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "model", "churn_model.pkl")

# The trained dataset stores charges in its original currency scale. Users enter ₹.
INR_PER_MODEL_CURRENCY = 83.0

logging.basicConfig(
    level=logging.WARNING,   # keep output clean for interactive use
    format="%(levelname)s  %(message)s",
)


# ---------------------------------------------------------------------------
# Question definitions — mirroring the original dataset columns
# ---------------------------------------------------------------------------

QUESTIONS = [
    {
        "key": "SeniorCitizen",
        "prompt": "Is the customer a senior citizen? (0 = No, 1 = Yes)",
        "type": "int",
        "valid": ["0", "1"],
    },
    {
        "key": "gender",
        "prompt": "Customer gender (Male / Female)",
        "type": "str",
        "valid": ["Male", "Female"],
    },
    {
        "key": "Partner",
        "prompt": "Does the customer have a partner? (Yes / No)",
        "type": "str",
        "valid": ["Yes", "No"],
    },
    {
        "key": "Dependents",
        "prompt": "Does the customer have dependents? (Yes / No)",
        "type": "str",
        "valid": ["Yes", "No"],
    },
    {
        "key": "tenure",
        "prompt": "How many months has the customer been with the company? (0–72)",
        "type": "float",
        "range": (0, 72),
    },
    {
        "key": "PhoneService",
        "prompt": "Does the customer have phone service? (Yes / No)",
        "type": "str",
        "valid": ["Yes", "No"],
    },
    {
        "key": "MultipleLines",
        "prompt": "Multiple phone lines? (No phone service / No / Yes)",
        "type": "str",
        "valid": ["No phone service", "No", "Yes"],
    },
    {
        "key": "InternetService",
        "prompt": "Internet service provider (DSL / Fiber optic / No)",
        "type": "str",
        "valid": ["DSL", "Fiber optic", "No"],
    },
    {
        "key": "OnlineSecurity",
        "prompt": "Online security add-on? (No internet service / No / Yes)",
        "type": "str",
        "valid": ["No internet service", "No", "Yes"],
    },
    {
        "key": "OnlineBackup",
        "prompt": "Online backup add-on? (No internet service / No / Yes)",
        "type": "str",
        "valid": ["No internet service", "No", "Yes"],
    },
    {
        "key": "DeviceProtection",
        "prompt": "Device protection add-on? (No internet service / No / Yes)",
        "type": "str",
        "valid": ["No internet service", "No", "Yes"],
    },
    {
        "key": "TechSupport",
        "prompt": "Tech support add-on? (No internet service / No / Yes)",
        "type": "str",
        "valid": ["No internet service", "No", "Yes"],
    },
    {
        "key": "StreamingTV",
        "prompt": "Streaming TV add-on? (No internet service / No / Yes)",
        "type": "str",
        "valid": ["No internet service", "No", "Yes"],
    },
    {
        "key": "StreamingMovies",
        "prompt": "Streaming movies add-on? (No internet service / No / Yes)",
        "type": "str",
        "valid": ["No internet service", "No", "Yes"],
    },
    {
        "key": "Contract",
        "prompt": "Contract type (Month-to-month / One year / Two year)",
        "type": "str",
        "valid": ["Month-to-month", "One year", "Two year"],
    },
    {
        "key": "PaperlessBilling",
        "prompt": "Paperless billing? (Yes / No)",
        "type": "str",
        "valid": ["Yes", "No"],
    },
    {
        "key": "PaymentMethod",
        "prompt": (
            "Payment method\n"
            "  Options: Electronic check / Mailed check / "
            "Bank transfer (automatic) / Credit card (automatic)"
        ),
        "type": "str",
        "valid": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    },
    {
        "key": "MonthlyCharges",
        "prompt": "Monthly charges in ₹ (e.g. 5,000)",
        "type": "float",
        "range": (0, 200 * INR_PER_MODEL_CURRENCY),
    },
    {
        "key": "TotalCharges",
        "prompt": "Total charges to date in ₹ (e.g. 60,000)",
        "type": "float",
        "range": (0, 10_000 * INR_PER_MODEL_CURRENCY),
    },
]


# ---------------------------------------------------------------------------
# Input collection
# ---------------------------------------------------------------------------

def _ask(question: dict) -> object:
    """
    Prompt the user for a single field with basic validation and retry.

    Parameters
    ----------
    question : dict
        One entry from QUESTIONS.

    Returns
    -------
    Validated user input cast to the correct type.
    """
    prompt_text = f"\n  {question['prompt']}: "
    while True:
        try:
            raw = input(prompt_text).strip()

            if question["type"] == "int":
                value = int(raw)
                if "valid" in question and str(value) not in question["valid"]:
                    raise ValueError
                return value

            elif question["type"] == "float":
                value = float(raw)
                lo, hi = question.get("range", (-1e9, 1e9))
                if not (lo <= value <= hi):
                    print(f"    ⚠  Please enter a value between {lo} and {hi}.")
                    continue
                return value

            else:  # str
                if "valid" in question:
                    # Case-insensitive matching
                    matches = [v for v in question["valid"] if v.lower() == raw.lower()]
                    if not matches:
                        print(f"    ⚠  Valid options: {question['valid']}")
                        continue
                    return matches[0]   # return the properly-cased version
                return raw

        except (ValueError, KeyboardInterrupt):
            print("    ⚠  Invalid input. Please try again.")


def collect_customer_data() -> dict:
    """
    Walk through all QUESTIONS and return a dict of raw feature values.

    Returns
    -------
    dict
        {column_name: value}  — mirrors the original dataset columns.
    """
    print("\n" + "=" * 60)
    print("  CUSTOMER CHURN PREDICTION — Enter Customer Details")
    print("=" * 60)
    print("  (Press Ctrl+C at any time to exit)\n")

    data = {}
    try:
        for q in QUESTIONS:
            data[q["key"]] = _ask(q)
    except KeyboardInterrupt:
        print("\n\n  Prediction cancelled by user.")
        sys.exit(0)

    return data


# ---------------------------------------------------------------------------
# Preprocessing for a single row
# ---------------------------------------------------------------------------

def preprocess_input(raw: dict, feature_names: list, scaler) -> np.ndarray:
    """
    Convert raw user input dict into a single feature vector that matches
    the training pipeline's output (same columns, same encoding, same scale).

    The logic mirrors preprocessing.py but is applied row-by-row.

    Parameters
    ----------
    raw : dict
        Raw values collected from the user.
    feature_names : list of str
        Ordered list of column names expected by the model.
    scaler : StandardScaler or None

    Returns
    -------
    np.ndarray
        Shape (1, n_features) — ready for model.predict().
    """
    df = pd.DataFrame([raw])

    # Convert rupees to the scale used when the model was trained.
    df[["MonthlyCharges", "TotalCharges"]] /= INR_PER_MODEL_CURRENCY

    # --- Engineer the same features as training ---
    bins = [0, 12, 24, 36, 48, 60, 72]
    labels = ["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=bins, labels=labels, include_lowest=True
    )
    tenure_val = float(df["tenure"].iloc[0])
    df["monthly_per_tenure"] = (
        df["MonthlyCharges"] / tenure_val
        if tenure_val > 0
        else df["MonthlyCharges"]
    )

    # --- Label encode binary columns (same as training) ---
    binary_map = {
        "gender": {"Male": 1, "Female": 0},
        "Partner": {"Yes": 1, "No": 0},
        "Dependents": {"Yes": 1, "No": 0},
        "PhoneService": {"Yes": 1, "No": 0},
        "PaperlessBilling": {"Yes": 1, "No": 0},
    }
    for col, mapping in binary_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)

    # --- One-hot encode multi-class columns ---
    ohe_cols = [
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
    df = pd.get_dummies(df, columns=[c for c in ohe_cols if c in df.columns], drop_first=True)

    # Convert booleans to int
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Keep exactly the columns, order, and default values used during training.
    df = df.reindex(columns=feature_names, fill_value=0)

    # Keep column names while scaling to match the training data exactly.
    X = df.astype(float)

    if scaler is not None:
        X = scaler.transform(X)

    return X


# ---------------------------------------------------------------------------
# Retention recommendations
# ---------------------------------------------------------------------------

def get_retention_actions(raw: dict, churn_prob: float) -> list:
    """
    Generate personalised retention recommendations based on the customer's
    profile and churn probability.

    Parameters
    ----------
    raw : dict
        Raw customer input data.
    churn_prob : float
        Predicted probability of churn (0–1).

    Returns
    -------
    list of str
        Ordered list of suggested actions.
    """
    actions = []

    if raw.get("Contract") == "Month-to-month":
        actions.append("📋  Offer a discounted annual or two-year contract.")

    if raw.get("InternetService") == "Fiber optic" and raw.get("TechSupport") == "No":
        actions.append("🔧  Recommend a TechSupport package — Fiber optic users value reliability.")

    if raw.get("OnlineSecurity") == "No":
        actions.append("🔒  Bundle Online Security add-on at a reduced rate.")

    if float(raw.get("tenure", 0)) < 12:
        actions.append("🎁  Enrol customer in a loyalty rewards programme (early tenure).")

    if float(raw.get("MonthlyCharges", 0)) > 70 * INR_PER_MODEL_CURRENCY:
        actions.append("💰  Offer a personalised discount or a lower-tier plan review.")

    if raw.get("PaymentMethod") == "Electronic check":
        actions.append("🏦  Incentivise switch to automatic bank transfer (reduces churn risk).")

    if raw.get("Dependents") == "No" and raw.get("Partner") == "No":
        actions.append("👥  Promote family/companion plan bundles.")

    if churn_prob > 0.75:
        actions.append("📞  Priority: Assign a dedicated customer success manager immediately.")

    if not actions:
        actions.append("✅  Customer profile looks healthy — maintain standard engagement.")

    return actions


# ---------------------------------------------------------------------------
# Local SHAP-based explanations (top contributing features)
# ---------------------------------------------------------------------------

def get_top_reasons(
    model,
    X_input: np.ndarray,
    feature_names: list,
    top_n: int = 5,
) -> list:
    """
    Return the top_n features most responsible for this prediction.
    Uses SHAP values if available, otherwise falls back to feature importances.

    Parameters
    ----------
    model : trained estimator
    X_input : np.ndarray, shape (1, n_features)
    feature_names : list
    top_n : int

    Returns
    -------
    list of str
        Human-readable feature influence statements.
    """
    reasons = []

    if SHAP_AVAILABLE:
        try:
            model_name = type(model).__name__
            if model_name in ("RandomForestClassifier", "DecisionTreeClassifier", "XGBClassifier"):
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_input)
                if isinstance(shap_vals, list):
                    vals = shap_vals[1][0]
                else:
                    vals = shap_vals[0]
            else:
                background = np.zeros((1, X_input.shape[1]))
                explainer = shap.KernelExplainer(model.predict_proba, background)
                shap_vals = explainer.shap_values(X_input)
                vals = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]

            top_indices = np.argsort(np.abs(vals))[::-1][:top_n]
            for i in top_indices:
                direction = "↑ increases" if vals[i] > 0 else "↓ decreases"
                reasons.append(
                    f"  • {feature_names[i]:<35}  {direction} churn risk  (SHAP: {vals[i]:+.3f})"
                )
            return reasons
        except Exception as exc:
            logger.debug("SHAP explanation failed: %s", exc)

    # Fallback: model feature importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return ["  • Feature importance not available for this model type."]

    top_indices = np.argsort(importances)[::-1][:top_n]
    for i in top_indices:
        reasons.append(
            f"  • {feature_names[i]:<35}  importance: {importances[i]:.4f}"
        )
    return reasons


# ---------------------------------------------------------------------------
# Main prediction flow
# ---------------------------------------------------------------------------

def run_prediction(model_path: str = DEFAULT_MODEL_PATH) -> None:
    """
    End-to-end interactive prediction session.

    Parameters
    ----------
    model_path : str
        Path to the saved model bundle (.pkl).
    """
    # Load saved model bundle
    bundle = load_model(model_path)
    model = bundle["model"]
    model_name = bundle["model_name"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]

    while True:
        # Collect customer info
        raw = collect_customer_data()

        # Preprocess
        try:
            X_input = preprocess_input(raw, feature_names, scaler)
        except Exception as exc:
            print(f"\n  ⚠  Preprocessing error: {exc}")
            continue

        # Predict
        y_pred = model.predict(X_input)[0]
        churn_label = "Yes" if y_pred == 1 else "No"

        if hasattr(model, "predict_proba"):
            churn_prob = model.predict_proba(X_input)[0][1]
        else:
            churn_prob = float(y_pred)

        # ── Output ──────────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("  PREDICTION RESULT")
        print("=" * 60)
        print(f"  Predicted Churn    : {'🔴  Yes — Customer likely to leave' if churn_label == 'Yes' else '🟢  No  — Customer likely to stay'}")
        print(f"  Probability of Churn: {churn_prob * 100:.1f}%")

        # Risk tier
        if churn_prob >= 0.75:
            risk = "🔴  HIGH RISK"
        elif churn_prob >= 0.45:
            risk = "🟡  MEDIUM RISK"
        else:
            risk = "🟢  LOW RISK"
        print(f"  Risk Level         : {risk}")

        # Top contributing features
        print("\n  TOP FACTORS INFLUENCING THIS PREDICTION:")
        print("-" * 60)
        reasons = get_top_reasons(model, X_input, feature_names, top_n=5)
        for r in reasons:
            print(r)

        # Retention actions
        actions = get_retention_actions(raw, churn_prob)
        print("\n  SUGGESTED RETENTION ACTIONS:")
        print("-" * 60)
        for a in actions:
            print(f"  {a}")

        print("=" * 60)

        # Ask to predict another customer
        again = input("\n  Predict another customer? (yes / no): ").strip().lower()
        if again not in ("yes", "y"):
            print("\n  Thank you for using the Churn Prediction System. Goodbye!\n")
            break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Customer Churn Prediction — Interactive Terminal Script"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the saved model .pkl file (default: model/churn_model.pkl)",
    )
    args = parser.parse_args()
    run_prediction(model_path=args.model)

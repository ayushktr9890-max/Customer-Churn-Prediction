"""
train.py
--------
Train multiple classification models on the Telco Customer Churn dataset,
compare their performance, select the best model, and save it to disk.

Usage
-----
    python src/train.py

Outputs
-------
    model/churn_model.pkl        — best trained model
    outputs/reports/metrics.csv — metrics comparison table
    outputs/plots/               — confusion matrix, ROC curve, feature importance
"""

import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving plots

# Ensure src/ is on the path when running as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from utils import (
    load_dataset,
    save_model,
    evaluate_model,
    print_metrics_table,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    ensure_dir,
)
from preprocessing import run_preprocessing_pipeline

# Optional XGBoost — skip gracefully if not installed
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not installed — skipping XGBClassifier.")

# Optional SHAP — skip gracefully if not installed
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not installed — skipping SHAP analysis.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "customer_churn.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "churn_model.pkl")
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_models() -> dict:
    """
    Return a dictionary of {name: estimator} for all models to train.

    Random seed is fixed for reproducibility.
    Class-weight='balanced' helps with the class imbalance (~73% No / ~27% Yes).
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
            solver="lbfgs",
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=10,
            random_state=42,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=3,    # compensate for class imbalance
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )

    return models


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_all_models(
    models: dict,
    X_train,
    X_test,
    y_train,
    y_test,
) -> tuple[dict, list]:
    """
    Train every model in the dict and collect evaluation metrics.

    Parameters
    ----------
    models : dict
        {name: estimator}
    X_train, X_test, y_train, y_test : arrays

    Returns
    -------
    tuple : (trained_models dict, metrics list)
    """
    trained = {}
    all_metrics = []

    for name, model in models.items():
        logger.info("Training: %s ...", name)
        try:
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test, model_name=name)
            trained[name] = model
            all_metrics.append(metrics)
        except Exception as exc:
            logger.error("Error training %s: %s", name, exc)

    return trained, all_metrics


# ---------------------------------------------------------------------------
# Best model selection
# ---------------------------------------------------------------------------

def select_best_model(metrics: list, trained_models: dict, metric: str = "roc_auc"):
    """
    Select the model with the highest value of the chosen metric.

    Parameters
    ----------
    metrics : list of dict
    trained_models : dict
    metric : str
        Metric to rank by. Default is ROC-AUC.

    Returns
    -------
    tuple : (best_name, best_model)
    """
    valid = [m for m in metrics if isinstance(m[metric], (int, float))]
    best = max(valid, key=lambda m: m[metric])
    best_name = best["model"]
    logger.info(
        "Best model: %s  (ROC-AUC = %.4f)", best_name, best[metric]
    )
    return best_name, trained_models[best_name]


# ---------------------------------------------------------------------------
# SHAP analysis
# ---------------------------------------------------------------------------

def run_shap_analysis(model, X_test, feature_names: list, save_dir: str) -> None:
    """
    Generate SHAP summary plots for the best model.

    Parameters
    ----------
    model : trained estimator
    X_test : array-like
    feature_names : list of str
    save_dir : str
        Directory to save the SHAP plots.
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available — skipping.")
        return

    logger.info("Running SHAP analysis...")
    try:
        ensure_dir(save_dir)

        # Tree-based models use TreeExplainer; others use KernelExplainer
        model_name = type(model).__name__
        if model_name in ("RandomForestClassifier", "DecisionTreeClassifier", "XGBClassifier"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            # For binary classification RF/DT, shap_values is a list [class0, class1]
            if isinstance(shap_values, list):
                shap_vals = shap_values[1]
            else:
                shap_vals = shap_values
        else:
            # Use a small sample for KernelExplainer (computationally heavy)
            X_test_sample = X_test[:200]
            background = shap.sample(X_test, 100)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_test_sample)
            shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values
            X_test = X_test_sample  # align X_test to the sample used

        X_test_df = pd.DataFrame(X_test, columns=feature_names)

        # Summary bar plot
        plt.figure()
        shap.summary_plot(shap_vals, X_test_df, plot_type="bar", show=False)
        bar_path = os.path.join(save_dir, "shap_summary_bar.png")
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("SHAP bar plot saved: %s", bar_path)

        # Summary beeswarm plot
        plt.figure()
        shap.summary_plot(shap_vals, X_test_df, show=False)
        beeswarm_path = os.path.join(save_dir, "shap_summary_beeswarm.png")
        plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("SHAP beeswarm plot saved: %s", beeswarm_path)

    except Exception as exc:
        logger.error("SHAP analysis failed: %s", exc)


# ---------------------------------------------------------------------------
# Save metrics report
# ---------------------------------------------------------------------------

def save_metrics_report(metrics: list, save_path: str) -> None:
    """
    Save the model comparison table as a CSV file.

    Parameters
    ----------
    metrics : list of dict
    save_path : str
    """
    ensure_dir(os.path.dirname(save_path))
    df = pd.DataFrame(metrics)
    df.to_csv(save_path, index=False)
    logger.info("Metrics report saved to: %s", save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ensure_dir(PLOTS_DIR)
    ensure_dir(REPORTS_DIR)
    ensure_dir(os.path.dirname(MODEL_PATH))

    # ── Load data ──────────────────────────────────────────────────────────
    df = load_dataset(DATA_PATH)

    # ── Preprocessing ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, scaler, feature_names, encoding_info = (
        run_preprocessing_pipeline(df)
    )

    # ── Train models ──────────────────────────────────────────────────────
    models = get_models()
    trained_models, all_metrics = train_all_models(
        models, X_train, X_test, y_train, y_test
    )

    # ── Print comparison table ────────────────────────────────────────────
    print_metrics_table(all_metrics)

    # ── Save metrics CSV ──────────────────────────────────────────────────
    save_metrics_report(
        all_metrics,
        os.path.join(REPORTS_DIR, "model_metrics.csv"),
    )

    # ── Select best model ─────────────────────────────────────────────────
    best_name, best_model = select_best_model(all_metrics, trained_models)

    # ── Evaluation plots for best model ───────────────────────────────────
    y_pred = best_model.predict(X_test)

    plot_confusion_matrix(
        y_test,
        y_pred,
        save_path=os.path.join(PLOTS_DIR, "confusion_matrix.png"),
        title=f"Confusion Matrix — {best_name}",
    )

    plot_roc_curve(
        best_model,
        X_test,
        y_test,
        model_name=best_name,
        save_path=os.path.join(PLOTS_DIR, "roc_curve.png"),
    )

    # ── Feature importance ────────────────────────────────────────────────
    importances = None
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_[0])

    if importances is not None:
        plot_feature_importance(
            feature_names,
            importances,
            top_n=15,
            save_path=os.path.join(PLOTS_DIR, "feature_importance.png"),
            title=f"Feature Importance — {best_name}",
        )

    # ── SHAP analysis ─────────────────────────────────────────────────────
    run_shap_analysis(best_model, X_test, feature_names, PLOTS_DIR)

    # ── Save the best model + metadata ────────────────────────────────────
    model_bundle = {
        "model": best_model,
        "model_name": best_name,
        "scaler": scaler,
        "feature_names": feature_names,
        "encoding_info": encoding_info,
    }
    save_model(model_bundle, MODEL_PATH)

    print(f"\n✅  Training complete. Best model: {best_name}")
    print(f"    Model saved to: {MODEL_PATH}")
    print(f"    Plots saved to: {PLOTS_DIR}")
    print(f"    Report saved to: {REPORTS_DIR}/model_metrics.csv\n")


if __name__ == "__main__":
    main()

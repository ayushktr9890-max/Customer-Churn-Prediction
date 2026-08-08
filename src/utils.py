"""
utils.py
--------
Utility functions shared across the Customer Churn Prediction project.
Covers file I/O, metric display, and model persistence helpers.
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load a CSV dataset from the given filepath.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'.\n"
            "Place your dataset in the project root as customer_churn.csv"
        )
    logger.info("Loading dataset from: %s", filepath)
    df = pd.read_csv(filepath)
    logger.info("Dataset loaded successfully — shape: %s", df.shape)
    return df


def save_model(model, filepath: str) -> None:
    """
    Serialize and save a trained model using pickle.

    Parameters
    ----------
    model : sklearn estimator
        Any trained sklearn-compatible model.
    filepath : str
        Destination path for the .pkl file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved to: %s", filepath)


def load_model(filepath: str):
    """
    Load a pickled model from disk.

    Parameters
    ----------
    filepath : str
        Path to the .pkl file.

    Returns
    -------
    Trained model object.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Model not found at '{filepath}'.\n"
            "Please run src/train.py first to train and save the model."
        )
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded from: %s", filepath)
    return model


def ensure_dir(path: str) -> None:
    """Create directory (and parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
) -> dict:
    """
    Compute a full set of classification metrics for a trained model.

    Parameters
    ----------
    model : sklearn estimator
        Trained model with predict / predict_proba interface.
    X_test : array-like
        Test feature matrix.
    y_test : array-like
        True binary labels.
    model_name : str
        Human-readable name used in log output.

    Returns
    -------
    dict
        Keys: accuracy, precision, recall, f1, roc_auc.
    """
    y_pred = model.predict(X_test)

    # ROC-AUC needs probability scores
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = float("nan")

    metrics = {
        "model": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc, 4) if not np.isnan(roc_auc) else "N/A",
    }

    logger.info(
        "%s — Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f | ROC-AUC: %s",
        model_name,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["roc_auc"],
    )
    return metrics


def print_metrics_table(results: list) -> None:
    """
    Print a formatted comparison table of model metrics.

    Parameters
    ----------
    results : list of dict
        Each dict must have keys: model, accuracy, precision, recall, f1, roc_auc.
    """
    df = pd.DataFrame(results).set_index("model")
    print("\n" + "=" * 65)
    print("           MODEL PERFORMANCE COMPARISON")
    print("=" * 65)
    print(df.to_string())
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = None,
    title: str = "Confusion Matrix",
) -> None:
    """
    Plot and optionally save a confusion matrix heatmap.

    Parameters
    ----------
    y_test : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    save_path : str, optional
        If provided, saves the figure to this path.
    title : str
        Plot title.
    """
    matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved to: %s", save_path)
    plt.show()
    plt.close()


def plot_roc_curve(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    save_path: str = None,
) -> None:
    """
    Plot and optionally save the ROC curve for a binary classifier.

    Parameters
    ----------
    model : sklearn estimator
    X_test : array-like
    y_test : array-like
    model_name : str
    save_path : str, optional
    """
    if not hasattr(model, "predict_proba"):
        logger.warning("Model does not support predict_proba — skipping ROC curve.")
        return

    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curve — {model_name}", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("ROC curve saved to: %s", save_path)
    plt.show()
    plt.close()


def plot_feature_importance(
    feature_names: list,
    importances: np.ndarray,
    top_n: int = 15,
    save_path: str = None,
    title: str = "Feature Importance",
) -> None:
    """
    Plot a horizontal bar chart of feature importances.

    Parameters
    ----------
    feature_names : list of str
    importances : array-like of float
    top_n : int
        Number of top features to display.
    save_path : str, optional
    title : str
    """
    top_indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = importances[top_indices]

    plt.figure(figsize=(10, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))  # type: ignore[attr-defined]
    plt.barh(range(top_n), top_importances[::-1], color=colors[::-1])
    plt.yticks(range(top_n), top_features[::-1], fontsize=10)
    plt.xlabel("Importance Score", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Feature importance plot saved to: %s", save_path)
    plt.show()
    plt.close()

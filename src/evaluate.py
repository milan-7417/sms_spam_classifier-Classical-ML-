"""
Model evaluation utilities for SMS Spam Classification.

Computes comprehensive classification metrics, confusion matrices,
model comparison summaries, and feature importance analyses.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Classifier",
    pos_label: int = 1,
) -> Dict[str, Any]:
    """
    Calculate thorough classification metrics with special attention to the spam class.

    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth binary labels (0 = ham, 1 = spam).
    y_pred : np.ndarray
        Predicted binary labels.
    model_name : str
        Human-readable name of the evaluated model.
    pos_label : int
        Label considered positive (default 1 for spam).

    Returns:
    --------
    Dict containing accuracy, spam precision, spam recall, spam f1, confusion matrix,
    and detailed classification reports.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=["ham", "spam"],
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=["ham", "spam"],
        zero_division=0,
    )

    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "raw": [[tn, fp], [fn, tp]],
        },
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
    }


def format_comparison_table(results: List[Dict[str, Any]]) -> str:
    """Format a clean markdown/text table comparing all evaluated models."""
    header = (
        f"{'Model':<25} | {'Accuracy':<10} | {'Precision (Spam)':<18} | "
        f"{'Recall (Spam)':<15} | {'F1 (Spam)':<10}"
    )
    divider = "-" * len(header)
    rows = [divider, header, divider]

    for r in results:
        rows.append(
            f"{r['model_name']:<25} | "
            f"{r['accuracy']:<10.4f} | "
            f"{r['precision']:<18.4f} | "
            f"{r['recall']:<15.4f} | "
            f"{r['f1_score']:<10.4f}"
        )
    rows.append(divider)
    return "\n".join(rows)


def get_top_features(
    model: Any,
    vectorizer: Any,
    top_n: int = 15,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """
    Extract the most informative features (words/n-grams) for spam and ham.

    Parameters:
    -----------
    model : trained scikit-learn classifier
    vectorizer : fitted TfidfVectorizer
    top_n : number of top features to return

    Returns:
    --------
    (top_spam_features, top_ham_features) as lists of (feature_name, weight_or_log_ratio)
    """
    feature_names = np.array(vectorizer.get_feature_names_out())

    # Check if linear model with coef_ (e.g., LogisticRegression, LinearSVC)
    if hasattr(model, "coef_"):
        coef = model.coef_[0]
        top_spam_idx = np.argsort(coef)[-top_n:][::-1]
        top_ham_idx = np.argsort(coef)[:top_n]

        top_spam = [(feature_names[i], round(float(coef[i]), 4)) for i in top_spam_idx]
        top_ham = [(feature_names[i], round(float(coef[i]), 4)) for i in top_ham_idx]
        return top_spam, top_ham

    # For Naive Bayes (MultinomialNB)
    if hasattr(model, "feature_log_prob_"):
        # Log-ratio of spam probability over ham probability
        log_prob_ratio = model.feature_log_prob_[1] - model.feature_log_prob_[0]
        top_spam_idx = np.argsort(log_prob_ratio)[-top_n:][::-1]
        top_ham_idx = np.argsort(log_prob_ratio)[:top_n]

        top_spam = [(feature_names[i], round(float(log_prob_ratio[i]), 4)) for i in top_spam_idx]
        top_ham = [(feature_names[i], round(float(log_prob_ratio[i]), 4)) for i in top_ham_idx]
        return top_spam, top_ham

    return [], []

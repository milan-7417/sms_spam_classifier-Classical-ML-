"""
Prediction module for SMS Spam Classification.

Loads saved classical ML model artifacts and provides clean, robust inference
with proper handling of probabilities vs decision margin scores.
"""

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib

from src.preprocessing import clean_text


def load_model_and_vectorizer(
    models_dir: Optional[Path] = None,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Load the trained classifier, TF-IDF vectorizer, and metadata metrics.

    Parameters:
    -----------
    models_dir : Optional[Path]
        Directory where artifacts are stored. Defaults to project_root / 'models'.

    Returns:
    --------
    (classifier, vectorizer, metadata)

    Raises:
    -------
    FileNotFoundError: If the model or vectorizer joblib files do not exist.
    """
    if models_dir is None:
        models_dir = Path(__file__).resolve().parent.parent / "models"

    model_path = models_dir / "spam_classifier.joblib"
    vectorizer_path = models_dir / "tfidf_vectorizer.joblib"
    metrics_path = models_dir / "metrics.json"

    if not model_path.exists() or not vectorizer_path.exists():
        raise FileNotFoundError(
            "Trained model artifacts not found! "
            f"Missing '{model_path.name}' or '{vectorizer_path.name}' in {models_dir}. "
            "Please run the training pipeline first: 'python src/train.py'."
        )

    classifier = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    metadata: Dict[str, Any] = {}
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    return classifier, vectorizer, metadata


def predict_message(
    message: str,
    classifier: Any,
    vectorizer: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Validate, preprocess, and classify a single SMS message.

    Parameters:
    -----------
    message : str
        The raw SMS message text input.
    classifier : scikit-learn classifier
        Trained classifier (LinearSVC, MultinomialNB, or LogisticRegression).
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer.
    metadata : Optional[Dict[str, Any]]
        Optional training metadata containing model name and metrics.

    Returns:
    --------
    Dict containing prediction ('spam'|'ham'), label ('SPAM'|'HAM'), risk indicator,
    confidence or decision score, model details, and explanation.
    """
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Please provide a valid, non-empty SMS message to analyze.")

    # Guard against excessively long payload (SMS abuse / DoS protection)
    max_chars = 1600
    if len(message) > max_chars:
        raise ValueError(
            f"Message exceeds maximum allowed length of {max_chars} characters (received {len(message)})."
        )

    # Clean and preprocess message
    cleaned = clean_text(message)
    if not cleaned.strip():
        # Input consisted only of stripped characters/symbols
        cleaned = message.strip().lower()

    # Transform text using fitted TF-IDF vectorizer
    X_vec = vectorizer.transform([cleaned])

    # Run prediction (0 = ham, 1 = spam)
    pred_idx = int(classifier.predict(X_vec)[0])
    is_spam = pred_idx == 1

    prediction = "spam" if is_spam else "ham"
    label = "SPAM" if is_spam else "HAM"
    risk = "High Risk" if is_spam else "Low Risk"

    model_name = (
        metadata.get("best_model_name", classifier.__class__.__name__)
        if metadata
        else classifier.__class__.__name__
    )

    # Determine confidence or decision score
    # Note: LinearSVC does not support predict_proba(). We handle this explicitly
    # and accurately without displaying fabricated probabilities.
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(X_vec)[0]
        confidence_val = round(float(probs[pred_idx]), 4)
        score_type = "probability"
        score_label = "Confidence"
        score_display = f"{confidence_val * 100:.1f}%"
        decision_score_val = None
    elif hasattr(classifier, "decision_function"):
        margin_score = float(classifier.decision_function(X_vec)[0])
        score_type = "decision_score"
        score_label = "Decision Margin Score"
        # Signed distance from separating hyperplane
        decision_score_val = round(margin_score, 3)
        score_display = f"{margin_score:+.2f} (margin)"
        # Calibrate distance via sigmoid for relative certainty indicator
        confidence_val = round(1.0 / (1.0 + math.exp(-abs(margin_score))), 4)
    else:
        score_type = "discrete"
        score_label = "Classification Result"
        score_display = "Categorical"
        confidence_val = 1.0
        decision_score_val = None

    if is_spam:
        explanation = (
            "This message contains keywords, formatting, or tokens "
            "(e.g., promotional terms, urgent calls to action, or contact patterns) "
            "frequently characteristic of spam."
        )
    else:
        explanation = (
            "This message displays natural conversational language and syntax "
            "typical of legitimate (ham) personal or informational communication."
        )

    return {
        "prediction": prediction,
        "label": label,
        "risk": risk,
        "confidence": confidence_val,
        "score_display": score_display,
        "score_label": score_label,
        "score_type": score_type,
        "decision_score": decision_score_val,
        "model_name": model_name,
        "explanation": explanation,
        "cleaned_text": cleaned,
    }

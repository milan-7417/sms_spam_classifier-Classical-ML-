"""
Training script for SMS Spam Classifier.

Executes end-to-end Classical ML workflow:
1. Loads dataset from data/SMSSpamCollection
2. Conducts exploratory data analysis (EDA) and reports class distribution
3. Performs stratified train/test split to maintain class ratio
4. Normalizes text using custom preprocessing
5. Fits TF-IDF vectorizer strictly on training data
6. Trains 3 classical ML models:
   - Multinomial Naive Bayes
   - Logistic Regression
   - Linear SVM (LinearSVC)
7. Evaluates all models on the unseen test set
8. Identifies the best model based on Spam F1-Score and balanced precision/recall
9. Saves the best model, TF-IDF vectorizer, and detailed evaluation metrics
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path when running script directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.evaluate import evaluate_model, format_comparison_table, get_top_features
from src.preprocessing import clean_text


def load_dataset(data_path: Path) -> pd.DataFrame:
    """
    Load the UCI SMS Spam Collection dataset from the local file path.
    The file is a tab-separated text dataset containing: label<TAB>message.
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at expected path: {data_path}. "
            "Please ensure the existing dataset file is in the data/ folder."
        )

    # Read tab-separated text dataset with quotes preserved
    df = pd.read_csv(
        data_path,
        sep="\t",
        names=["label", "message"],
        quoting=3,
        encoding="utf-8",
    )

    # Basic validation
    df = df.dropna(subset=["label", "message"])
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str)

    # Validate classes
    valid_labels = {"ham", "spam"}
    df = df[df["label"].isin(valid_labels)].reset_index(drop=True)
    return df


def perform_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate and display dataset statistics and class distribution."""
    total_messages = len(df)
    class_counts = df["label"].value_counts().to_dict()
    ham_count = class_counts.get("ham", 0)
    spam_count = class_counts.get("spam", 0)
    duplicate_count = int(df.duplicated(subset=["message"]).sum())

    ham_pct = (ham_count / total_messages) * 100 if total_messages else 0
    spam_pct = (spam_count / total_messages) * 100 if total_messages else 0

    print("=" * 70)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)
    print(f"Total SMS Messages   : {total_messages}")
    print(f"Legitimate (Ham)     : {ham_count} ({ham_pct:.2f}%)")
    print(f"Spam                 : {spam_count} ({spam_pct:.2f}%)")
    print(f"Duplicate Messages   : {duplicate_count} "
          f"({duplicate_count / total_messages * 100:.2f}%)")
    print(f"Missing Values       : {df.isnull().sum().sum()}")
    print("=" * 70)

    return {
        "total_messages": total_messages,
        "ham_count": ham_count,
        "spam_count": spam_count,
        "ham_pct": round(ham_pct, 2),
        "spam_pct": round(spam_pct, 2),
        "duplicate_count": duplicate_count,
    }


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series, Any, Any]:
    """
    Preprocess messages and perform a stratified train/test split.
    Ham is encoded as 0, Spam is encoded as 1.
    """
    print("\n[Step 1/5] Preprocessing text messages...")
    df["cleaned_message"] = df["message"].apply(clean_text)

    # Encode target labels
    label_map = {"ham": 0, "spam": 1}
    y = df["label"].map(label_map).values

    print(f"[Step 2/5] Performing stratified train/test split (test_size={test_size}, random_state={random_state})...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["cleaned_message"],
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"  Training samples : {len(X_train_raw)} (Ham: {(y_train == 0).sum()}, Spam: {(y_train == 1).sum()})")
    print(f"  Testing samples  : {len(X_test_raw)} (Ham: {(y_test == 0).sum()}, Spam: {(y_test == 1).sum()})")

    return X_train_raw, X_test_raw, y_train, y_test


def fit_vectorizer(
    X_train_raw: pd.Series,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
) -> Tuple[TfidfVectorizer, Any]:
    """
    Fit TF-IDF Vectorizer strictly on training data to avoid data leakage.
    Uses unigrams + bigrams and sublinear term frequency scaling.
    """
    print(f"\n[Step 3/5] Fitting TF-IDF Vectorizer (ngram_range={ngram_range}, min_df={min_df}, sublinear_tf={sublinear_tf})...")
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
    )
    X_train_vec = vectorizer.fit_transform(X_train_raw)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)} n-grams extracted.")
    return vectorizer, X_train_vec


def train_and_evaluate_models(
    X_train_vec: Any,
    y_train: Any,
    X_test_vec: Any,
    y_test: Any,
    vectorizer: TfidfVectorizer,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
    """
    Train and evaluate candidate classical ML models:
    1. Multinomial Naive Bayes
    2. Logistic Regression
    3. Linear SVM (LinearSVC)
    """
    print("\n[Step 4/5] Training and evaluating candidate classical ML models...")

    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.2),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=5.0,
        ),
        "Linear SVM": LinearSVC(
            max_iter=2000,
            random_state=42,
            C=1.0,
        ),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        metrics = evaluate_model(y_test, y_pred, model_name=name)
        results.append(metrics)
        trained_models[name] = model

    print("\n" + "=" * 70)
    print("MODEL COMPARISON ON TEST SET (Unseen Data)")
    print("=" * 70)
    print(format_comparison_table(results))
    print("=" * 70)

    # Model selection based on highest Spam F1-Score
    best_result = max(results, key=lambda x: (x["f1_score"], x["recall"], x["accuracy"]))
    best_model_name = best_result["model_name"]
    print(f"\n>> Best performing model selected: {best_model_name} (Spam F1: {best_result['f1_score']:.4f})")

    # Display Top Informative Features for best model
    best_model = trained_models[best_model_name]
    top_spam, top_ham = get_top_features(best_model, vectorizer, top_n=12)

    print("\nTop 12 Most Informative Features for SPAM:")
    for word, score in top_spam:
        print(f"  + {word:<20} (weight/ratio: {score:+7.3f})")

    print("\nTop 12 Most Informative Features for HAM (Legitimate):")
    for word, score in top_ham:
        print(f"  - {word:<20} (weight/ratio: {score:+7.3f})")

    return results, trained_models, best_model_name


def save_artifacts(
    models_dir: Path,
    best_model: Any,
    vectorizer: TfidfVectorizer,
    eda_stats: Dict[str, Any],
    all_results: List[Dict[str, Any]],
    best_model_name: str,
) -> None:
    """Save the best trained model, vectorizer, and evaluation metrics JSON."""
    print(f"\n[Step 5/5] Saving model artifacts to {models_dir}...")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "spam_classifier.joblib"
    vectorizer_path = models_dir / "tfidf_vectorizer.joblib"
    metrics_path = models_dir / "metrics.json"

    # Save binary artifacts using joblib
    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"  Saved trained classifier: {model_path}")
    print(f"  Saved TF-IDF vectorizer : {vectorizer_path}")

    # Find metrics of best model
    best_metrics = next(r for r in all_results if r["model_name"] == best_model_name)
    top_spam, top_ham = get_top_features(best_model, vectorizer, top_n=15)

    payload = {
        "best_model_name": best_model_name,
        "eda": eda_stats,
        "best_metrics": best_metrics,
        "all_models_comparison": all_results,
        "top_features": {
            "spam": [{"feature": feat, "score": score} for feat, score in top_spam],
            "ham": [{"feature": feat, "score": score} for feat, score in top_ham],
        },
        "model_type": best_model.__class__.__name__,
        "supports_probability": hasattr(best_model, "predict_proba"),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved evaluation metrics : {metrics_path}")


def main() -> None:
    """Main execution function for reproducible model training."""
    project_root = Path(__file__).resolve().parent.parent
    data_file = project_root / "data" / "SMSSpamCollection"
    models_dir = project_root / "models"

    print("======================================================================")
    print("         SPAMGUARD — SMS SPAM CLASSIFIER TRAINING PIPELINE            ")
    print("======================================================================")

    # 1. Load dataset
    df = load_dataset(data_file)

    # 2. EDA
    eda_stats = perform_eda(df)

    # 3. Data Split
    X_train_raw, X_test_raw, y_train, y_test = prepare_data(df, test_size=0.2, random_state=42)

    # 4. Feature Engineering
    vectorizer, X_train_vec = fit_vectorizer(X_train_raw)
    X_test_vec = vectorizer.transform(X_test_raw)

    # 5. Model Training & Comparison
    results, trained_models, best_model_name = train_and_evaluate_models(
        X_train_vec, y_train, X_test_vec, y_test, vectorizer
    )

    # 6. Save Artifacts
    best_model = trained_models[best_model_name]
    save_artifacts(
        models_dir=models_dir,
        best_model=best_model,
        vectorizer=vectorizer,
        eda_stats=eda_stats,
        all_results=results,
        best_model_name=best_model_name,
    )

    print("\n Training and evaluation completed successfully!")


if __name__ == "__main__":
    main()

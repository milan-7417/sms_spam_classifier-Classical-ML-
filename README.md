# SpamGuard — SMS Spam Classifier

![Python](https://img.shields.io/badge/Python-3.13.5-blue)
![Flask](https://img.shields.io/badge/Flask-3.1.2-black)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458)
![NumPy](https://img.shields.io/badge/NumPy-Data_Processing-013243)
![Joblib](https://img.shields.io/badge/Joblib-Model_Persistence-green)
![TF--IDF](https://img.shields.io/badge/TF--IDF-NLP-purple)
![HTML5](https://img.shields.io/badge/HTML5-E34F26)
![CSS3](https://img.shields.io/badge/CSS3-1572B6)
![JavaScript](https://img.shields.io/badge/JavaScript-ES6-F7DF1E)
![Jinja2](https://img.shields.io/badge/Jinja2-Templates-B41717)
![Render](https://img.shields.io/badge/Render-Deployment-46E3B7)

SpamGuard is a high-accuracy, production-ready SMS Spam Classification web application built with classical Natural Language Processing (NLP) and Machine Learning (scikit-learn), powered by a **FastAPI** backend and custom vanilla web frontend.

---

## Overview

Text messaging and SMS remain primary channels for transactional notifications, authentication codes, and personal messages. However, they are continuously targeted by unsolicited promotional spam, phishing scams, and fraudulent prize claims. 

SpamGuard demonstrates how classical ML algorithms—when paired with domain-specific text preprocessing and n-gram TF-IDF vectorization—can achieve **99.28% test accuracy** and **0.9726 F1-score** on unseen data while running inference in under 2 milliseconds without requiring heavy neural networks or external cloud APIs.

---

## Features

- **FastAPI Asynchronous Backend:** High-performance REST API with automated Pydantic schema validation and JSON serialization.
- **Clean, Distraction-Free Playground:** Minimalist user interface on the home page dedicated solely to message analysis and dynamic risk scoring.
- **Dedicated Documentation Page (`/documentation`):** Concise architectural breakdown explaining how the model was trained, how the runtime NLP pipeline functions, and the comparative model evaluation metrics.
- **Classical NLP Normalization Pipeline:** Custom preprocessing in `src/preprocessing.py` that normalizes URLs (`urladdr`), emails (`emailaddr`), numbers (`numtoken`), and currency symbols (`currsym`) while retaining spam-critical vocabulary (e.g., "free", "win", "call").
- **TF-IDF Feature Representation:** Sublinear term-frequency scaling with unigram and bigram extraction (11,703 features).
- **Comparative Multi-Model Benchmarking:** Evaluates Multinomial Naive Bayes, Logistic Regression, and Linear SVM under identical stratified test splits.
- **Genuine Decision Margin Scores:** Transparent mathematical scores (signed distance from the separating hyperplane for Linear SVM), avoiding fabricated probabilities.
- **Explainable Insights & Token Inspection:** Displays normalized n-gram tokens and explains why an SMS triggered spam or legitimate classification.
- **Custom Modern Frontend:** Clean, dark-mode user interface designed with vanilla HTML5, CSS3, and JavaScript—zero bloated frameworks, fully responsive, and accessible.

---

## Machine Learning Pipeline

```
Raw SMS Text
     │
     ▼
Text Preprocessing & Normalization
(lowercasing, punctuation stripping, tokenizing URLs, currencies, and numbers)
     │
     ▼
TF-IDF Vectorization
(unigram + bigram n-grams, sublinear TF scaling, min_df=2, max_df=0.95)
     │
     ▼
Classical ML Classifier (Linear SVM)
(maximum-margin decision boundary)
     │
     ▼
Prediction Output: SPAM (High Risk) / HAM (Low Risk)
+ Decision Margin Score + Explanation
```

---

## Dataset

- **Source:** UCI Machine Learning Repository — SMS Spam Collection
- **Official Repository Link:** [https://archive.ics.uci.edu/dataset/228/sms+spam+collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Dataset Size:** 5,574 SMS messages
- **Task:** Binary text classification
- **Class Labels:**
  - `ham`: Legitimate message (4,827 instances, ~86.6%)
  - `spam`: Unsolicited / fraudulent message (747 instances, ~13.4%)
- **Data Format:** Tab-separated text (`label<TAB>message`) stored locally under `data/SMSSpamCollection`.

### Citation

```bibtex
Almeida, T. & Hidalgo, J. (2011).
SMS Spam Collection.
UCI Machine Learning Repository.
https://doi.org/10.24432/C5CC84
```

> **Dataset Attribution:** The SMS Spam Collection dataset is provided by the UCI Machine Learning Repository and is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). This project does not own or distribute modified derivatives of the underlying dataset.

---

## Tech Stack

- 🐍 **Language:** Python 3.10+
- ⚡ **Backend Framework:** FastAPI, Uvicorn, Pydantic, Jinja2
- 🤖 **Machine Learning & NLP:** scikit-learn (Linear SVM, Multinomial Naive Bayes, Logistic Regression, TF-IDF)
- 🐼 **Data Processing:** pandas, NumPy
- 💾 **Model Serialization:** Joblib
- 🌐 **Frontend:** Semantic HTML5, Custom CSS3 (Glassmorphism & Design Tokens), Vanilla JavaScript (ES6+ `fetch`)
- 🚀 **Cloud Deployment:** Render

---

## Project Structure

```
Spam_classifier/
│
├── data/
│   └── SMSSpamCollection            # UCI SMS Spam Collection dataset (tab-separated)
│
├── src/
│   ├── __init__.py                  # Package initializer
│   ├── preprocessing.py             # Normalization, tokenization, and regex cleaning
│   ├── train.py                     # End-to-end training, split, evaluation, and serialization
│   ├── evaluate.py                  # Classification metrics, confusion matrix, top features
│   └── predict.py                   # Artifact loading, input validation, and inference
│
├── models/
│   ├── spam_classifier.joblib       # Best trained estimator (Linear SVM)
│   ├── tfidf_vectorizer.joblib      # Fitted TF-IDF vectorizer (11,703 features)
│   └── metrics.json                 # Actual test metrics and model comparison data
│
├── templates/
│   ├── index.html                   # Clean SMS classifier interface
│   └── docs.html                    # Dedicated documentation & architecture page
│
├── static/
│   ├── css/
│   │   └── style.css                # Polished design system and responsive styles
│   └── js/
│       └── app.js                   # Client-side validation, async fetch, and UI states
│
├── app.py                           # FastAPI server and REST API endpoints
├── requirements.txt                 # Project dependencies
├── README.md                        # Documentation and architecture guide
└── .gitignore                       # Git ignore configuration
```

---

## Installation

### 1. Clone or Open the Workspace

Open a terminal in the `Spam_classifier/` project root directory.

### 2. Set Up a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS / Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Training

To train all candidate models on the local dataset, evaluate performance, and generate the model artifacts:

```bash
python src/train.py
```

The training pipeline performs:
1. Local data validation from `data/SMSSpamCollection`.
2. Exploratory Data Analysis (reporting message counts, class ratios, and duplicates).
3. Stratified 80/20 train/test split (`random_state=42`).
4. Fitting the TF-IDF vectorizer strictly on training data to prevent leakage.
5. Training Multinomial Naive Bayes, Logistic Regression, and Linear SVM.
6. Selecting the best model based on Spam F1-Score.
7. Saving `models/spam_classifier.joblib`, `models/tfidf_vectorizer.joblib`, and `models/metrics.json`.

---

## Running the Application

Ensure the models have been trained, then launch the FastAPI server:

```bash
python app.py
```

Open your browser and navigate to:

```
http://127.0.0.1:8000
```

- **Live Classifier:** `http://127.0.0.1:8000/`
- **Documentation & Architecture:** `http://127.0.0.1:8000/documentation`
- **Interactive Swagger UI:** `http://127.0.0.1:8000/docs`

---

## Model Evaluation

All models were evaluated on the exact same unseen test set of **1,115 messages** (966 Ham, 149 Spam) using an 80/20 stratified split (`random_state=42`).

### Model Comparison Table (Actual Test Results)

| Model Architecture | Accuracy | Precision (Spam) | Recall (Spam) | F1-Score (Spam) |
|---|---:|---:|---:|---:|
| **Multinomial Naive Bayes** | 0.9883 | 0.9928 | 0.9195 | 0.9547 |
| **Logistic Regression** | 0.9910 | 1.0000 | 0.9329 | 0.9653 |
| **Linear SVM (LinearSVC)** | **0.9928** | **0.9930** | **0.9530** | **0.9726** |

### Selected Final Model: Linear SVM

**Linear SVM** was selected as the production classifier because it achieved the highest Spam F1-Score (**0.9726**) and the highest Spam Recall (**95.30%**), successfully catching 142 out of 149 spam messages while yielding only a single false alarm on 966 legitimate messages.

### Test Confusion Matrix (Linear SVM)

| | Predicted Ham | Predicted Spam |
|---|---:|---:|
| **Actual Ham (966)** | **965** (TN) | **1** (FP) |
| **Actual Spam (149)** | **7** (FN) | **142** (TP) |

---

## REST API Specification

### POST `/predict`

Analyze an SMS message and receive a classification response.

**Request Body:**
```json
{
  "message": "WINNER!! As a valued customer you have won a £900 prize reward! Call 09061701461 to claim now."
}
```

**Response (`200 OK`):**
```json
{
  "prediction": "spam",
  "label": "SPAM",
  "risk": "High Risk",
  "confidence": 0.8798,
  "score_display": "+1.99 (margin)",
  "score_label": "Decision Margin Score",
  "score_type": "decision_score",
  "decision_score": 1.99,
  "model_name": "Linear SVM",
  "explanation": "This message contains keywords, formatting, or tokens frequently characteristic of spam.",
  "cleaned_text": "winner you have won a currsym numtoken numtoken cash prize call numtoken to claim your reward"
}
```

### GET `/metrics`

Returns JSON containing the training metadata, comparison table, and confusion matrix.

---

## Limitations

- **Dataset Size:** The dataset contains 5,574 messages. While sufficient for high classical ML performance, larger corpora are needed for edge-case coverage.
- **SMS Era vs. Modern Messaging:** The dataset predominantly reflects SMS conventions from the late 2000s/early 2010s; modern spam on platforms like WhatsApp or Telegram frequently incorporates obfuscated emojis, shortened link redirections, or conversational social engineering.
- **Class Imbalance:** Legitimate messages comprise ~86.6% of the dataset. Stratified sampling and F1-score tracking are mandatory to avoid false accuracy confidence.
- **Evolving Spam Tactics:** Spammers adapt vocabulary over time (adversarial text perturbations, zero-width spaces, leetspeak), requiring recurring retrains or adaptive dictionaries.
- **Probabilistic Nature:** Predictions are statistical estimates and cannot guarantee 100% security against novel attack vectors.

---

## Future Improvements

- **Character N-Gram Augmentation:** Incorporating character n-grams (e.g. 3-grams to 5-grams) to make the model resilient against spelling obfuscation (e.g., `"w!n"`, `"f-r-e-e"`).
- **Multilingual Support:** Training on multi-language spam datasets for cross-border messaging platforms.
- **Active / Online Learning:** Allowing user feedback loops to flag false negatives and retrain incremental models.
- **Transformer Benchmark:** Creating an offline benchmark against distilled transformer architectures (e.g. DistilBERT) to document trade-offs in inference latency vs. marginal accuracy gains.

---

## License

- **Code License:** Distributed under the [MIT License](https://opensource.org/licenses/MIT).
- **Dataset License:** The UCI SMS Spam Collection dataset is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). All credit and attribution belong to the original authors (Almeida & Hidalgo, 2011).

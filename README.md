<div align="center">

# 📩 Spam Detector
### *A lightweight ML pipeline that classifies SMS messages as spam or ham*

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Naive%20Bayes-F7931E?logo=scikitlearn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20Demo-000000?logo=flask&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data%20Handling-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

---

## 📖 Table of Contents
- [Overview](#-overview)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Folder Structure](#-folder-structure)
- [Setup](#-setup)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Dataset](#-dataset)
- [Roadmap](#-roadmap)

---

## 🔭 Overview

**Spam Detector** is a minimal, end-to-end text classification project: it trains a **Naive Bayes** model on labeled SMS messages, saves the trained model and vectorizer as reusable artifacts, and exposes predictions through both a **command-line tool** and a **Flask web demo**.

**What it does:**
- Trains a bag-of-words spam classifier from a CSV dataset
- Saves model + vectorizer as portable `.joblib` artifacts (train once, predict anywhere)
- Classifies new messages via CLI (`predict.py`) or browser (`app.py`)
- Reports accuracy and a full classification report on every training run

---

## 🏗 Architecture

```
┌─────────────────┐
│    spam.csv        │
│ (labeled dataset)   │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│      train.py           │
│  CountVectorizer +       │
│  MultinomialNB fit        │
└─────────┬─────────────────┘
          ▼
┌─────────────────────┐
│   artifacts/             │
│  model.joblib             │
│  vectorizer.joblib          │
└─────────┬─────────────────┘
          │
   ┌──────┴───────┐
   ▼               ▼
┌───────────┐  ┌───────────┐
│ predict.py  │  │  app.py     │
│ (CLI)        │  │ (Flask web) │
└───────────┘  └───────────┘
```

`train.py` is the only file that touches raw data — every prediction path afterward just loads the saved artifacts, so training and inference are fully decoupled.

---

## 🧰 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Data handling** | pandas | Load and normalize the CSV dataset |
| **Feature extraction** | scikit-learn `CountVectorizer` | Bag-of-words text vectorization (English stop words removed) |
| **Model** | scikit-learn `MultinomialNB` | Naive Bayes classifier, well-suited to word-count features |
| **Serialization** | joblib | Persist the trained model and vectorizer to disk |
| **CLI interface** | Python stdlib (`sys.argv`) | `predict.py` — quick single-message predictions |
| **Web interface** | Flask | `app.py` — browser form for interactive predictions |
| **Production server** | gunicorn | WSGI server for deploying the Flask app |

---

## 🗂 Folder Structure

```
punith123-main/
├── train.py              # Loads data, trains the model, saves artifacts
├── predict.py              # CLI — classify a single message
├── app.py                    # Flask web demo
├── spam_detector.py             # Standalone data-loading sanity check
├── spam.csv                       # Labeled dataset (v1 = label, v2 = text)
├── artifacts/                        # Created by train.py — model.joblib, vectorizer.joblib
├── requirements.txt                     # Python dependencies
└── README.md                              # You are here
```

---

## 🚀 Setup

### Prerequisites
- Python 3.8+

### 1. Create and activate a virtual environment
```bash
python -m venv env
source env/Scripts/activate     # Git Bash on Windows
# or: source env/bin/activate   # macOS/Linux
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Train the model
```bash
python train.py
```
This prints the dataset's label distribution, trains the classifier, evaluates it on a held-out test split, and saves `model.joblib` + `vectorizer.joblib` into `artifacts/`.

### Predict from the command line
```bash
python predict.py "Congratulations! You've won a free prize, claim now!"
```

### Run the web demo
```bash
python app.py
```
Visit `http://localhost:5000`, paste a message into the textarea, and submit to see the prediction.

---

## 🧠 How It Works

1. **Load & normalize** — `spam.csv` columns `v1`/`v2` are renamed to `label`/`text`.
2. **Split safely** — `train.py` checks class sizes before splitting; if any class has fewer than 2 examples, it disables stratified splitting to avoid a crash, and pads the test set so every class can still appear in it.
3. **Vectorize** — `CountVectorizer` turns raw text into word-count feature vectors, with English stop words removed.
4. **Train** — `MultinomialNB` fits on the vectorized training set.
5. **Evaluate** — accuracy and a full precision/recall/F1 report print to the console.
6. **Persist** — the fitted model and vectorizer are saved separately as `.joblib` files, so any script can load them without retraining.

---

## 📊 Dataset

`spam.csv` uses the classic SMS Spam Collection format — two relevant columns:
- `v1` — label (`spam` or `ham`)
- `v2` — the raw message text

> ⚠️ **Note:** the dataset currently bundled in this repo is a very small sample (only a handful of rows). For meaningful accuracy, swap in the full [SMS Spam Collection dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (~5,500 messages) before training for real use.

---

## 🛣 Roadmap

- [ ] Swap in the full SMS Spam Collection dataset for real accuracy
- [ ] Add TF-IDF as an alternative to raw CountVectorizer and compare performance
- [ ] Add a `/predict` JSON API endpoint to `app.py` for programmatic use
- [ ] Add unit tests for `train.py` and `predict.py`
- [ ] Add a confusion matrix plot to the training output

---

<div align="center">

Built as a hands-on intro to the classic ML pipeline: data → features → model → deployment.

</div>
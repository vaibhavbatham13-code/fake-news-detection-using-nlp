"""
Train the fake news classifier and save it to disk.

Kaggle dataset:
python train_model.py --fake data/Fake.csv --true data/True.csv

Quick sample test:
python train_model.py
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split

from sample_data import SAMPLE_DATA
from utilis import build_pipeline, clean_text, top_terms


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"

MODEL_PATH = MODEL_DIR / "fake_news_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
CONFUSION_PATH = MODEL_DIR / "confusion_matrix.png"
TOP_WORDS_PATH = MODEL_DIR / "top_words.png"
ROC_PATH = MODEL_DIR / "roc_curve.png"


def load_sample_data():
    return pd.DataFrame(SAMPLE_DATA, columns=["text", "label"])


def load_kaggle_data(fake_csv, true_csv):
    fake = pd.read_csv(fake_csv)
    true = pd.read_csv(true_csv)

    fake["label"] = 1
    true["label"] = 0

    for df in (fake, true):
        df["title"] = df["title"].fillna("")
        df["text"] = df["title"] + " " + df["text"].fillna("")

    combined = pd.concat(
        [
            fake[["text", "label"]],
            true[["text", "label"]],
        ],
        ignore_index=True,
    )

    return combined


def build_model():
    classifier = LogisticRegression(
        max_iter=2000,
        C=1.0,
        class_weight="balanced",
    )

    return build_pipeline(classifier=classifier, use_style=True)


def save_confusion_matrix(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Real", "Fake"],
        cmap="Blues",
        ax=ax,
    )

    ax.set_title("Fake News Detection - Confusion Matrix")
    fig.tight_layout()
    fig.savefig(CONFUSION_PATH, dpi=150)
    plt.close(fig)


def save_roc_curve(y_test, y_prob):
    fig, ax = plt.subplots(figsize=(6, 5))

    RocCurveDisplay.from_predictions(
        y_test,
        y_prob,
        ax=ax,
    )

    ax.set_title("Fake News Detection - ROC Curve")
    fig.tight_layout()
    fig.savefig(ROC_PATH, dpi=150)
    plt.close(fig)


def save_top_words(model):
    fake_terms, real_terms = top_terms(model, n=15)

    terms = list(reversed(real_terms)) + fake_terms
    labels = [term for term, _ in terms]
    weights = [weight for _, weight in terms]

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.barh(labels, weights)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Words Influencing Fake and Real Predictions")
    ax.set_xlabel("Logistic Regression coefficient")
    fig.tight_layout()
    fig.savefig(TOP_WORDS_PATH, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Train the fake news classifier."
    )

    parser.add_argument("--fake", help="Path to Fake.csv")
    parser.add_argument("--true", help="Path to True.csv")

    args = parser.parse_args()

    if bool(args.fake) != bool(args.true):
        parser.error("Provide both --fake and --true.")

    if args.fake and args.true:
        print("Loading Kaggle dataset...")
        df = load_kaggle_data(args.fake, args.true)
        dataset_name = "Kaggle Fake and Real News"
    else:
        print("Using built-in sample data.")
        print("Sample accuracy is not meaningful.")
        df = load_sample_data()
        dataset_name = "Built-in sample"

    print(f"\nOriginal articles: {len(df)}")
    print("\nLabel distribution:")
    print(df["label"].value_counts())

    df["clean"] = df["text"].apply(clean_text)
    df = df[df["clean"].str.len() > 0].copy()

    before = len(df)

    # Remove duplicate cleaned articles.
    # Keep the first occurrence only.
    df = df.drop_duplicates(
        subset=["clean"],
        keep="first",
    ).reset_index(drop=True)

    print(f"\nDuplicates removed: {before - len(df)}")
    print(f"Final dataset size: {len(df)}")

    print("\nFinal label distribution:")
    print(df["label"].value_counts())

    if df["label"].nunique() != 2:
        raise ValueError("Training requires both Real and Fake labels.")

    # IMPORTANT:
    # Split raw text, not cleaned text.
    # The shared model pipeline performs cleaning internally.
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    model = build_model()

    print("\nTraining model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    classes = list(model.named_steps["clf"].classes_)
    fake_index = classes.index(1)
    y_prob = model.predict_proba(X_test)[:, fake_index]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"\nTest accuracy: {accuracy:.4f}")
    print(f"Fake precision: {precision:.4f}")
    print(f"Fake recall: {recall:.4f}")
    print(f"Fake F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Real", "Fake"],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    save_confusion_matrix(y_test, y_pred)
    save_roc_curve(y_test, y_prob)
    save_top_words(model)

    metrics = {
        "dataset": dataset_name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "selected_model": "Logistic Regression",
        "test": {
            "Logistic Regression": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "roc_auc": float(roc_auc),
            }
        },
        "warnings": [
            "Performance on this test split does not guarantee real-world accuracy.",
            "Independent evaluation on unseen news sources is still required.",
        ],
    }

    joblib.dump(model, MODEL_PATH)

    METRICS_PATH.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )

    print("\nModel and evaluation files saved:")
    print(MODEL_PATH)
    print(METRICS_PATH)
    print(CONFUSION_PATH)
    print(TOP_WORDS_PATH)
    print(ROC_PATH)


if __name__ == "__main__":
    main()

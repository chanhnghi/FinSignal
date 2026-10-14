"""Train one small financial sentiment model and evaluate it on held-out data."""

from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, ConfusionMatrixDisplay, f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
LABELS = ["negative", "neutral", "positive"]


def main():
    # Read the original labeled sentences.
    rows = []
    path = ROOT / "data/raw/Sentences_75Agree.txt"
    for line in path.read_text(encoding="iso-8859-1").splitlines():
        text, label = line.rsplit("@", 1)
        text, label = " ".join(text.split()), label.strip()
        if not text or label not in LABELS:
            raise ValueError("Found an empty sentence or invalid label.")
        rows.append({"text": text, "label": label, "key": text.casefold()})
    data = pd.DataFrame(rows)
    raw_count = len(data)

    # Remove conflicting groups and exact duplicates before splitting.
    counts = data.groupby("key")["label"].nunique()
    conflicts = counts[counts > 1].index
    data = data[~data["key"].isin(conflicts)].drop_duplicates("key")

    # Fixed 80/20 split. Do not tune the model using the test results.
    train, test = train_test_split(
        data, test_size=0.2, stratify=data["label"], random_state=42,
    )

    # TF-IDF learns a vocabulary; Logistic Regression learns class weights.
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 1), min_df=2, sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42,
        )),
    ])
    model.fit(train["text"], train["label"])
    predictions = model.predict(test["text"])

    # Compare against always choosing the most common training label.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(train["text"], train["label"])
    baseline_predictions = baseline.predict(test["text"])

    metrics = {
        "model": "TF-IDF unigrams + balanced Logistic Regression",
        "seed": 42,
        "raw_rows": raw_count,
        "clean_rows": len(data),
        "conflicting_groups_removed": len(conflicts),
        "train_rows": len(train),
        "test_rows": len(test),
        "test_accuracy": accuracy_score(test["label"], predictions),
        "test_macro_f1": f1_score(
            test["label"], predictions, labels=LABELS, average="macro",
        ),
        "baseline_macro_f1": f1_score(
            test["label"], baseline_predictions, labels=LABELS,
            average="macro", zero_division=0,
        ),
        "classification_report": classification_report(
            test["label"], predictions, labels=LABELS,
            output_dict=True, zero_division=0,
        ),
    }
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    joblib.dump(model, output / "model.joblib", compress=3)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8",
    )
    table = test[["text", "label"]].copy()
    table["prediction"] = predictions
    table.to_csv(output / "test_predictions.csv", index=False)
    table[table["label"] != table["prediction"]].to_csv(
        output / "errors.csv", index=False,
    )
    ConfusionMatrixDisplay.from_predictions(
        test["label"], predictions, labels=LABELS,
    )
    plt.tight_layout()
    plt.savefig(output / "confusion_matrix.png", dpi=140)
    plt.close()

    print(f"Train: {len(train)} | Test: {len(test)}")
    print(f"Test Macro-F1: {metrics['test_macro_f1']:.4f}")
    print(f"Test accuracy: {metrics['test_accuracy']:.2%}")
    print(f"Baseline Macro-F1: {metrics['baseline_macro_f1']:.4f}")
    print(classification_report(test["label"], predictions, zero_division=0))
    print("Saved model, metrics, predictions, errors and confusion matrix to outputs/.")


if __name__ == "__main__":
    main()

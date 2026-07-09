import json
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.svm import LinearSVC

from search_alternative_baselines import evaluate_model, load_preprocessed, prepare_features

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    df = load_preprocessed()
    prepared = prepare_features(df, max_features=1500)

    best_spec = {
        "kind": "text",
        "needs_scaled": False,
        "model": LinearSVC(C=1.0, class_weight="balanced", random_state=42),
    }
    metrics = evaluate_model("linearsvc_only", best_spec, prepared)

    model = best_spec["model"]
    model.fit(prepared["X_train_tfidf"], prepared["y_train_text"])
    pred = model.predict(prepared["X_test_tfidf"])

    print("=" * 60)
    print("BEST ALTERNATIVE BASELINE: LinearSVC")
    print(f"ACCURACY        : {metrics['accuracy'] * 100:.2f}%")
    print(f"ERROR RATE      : {metrics['error_rate'] * 100:.2f}%")
    print(f"BALANCED ACC    : {metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"MACRO F1        : {metrics['macro_f1'] * 100:.2f}%")
    print("=" * 60)
    print("\nClassification Report")
    print(classification_report(prepared["y_test_text"], pred, digits=3, zero_division=0))

    output_json = OUTPUT_DIR / "best_alternative_baseline_metrics.json"
    output_txt = OUTPUT_DIR / "best_alternative_baseline_report.txt"
    output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    output_txt.write_text(
        classification_report(prepared["y_test_text"], pred, digits=3, zero_division=0),
        encoding="utf-8",
    )
    print("\nSaved:", output_json)
    print("Saved:", output_txt)


if __name__ == "__main__":
    main()

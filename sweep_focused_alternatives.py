import itertools
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from search_alternative_baselines import evaluate_model, load_preprocessed, prepare_features

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


def build_model(config):
    model_family = config["model_family"]
    if model_family == "linearsvc":
        return {
            "kind": "text",
            "needs_scaled": False,
            "model": LinearSVC(
                C=config["c_value"],
                class_weight=config["class_weight"],
                random_state=42,
            ),
        }
    if model_family == "linearsvc_calibrated":
        return {
            "kind": "text",
            "needs_scaled": False,
            "model": CalibratedClassifierCV(
                estimator=LinearSVC(
                    C=config["c_value"],
                    class_weight=config["class_weight"],
                    random_state=42,
                ),
                cv=3,
            ),
        }
    if model_family == "complement_nb":
        return {
            "kind": "text",
            "needs_scaled": False,
            "model": ComplementNB(alpha=config["alpha"]),
        }
    if model_family == "ros_complement_nb":
        return {
            "kind": "text",
            "needs_scaled": False,
            "model": ImbPipeline(
                [
                    ("ros", RandomOverSampler(random_state=42)),
                    ("clf", ComplementNB(alpha=config["alpha"])),
                ]
            ),
        }
    if model_family == "ensemble_cnb_svc":
        svc = CalibratedClassifierCV(
            estimator=LinearSVC(
                C=config["c_value"],
                class_weight=config["class_weight"],
                random_state=42,
            ),
            cv=3,
        )
        return {
            "kind": "text",
            "needs_scaled": False,
            "model": VotingClassifier(
                estimators=[
                    ("cnb", ComplementNB(alpha=config["alpha"])),
                    ("svc", svc),
                ],
                voting="soft",
                weights=[config["weight_cnb"], config["weight_svc"]],
            ),
        }
    if model_family == "ensemble_cnb_logreg":
        return {
            "kind": "text",
            "needs_scaled": False,
            "model": VotingClassifier(
                estimators=[
                    ("cnb", ComplementNB(alpha=config["alpha"])),
                    (
                        "logreg",
                        LogisticRegression(
                            max_iter=3000,
                            class_weight=config["class_weight"],
                            solver="liblinear",
                            C=config["c_value"],
                            random_state=42,
                        ),
                    ),
                ],
                voting="soft",
                weights=[config["weight_cnb"], config["weight_svc"]],
            ),
        }
    raise ValueError(f"Unknown model family: {model_family}")


def generate_configs():
    families = [
        "linearsvc",
        "linearsvc_calibrated",
        "complement_nb",
        "ros_complement_nb",
        "ensemble_cnb_svc",
        "ensemble_cnb_logreg",
    ]
    configs = []
    for model_family in families:
        if model_family in {"linearsvc", "linearsvc_calibrated"}:
            for max_features, c_value, class_weight in itertools.product(
                [1200, 1500, 2000, 2500],
                [0.5, 1.0, 1.5, 2.0],
                ["balanced", None],
            ):
                configs.append(
                    {
                        "model_family": model_family,
                        "max_features": max_features,
                        "c_value": c_value,
                        "class_weight": class_weight,
                    }
                )
        elif model_family in {"complement_nb", "ros_complement_nb"}:
            for max_features, alpha in itertools.product(
                [1200, 1500, 2000, 2500],
                [0.2, 0.5, 0.8, 1.0],
            ):
                configs.append(
                    {
                        "model_family": model_family,
                        "max_features": max_features,
                        "alpha": alpha,
                    }
                )
        else:
            for max_features, alpha, c_value, class_weight, weight_cnb, weight_svc in itertools.product(
                [1200, 1500, 2000],
                [0.2, 0.5, 0.8],
                [0.5, 1.0, 1.5],
                ["balanced", None],
                [1, 2],
                [1, 2],
            ):
                configs.append(
                    {
                        "model_family": model_family,
                        "max_features": max_features,
                        "alpha": alpha,
                        "c_value": c_value,
                        "class_weight": class_weight,
                        "weight_cnb": weight_cnb,
                        "weight_svc": weight_svc,
                    }
                )
    return configs


def main():
    random.seed(42)
    np.random.seed(42)

    df = load_preprocessed()
    configs = generate_configs()
    output_csv = OUTPUT_DIR / "focused_alternative_sweep_results.csv"
    output_json = OUTPUT_DIR / "focused_alternative_sweep_best.json"
    rows = []
    save_every = 10

    print("=== Focused Sweep Around LinearSVC / ComplementNB / Light Ensembles ===")
    for idx, config in enumerate(configs, start=1):
        prepared = prepare_features(df, max_features=config["max_features"])
        spec = build_model(config)
        row = {**config, **evaluate_model(config["model_family"], spec, prepared)}
        rows.append(row)

        if idx % save_every == 0:
            partial = pd.DataFrame(rows).sort_values(
                ["error_rate", "accuracy", "macro_f1", "balanced_accuracy"],
                ascending=[True, False, False, False],
            )
            partial.to_csv(output_csv, index=False)
            output_json.write_text(json.dumps(partial.iloc[0].to_dict(), indent=2), encoding="utf-8")

        print(
            f"{idx:03d}/{len(configs)} "
            f"{row['model_family']} "
            f"err={row['error_rate']:.4f} acc={row['accuracy']:.4f} "
            f"f1={row['macro_f1']:.4f} cfg={config}"
        )

    results = pd.DataFrame(rows).sort_values(
        ["error_rate", "accuracy", "macro_f1", "balanced_accuracy"],
        ascending=[True, False, False, False],
    )
    results.to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(results.iloc[0].to_dict(), indent=2), encoding="utf-8")

    print("\nTop Results")
    print(results.head(15).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nSaved:", output_csv)
    print("Saved:", output_json)


if __name__ == "__main__":
    main()

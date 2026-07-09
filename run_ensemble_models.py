from pathlib import Path

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

from fatalitas_core import (
    ProbabilityVotingEnsemble,
    XGBoostLabelEncodedClassifier,
    build_tfidf,
    cross_validate_model,
    holdout_evaluate,
    load_dataset,
    make_calibrated_svc,
    make_logreg,
)

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


def build_ensemble_specs():
    tfidf = build_tfidf()
    cnb = ComplementNB(alpha=0.5)
    svc_prob = make_calibrated_svc()
    logreg = make_logreg()

    specs = {
        "ensemble_soft_svc_cnb_logreg": {
            "target": "label",
            "estimator": Pipeline(
                [
                    ("tfidf", tfidf),
                    (
                        "clf",
                        ProbabilityVotingEnsemble(
                            estimators=[("svc", svc_prob), ("cnb", cnb), ("logreg", logreg)],
                            weights=[1.0, 1.2, 1.0],
                        ),
                    ),
                ]
            ),
        },
        "ensemble_soft_roscnb_logreg": {
            "target": "label",
            "estimator": ImbPipeline(
                [
                    ("tfidf", tfidf),
                    ("ros", RandomOverSampler(random_state=42)),
                    (
                        "clf",
                        ProbabilityVotingEnsemble(
                            estimators=[("cnb", cnb), ("logreg", logreg)],
                            weights=[1.3, 1.0],
                        ),
                    ),
                ]
            ),
        },
    }

    try:
        import xgboost as xgb

        _ = xgb
        xgb_clf = XGBoostLabelEncodedClassifier(n_estimators=120, learning_rate=0.07, max_depth=3)
        specs["ensemble_soft_svm_xgboost"] = {
            "target": "label",
            "estimator": Pipeline(
                [
                    ("tfidf", tfidf),
                    (
                        "clf",
                        ProbabilityVotingEnsemble(
                            estimators=[("svc", svc_prob), ("xgb", xgb_clf)],
                            weights=[1.0, 1.0],
                        ),
                    ),
                ]
            ),
        }
        specs["ensemble_soft_cnb_xgboost"] = {
            "target": "label",
            "estimator": Pipeline(
                [
                    ("tfidf", tfidf),
                    (
                        "clf",
                        ProbabilityVotingEnsemble(
                            estimators=[("cnb", cnb), ("xgb", xgb_clf)],
                            weights=[1.2, 1.0],
                        ),
                    ),
                ]
            ),
        }
    except Exception as exc:
        print(f"[info] XGBoost dilewati di ensemble runner: {exc}")

    return specs


def main():
    df = load_dataset(use_normalization=True)
    specs = build_ensemble_specs()
    rows = []

    print("=== Ensemble Comparative Study ===")
    for name, spec in specs.items():
        cv_metrics = cross_validate_model(
            spec["estimator"], df["Teks_Bersih"], df["TingkatFatalitas"], df["TingkatFatalitas"]
        )
        holdout_metrics = holdout_evaluate(
            spec["estimator"], df["Teks_Bersih"], df["TingkatFatalitas"], df["TingkatFatalitas"]
        )
        row = {
            "ensemble": name,
            "cv_accuracy": cv_metrics["accuracy"],
            "cv_error_rate": cv_metrics["error_rate"],
            "cv_balanced_accuracy": cv_metrics["balanced_accuracy"],
            "cv_macro_f1": cv_metrics["macro_f1"],
            "cv_quadratic_kappa": cv_metrics["quadratic_kappa"],
            "holdout_accuracy": holdout_metrics["accuracy"],
            "holdout_error_rate": holdout_metrics["error_rate"],
            "holdout_balanced_accuracy": holdout_metrics["balanced_accuracy"],
            "holdout_macro_f1": holdout_metrics["macro_f1"],
            "holdout_quadratic_kappa": holdout_metrics["quadratic_kappa"],
        }
        rows.append(row)
        print(
            f"{name}: cv_macro_f1={row['cv_macro_f1']:.4f}, "
            f"cv_error={row['cv_error_rate']:.4f}, holdout_macro_f1={row['holdout_macro_f1']:.4f}"
        )

    results = pd.DataFrame(rows).sort_values(
        ["cv_macro_f1", "cv_balanced_accuracy", "cv_quadratic_kappa", "cv_accuracy"],
        ascending=False,
    )
    output_path = OUTPUT_DIR / "ensemble_models_results.csv"
    results.to_csv(output_path, index=False)
    print("\nSaved:", output_path)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()

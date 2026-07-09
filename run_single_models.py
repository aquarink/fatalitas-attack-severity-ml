from pathlib import Path

import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from fatalitas_core import (
    CLASS_ORDER,
    BucketedRegressor,
    OrdinalReductionClassifier,
    XGBoostLabelEncodedClassifier,
    build_tfidf,
    cross_validate_model,
    holdout_evaluate,
    load_dataset,
    make_calibrated_svc,
    make_logreg,
    make_ridge_bucket,
)

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


def build_single_model_specs():
    tfidf = build_tfidf()
    ordinal_logreg = OrdinalReductionClassifier(base_estimator=make_logreg(), class_order=CLASS_ORDER)
    ordinal_svm = OrdinalReductionClassifier(base_estimator=make_calibrated_svc(), class_order=CLASS_ORDER)

    specs = {
        "tfidf_complement_nb": {
            "target": "label",
            "estimator": Pipeline([("tfidf", tfidf), ("clf", ComplementNB(alpha=0.5))]),
        },
        "tfidf_linear_svc": {
            "target": "label",
            "estimator": Pipeline(
                [("tfidf", tfidf), ("clf", LinearSVC(C=1.0, class_weight="balanced", random_state=42))]
            ),
        },
        "tfidf_logistic_regression": {
            "target": "label",
            "estimator": Pipeline([("tfidf", tfidf), ("clf", make_logreg())]),
        },
        "tfidf_ros_complement_nb": {
            "target": "label",
            "estimator": ImbPipeline(
                [("tfidf", tfidf), ("ros", RandomOverSampler(random_state=42)), ("clf", ComplementNB(alpha=0.5))]
            ),
        },
        "tfidf_smote_svd_logreg": {
            "target": "label",
            "estimator": ImbPipeline(
                [
                    ("tfidf", tfidf),
                    ("svd", TruncatedSVD(n_components=100, random_state=42)),
                    ("smote", SMOTE(random_state=42, k_neighbors=3)),
                    ("clf", LogisticRegression(max_iter=3000, solver="lbfgs", random_state=42)),
                ]
            ),
        },
        "ordinal_logistic_regression": {
            "target": "label",
            "estimator": Pipeline([("tfidf", tfidf), ("clf", ordinal_logreg)]),
        },
        "ordinal_svm": {
            "target": "label",
            "estimator": Pipeline([("tfidf", tfidf), ("clf", ordinal_svm)]),
        },
        "ridge_regression_bucket": {
            "target": "deaths",
            "estimator": Pipeline([("tfidf", tfidf), ("clf", make_ridge_bucket())]),
        },
    }

    try:
        import xgboost as xgb

        _ = xgb
        specs["tfidf_xgboost"] = {
            "target": "label",
            "estimator": Pipeline([("tfidf", tfidf), ("clf", XGBoostLabelEncodedClassifier())]),
        }
        specs["xgboost_regression_bucket"] = {
            "target": "deaths",
            "estimator": Pipeline(
                [
                    ("tfidf", tfidf),
                    (
                        "clf",
                        BucketedRegressor(
                            xgb.XGBRegressor(
                                n_estimators=300,
                                learning_rate=0.05,
                                max_depth=4,
                                subsample=0.9,
                                colsample_bytree=0.9,
                                objective="reg:squarederror",
                                random_state=42,
                            )
                        ),
                    ),
                ]
            ),
        }
    except Exception as exc:
        print(f"[info] XGBoost dilewati di single model runner: {exc}")

    return specs


def main():
    df = load_dataset(use_normalization=True)
    specs = build_single_model_specs()
    rows = []

    print("=== Single-Model Comparative Study ===")
    for name, spec in specs.items():
        y_fit = df["TingkatFatalitas"] if spec["target"] == "label" else df["KorbanTewas"]
        cv_metrics = cross_validate_model(spec["estimator"], df["Teks_Bersih"], y_fit, df["TingkatFatalitas"])
        holdout_metrics = holdout_evaluate(spec["estimator"], df["Teks_Bersih"], y_fit, df["TingkatFatalitas"])

        row = {
            "model": name,
            "target_fit": spec["target"],
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
    output_path = OUTPUT_DIR / "single_models_results.csv"
    results.to_csv(output_path, index=False)
    print("\nSaved:", output_path)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()

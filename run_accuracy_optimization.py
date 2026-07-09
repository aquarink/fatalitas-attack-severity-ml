from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
from sklearn.svm import LinearSVC, SVC

from fatalitas_core import (
    XGBoostLabelEncodedClassifier,
    cross_validate_model,
    holdout_evaluate,
    load_dataset,
)

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


SAFE_CATEGORICAL_COLS = [
    "BentukSerangan",
    "JenisTarget",
    "AfiliasiJaringan",
    "NamaPelaku",
    "TargetLokasi",
]


def make_hybrid_preprocessor(text_col: str, categorical_cols):
    transformers = [
        (
            "text",
            TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1, sublinear_tf=True),
            text_col,
        )
    ]
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="NA")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                list(categorical_cols),
            )
        )
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=1.0)


def build_search_space():
    specs = []
    feature_sets = [
        ("text_only", []),
        ("text_plus_attack_target", ["BentukSerangan", "JenisTarget"]),
        ("text_plus_actor", ["AfiliasiJaringan", "NamaPelaku"]),
        ("text_plus_location", ["TargetLokasi"]),
        ("text_plus_all_safe_cat", SAFE_CATEGORICAL_COLS),
    ]

    for feature_name, cat_cols in feature_sets:
        pre = make_hybrid_preprocessor("Teks_Bersih", cat_cols)
        specs.append(
            {
                "name": f"logreg_{feature_name}",
                "estimator": Pipeline(
                    [
                        ("pre", pre),
                        (
                            "clf",
                            LogisticRegression(
                                max_iter=4000,
                                class_weight=None,
                                solver="liblinear",
                                C=2.0,
                                random_state=42,
                            ),
                        ),
                    ]
                ),
            }
        )
        specs.append(
            {
                "name": f"logreg_balanced_{feature_name}",
                "estimator": Pipeline(
                    [
                        ("pre", pre),
                        (
                            "clf",
                            LogisticRegression(
                                max_iter=4000,
                                class_weight="balanced",
                                solver="liblinear",
                                C=2.0,
                                random_state=42,
                            ),
                        ),
                    ]
                ),
            }
        )
        specs.append(
            {
                "name": f"linearsvc_balanced_{feature_name}",
                "estimator": Pipeline(
                    [
                        ("pre", pre),
                        ("clf", LinearSVC(C=1.0, class_weight="balanced", random_state=42)),
                    ]
                ),
            }
        )
        specs.append(
            {
                "name": f"svc_rbf_balanced_{feature_name}",
                "estimator": Pipeline(
                    [
                        ("pre", pre),
                        ("scale", MaxAbsScaler()),
                        (
                            "clf",
                            SVC(
                                kernel="rbf",
                                C=2.0,
                                gamma="scale",
                                probability=True,
                                class_weight="balanced",
                                random_state=42,
                            ),
                        ),
                    ]
                ),
            }
        )
        specs.append(
            {
                "name": f"svc_rbf_plain_{feature_name}",
                "estimator": Pipeline(
                    [
                        ("pre", pre),
                        ("scale", MaxAbsScaler()),
                        ("clf", SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, random_state=42)),
                    ]
                ),
            }
        )
        try:
            import xgboost as xgb

            _ = xgb
            specs.append(
                {
                    "name": f"xgboost_{feature_name}",
                    "estimator": Pipeline(
                        [
                            ("pre", pre),
                            (
                                "clf",
                                XGBoostLabelEncodedClassifier(
                                    n_estimators=160,
                                    learning_rate=0.06,
                                    max_depth=3,
                                    subsample=0.9,
                                    colsample_bytree=0.9,
                                ),
                            ),
                        ]
                    ),
                }
            )
        except Exception:
            pass

    return specs


def main():
    df = load_dataset(use_normalization=True)
    for column in SAFE_CATEGORICAL_COLS:
        df[column] = df[column].fillna("NA").astype(str)

    rows = []
    print("=== Accuracy-Oriented Search ===")
    for spec in build_search_space():
        cv_metrics = cross_validate_model(
            spec["estimator"], df, df["TingkatFatalitas"], df["TingkatFatalitas"]
        )
        holdout_metrics = holdout_evaluate(
            spec["estimator"], df, df["TingkatFatalitas"], df["TingkatFatalitas"]
        )
        row = {
            "model": spec["name"],
            "cv_accuracy": cv_metrics["accuracy"],
            "cv_error_rate": cv_metrics["error_rate"],
            "cv_macro_f1": cv_metrics["macro_f1"],
            "cv_balanced_accuracy": cv_metrics["balanced_accuracy"],
            "holdout_accuracy": holdout_metrics["accuracy"],
            "holdout_error_rate": holdout_metrics["error_rate"],
            "holdout_macro_f1": holdout_metrics["macro_f1"],
            "holdout_balanced_accuracy": holdout_metrics["balanced_accuracy"],
        }
        rows.append(row)
        print(
            f"{row['model']}: holdout_acc={row['holdout_accuracy']:.4f}, "
            f"holdout_err={row['holdout_error_rate']:.4f}, cv_acc={row['cv_accuracy']:.4f}"
        )

    results = pd.DataFrame(rows).sort_values(
        ["holdout_accuracy", "cv_accuracy", "holdout_macro_f1", "cv_macro_f1"],
        ascending=False,
    )
    output_path = OUTPUT_DIR / "accuracy_optimization_results.csv"
    results.to_csv(output_path, index=False)
    print("\nSaved:", output_path)
    print(results.head(15).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()

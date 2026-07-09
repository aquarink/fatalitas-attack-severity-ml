import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

DATA_FILE = Path("MtblDataSerAsSelPakistan2425.xlsx")
RANDOM_STATE = 42
CLASS_ORDER = ["0-2 Rendah", "3-5 Sedang", "6-7 Tinggi", "8+ Sangat Tinggi"]
NORMALIZATION_MAP = {
    "ttp": "tehrik i taliban pakistan",
    "blf": "baloch liberation front",
    "bla": "baloch liberation army",
    "iskp": "islamic state khorasan province",
    "sfs": "security forces",
    "fc": "frontier corps",
    "ctd": "counter terrorism department",
    "govt": "government",
    "ied": "improvised explosive device",
    "vbied": "vehicle borne improvised explosive device",
    "kp": "khyber pakhtunkhwa",
}
STOP_WORDS = ENGLISH_STOP_WORDS.union(
    {"report", "reports", "said", "added", "stated", "according", "pakistan", "area", "district", "province"}
)


def assign_fatality_label(value: float) -> str:
    if value <= 2:
        return "0-2 Rendah"
    if value <= 5:
        return "3-5 Sedang"
    if value <= 7:
        return "6-7 Tinggi"
    return "8+ Sangat Tinggi"


def bucket_from_deaths(values) -> np.ndarray:
    labels = []
    for value in values:
        labels.append(assign_fatality_label(max(0.0, float(value))))
    return np.array(labels)


def clean_text(text: str) -> str:
    text = str(text).lower()
    for short, full in NORMALIZATION_MAP.items():
        text = re.sub(rf"\b{re.escape(short)}\b", full, text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [token for token in text.split() if token not in STOP_WORDS and len(token) > 2]
    return " ".join(tokens)


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.dropna(subset=["RgksPeristiwa"]).copy()
    df["RgksPeristiwa"] = df["RgksPeristiwa"].astype(str)
    df = df.drop_duplicates(subset=["RgksPeristiwa"]).reset_index(drop=True)
    df["KorbanTewas"] = df["KorbanTewas"].fillna(0)
    df["TingkatFatalitas"] = df["KorbanTewas"].apply(assign_fatality_label)
    df["Teks_Bersih"] = df["RgksPeristiwa"].apply(clean_text)
    return df


class XGBoostLabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.objective = objective
        self.eval_metric = eval_metric
        self.random_state = random_state

    def fit(self, X, y):
        import xgboost as xgb

        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.model_ = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective=self.objective,
            eval_metric=self.eval_metric,
            num_class=len(self.label_encoder_.classes_),
            random_state=self.random_state,
        )
        self.model_.fit(X, y_encoded)
        self.classes_ = self.label_encoder_.classes_
        return self

    def predict(self, X):
        pred_encoded = self.model_.predict(X)
        return self.label_encoder_.inverse_transform(pred_encoded.astype(int))

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


class OrdinalReductionClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, class_order=None):
        self.base_estimator = base_estimator
        self.class_order = class_order if class_order is not None else CLASS_ORDER

    def fit(self, X, y):
        self.classes_ = np.array(self.class_order)
        self.threshold_models_ = []
        y_series = pd.Series(y)
        for idx in range(len(self.classes_) - 1):
            threshold_label = self.classes_[idx]
            binary_target = (y_series.isin(self.classes_[idx + 1 :])).astype(int)
            model = clone(self.base_estimator)
            model.fit(X, binary_target)
            self.threshold_models_.append(model)
        return self

    def predict_proba(self, X):
        gt_probs = []
        for model in self.threshold_models_:
            threshold_prob = model.predict_proba(X)[:, 1]
            gt_probs.append(threshold_prob)
        gt_probs = np.column_stack(gt_probs)
        gt_probs = np.minimum.accumulate(gt_probs, axis=1)

        n_samples = gt_probs.shape[0]
        n_classes = len(self.classes_)
        class_probs = np.zeros((n_samples, n_classes))
        class_probs[:, 0] = 1.0 - gt_probs[:, 0]
        for idx in range(1, n_classes - 1):
            class_probs[:, idx] = gt_probs[:, idx - 1] - gt_probs[:, idx]
        class_probs[:, -1] = gt_probs[:, -1]
        class_probs = np.clip(class_probs, 0.0, 1.0)
        class_probs = class_probs / class_probs.sum(axis=1, keepdims=True)
        return class_probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


class BucketedRegressor(BaseEstimator, ClassifierMixin):
    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, X, y):
        self.classes_ = np.array(CLASS_ORDER)
        self.regressor_ = clone(self.regressor)
        self.regressor_.fit(X, y)
        return self

    def predict(self, X):
        pred = self.regressor_.predict(X)
        return bucket_from_deaths(pred)


def quadratic_kappa(y_true, y_pred):
    label_map = {label: idx for idx, label in enumerate(CLASS_ORDER)}
    y_true_num = [label_map[label] for label in y_true]
    y_pred_num = [label_map[label] for label in y_pred]
    return cohen_kappa_score(y_true_num, y_pred_num, weights="quadratic")


def build_models():
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    linear_svc_prob = CalibratedClassifierCV(
        estimator=LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE),
        cv=3,
    )
    ordinal_logreg = OrdinalReductionClassifier(
        base_estimator=LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        )
    )
    ordinal_svm = OrdinalReductionClassifier(base_estimator=linear_svc_prob)

    models = {
        "complement_nb": {
            "target": "label",
            "estimator": Pipeline([("tfidf", tfidf), ("clf", ComplementNB(alpha=0.5))]),
        },
        "linear_svc": {
            "target": "label",
            "estimator": Pipeline(
                [("tfidf", tfidf), ("clf", LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE))]
            ),
        },
        "logistic_regression": {
            "target": "label",
            "estimator": Pipeline(
                [
                    ("tfidf", tfidf),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=3000,
                            class_weight="balanced",
                            solver="liblinear",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        },
        "ros_complement_nb": {
            "target": "label",
            "estimator": ImbPipeline(
                [("tfidf", tfidf), ("ros", RandomOverSampler(random_state=RANDOM_STATE)), ("clf", ComplementNB(alpha=0.5))]
            ),
        },
        "smote_svd_logreg": {
            "target": "label",
            "estimator": ImbPipeline(
                [
                    ("tfidf", tfidf),
                    ("svd", TruncatedSVD(n_components=100, random_state=RANDOM_STATE)),
                    ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=3000,
                            solver="lbfgs",
                            multi_class="auto",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        },
        "ordinal_logistic": {
            "target": "label",
            "estimator": Pipeline([("tfidf", tfidf), ("clf", ordinal_logreg)]),
        },
        "ordinal_svm": {
            "target": "label",
            "estimator": Pipeline([("tfidf", tfidf), ("clf", ordinal_svm)]),
        },
        "ridge_regression_bucket": {
            "target": "deaths",
            "estimator": Pipeline([("tfidf", tfidf), ("clf", BucketedRegressor(Ridge(alpha=1.0)))]),
        },
    }

    try:
        import xgboost as xgb

        _ = xgb
        models["xgboost_classifier"] = {
            "target": "label",
            "estimator": Pipeline([("tfidf", tfidf), ("clf", XGBoostLabelEncodedClassifier())]),
        }
        models["xgboost_regression_bucket"] = {
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
                                random_state=RANDOM_STATE,
                            )
                        ),
                    ),
                ]
            ),
        }
    except Exception as exc:
        print(f"[info] XGBoost dilewati: {exc}")

    return models


def evaluate_models(models, x_text, y_label, y_deaths):
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    x_text = pd.Series(x_text).reset_index(drop=True)
    y_label = pd.Series(y_label).reset_index(drop=True)
    y_deaths = pd.Series(y_deaths).reset_index(drop=True)

    for name, spec in models.items():
        model = spec["estimator"]
        target_type = spec["target"]
        fold_metrics = []
        for train_idx, test_idx in cv.split(x_text, y_label):
            x_train = x_text.iloc[train_idx]
            x_test = x_text.iloc[test_idx]
            y_train = y_label.iloc[train_idx] if target_type == "label" else y_deaths.iloc[train_idx]
            y_test_label = y_label.iloc[test_idx]

            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            fold_metrics.append(
                {
                    "accuracy": accuracy_score(y_test_label, pred),
                    "balanced_accuracy": balanced_accuracy_score(y_test_label, pred),
                    "macro_f1": f1_score(y_test_label, pred, average="macro"),
                    "quadratic_kappa": quadratic_kappa(y_test_label, pred),
                }
            )

        rows.append(
            {
                "model": name,
                "model_type": target_type,
                "accuracy_mean": np.mean([m["accuracy"] for m in fold_metrics]),
                "error_rate_mean": 1.0 - np.mean([m["accuracy"] for m in fold_metrics]),
                "balanced_accuracy_mean": np.mean([m["balanced_accuracy"] for m in fold_metrics]),
                "macro_f1_mean": np.mean([m["macro_f1"] for m in fold_metrics]),
                "quadratic_kappa_mean": np.mean([m["quadratic_kappa"] for m in fold_metrics]),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["macro_f1_mean", "balanced_accuracy_mean", "quadratic_kappa_mean", "accuracy_mean"],
        ascending=False,
    )


def holdout_report(model_name, spec, x_text, y_label, y_deaths):
    x_train, x_test, y_train_label, y_test_label, y_train_deaths, y_test_deaths = train_test_split(
        x_text,
        y_label,
        y_deaths,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_label,
    )
    model = spec["estimator"]
    y_train = y_train_label if spec["target"] == "label" else y_train_deaths
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print("\n=== Holdout Evaluation ===")
    print(f"Model               : {model_name}")
    print(f"Target Fit          : {spec['target']}")
    print(f"Accuracy            : {accuracy_score(y_test_label, pred):.4f}")
    print(f"Error Rate          : {1.0 - accuracy_score(y_test_label, pred):.4f}")
    print(f"Balanced Accuracy   : {balanced_accuracy_score(y_test_label, pred):.4f}")
    print(f"Macro F1            : {f1_score(y_test_label, pred, average='macro'):.4f}")
    print(f"Quadratic Kappa     : {quadratic_kappa(y_test_label, pred):.4f}")
    print("Confusion Matrix")
    print(confusion_matrix(y_test_label, pred, labels=CLASS_ORDER))


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_FILE}")

    df = load_dataset(DATA_FILE)
    print("=== Dataset Summary ===")
    print(f"Jumlah baris setelah cleaning: {len(df)}")
    print("Distribusi kelas:")
    print(df["TingkatFatalitas"].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int).to_string())

    models = build_models()
    results = evaluate_models(models, df["Teks_Bersih"], df["TingkatFatalitas"], df["KorbanTewas"])

    print("\n=== Comparative Results (4-Fold CV) ===")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    best_name = results.iloc[0]["model"]
    holdout_report(best_name, models[best_name], df["Teks_Bersih"], df["TingkatFatalitas"], df["KorbanTewas"])


if __name__ == "__main__":
    main()

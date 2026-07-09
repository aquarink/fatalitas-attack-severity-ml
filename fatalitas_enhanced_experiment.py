import re
import warnings
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

DATA_FILE = Path("MtblDataSerAsSelPakistan2425.xlsx")
RANDOM_STATE = 42
CLASS_ORDER = [
    "0-2 Rendah",
    "3-5 Sedang",
    "6-7 Tinggi",
    "8+ Sangat Tinggi",
]
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
    {
        "report",
        "reports",
        "said",
        "added",
        "stated",
        "according",
        "pakistan",
        "area",
        "district",
        "province",
    }
)


def assign_fatality_label(value: float) -> str:
    if value <= 2:
        return "0-2 Rendah"
    if value <= 5:
        return "3-5 Sedang"
    if value <= 7:
        return "6-7 Tinggi"
    return "8+ Sangat Tinggi"


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


def build_models():
    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )

    models = {
        "linear_svc_balanced": Pipeline(
            [
                ("tfidf", vectorizer),
                ("clf", LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "linear_svc_calibrated": Pipeline(
            [
                ("tfidf", vectorizer),
                (
                    "clf",
                    CalibratedClassifierCV(
                        estimator=LinearSVC(
                            C=1.0,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                        cv=3,
                    ),
                ),
            ]
        ),
        "logreg_balanced": Pipeline(
            [
                ("tfidf", vectorizer),
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
        "complement_nb": Pipeline(
            [
                ("tfidf", vectorizer),
                ("clf", ComplementNB(alpha=0.5)),
            ]
        ),
        "ros_complement_nb": ImbPipeline(
            [
                ("tfidf", vectorizer),
                ("ros", RandomOverSampler(random_state=RANDOM_STATE)),
                ("clf", ComplementNB(alpha=0.5)),
            ]
        ),
    }

    try:
        import xgboost as xgb
        _ = xgb
        models["xgboost_text"] = Pipeline(
            [
                ("tfidf", vectorizer),
                ("clf", XGBoostLabelEncodedClassifier()),
            ]
        )
    except Exception as exc:
        print(f"[info] XGBoost dilewati: {exc}")

    return models


def evaluate_models(models, x_text, y_label) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "macro_f1": make_scorer(f1_score, average="macro"),
    }
    rows = []
    for name, model in models.items():
        scores = cross_validate(model, x_text, y_label, cv=cv, scoring=scoring, n_jobs=1)
        rows.append(
            {
                "model": name,
                "accuracy_mean": scores["test_accuracy"].mean(),
                "accuracy_std": scores["test_accuracy"].std(),
                "balanced_accuracy_mean": scores["test_balanced_accuracy"].mean(),
                "balanced_accuracy_std": scores["test_balanced_accuracy"].std(),
                "macro_f1_mean": scores["test_macro_f1"].mean(),
                "macro_f1_std": scores["test_macro_f1"].std(),
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["macro_f1_mean", "balanced_accuracy_mean", "accuracy_mean"],
        ascending=False,
    )
    return result


def fit_holdout_report(model_name, model, x_text, y_label):
    x_train, x_test, y_train, y_test = train_test_split(
        x_text,
        y_label,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_label,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    print("\n=== Holdout Evaluation ===")
    print(f"Model: {model_name}")
    print(f"Accuracy           : {accuracy_score(y_test, pred):.4f}")
    print(f"Balanced Accuracy  : {balanced_accuracy_score(y_test, pred):.4f}")
    print(f"Macro F1           : {f1_score(y_test, pred, average='macro'):.4f}")
    print("\nClassification Report")
    print(classification_report(y_test, pred, labels=CLASS_ORDER, digits=3, zero_division=0))
    print("Confusion Matrix")
    print(confusion_matrix(y_test, pred, labels=CLASS_ORDER))

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(x_test)
        roc_auc = roc_auc_score(y_test, prob, labels=CLASS_ORDER, multi_class="ovr", average="macro")
        print(f"Macro ROC-AUC      : {roc_auc:.4f}")


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_FILE}")

    df = load_dataset(DATA_FILE)
    print("=== Dataset Summary ===")
    print(f"Jumlah baris setelah cleaning: {len(df)}")
    print("Distribusi kelas:")
    print(df["TingkatFatalitas"].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int).to_string())
    print("\nPersentase kelas:")
    print((df["TingkatFatalitas"].value_counts(normalize=True).reindex(CLASS_ORDER).fillna(0) * 100).round(2).to_string())

    models = build_models()
    results = evaluate_models(models, df["Teks_Bersih"], df["TingkatFatalitas"])
    print("\n=== Cross-Validation Comparison ===")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    best_name = results.iloc[0]["model"]
    best_model = models[best_name]
    fit_holdout_report(best_name, best_model, df["Teks_Bersih"], df["TingkatFatalitas"])


if __name__ == "__main__":
    main()

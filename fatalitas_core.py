import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
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
    return np.array([assign_fatality_label(max(0.0, float(value))) for value in values])


def clean_text(text: str, use_normalization: bool = True) -> str:
    text = str(text).lower()
    if use_normalization:
        for short, full in NORMALIZATION_MAP.items():
            text = re.sub(rf"\b{re.escape(short)}\b", full, text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [token for token in text.split() if token not in STOP_WORDS and len(token) > 2]
    return " ".join(tokens)


def load_dataset(path: Path = DATA_FILE, use_normalization: bool = True) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.dropna(subset=["RgksPeristiwa"]).copy()
    df["RgksPeristiwa"] = df["RgksPeristiwa"].astype(str)
    df = df.drop_duplicates(subset=["RgksPeristiwa"]).reset_index(drop=True)
    df["KorbanTewas"] = df["KorbanTewas"].fillna(0)
    df["TingkatFatalitas"] = df["KorbanTewas"].apply(assign_fatality_label)
    df["Teks_Bersih"] = df["RgksPeristiwa"].apply(lambda value: clean_text(value, use_normalization))
    return df


def build_tfidf(max_features: int = 3000) -> TfidfVectorizer:
    return TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def quadratic_kappa(y_true, y_pred) -> float:
    label_map = {label: idx for idx, label in enumerate(CLASS_ORDER)}
    y_true_num = [label_map[label] for label in y_true]
    y_pred_num = [label_map[label] for label in y_pred]
    return cohen_kappa_score(y_true_num, y_pred_num, weights="quadratic")


def summarize_predictions(y_true, y_pred) -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    return {
        "accuracy": accuracy,
        "error_rate": 1.0 - accuracy,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "quadratic_kappa": quadratic_kappa(y_true, y_pred),
    }


def cross_validate_model(estimator, x_text, y_fit, y_eval, n_splits: int = 4) -> dict:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    x_text = x_text.reset_index(drop=True) if isinstance(x_text, pd.DataFrame) else pd.Series(x_text).reset_index(drop=True)
    y_fit = pd.Series(y_fit).reset_index(drop=True)
    y_eval = pd.Series(y_eval).reset_index(drop=True)

    fold_metrics = []
    for train_idx, test_idx in cv.split(x_text, y_eval):
        x_train = x_text.iloc[train_idx]
        x_test = x_text.iloc[test_idx]
        estimator_fold = clone(estimator)
        estimator_fold.fit(x_train, y_fit.iloc[train_idx])
        pred = estimator_fold.predict(x_test)
        fold_metrics.append(summarize_predictions(y_eval.iloc[test_idx], pred))

    return {metric: float(np.mean([fold[metric] for fold in fold_metrics])) for metric in fold_metrics[0]}


def holdout_evaluate(estimator, x_text, y_fit, y_eval, test_size: float = 0.2) -> dict:
    x_text = x_text.reset_index(drop=True) if isinstance(x_text, pd.DataFrame) else pd.Series(x_text).reset_index(drop=True)
    y_fit = pd.Series(y_fit).reset_index(drop=True)
    y_eval = pd.Series(y_eval).reset_index(drop=True)
    x_train, x_test, y_fit_train, _, y_eval_train, y_eval_test = train_test_split(
        x_text,
        y_fit,
        y_eval,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y_eval,
    )
    estimator.fit(x_train, y_fit_train)
    pred = estimator.predict(x_test)
    result = summarize_predictions(y_eval_test, pred)
    result["predictions"] = pred
    result["truth"] = np.array(y_eval_test)
    return result


class XGBoostLabelEncodedClassifier(ClassifierMixin, BaseEstimator):
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


class OrdinalReductionClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, base_estimator, class_order=None):
        self.base_estimator = base_estimator
        self.class_order = class_order if class_order is not None else CLASS_ORDER

    def fit(self, X, y):
        self.classes_ = np.array(self.class_order)
        self.threshold_models_ = []
        y_series = pd.Series(y)
        for idx in range(len(self.classes_) - 1):
            binary_target = (y_series.isin(self.classes_[idx + 1 :])).astype(int)
            threshold_model = clone(self.base_estimator)
            threshold_model.fit(X, binary_target)
            self.threshold_models_.append(threshold_model)
        return self

    def predict_proba(self, X):
        gt_probs = np.column_stack([model.predict_proba(X)[:, 1] for model in self.threshold_models_])
        gt_probs = np.minimum.accumulate(gt_probs, axis=1)

        class_probs = np.zeros((gt_probs.shape[0], len(self.classes_)))
        class_probs[:, 0] = 1.0 - gt_probs[:, 0]
        for idx in range(1, len(self.classes_) - 1):
            class_probs[:, idx] = gt_probs[:, idx - 1] - gt_probs[:, idx]
        class_probs[:, -1] = gt_probs[:, -1]
        class_probs = np.clip(class_probs, 0.0, 1.0)
        return class_probs / class_probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class BucketedRegressor(ClassifierMixin, BaseEstimator):
    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, X, y):
        self.regressor_ = clone(self.regressor)
        self.regressor_.fit(X, y)
        self.classes_ = np.array(CLASS_ORDER)
        return self

    def predict(self, X):
        return bucket_from_deaths(self.regressor_.predict(X))


class ProbabilityVotingEnsemble(ClassifierMixin, BaseEstimator):
    def __init__(self, estimators, weights=None):
        self.estimators = estimators
        self.weights = weights

    def fit(self, X, y):
        self.classes_ = np.array(CLASS_ORDER)
        self.estimators_ = []
        for name, estimator in self.estimators:
            model = clone(estimator)
            model.fit(X, y)
            self.estimators_.append((name, model))
        return self

    def predict_proba(self, X):
        probas = []
        for _, model in self.estimators_:
            proba = model.predict_proba(X)
            probas.append(proba)
        stacked = np.stack(probas, axis=0)
        if self.weights is None:
            return stacked.mean(axis=0)
        weights = np.array(self.weights, dtype=float).reshape(-1, 1, 1)
        return (stacked * weights).sum(axis=0) / weights.sum()

    def predict(self, X):
        mean_proba = self.predict_proba(X)
        return self.classes_[np.argmax(mean_proba, axis=1)]


def make_calibrated_svc() -> CalibratedClassifierCV:
    return CalibratedClassifierCV(
        estimator=LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE),
        cv=3,
    )


def make_logreg() -> LogisticRegression:
    return LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE,
    )


def make_ridge_bucket() -> BucketedRegressor:
    return BucketedRegressor(Ridge(alpha=1.0))

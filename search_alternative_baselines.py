import json
import random
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, wordpunct_tokenize
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

DATA_FILE = Path("MtblDataSerAsSelPakistan2425.xlsx")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


def _safe_word_tokenize(text: str):
    try:
        return word_tokenize(text)
    except Exception:
        return wordpunct_tokenize(text)


def tentukan_fatalitas(korban):
    if korban <= 2:
        return "0-2 Rendah"
    if korban <= 5:
        return "3-5 Sedang"
    if korban <= 7:
        return "6-7 Tinggi"
    return "8+ Sangat Tinggi"


def text_cleaning(text, stop_words, kamus_normalisasi, lemmatizer):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = _safe_word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [kamus_normalisasi.get(word, word) for word in tokens]
    text_normalized = " ".join(tokens)
    tokens = text_normalized.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if len(word) > 2]
    return " ".join(tokens)


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " ").replace("-", " ").lower())
    synonyms.discard(word)
    return sorted(synonyms)


def synonym_replacement(tokens, rng, n):
    new_tokens = tokens.copy()
    random_word_list = sorted(set(tokens))
    rng.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if synonyms:
            synonym = rng.choice(synonyms)
            new_tokens = [synonym if word == random_word else word for word in new_tokens]
            num_replaced += 1
        if num_replaced >= n:
            break
    return " ".join(new_tokens)


def load_preprocessed():
    df = pd.read_excel(DATA_FILE)
    df = df.dropna(subset=["RgksPeristiwa"])
    df["RgksPeristiwa"] = df["RgksPeristiwa"].astype(str)
    df = df.drop_duplicates(subset=["RgksPeristiwa"]).reset_index(drop=True)
    df["KorbanTewas"] = df["KorbanTewas"].fillna(0)
    df["TingkatFatalitas"] = df["KorbanTewas"].apply(tentukan_fatalitas)
    kamus_normalisasi = {
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
    stop_words = set(stopwords.words("english")).union(
        {"report", "reports", "said", "added", "stated", "according", "pakistan", "area", "district", "province"}
    )
    lemmatizer = WordNetLemmatizer()
    df["Teks_Bersih"] = df["RgksPeristiwa"].apply(
        lambda text: text_cleaning(text, stop_words, kamus_normalisasi, lemmatizer)
    )
    return df


def build_augmented_train(train_data, aug_seed=21, augment_repeats=3, synonym_n=2):
    rng = random.Random(aug_seed)
    data_tambahan = []
    for _, row in train_data.iterrows():
        if row["TingkatFatalitas"] in ["6-7 Tinggi", "8+ Sangat Tinggi"]:
            tokens = str(row["Teks_Bersih"]).split()
            if len(tokens) > 5:
                for _ in range(augment_repeats):
                    teks_baru = synonym_replacement(tokens, rng=rng, n=synonym_n)
                    data_tambahan.append({"Teks_Bersih": teks_baru, "TingkatFatalitas": row["TingkatFatalitas"]})
    return pd.concat([train_data, pd.DataFrame(data_tambahan)], ignore_index=True)


def prepare_features(df, max_features=1500):
    X = df["Teks_Bersih"]
    y = df["TingkatFatalitas"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    train_data = pd.DataFrame({"Teks_Bersih": X_train, "TingkatFatalitas": y_train})
    df_augmented = build_augmented_train(train_data)

    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(df_augmented["Teks_Bersih"])
    X_test_tfidf = tfidf.transform(X_test)

    scaler = MaxAbsScaler()
    X_train_scaled = scaler.fit_transform(X_train_tfidf)
    X_test_scaled = scaler.transform(X_test_tfidf)

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(df_augmented["TingkatFatalitas"])
    y_test_encoded = le.transform(y_test)

    return {
        "X_train_tfidf": X_train_tfidf,
        "X_test_tfidf": X_test_tfidf,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train_text": df_augmented["TingkatFatalitas"].reset_index(drop=True),
        "y_test_text": y_test.reset_index(drop=True),
        "y_train_encoded": y_train_encoded,
        "y_test_encoded": y_test_encoded,
        "label_encoder": le,
    }


def build_models():
    models = {
        "svm_rbf_only": {"kind": "encoded", "needs_scaled": True, "model": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=42)},
        "linearsvc_only": {"kind": "text", "needs_scaled": False, "model": LinearSVC(C=1.0, class_weight="balanced", random_state=42)},
        "linearsvc_calibrated": {"kind": "text", "needs_scaled": False, "model": CalibratedClassifierCV(estimator=LinearSVC(C=1.0, class_weight="balanced", random_state=42), cv=3)},
        "logreg_only": {"kind": "text", "needs_scaled": False, "model": LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", random_state=42)},
        "complement_nb": {"kind": "text", "needs_scaled": False, "model": ComplementNB(alpha=0.5)},
        "ros_complement_nb": {
            "kind": "text",
            "needs_scaled": False,
            "model": ImbPipeline([("ros", RandomOverSampler(random_state=42)), ("clf", ComplementNB(alpha=0.5))]),
        },
    }

    try:
        import xgboost as xgb

        models["xgboost_only"] = {
            "kind": "encoded",
            "needs_scaled": True,
            "model": xgb.XGBClassifier(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                random_state=42,
                eval_metric="mlogloss",
            ),
        }
        models["ensemble_svm_xgb"] = {
            "kind": "encoded",
            "needs_scaled": True,
            "model": VotingClassifier(
                estimators=[
                    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=42)),
                    ("xgb", xgb.XGBClassifier(
                        n_estimators=300,
                        learning_rate=0.03,
                        max_depth=4,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="multi:softprob",
                        random_state=42,
                        eval_metric="mlogloss",
                    )),
                ],
                voting="soft",
            ),
        }
    except Exception as exc:
        print(f"[info] XGBoost dilewati: {exc}")

    models["ensemble_cnb_logreg"] = {
        "kind": "text",
        "needs_scaled": False,
        "model": VotingClassifier(
            estimators=[
                ("cnb", ComplementNB(alpha=0.5)),
                ("logreg", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", random_state=42)),
            ],
            voting="soft",
        ),
    }
    models["ensemble_cnb_calibrated_svc"] = {
        "kind": "text",
        "needs_scaled": False,
        "model": VotingClassifier(
            estimators=[
                ("cnb", ComplementNB(alpha=0.5)),
                ("svc", CalibratedClassifierCV(estimator=LinearSVC(C=1.0, class_weight="balanced", random_state=42), cv=3)),
            ],
            voting="soft",
        ),
    }
    return models


def evaluate_model(name, spec, prepared):
    if spec["kind"] == "encoded":
        X_train = prepared["X_train_scaled"] if spec["needs_scaled"] else prepared["X_train_tfidf"]
        X_test = prepared["X_test_scaled"] if spec["needs_scaled"] else prepared["X_test_tfidf"]
        y_train = prepared["y_train_encoded"]
        y_test = prepared["y_test_encoded"]
        model = spec["model"]
        if name == "xgboost_only":
            model.set_params(num_class=len(prepared["label_encoder"].classes_))
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
    else:
        X_train = prepared["X_train_tfidf"]
        X_test = prepared["X_test_tfidf"]
        y_train = prepared["y_train_text"]
        y_test = prepared["y_test_text"]
        model = spec["model"]
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    return {
        "model": name,
        "accuracy": accuracy,
        "error_rate": 1.0 - accuracy,
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
    }


def main():
    random.seed(42)
    np.random.seed(42)

    df = load_preprocessed()
    prepared = prepare_features(df, max_features=1500)
    models = build_models()
    rows = []

    print("=== Alternative Baselines Around Notebook Recipe ===")
    for name, spec in models.items():
        row = evaluate_model(name, spec, prepared)
        rows.append(row)
        print(
            f"{name}: acc={row['accuracy']:.4f} err={row['error_rate']:.4f} "
            f"bal_acc={row['balanced_accuracy']:.4f} macro_f1={row['macro_f1']:.4f}"
        )

    results = pd.DataFrame(rows).sort_values(
        ["error_rate", "accuracy", "macro_f1", "balanced_accuracy"],
        ascending=[True, False, False, False],
    )
    csv_path = OUTPUT_DIR / "alternative_baselines_results.csv"
    json_path = OUTPUT_DIR / "alternative_baselines_best.json"
    results.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(results.iloc[0].to_dict(), indent=2), encoding="utf-8")

    print("\nSaved:", csv_path)
    print("Saved:", json_path)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()

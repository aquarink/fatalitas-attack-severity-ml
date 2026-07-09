import itertools
import json
import random
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, wordpunct_tokenize
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.svm import SVC

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


def build_augmented_train(train_data, aug_seed, augment_repeats, synonym_n):
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


def run_config(df, cfg):
    import xgboost as xgb

    X = df["Teks_Bersih"]
    y = df["TingkatFatalitas"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    train_data = pd.DataFrame({"Teks_Bersih": X_train, "TingkatFatalitas": y_train})
    df_augmented = build_augmented_train(
        train_data=train_data,
        aug_seed=cfg["aug_seed"],
        augment_repeats=cfg["augment_repeats"],
        synonym_n=cfg["synonym_n"],
    )

    tfidf = TfidfVectorizer(max_features=cfg["max_features"], ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(df_augmented["Teks_Bersih"])
    X_test_tfidf = tfidf.transform(X_test)

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(df_augmented["TingkatFatalitas"])
    y_test_encoded = le.transform(y_test)

    scaler = MaxAbsScaler()
    X_train_final = scaler.fit_transform(X_train_tfidf)
    X_test_final = scaler.transform(X_test_tfidf)

    model_svm = SVC(
        kernel="rbf",
        C=cfg["svm_c"],
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=42,
    )
    model_xgb = xgb.XGBClassifier(
        n_estimators=cfg["xgb_estimators"],
        learning_rate=cfg["xgb_lr"],
        max_depth=cfg["xgb_depth"],
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(le.classes_),
        random_state=42,
        eval_metric="mlogloss",
    )
    ensemble = VotingClassifier(
        estimators=[("svm", model_svm), ("xgb", model_xgb)],
        voting="soft",
    )
    ensemble.fit(X_train_final, y_train_encoded)
    pred = ensemble.predict(X_test_final)
    acc = accuracy_score(y_test_encoded, pred)
    return {
        "accuracy": acc,
        "error_rate": 1.0 - acc,
        "balanced_accuracy": balanced_accuracy_score(y_test_encoded, pred),
        "macro_f1": f1_score(y_test_encoded, pred, average="macro"),
        "n_train_augmented": int(len(df_augmented)),
    }


def main():
    df = load_preprocessed()
    search_space = {
        "aug_seed": [21, 42],
        "augment_repeats": [2, 3],
        "synonym_n": [2, 3],
        "max_features": [1500],
        "svm_c": [1.0],
        "xgb_estimators": [300],
        "xgb_lr": [0.03],
        "xgb_depth": [4, 6],
    }

    keys = list(search_space)
    configs = [dict(zip(keys, values)) for values in itertools.product(*(search_space[key] for key in keys))]
    rows = []
    output_csv = OUTPUT_DIR / "ipynb_baseline_search_results.csv"
    output_json = OUTPUT_DIR / "ipynb_baseline_best_config.json"

    print("=== Search Around Notebook Baseline ===")
    for idx, cfg in enumerate(configs, start=1):
        metrics = run_config(df, cfg)
        row = {**cfg, **metrics}
        rows.append(row)
        partial = pd.DataFrame(rows).sort_values(
            ["error_rate", "accuracy", "macro_f1", "balanced_accuracy"],
            ascending=[True, False, False, False],
        )
        partial.to_csv(output_csv, index=False)
        output_json.write_text(json.dumps(partial.iloc[0].to_dict(), indent=2), encoding="utf-8")
        print(
            f"{idx:02d}/{len(configs)} "
            f"err={row['error_rate']:.4f} acc={row['accuracy']:.4f} "
            f"f1={row['macro_f1']:.4f} cfg={cfg}"
        )

    results = pd.DataFrame(rows).sort_values(
        ["error_rate", "accuracy", "macro_f1", "balanced_accuracy"],
        ascending=[True, False, False, False],
    )
    results.to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(results.iloc[0].to_dict(), indent=2), encoding="utf-8")

    print("\nBest Config")
    print(results.head(10).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nSaved:", output_csv)
    print("Saved:", output_json)


if __name__ == "__main__":
    main()

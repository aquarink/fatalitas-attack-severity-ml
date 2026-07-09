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
from sklearn.metrics import accuracy_score, auc, balanced_accuracy_score, classification_report, f1_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, label_binarize
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


def text_cleaning_15_steps(text, stop_words, kamus_normalisasi, lemmatizer):
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


def synonym_replacement(tokens, rng, n=2):
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


def main():
    import xgboost as xgb

    random.seed(42)
    np.random.seed(42)
    rng = random.Random(42)

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_FILE}")

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
        lambda text: text_cleaning_15_steps(text, stop_words, kamus_normalisasi, lemmatizer)
    )

    X = df["Teks_Bersih"]
    y = df["TingkatFatalitas"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    train_data = pd.DataFrame({"Teks_Bersih": X_train, "TingkatFatalitas": y_train})

    data_tambahan = []
    for _, row in train_data.iterrows():
        if row["TingkatFatalitas"] in ["6-7 Tinggi", "8+ Sangat Tinggi"]:
            tokens = str(row["Teks_Bersih"]).split()
            if len(tokens) > 5:
                for _ in range(2):
                    teks_baru = synonym_replacement(tokens, rng=rng, n=3)
                    data_tambahan.append({"Teks_Bersih": teks_baru, "TingkatFatalitas": row["TingkatFatalitas"]})

    df_augmented = pd.concat([train_data, pd.DataFrame(data_tambahan)], ignore_index=True)

    tfidf = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(df_augmented["Teks_Bersih"])
    X_test_tfidf = tfidf.transform(X_test)

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(df_augmented["TingkatFatalitas"])
    y_test_encoded = le.transform(y_test)
    nama_kelas = le.classes_

    scaler = MaxAbsScaler()
    X_train_final = scaler.fit_transform(X_train_tfidf)
    X_test_final = scaler.transform(X_test_tfidf)

    model_svm = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=42,
    )
    model_xgb = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(nama_kelas),
        random_state=42,
        eval_metric="mlogloss",
    )
    ensemble_voting = VotingClassifier(
        estimators=[("svm", model_svm), ("xgb", model_xgb)],
        voting="soft",
    )

    ensemble_voting.fit(X_train_final, y_train_encoded)
    y_pred = ensemble_voting.predict(X_test_final)
    akurasi = accuracy_score(y_test_encoded, y_pred)
    error_rate = 1.0 - akurasi
    balanced_acc = balanced_accuracy_score(y_test_encoded, y_pred)
    macro_f1 = f1_score(y_test_encoded, y_pred, average="macro")

    print("=" * 60)
    print(f"AKURASI FINAL ENSEMBLE (SVM + XGBoost): {akurasi * 100:.2f}%")
    print(f"ERROR RATE                           : {error_rate * 100:.2f}%")
    print(f"BALANCED ACCURACY                    : {balanced_acc * 100:.2f}%")
    print(f"MACRO F1                             : {macro_f1 * 100:.2f}%")
    print("=" * 60)
    print("\nClassification Report")
    print(classification_report(y_test_encoded, y_pred, target_names=nama_kelas, digits=3, zero_division=0))

    y_score = ensemble_voting.predict_proba(X_test_final)
    y_test_bin = label_binarize(y_test_encoded, classes=list(range(len(nama_kelas))))
    roc_auc = {}
    for i in range(y_test_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[nama_kelas[i]] = auc(fpr, tpr)

    mean_auc = float(np.mean(list(roc_auc.values())))
    print("Macro AUC:", f"{mean_auc:.3f}")
    for kelas, nilai in roc_auc.items():
        print(f"{kelas}: {nilai:.3f}")

    metrics = {
        "accuracy": akurasi,
        "error_rate": error_rate,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "macro_auc": mean_auc,
        "class_auc": roc_auc,
        "class_names": list(nama_kelas),
        "n_train_augmented": int(len(df_augmented)),
        "n_test": int(len(y_test)),
    }
    output_path = OUTPUT_DIR / "ipynb_baseline_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from fatalitas_core import (
    CLASS_ORDER,
    DATA_FILE,
    RANDOM_STATE,
    XGBoostLabelEncodedClassifier,
    build_tfidf,
    cross_validate_model,
    load_dataset,
    quadratic_kappa,
    summarize_predictions,
)

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


class CatBoostLabelEncodedClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        iterations=250,
        learning_rate=0.05,
        depth=5,
        loss_function="MultiClass",
        random_seed=RANDOM_STATE,
        verbose=False,
    ):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.loss_function = loss_function
        self.random_seed = random_seed
        self.verbose = verbose

    def fit(self, X, y):
        from catboost import CatBoostClassifier

        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.model_ = CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            loss_function=self.loss_function,
            random_seed=self.random_seed,
            verbose=self.verbose,
            allow_writing_files=False,
        )
        self.model_.fit(X, y_encoded)
        self.classes_ = self.label_encoder_.classes_
        return self

    def predict(self, X):
        pred = np.asarray(self.model_.predict(X)).reshape(-1).astype(int)
        return self.label_encoder_.inverse_transform(pred)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def evaluate_sklearn_holdout(estimator, x_text, y_label, test_size=0.2):
    x_train, x_test, y_train, y_test = train_test_split(
        x_text,
        y_label,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y_label,
    )
    estimator.fit(x_train, y_train)
    pred = estimator.predict(x_test)
    return summarize_predictions(y_test, pred)


def run_xgboost(df, include_cv=False):
    estimator = Pipeline(
        [
            ("tfidf", build_tfidf()),
            ("clf", XGBoostLabelEncodedClassifier(n_estimators=120, learning_rate=0.07, max_depth=3)),
        ]
    )
    holdout_metrics = evaluate_sklearn_holdout(estimator, df["Teks_Bersih"], df["TingkatFatalitas"])
    result = {"model": "NLP_XGBoost", **prefix_metrics("holdout", holdout_metrics)}
    if include_cv:
        cv_metrics = cross_validate_model(estimator, df["Teks_Bersih"], df["TingkatFatalitas"], df["TingkatFatalitas"])
        result.update(prefix_metrics("cv", cv_metrics))
    return result


def run_catboost(df, include_cv=False):
    estimator = Pipeline(
        [
            ("tfidf", build_tfidf(max_features=5000)),
            ("svd", TruncatedSVD(n_components=120, random_state=RANDOM_STATE)),
            ("clf", CatBoostLabelEncodedClassifier(iterations=100, learning_rate=0.07, depth=4)),
        ]
    )
    holdout_metrics = evaluate_sklearn_holdout(estimator, df["Teks_Bersih"], df["TingkatFatalitas"])
    result = {"model": "NLP_CatBoost", **prefix_metrics("holdout", holdout_metrics)}
    if include_cv:
        cv_metrics = cross_validate_model(estimator, df["Teks_Bersih"], df["TingkatFatalitas"], df["TingkatFatalitas"])
        result.update(prefix_metrics("cv", cv_metrics))
    return result


def prefix_metrics(prefix, metrics):
    return {
        f"{prefix}_accuracy": metrics["accuracy"],
        f"{prefix}_error_rate": metrics["error_rate"],
        f"{prefix}_balanced_accuracy": metrics["balanced_accuracy"],
        f"{prefix}_macro_f1": metrics["macro_f1"],
        f"{prefix}_quadratic_kappa": metrics["quadratic_kappa"],
    }


def run_bilstm(df, epochs=5, batch_size=32, max_features=4000, max_len=50, embedding_dim=48, hidden_dim=48):
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        print(f"[skip] BiLSTM dilewati karena PyTorch belum siap: {exc}")
        return None

    torch.manual_seed(RANDOM_STATE)
    torch.set_num_threads(1)
    np.random.seed(RANDOM_STATE)

    vectorizer = TfidfVectorizer(max_features=max_features, token_pattern=r"(?u)\b\w+\b")
    vectorizer.fit(df["Teks_Bersih"])
    vocabulary = vectorizer.vocabulary_

    def encode(text):
        ids = [vocabulary[token] + 1 for token in str(text).split() if token in vocabulary][:max_len]
        if len(ids) < max_len:
            ids.extend([0] * (max_len - len(ids)))
        return ids

    label_to_id = {label: idx for idx, label in enumerate(CLASS_ORDER)}
    x = np.array([encode(text) for text in df["Teks_Bersih"]], dtype=np.int64)
    y = df["TingkatFatalitas"].map(label_to_id).to_numpy(dtype=np.int64)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    test_tensor = torch.tensor(x_test)

    class BiLSTMClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(len(vocabulary) + 1, embedding_dim, padding_idx=0)
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                batch_first=True,
                bidirectional=True,
                dropout=0.2,
            )
            self.dropout = nn.Dropout(0.35)
            self.classifier = nn.Linear(hidden_dim * 2, len(CLASS_ORDER))

        def forward(self, inputs):
            embedded = self.embedding(inputs)
            output, _ = self.lstm(embedded)
            pooled = output.mean(dim=1)
            return self.classifier(self.dropout(pooled))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        print(f"BiLSTM epoch={epoch}/{epochs}, loss={np.mean(losses):.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        logits = model(test_tensor.to(device)).cpu().numpy()
    pred = logits.argmax(axis=1)
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    y_true_label = [id_to_label[idx] for idx in y_test]
    y_pred_label = [id_to_label[idx] for idx in pred]
    metrics = summarize_predictions(y_true_label, y_pred_label)
    return {"model": "BiLSTM", **prefix_metrics("holdout", metrics)}


def run_bert(df, epochs=1, batch_size=8, max_len=128, model_name="distilbert-base-uncased"):
    try:
        from datasets import Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
    except Exception as exc:
        print(f"[skip] BERT dilewati karena dependency belum siap: {exc}")
        return None

    label_to_id = {label: idx for idx, label in enumerate(CLASS_ORDER)}
    working = df[["RgksPeristiwa", "TingkatFatalitas"]].rename(columns={"RgksPeristiwa": "text"}).copy()
    working["label"] = working["TingkatFatalitas"].map(label_to_id)
    train_df, test_df = train_test_split(
        working[["text", "label"]],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=working["label"],
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_len)

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True)).map(tokenize, batched=True)
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True)).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(CLASS_ORDER),
        id2label=dict(enumerate(CLASS_ORDER)),
        label2id=label_to_id,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        pred = np.argmax(logits, axis=1)
        label_names = [CLASS_ORDER[idx] for idx in labels]
        pred_names = [CLASS_ORDER[idx] for idx in pred]
        return {
            "accuracy": accuracy_score(labels, pred),
            "balanced_accuracy": balanced_accuracy_score(labels, pred),
            "macro_f1": f1_score(labels, pred, average="macro"),
            "quadratic_kappa": quadratic_kappa(label_names, pred_names),
        }

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "bert_requested_holdout"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        report_to=[],
        seed=RANDOM_STATE,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    holdout = {
        "accuracy": metrics["eval_accuracy"],
        "error_rate": 1.0 - metrics["eval_accuracy"],
        "balanced_accuracy": metrics["eval_balanced_accuracy"],
        "macro_f1": metrics["eval_macro_f1"],
        "quadratic_kappa": metrics["eval_quadratic_kappa"],
    }
    return {"model": "BERT", "bert_model": model_name, **prefix_metrics("holdout", holdout)}


def parse_args():
    parser = argparse.ArgumentParser(description="Uji NLP_XGBoost, NLP_CatBoost, BiLSTM, dan BERT.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["xgboost", "catboost", "bilstm", "bert"],
        choices=["xgboost", "catboost", "bilstm", "bert"],
        help="Model yang dijalankan.",
    )
    parser.add_argument("--bilstm-epochs", type=int, default=8)
    parser.add_argument("--bert-epochs", type=int, default=1)
    parser.add_argument("--bert-model", default="distilbert-base-uncased")
    parser.add_argument("--cv", action="store_true", help="Tambahkan 4-fold CV untuk XGBoost dan CatBoost.")
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_dataset(DATA_FILE, use_normalization=True)
    rows = []

    print("=== Requested NLP Models Trial ===", flush=True)
    print(f"Dataset rows: {len(df)}", flush=True)
    print("Class distribution:", flush=True)
    print(df["TingkatFatalitas"].value_counts().reindex(CLASS_ORDER).to_string(), flush=True)

    runners = {
        "xgboost": lambda: run_xgboost(df, include_cv=args.cv),
        "catboost": lambda: run_catboost(df, include_cv=args.cv),
        "bilstm": lambda: run_bilstm(df, epochs=args.bilstm_epochs),
        "bert": lambda: run_bert(df, epochs=args.bert_epochs, model_name=args.bert_model),
    }

    for model_name in args.models:
        print(f"\n--- Running {model_name} ---", flush=True)
        try:
            result = runners[model_name]()
        except Exception as exc:
            result = {"model": model_name, "status": "failed", "error": str(exc)}
            print(f"[failed] {model_name}: {exc}", flush=True)
        if result is not None:
            rows.append(result)
            printable = {key: value for key, value in result.items() if isinstance(value, (int, float, str))}
            print(json.dumps(printable, indent=2), flush=True)

    results = pd.DataFrame(rows)
    output_path = OUTPUT_DIR / "requested_nlp_models_results.csv"
    if output_path.exists():
        previous = pd.read_csv(output_path)
        results = pd.concat([previous, results], ignore_index=True)
        results = results.drop_duplicates(subset=["model"], keep="last")
    results.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}", flush=True)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)


if __name__ == "__main__":
    main()

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

from fatalitas_core import CLASS_ORDER, DATA_FILE, RANDOM_STATE, load_dataset

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "quadratic_kappa": cohen_kappa_score(labels, preds, weights="quadratic"),
    }


def main():
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        print("Dependency BERT belum siap.")
        print("Install dulu: pip install transformers datasets accelerate torch")
        print(f"Detail: {exc}")
        return

    model_name = "distilbert-base-uncased"
    df = load_dataset(DATA_FILE, use_normalization=False)
    label_to_id = {label: idx for idx, label in enumerate(CLASS_ORDER)}
    df["label_id"] = df["TingkatFatalitas"].map(label_to_id)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    print("=== BERT Comparative Baseline ===")
    for fold_id, (train_idx, test_idx) in enumerate(cv.split(df["RgksPeristiwa"], df["label_id"]), start=1):
        train_df = df.iloc[train_idx][["RgksPeristiwa", "label_id"]].rename(columns={"RgksPeristiwa": "text"})
        test_df = df.iloc[test_idx][["RgksPeristiwa", "label_id"]].rename(columns={"RgksPeristiwa": "text"})

        train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
        test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

        def tokenize(batch):
            return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

        train_ds = train_ds.map(tokenize, batched=True)
        test_ds = test_ds.map(tokenize, batched=True)
        train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label_id"])
        test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label_id"])

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(CLASS_ORDER), id2label=dict(enumerate(CLASS_ORDER)), label2id=label_to_id
        )

        training_args = TrainingArguments(
            output_dir=str(OUTPUT_DIR / f"bert_fold_{fold_id}"),
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=2e-5,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="no",
            logging_strategy="no",
            report_to=[],
            seed=RANDOM_STATE,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds.rename_column("label_id", "labels"),
            eval_dataset=test_ds.rename_column("label_id", "labels"),
            compute_metrics=compute_metrics,
        )
        trainer.train()
        metrics = trainer.evaluate()
        rows.append(
            {
                "fold": fold_id,
                "accuracy": metrics["eval_accuracy"],
                "error_rate": 1.0 - metrics["eval_accuracy"],
                "balanced_accuracy": metrics["eval_balanced_accuracy"],
                "macro_f1": metrics["eval_macro_f1"],
                "quadratic_kappa": metrics["eval_quadratic_kappa"],
            }
        )
        print(
            f"fold={fold_id}: accuracy={metrics['eval_accuracy']:.4f}, "
            f"macro_f1={metrics['eval_macro_f1']:.4f}, error_rate={1.0 - metrics['eval_accuracy']:.4f}"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "model": "distilbert-base-uncased",
                "cv_accuracy": results["accuracy"].mean(),
                "cv_error_rate": results["error_rate"].mean(),
                "cv_balanced_accuracy": results["balanced_accuracy"].mean(),
                "cv_macro_f1": results["macro_f1"].mean(),
                "cv_quadratic_kappa": results["quadratic_kappa"].mean(),
            }
        ]
    )
    output_path = OUTPUT_DIR / "bert_baseline_results.csv"
    summary.to_csv(output_path, index=False)
    print("\nSaved:", output_path)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()

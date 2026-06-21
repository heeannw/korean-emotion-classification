"""
KLUE-RoBERTa 기반 한국어 70감정 분류 파인튜닝 스크립트
- Colab GPU 환경에서 실행 권장
- 로컬 실행 시: python train_transformer.py --data data/감정분석_70감정_추가증강.csv

Colab 설치 명령어:
    !pip install torch==2.3.0 numpy==1.26.4 transformers==4.44.2 \
                 accelerate==0.34.2 datasets==2.21.0 scikit-learn
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

TEXT_COLUMN = "문장 예시"
LABEL_COLUMN = "감정명"
MODEL_NAME = "klue/roberta-base"
MAX_LEN = 128


class EmotionDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: AutoTokenizer,
        max_len: int = MAX_LEN,
    ) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def load_data(csv_path: Path) -> tuple[list[str], list[int], LabelEncoder]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig").dropna(
        subset=[TEXT_COLUMN, LABEL_COLUMN]
    )
    # 충돌 텍스트: 다수결 처리
    df["_norm"] = df[TEXT_COLUMN].str.strip().str.lower()
    conflict_mask = df.groupby("_norm")[LABEL_COLUMN].transform("nunique") > 1
    if conflict_mask.any():
        majority = (
            df[conflict_mask]
            .groupby("_norm")[LABEL_COLUMN]
            .agg(lambda s: s.value_counts().index[0])
        )
        df.loc[conflict_mask, LABEL_COLUMN] = df.loc[conflict_mask, "_norm"].map(majority)

    df = df.drop_duplicates(["_norm", LABEL_COLUMN]).reset_index(drop=True)

    encoder = LabelEncoder()
    labels = encoder.fit_transform(df[LABEL_COLUMN].tolist()).tolist()
    texts = df[TEXT_COLUMN].str.strip().tolist()
    print(f"  데이터: {len(texts)}행, {len(encoder.classes_)}클래스")
    return texts, labels, encoder


def compute_metrics(pred: object) -> dict[str, float]:
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="weighted")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/감정분석_70감정_추가증강.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/roberta_emotion"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ["WANDB_DISABLED"] = "true"
    torch.manual_seed(args.seed)

    print("=== 데이터 로딩 ===")
    texts, labels, encoder = load_data(args.data)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.15, random_state=args.seed, stratify=labels
    )
    print(f"  학습: {len(train_texts)}, 검증: {len(val_texts)}")

    print(f"\n=== 토크나이저 로딩: {MODEL_NAME} ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer)

    print(f"\n=== 모델 로딩: {MODEL_NAME} ===")
    id2label = {i: label for i, label in enumerate(encoder.classes_)}
    label2id = {label: i for i, label in enumerate(encoder.classes_)}
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(encoder.classes_),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        label_smoothing_factor=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("\n=== 학습 시작 ===")
    trainer.train()

    print("\n=== 최종 평가 ===")
    results = trainer.evaluate()
    print(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"\n=== 모델 저장: {args.output_dir} ===")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    label_classes_path = Path(args.output_dir) / "label_classes.json"
    label_classes_path.write_text(
        json.dumps(encoder.classes_.tolist(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  label_classes.json 저장 완료")

    metrics_path = Path(args.output_dir) / "metrics.json"
    metrics_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n최종 정확도: {results.get('eval_accuracy', 'N/A'):.4f}")
    print(f"최종 F1: {results.get('eval_f1', 'N/A'):.4f}")


if __name__ == "__main__":
    main()

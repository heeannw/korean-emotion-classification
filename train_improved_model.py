from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.emotion_model import fit_model, load_and_clean_data, predict, save_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the cleaned Korean 70-emotion classifier")
    parser.add_argument("--data", type=Path, default=Path("data/감정분석_70감정_추가증강.csv"))
    parser.add_argument("--model-out", type=Path, default=Path("artifacts/emotion_tfidf_svm_v5.joblib"))
    parser.add_argument("--metrics-out", type=Path, default=Path("artifacts/metrics_v5.json"))
    args = parser.parse_args()

    data, cleaning = load_and_clean_data(args.data)
    train_data, validation_data = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data["label"],
    )

    evaluation_model = fit_model(train_data["text"].tolist(), train_data["label"].tolist())
    predictions, _ = predict(evaluation_model, validation_data["text"].tolist())
    actual = validation_data["label"].to_numpy()
    metrics = {
        **cleaning,
        "split_seed": 42,
        "validation_rows": len(validation_data),
        "accuracy": float(accuracy_score(actual, predictions)),
        "weighted_f1": float(f1_score(actual, predictions, average="weighted")),
        "macro_f1": float(f1_score(actual, predictions, average="macro")),
        "weighted_precision": float(
            precision_score(actual, predictions, average="weighted", zero_division=0)
        ),
        "weighted_recall": float(recall_score(actual, predictions, average="weighted")),
    }

    final_model = fit_model(data["text"].tolist(), data["label"].tolist())
    save_model(final_model, args.model_out)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

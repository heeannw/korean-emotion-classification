from __future__ import annotations

import argparse
from pathlib import Path

from src.emotion_model import load_model, predict


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a Korean emotion label")
    parser.add_argument("text", nargs="+", help="One or more Korean sentences")
    parser.add_argument("--model", type=Path, default=Path("artifacts/emotion_tfidf_svm_v5.joblib"))
    args = parser.parse_args()

    model = load_model(args.model)
    labels, scores = predict(model, args.text)
    for text, label, score in zip(args.text, labels, scores):
        print(f"{text}\t{label}\tdecision_score={score:.4f}")


if __name__ == "__main__":
    main()

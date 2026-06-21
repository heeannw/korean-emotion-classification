"""
SVM 모델의 오류 패턴 분석
- 클래스별 F1 하위 10개 (개선 우선순위)
- 혼동 행렬 상위 오류 쌍

사용법:
    python analyze_errors.py
    python analyze_errors.py --model artifacts/emotion_tfidf_svm_v5.joblib
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.emotion_model import fit_model, load_and_clean_data, predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/감정분석_70감정_추가증강.csv"))
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--top-errors", type=int, default=20)
    args = parser.parse_args()

    print("=== 데이터 로딩 ===")
    data, info = load_and_clean_data(args.data)
    print(f"  정제 후: {info['clean_rows']}행, {info['classes']}클래스")

    train_data, val_data = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data["label"]
    )

    print("\n=== 모델 학습 (검증용) ===")
    if args.model and args.model.exists():
        from src.emotion_model import load_model
        model = load_model(args.model)
        print(f"  저장된 모델 로딩: {args.model}")
    else:
        model = fit_model(train_data["text"].tolist(), train_data["label"].tolist())
        print("  새 모델 학습 완료")

    predictions, scores = predict(model, val_data["text"].tolist())
    actual = val_data["label"].to_numpy()

    # 전체 리포트
    report = classification_report(
        actual, predictions, output_dict=True, zero_division=0
    )
    overall_acc = report["accuracy"]
    overall_f1  = report["weighted avg"]["f1-score"]
    print(f"\n=== 전체 성능 ===")
    print(f"  Accuracy:    {overall_acc:.4f}")
    print(f"  Weighted F1: {overall_f1:.4f}")

    # 클래스별 F1 하위 분석
    class_metrics = {
        label: v for label, v in report.items()
        if isinstance(v, dict) and label not in ("accuracy", "macro avg", "weighted avg")
    }
    df_metrics = pd.DataFrame(class_metrics).T.sort_values("f1-score")
    print(f"\n=== F1 하위 10 클래스 (개선 우선순위) ===")
    print(df_metrics[["f1-score", "support"]].head(10).to_string())

    # 혼동 행렬 상위 오류
    classes = np.unique(actual)
    cm = confusion_matrix(actual, predictions, labels=classes)
    np.fill_diagonal(cm, 0)  # 정답 제거

    errors = []
    for i, true_cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            if cm[i, j] > 0:
                errors.append({
                    "실제": true_cls,
                    "예측": pred_cls,
                    "횟수": int(cm[i, j]),
                })
    df_errors = pd.DataFrame(errors).sort_values("횟수", ascending=False)
    print(f"\n=== 혼동 상위 {args.top_errors}쌍 ===")
    print(df_errors.head(args.top_errors).to_string(index=False))

    # 오분류 예시 출력
    wrong_mask = predictions != actual
    wrong_df = val_data.copy()
    wrong_df["predicted"] = predictions
    wrong_df["score"] = scores
    wrong_df = wrong_df[wrong_mask].sort_values("score", ascending=False)

    print(f"\n=== 오분류 예시 (신뢰도 높은 오답 5개) ===")
    for _, row in wrong_df.head(5).iterrows():
        print(f"  문장: {row['text']}")
        print(f"  실제: {row['label']}  예측: {row['predicted']}  점수: {row['score']:.2f}")
        print()


if __name__ == "__main__":
    main()

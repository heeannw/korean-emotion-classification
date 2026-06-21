from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

TEXT_COLUMN = "문장 예시"
LABEL_COLUMN = "감정명"


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).lower()


def load_and_clean_data(csv_path: str | Path) -> tuple[pd.DataFrame, dict[str, int]]:
    data = pd.read_csv(csv_path, encoding="utf-8-sig").dropna(
        subset=[TEXT_COLUMN, LABEL_COLUMN]
    ).copy()
    data["text"] = data[TEXT_COLUMN].astype(str).str.strip()
    data["label"] = data[LABEL_COLUMN].astype(str).str.strip()
    data["normalized_text"] = data["text"].map(normalize_text)

    original_rows = len(data)
    conflicting = data.groupby("normalized_text")["label"].nunique()
    conflicting_texts = set(conflicting[conflicting > 1].index)
    conflict_rows = int(data["normalized_text"].isin(conflicting_texts).sum())
    data = data[~data["normalized_text"].isin(conflicting_texts)]

    before_dedup = len(data)
    data = data.drop_duplicates(["normalized_text", "label"]).reset_index(drop=True)
    report = {
        "original_rows": original_rows,
        "conflicting_texts": len(conflicting_texts),
        "conflict_rows_removed": conflict_rows,
        "duplicate_rows_removed": before_dedup - len(data),
        "clean_rows": len(data),
        "classes": int(data["label"].nunique()),
    }
    return data, report


def build_components() -> tuple[TfidfVectorizer, LinearSVC]:
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        max_features=250_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    classifier = LinearSVC(C=2.0, class_weight="balanced")
    return vectorizer, classifier


def fit_model(texts: list[str], labels: list[str]) -> dict[str, object]:
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    vectorizer, classifier = build_components()
    features = vectorizer.fit_transform(texts)
    classifier.fit(features, encoded_labels)
    return {"vectorizer": vectorizer, "classifier": classifier, "classes": encoder.classes_}


def predict(model: dict[str, object], texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    features = model["vectorizer"].transform(texts)
    decision = model["classifier"].decision_function(features)
    indices = decision.argmax(axis=1)
    return model["classes"][indices], decision.max(axis=1)


def save_model(model: dict[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)


def load_model(path: str | Path) -> dict[str, object]:
    return joblib.load(path)

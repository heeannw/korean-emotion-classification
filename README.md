# 한국어 70종 감정 분류

한국어 문장을 70개 감정 범주로 분류하는 프로젝트입니다. 초기 KcELECTRA v4의 오류를 분석한 뒤 데이터 충돌과 중복을 제거하고, 재현 가능한 경량 v5 분류기를 추가했습니다.

## 개선 결과

동일한 seed 42, 계층화 80:20 분할에서 측정한 결과입니다.

| 모델 | Accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|
| KcELECTRA v4 기존 저장 평가 | 0.4835 | 0.5031 | - |
| TF-IDF + Linear SVM v5 | **0.6426** | **0.6571** | **0.6499** |

v5는 기존 저장 평가 대비 정확도 **15.91%p**, Weighted F1 **15.40%p** 개선되었습니다.

## 데이터 품질 개선

원본 34,459행을 검사한 결과 같은 문장에 여러 감정 라벨이 붙은 충돌 문장과 중복 데이터가 성능을 제한하고 있었습니다.

- 라벨 충돌 문장: 2,061개
- 충돌 행 제거: 7,976행
- 완전 중복 제거: 1,709행
- 최종 정제 데이터: 24,774행, 70개 클래스

정제는 `src/emotion_model.py`에서 자동으로 수행되며, 검증 데이터는 학습에 사용하지 않습니다.

## 저장소 구성

- `train_improved_model.py`: 정제, 검증, 전체 데이터 재학습 및 모델 저장
- `predict.py`: 저장된 v5 모델로 문장 감정 예측
- `src/emotion_model.py`: 공용 전처리·학습·추론 로직
- `artifacts/emotion_tfidf_svm_v5.joblib`: 전체 정제 데이터로 학습한 경량 모델
- `artifacts/metrics_v5.json`: 데이터 정제 및 검증 결과
- `notebooks/감정분석모델_70감정.ipynb`: 초기 KcELECTRA v4 실험
- `model_configs/KcELECTRA_70Emotion_v4/`: v4 설정, 라벨, 토크나이저

## 실행 방법

```bash
pip install -r requirements.txt
python train_improved_model.py
python predict.py "오늘 너무 행복하고 기분이 좋아"
```

## 모델 선택 이유

v5는 문자 2~5 gram TF-IDF와 클래스 균형 Linear SVM을 사용합니다. CPU에서 수 초 안에 학습할 수 있고 모델 크기가 약 5MB라 배포가 간단합니다. 향후 GPU 환경에서는 동일한 정제 데이터를 사용해 KcELECTRA를 재학습하는 것이 다음 개선 단계입니다.

## KcELECTRA 가중치

초기 v4 가중치는 저장소 용량을 줄이기 위해 포함하지 않았습니다. 모델 구조 확인에 필요한 설정, 라벨, 토크나이저는 보존했습니다.

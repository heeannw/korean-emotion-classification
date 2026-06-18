# 한국어 70종 감정 분류 모델

한국어 문장을 70개 감정 범주로 분류하는 KcELECTRA 기반 프로젝트입니다. 여러 실험 중 최종 성능이 가장 좋았던 `KcELECTRA_70Emotion_v4`만 공개합니다.

## 주요 내용

- 한국어 감정 문장 데이터 전처리 및 증강
- KcELECTRA 기반 70종 감정 분류
- 계층화 학습·검증 데이터 분리
- Accuracy, Weighted F1, Precision, Recall 평가

## 검증 결과

| 지표 | 값 |
|---|---:|
| Accuracy | 0.4835 |
| Weighted F1 | 0.5031 |
| Weighted Precision | 0.5814 |
| Weighted Recall | 0.4835 |

## 저장소 구성

- `notebooks/감정분석모델_70감정.ipynb`: 최종 v4 모델 학습 및 평가
- `data/`: 감정 라벨과 문장 예시로 구성된 학습 데이터
- `model_configs/KcELECTRA_70Emotion_v4/`: 최종 모델 설정, 라벨, 토크나이저
- `requirements.txt`: 실행 의존성

## 실행 방법

```bash
pip install -r requirements.txt
```

노트북의 데이터 경로를 현재 환경의 `data/` 경로에 맞게 변경한 뒤 실행합니다. GPU 환경 사용을 권장합니다.

## 모델 가중치

학습된 가중치 파일은 저장소 용량과 GitHub 파일 크기 제한을 고려해 포함하지 않았습니다. 모델 구조를 확인하고 다시 학습하는 데 필요한 v4 설정, 라벨, 토크나이저는 포함되어 있습니다.

# IMU (Inertial Measurement Unit)

IMU Calibration and Sensitivity Analysis. 센서 보정 알고리즘 구현 및 성능 분석 프로젝트입니다.

## Project Structure

```text
IMU/
├── calibration/        # 센서 보정(가속도계, 지자기, 자이로스코프) 스크립트
├── experiments/        # 성능 실험 및 파라미터 스윕 분석 (main 실행부)
├── results/            # 시뮬레이션 및 필터링 결과 시각화 이미지
├── simulation/         # IMU 데이터 생성 및 노이즈/바이어스 시뮬레이션
├── tracking/           # 상보 필터(Complementary Filter) 및 쿼터니언 기반 추적 로직
└── utils/              # 수학적 연산(Quaternion) 및 시각화 유틸리티
```

## Core Features

- **Calibration**: 
    - Accelerometer: 12-param & Ellipsoid models
    - Magnetometer: Target norm optimization
    - Gyroscope: Static bias & LS scaling matrix
- **Tracking**: Quaternion-based complementary filter for robust orientation estimation.
- **Analysis**: Sensitivity analysis for bias and noise sigma levels.

## Experiments & Analysis

`experiments/` 디렉토리는 목적에 따라 크게 두 가지 역할로 나뉩니다.

1. **`main.py` (통합 쇼케이스/전체 파이프라인)**
    - "전체 시스템이 A부터 Z까지 유기적으로 연결되어 동작함"을 증명합니다.
    - 가속도계 보정 $\rightarrow$ 지자기 보정 $\rightarrow$ 자이로 보정 $\rightarrow$ 센서 퓨전(3D Tracking)으로 이어지는 전체 흐름을 한 번에 실행합니다.

2. **`experiment_*.py` (심층 분석/현미경 전문가)**
    - 특정 주제를 현미경처럼 정밀하게 파고드는 실험에 집중합니다.
    - **`experiment_sensitivity.py`**: 가우시안 노이즈($\sigma$)와 바이어스(Bias) 변화가 각 알고리즘의 오차(MSE) 추이에 미치는 영향을 심층적으로 분석합니다.

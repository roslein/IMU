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


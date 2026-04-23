# IMU (Inertial Measurement Unit) 프로젝트

IMU 센서 보정, 데이터 시뮬레이션 및 쿼터니언 기반 자세 추정 시스템입니다.

## 📁 Project Structure

```text
IMU/
├── calibration/        # 센서 보정(가속도계, 지자기, 자이로스코프) 알고리즘
├── tracking/           # 상보 필터(Complementary Filter) 및 쿼터니언 추적 로직
├── simulation/         # 물리적 노이즈/바이어스가 포함된 IMU 데이터 생성기
├── utils/              # 쿼터니언 수학(SLERP, TRIAD) 및 시각화 도구
├── experiments/        # 실행 및 분석 스크립트
│   ├── main.py                # [Entry Point] 전체 파이프라인 통합 테스트 런처
│   ├── experiment_sensitivity.py # 노이즈/바이어스 민감도 분석 실험
│   └── experiment_tracking.py    # 자세 추정 성능 정밀 평가 및 3D 시각화
└── results/            # 실험 결과(그래프, 보고서 이미지) 저장 폴더
```

## Core Features

- **Calibration**: 
    - Accelerometer: 12-param & Ellipsoid models
    - Magnetometer: Target norm optimization
    - Gyroscope: Static bias & LS scaling matrix
- **Tracking**: Quaternion-based complementary filter for robust orientation estimation.
- **Analysis**: Sensitivity analysis for bias and noise sigma levels.


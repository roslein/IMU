import numpy as np
import matplotlib.pyplot as plt

from sensor_simulation import (
    sim_acc_6_position_static, 
    sim_acc_multi_position_static,
    sim_mag_figure8_dynamic
)
from calib_accelerometer import calibrate_acc_ellipsoid, calibrate_acc_12param
from calib_magnetometer import calibrate_mag_ellipsoid
from utils_visualization import verify_calibration_pipeline

# --- 상단 임포트 영역에 추가 ---
from sensor_simulation import sim_gyro_static_for_bias, sim_gyro_rate_table_for_M
from calib_gyroscope import calibrate_gyroscope_full
from utils_visualization import verify_gyro_calibration_pipeline

# === 파이프라인 통합 실행 블록 (Execution Example) ===
if __name__ == "__main__":
    # ==========================================
    # 1. 가속도계 (Accelerometer) 파이프라인
    # ==========================================
    print("=== Accelerometer Calibration ===")
    # 시뮬레이션 데이터 생성 (Train & Test 분리)
    _, raw_6pos_train = sim_acc_6_position_static(n_samples_per_pos=100)
    _, raw_multi_train = sim_acc_multi_position_static(n_positions=50, n_samples_per_pos=10)
    _, raw_multi_test = sim_acc_multi_position_static(n_positions=30, n_samples_per_pos=10)
    
    # 알고리즘별 보정 파라미터 도출 (Train Set 사용)
    W_acc_ell, b_acc_ell = calibrate_acc_ellipsoid(raw_multi_train, n_samples_per_pos=10)
    W_acc_12p, b_acc_12p = calibrate_acc_12param(raw_6pos_train, n_samples_per_pos=100)
    
    # 보정 데이터 딕셔너리 생성
    dict_calibrated_acc = {
        "Ellipsoid Fitting": (W_acc_ell @ (raw_multi_test - b_acc_ell).T).T,
        "12-Param Iterative": (W_acc_12p @ (raw_multi_test - b_acc_12p).T).T
    }
    
    # 보정 결과 교차 검증 및 시각화 (Test Set 사용)
    verify_calibration_pipeline(
        raw_multi_test, 
        dict_calibrated_acc, 
        target_norm=1.0,
        title='3D Accelerometer Calibration Verification (Test Set)', 
        unit='g'
    )
    
    # ==========================================
    # 2. 지자기 센서 (Magnetometer) 파이프라인
    # ==========================================
    print("\n=== Magnetometer Calibration ===")
    # 8자 궤적 동적 데이터 생성 (학습/테스트 겸용, 원하면 별도 생성 가능)
    _, raw_mag_dynamic = sim_mag_figure8_dynamic(n_samples=1000)
    
    # 타원체 피팅 알고리즘 (Hard/Soft-iron 보정)
    W_mag_ellipsoid, b_mag_ellipsoid = calibrate_mag_ellipsoid(raw_mag_dynamic, target_norm=1.0)
    
    # 보정 데이터 생성
    dict_calibrated_mag = {
        "Ellipsoid Fitting (Hard/Soft-iron)": (W_mag_ellipsoid @ (raw_mag_dynamic - b_mag_ellipsoid).T).T
    }
    
    # 검증 및 시각화
    verify_calibration_pipeline(
        raw_mag_dynamic, 
        dict_calibrated_mag, 
        target_norm=1.0,
        title='3D Magnetometer Soft/Hard-iron Calibration', 
        unit='a.u.'
    )

# main.py 에 추가할 코드

# --- 기존 하단 메인 블록(지자기 센서 파이프라인 아래)에 추가 ---
    # ==========================================
    # 3. 자이로스코프 (Gyroscope) 파이프라인
    # ==========================================
    print("\n=== Gyroscope Calibration ===")
    
    # 시뮬레이션 데이터 생성 (정지 상태 및 Rate Table 회전 상태)
    _, raw_gyro_static = sim_gyro_static_for_bias(n_samples=1000)
    gt_gyro_rate, raw_gyro_rate = sim_gyro_rate_table_for_M(rate_dps=100.0, n_samples_per_axis=100)
    
    # 알고리즘 파라미터 도출 (선형 최소제곱법)
    sim_data_gyro = {
        'gyro_static': (None, raw_gyro_static), 
        'gyro_rate': (gt_gyro_rate, raw_gyro_rate)
    }
    W_gyro_est, b_gyro_est = calibrate_gyroscope_full(sim_data_gyro)
    
    # 보정 데이터 딕셔너리 생성
    dict_calibrated_gyro = {
        "Linear Least Squares": (W_gyro_est @ (raw_gyro_rate - b_gyro_est).T).T
    }
    
    # 검증 및 시각화 (GT 기준 오차 평가)
    verify_gyro_calibration_pipeline(
        gt_gyro_rate,
        raw_gyro_rate,
        dict_calibrated_gyro,
        title='3D Gyroscope Calibration Verification',
        unit='dps'
    )
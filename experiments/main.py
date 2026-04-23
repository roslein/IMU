import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation.sensor_simulation import (
    sim_acc_6_position_static, 
    sim_acc_multi_position_static,
    sim_mag_figure8_dynamic
)
from calibration.calib_accelerometer import calibrate_acc_ellipsoid, calibrate_acc_12param
from calibration.calib_magnetometer import calibrate_mag_ellipsoid
from utils.utils_visualization import verify_calibration_pipeline

# --- 상단 임포트 영역에 추가 ---
from simulation.sensor_simulation import sim_gyro_static_for_bias, sim_gyro_rate_table_for_M
from calibration.calib_gyroscope import calibrate_gyroscope_full
from utils.utils_visualization import verify_gyro_calibration_pipeline

# --- 상단 임포트 영역에 추가 ---
from scipy.spatial.transform import Rotation as R_scipy
from simulation.sensor_simulation import sim_imu_continuous_rotation
from tracking.complementary_quaternion import ComplementaryFilter

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

    # ==========================================
    # 4. 센서 퓨전 (Complementary Filter) 파이프라인
    # ==========================================
    print("\n=== IMU Sensor Fusion (Quaternion Complementary Filter) ===")

    # 1) 연속 회전 시뮬레이션 데이터 생성 (10초 동안 0.01초 간격)
    # [Gimbal Lock 회피 테스트]: pitch_amp를 90도(1.57rad)가 넘는 1.6으로 설정하여, 
    # 오일러 기반 필터라면 무조건 무한대로 터져버릴 극한의 자세(특이점)를 강제로 유발합니다.
    dt = 0.01
    time_span = 10.0
    time_data, gt_quats, meas_gyro, meas_acc, meas_mag = sim_imu_continuous_rotation(
        time_span=time_span, dt=dt, 
        roll_amp=0.5, roll_freq=0.1,
        pitch_amp=1.6, pitch_freq=0.15, 
        yaw_amp=0.8, yaw_freq=0.05,
        sigma_gyro=0.05, sigma_acc=0.02, sigma_mag=0.05
    )

    # 2) 상보 필터 초기화
    # alpha=0.98 : 자이로 98% 신뢰, 가속도/지자기 2% 신뢰
    comp_filter = ComplementaryFilter(alpha=0.98, dt=dt)
    
    est_quats = []

    # 3) 시간 흐름에 따른 센서 퓨전 실행 (실시간 처리와 동일한 구조)
    for i in range(len(time_data)):
        q_est = comp_filter.update(meas_gyro[i], meas_acc[i], meas_mag[i])
        est_quats.append(q_est)
    
    est_quats = np.array(est_quats)

    # 4) 결과 시각화를 위한 쿼터니언 성분(w, x, y, z) 직접 비교
    # 오일러 각 변환 시 발생하는 짐벌 락 플립(Jumping) 현상을 피하기 위해, 
    # 필터가 추정한 순수 쿼터니언의 4가지 요소를 직접 비교합니다.
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig.suptitle('IMU Sensor Fusion: Quaternion Tracking (w, x, y, z)', fontsize=16)

    labels = ['q_w (Scalar)', 'q_x', 'q_y', 'q_z']
    for i in range(4):
        # Ground Truth 데이터 (정답)
        axes[i].plot(time_data, gt_quats[:, i], label='Ground Truth (q)', color='black', linestyle='--', linewidth=2)
        # 상보 필터 추정 데이터
        axes[i].plot(time_data, est_quats[:, i], label='Estimated', color='blue', alpha=0.7, linewidth=2)
        # 쿼터니언의 이중 피복(q = -q) 특성상 위상이 뒤집힐 수 있으므로 -q 정답선도 옅게 표시
        axes[i].plot(time_data, -gt_quats[:, i], label='Negative GT (-q)', color='gray', linestyle=':', alpha=0.5)
        
        axes[i].set_ylabel(f'{labels[i]}')
        axes[i].grid(True)
        if i == 0:
            axes[i].legend(loc='upper right')
        axes[i].set_ylim(-1.2, 1.2)

    axes[3].set_xlabel('Time [s]')
    plt.tight_layout()
    
    # 보고서용 결과 그래프 이미지 자동 저장
    import os
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/stage3_tracking_result.png', dpi=150, bbox_inches='tight')
    
    plt.show()
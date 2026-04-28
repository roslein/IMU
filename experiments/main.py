import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 모듈 임포트 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation.sensor_simulation import (
    sim_acc_6_position_static, sim_acc_multi_position_static,
    sim_mag_figure8_dynamic, sim_gyro_static_for_bias, 
    sim_gyro_rate_table_for_M, sim_imu_continuous_rotation
)
from simulation.sensor_simulation import sim_mag_multi_position_static
from calibration.calib_accelerometer import calibrate_acc_ellipsoid, calibrate_acc_12param
from calibration.calib_magnetometer import calibrate_mag_ellipsoid
from calibration.calib_gyroscope import calibrate_gyroscope_full
from tracking.complementary_quaternion import ComplementaryFilter
from utils.utils_visualization import verify_calibration_pipeline, verify_gyro_calibration_pipeline
from utils.quaternion_math import q_angle_error, q_align_offset, quat_to_euler, q_mult #
from utils.utils_visualization import ( #
    plot_tracking_angle_error,
    plot_tracking_euler_comparison,
    animate_quaternion_tracking
)
from utils.quaternion_math import accel_mag_to_quaternion

if __name__ == "__main__":

    # -1.실험 결과 저장 디렉토리 설정 및 생성
    RESULT_DIR = r"D:\바탕화면\CS-Study-Tracker\IMU\results"
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
        print(f"Directory created: {RESULT_DIR}")

    # 실험 타입 설정 (파일명 구분을 위해)
    # main.py -> "noisy"
    exp_type = "noisy"

    print("=== [Full Pipeline Integration Test] ===")
    # 0. 공통 가상 센서 고유 오차(M, b) 정의
    M_acc_true = np.array([[1.05, 0.02, 0.01], [0.01, 0.98, 0.02], [0.02, 0.01, 1.02]])
    b_acc_true = np.array([0.05, -0.03, 0.1])
    
    M_mag_true = np.array([[1.1, 0.1, 0.0], [0.1, 0.9, -0.1], [0.0, -0.1, 1.2]])
    b_mag_true = np.array([0.3, -0.2, 0.1])
    
    M_gyro_true = np.array([[1.02, 0.01, -0.01], [0.01, 0.99, 0.02], [-0.02, 0.01, 1.05]])
    b_gyro_true = np.array([1.5, -0.8, 0.5])

    # 1. Accelerometer Calibration
    print("\n1. Accelerometer Stage...")
    _, raw_6pos = sim_acc_6_position_static(M_matrix=M_acc_true, b_vector=b_acc_true)
    # _, raw_6pos = sim_acc_6_position_static(sigma=0.0, M_matrix=M_acc_true, b_vector=b_acc_true)
    W_acc, b_acc = calibrate_acc_12param(raw_6pos, n_samples_per_pos=100)
    print(f"   - Accel Calibration Done. Bias: {b_acc}")

    # 2. Magnetometer Calibration
    print("\n2. Magnetometer Stage...")
    #_, raw_mag = sim_mag_figure8_dynamic(M_matrix=M_mag_true, b_vector=b_mag_true)
    # _, raw_mag = sim_mag_figure8_dynamic(sigma=0.0, M_matrix=M_mag_true, b_vector=b_mag_true)
    _, raw_mag_multi = sim_mag_multi_position_static(
        n_positions=50, n_samples_per_pos=10, 
        M_matrix=M_mag_true, b_vector=b_mag_true
    )
    W_mag, b_mag = calibrate_mag_ellipsoid(raw_mag_multi)
    print(f"   - Mag Calibration Done. Bias: {b_mag}")

    # 3. Gyroscope Calibration
    print("\n3. Gyroscope Stage...")
    _, raw_gyro_static = sim_gyro_static_for_bias(M_matrix=M_gyro_true, b_vector=b_gyro_true)
    gt_gyro_rate, raw_gyro_rate = sim_gyro_rate_table_for_M(M_matrix=M_gyro_true, b_vector=b_gyro_true)
    # _, raw_gyro_static = sim_gyro_static_for_bias(sigma=0.0, M_matrix=M_gyro_true, b_vector=b_gyro_true)
    # gt_gyro_rate, raw_gyro_rate = sim_gyro_rate_table_for_M(sigma=0.0, M_matrix=M_gyro_true, b_vector=b_gyro_true)
    sim_data_gyro = {'gyro_static': (None, raw_gyro_static), 'gyro_rate': (gt_gyro_rate, raw_gyro_rate)}
    W_gyro, b_gyro = calibrate_gyroscope_full(sim_data_gyro)
    print(f"   - Gyro Calibration Done. Bias: {b_gyro}")

    # 4. Sensor Fusion (Integration Test) & 보정
    print("\n4. Sensor Fusion Stage (Running 10s Simulation)...")
    dt = 0.01
    time_data, gt_quats, meas_gyro, meas_acc, meas_mag = sim_imu_continuous_rotation(
        time_span=10.0, dt=dt, pitch_amp=1.6,
        M_gyro=M_gyro_true, b_gyro=b_gyro_true,
        M_acc=M_acc_true, b_acc=b_acc_true,
        M_mag=M_mag_true, b_mag=b_mag_true
    )
    """
    time_data, gt_quats, meas_gyro, meas_acc, meas_mag = sim_imu_continuous_rotation(
        time_span=10.0, dt=dt, pitch_amp=1.6,
        sigma_gyro=0.0, sigma_acc=0.0, sigma_mag=0.0, # 👈 노이즈 완전 제거
        M_gyro=M_gyro_true, b_gyro=b_gyro_true,
        M_acc=M_acc_true, b_acc=b_acc_true,
        M_mag=M_mag_true, b_mag=b_mag_true
    )
    """
    calib_gyro = (W_gyro @ (meas_gyro - b_gyro).T).T
    calib_acc  = (W_acc  @ (meas_acc  - b_acc).T).T
    calib_mag  = (W_mag  @ (meas_mag  - b_mag).T).T

    # ... [보정 데이터(calib_*) 생성 이후 로직] ...

    # 0. 첫 번째 측정값으로 초기 자세(Initial Attitude) 계산
    q_init = accel_mag_to_quaternion(calib_acc[0], calib_mag[0])

    # 1. 보정 후(Calibrated) 데이터로 트래킹 진행
    comp_filter = ComplementaryFilter(alpha=0.98, dt=dt, q_init=q_init)
    est_quats = np.array([comp_filter.update(g, a, m) for g, a, m in zip(calib_gyro, calib_acc, calib_mag)])

    # 2. 에러 측정 및 오일러 각 변환
    angle_errors = np.array([q_angle_error(gt, est) for gt, est in zip(gt_quats, est_quats)]) #
    euler_gt = np.array([quat_to_euler(q) for q in gt_quats]) #
    euler_est = np.array([quat_to_euler(q) for q in est_quats]) #

    # 3. 종합 시각화 호출
    print("\n[1/3] 각도 오차(RMSE) 시각화 중...")
    plot_tracking_angle_error(time_data, angle_errors, 0.98,save_path=os.path.join(RESULT_DIR, f"{exp_type}_angle_error.png")) #

    print("\n[2/3] 오일러 각도 비교 (짐벌 락 극복 확인)...")
    plot_tracking_euler_comparison(time_data, euler_gt, euler_est, 0.98,save_path=os.path.join(RESULT_DIR, f"{exp_type}_euler_comparison.png")) #

    print("\n[3/3] 3D 쿼터니언 트래킹 애니메이션 실행...")
    animate_quaternion_tracking(time_data, gt_quats, est_quats,save_path=os.path.join(RESULT_DIR, f"{exp_type}_tracking_animation.gif")) #
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# 모듈 임포트 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation.sensor_simulation import (
    sim_acc_6_position_static, sim_mag_multi_position_static,
    sim_gyro_static_for_bias, sim_gyro_rate_table_for_M, sim_imu_static_pose
)
from calibration.calib_accelerometer import calibrate_acc_12param
from calibration.calib_magnetometer import calibrate_mag_ellipsoid
from calibration.calib_gyroscope import calibrate_gyroscope_full
from utils.quaternion_math import q_angle_error, quat_to_euler, accel_mag_to_quaternion

def estimate_static_orientation_window(acc_data, mag_data):
    """
    정적 제약 조건을 활용하여 특정 윈도우 내의 데이터를 평균낸 후 자세를 추정합니다.
    """
    acc_avg = np.mean(acc_data, axis=0)
    mag_avg = np.mean(mag_data, axis=0)
    return accel_mag_to_quaternion(acc_avg, mag_avg)

def plot_static_constraint_results(window_sizes, rmse_uncal, rmse_cal, save_prefix):
    """
    윈도우 크기에 따른 정적 자세 추정 오차(RMSE)의 변화를 시각화합니다.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(window_sizes, rmse_uncal, 'o-', color='gray', label='Uncalibrated (Static Estimator)')
    plt.plot(window_sizes, rmse_cal, 's-', color='crimson', label='Calibrated (Static Estimator)')
    
    plt.xscale('log') # 윈도우 크기 변화를 잘 보기 위해 로그 스케일 사용
    plt.title('Static Constraint: Orientation Error vs Window Size', fontsize=14)
    plt.xlabel('Window Size [seconds]', fontsize=12)
    plt.ylabel('Orientation RMSE [deg]', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    # 데이터 포인트 위에 값 표시 (가독성용)
    for x, y in zip(window_sizes, rmse_cal):
        plt.annotate(f'{y:.2f}\u00b0', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='crimson')

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_window_analysis.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    RESULT_DIR = r"D:\바탕화면\CS-Study-Tracker\IMU\results"
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    print("=== [Static Constraint: Orientation Estimation Experiment] ===")
    
    # 0. 공통 가상 센서 오차 정의 (기존과 동일)
    M_acc_true = np.array([[1.05, 0.02, 0.01], [0.01, 0.98, 0.02], [0.02, 0.01, 1.02]])
    b_acc_true = np.array([0.05, -0.03, 0.1])
    M_mag_true = np.array([[1.1, 0.1, 0.0], [0.1, 0.9, -0.1], [0.0, -0.1, 1.2]])
    b_mag_true = np.array([0.3, -0.2, 0.1])
    M_gyro_true = np.array([[1.02, 0.01, -0.01], [0.01, 0.99, 0.02], [-0.02, 0.01, 1.05]])
    b_gyro_true = np.array([1.5, -0.8, 0.5])

    # 1. 캘리브레이션 (기존과 동일)
    print("\n[Obtaining Calibration Parameters]")
    _, raw_6pos = sim_acc_6_position_static(n_samples_per_pos=100, M_matrix=M_acc_true, b_vector=b_acc_true)
    W_acc, b_acc = calibrate_acc_12param(raw_6pos, n_samples_per_pos=100)
    
    _, raw_mag_multi = sim_mag_multi_position_static(n_positions=50, n_samples_per_pos=10, M_matrix=M_mag_true, b_vector=b_mag_true)
    W_mag, b_mag = calibrate_mag_ellipsoid(raw_mag_multi)
    
    _, raw_gyro_static = sim_gyro_static_for_bias(M_matrix=M_gyro_true, b_vector=b_gyro_true)
    gt_gyro_rate, raw_gyro_rate = sim_gyro_rate_table_for_M(M_matrix=M_gyro_true, b_vector=b_gyro_true)
    sim_data_gyro = {'gyro_static': (None, raw_gyro_static), 'gyro_rate': (gt_gyro_rate, raw_gyro_rate)}
    W_gyro, b_gyro = calibrate_gyroscope_full(sim_data_gyro)

    # 2. 정적 데이터 생성 (10초 분량)
    print("\n[Running Static Pose Simulation]")
    dt = 0.01
    time_data, gt_quats, meas_gyro, meas_acc, meas_mag = sim_imu_static_pose(
        time_span=10.0, dt=dt, roll_deg=30.0, pitch_deg=45.0, yaw_deg=60.0,
        M_gyro=M_gyro_true, b_gyro=b_gyro_true,
        M_acc=M_acc_true, b_acc=b_acc_true,
        M_mag=M_mag_true, b_mag=b_mag_true
    )

    # 3. 보정 데이터 준비
    calib_acc  = (W_acc  @ (meas_acc  - b_acc).T).T
    calib_mag  = (W_mag  @ (meas_mag  - b_mag).T).T
    calib_gyro = (W_gyro @ (meas_gyro - b_gyro).T).T # 보정 후 자이로는 0에 가까워야 함

    # 4. 정적 제약 조건 실험 (윈도우 크기별 분석)
    print("\n[Analyzing Estimation Error vs Window Size]")
    window_sizes_sec = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    rmse_uncal = []
    rmse_cal = []
    
    # GT는 정적이므로 첫 번째 샘플만 사용
    q_gt = gt_quats[0] 

    for sec in window_sizes_sec:
        n_samples = int(sec / dt)
        if n_samples < 1: n_samples = 1
        
        # 전체 10초 데이터를 해당 윈도우 크기로 조각내서 각각 추정해봄
        n_windows = len(time_data) // n_samples
        errors_uncal_win = []
        errors_cal_win = []
        
        for w in range(n_windows):
            start = w * n_samples
            end = start + n_samples
            
            # Uncalibrated Estimation
            q_est_uncal = estimate_static_orientation_window(meas_acc[start:end], meas_mag[start:end])
            errors_uncal_win.append(q_angle_error(q_gt, q_est_uncal))
            
            # Calibrated Estimation
            q_est_cal = estimate_static_orientation_window(calib_acc[start:end], calib_mag[start:end])
            errors_cal_win.append(q_angle_error(q_gt, q_est_cal))
            
        rmse_uncal.append(np.sqrt(np.mean(np.array(errors_uncal_win)**2)))
        rmse_cal.append(np.sqrt(np.mean(np.array(errors_cal_win)**2)))

    # 5. 자이로 바이어스 관측 (정적 제약의 또 다른 활용)
    print("\n[Observing Gyro Bias in Static Pose]")
    gyro_bias_observed_uncal = np.mean(meas_gyro, axis=0)
    gyro_bias_observed_cal = np.mean(calib_gyro, axis=0)
    print(f" - True Gyro Bias:      {b_gyro_true}")
    print(f" - Observed Bias (Raw): {gyro_bias_observed_uncal} (Residual: {np.linalg.norm(gyro_bias_observed_uncal - b_gyro_true):.4f})")
    print(f" - Observed Bias (Cal): {gyro_bias_observed_cal} (Residual: {np.linalg.norm(gyro_bias_observed_cal):.4f})")

    # 6. 결과 시각화
    print("\n[Visualizing Static Constraint Analysis...]")
    save_prefix = os.path.join(RESULT_DIR, "static_constraint")
    plot_static_constraint_results(window_sizes_sec, rmse_uncal, rmse_cal, save_prefix)

    print("\n완료되었습니다.")

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
from tracking.complementary_quaternion import ComplementaryFilter
from utils.quaternion_math import q_angle_error, quat_to_euler, accel_mag_to_quaternion
from utils.utils_visualization import plot_tracking_angle_error, plot_tracking_euler_comparison

# 보정 전/후 비교 시각화용 함수 정의 (experiment_uncalibrated_tracking 방식 차용)
def plot_static_tracking_compare(time_data, errors_uncalib, errors_calib, euler_gt, euler_uncalib, euler_calib, save_prefix):
    # 1. 각도 오차 비교
    plt.figure(figsize=(10, 4))
    plt.plot(time_data, errors_uncalib, color='gray', alpha=0.6, label='Uncalibrated (Drift)')
    plt.plot(time_data, errors_calib, color='crimson', label='Calibrated (Stable)')
    
    rmse_uncalib = np.sqrt(np.mean(errors_uncalib**2))
    rmse_calib = np.sqrt(np.mean(errors_calib**2))
    
    plt.title('Static Object: Angle Error Drift over Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Angle Error [deg]')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 그래프 내부에 RMSE 텍스트 박스 추가
    info_text = f"Uncalib RMSE: {rmse_uncalib:.2f}\u00b0\nCalib RMSE: {rmse_calib:.2f}\u00b0"
    plt.text(0.02, 0.85, info_text, transform=plt.gca().transAxes, fontsize=11,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
             
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_angle_error.png", dpi=300)
    plt.show()

    # 2. 오일러 각 비교
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('Static Object: Euler Angle Drift Comparison', fontsize=16)
    labels = ['Roll (X)', 'Pitch (Y)', 'Yaw (Z)']
    colors = ['red', 'green', 'blue']
    
    for i in range(3):
        axes[i].plot(time_data, np.degrees(euler_gt[:, i]), color='black', linestyle='--', linewidth=2, label='Ground Truth')
        axes[i].plot(time_data, np.degrees(euler_uncalib[:, i]), color='gray', alpha=0.6, label='Uncalibrated')
        axes[i].plot(time_data, np.degrees(euler_calib[:, i]), color=colors[i], alpha=0.8, label='Calibrated')
        axes[i].set_ylabel(f'{labels[i]} [deg]')
        axes[i].grid(True)
        if i == 0:
            axes[i].legend(loc='upper left', bbox_to_anchor=(1, 1))
            
    axes[2].set_xlabel('Time [s]')
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_euler_comparison.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    RESULT_DIR = r"D:\바탕화면\CS-Study-Tracker\IMU\results"
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    print("=== [Static Object Tracking Experiment] ===")
    
    # 0. 공통 가상 센서 고유 오차(M, b) 정의
    M_acc_true = np.array([[1.05, 0.02, 0.01], [0.01, 0.98, 0.02], [0.02, 0.01, 1.02]])
    b_acc_true = np.array([0.05, -0.03, 0.1])
    M_mag_true = np.array([[1.1, 0.1, 0.0], [0.1, 0.9, -0.1], [0.0, -0.1, 1.2]])
    b_mag_true = np.array([0.3, -0.2, 0.1])
    M_gyro_true = np.array([[1.02, 0.01, -0.01], [0.01, 0.99, 0.02], [-0.02, 0.01, 1.05]])
    b_gyro_true = np.array([1.5, -0.8, 0.5]) # 이 자이로 바이어스가 치명적 누적 오차를 유발함

    # 1. 캘리브레이션 파라미터 획득
    print("\n[Obtaining Calibration Parameters]")
    _, raw_6pos = sim_acc_6_position_static(n_samples_per_pos=100, M_matrix=M_acc_true, b_vector=b_acc_true)
    W_acc, b_acc = calibrate_acc_12param(raw_6pos, n_samples_per_pos=100)
    
    _, raw_mag_multi = sim_mag_multi_position_static(n_positions=50, n_samples_per_pos=10, M_matrix=M_mag_true, b_vector=b_mag_true)
    W_mag, b_mag = calibrate_mag_ellipsoid(raw_mag_multi)
    
    _, raw_gyro_static = sim_gyro_static_for_bias(M_matrix=M_gyro_true, b_vector=b_gyro_true)
    gt_gyro_rate, raw_gyro_rate = sim_gyro_rate_table_for_M(M_matrix=M_gyro_true, b_vector=b_gyro_true)
    sim_data_gyro = {'gyro_static': (None, raw_gyro_static), 'gyro_rate': (gt_gyro_rate, raw_gyro_rate)}
    W_gyro, b_gyro = calibrate_gyroscope_full(sim_data_gyro)

    # 2. 정적 객체 시뮬레이션
    print("\n[Running Static Pose Simulation]")
    dt = 0.01
    time_data, gt_quats, meas_gyro, meas_acc, meas_mag = sim_imu_static_pose(
        time_span=10.0, dt=dt, roll_deg=30.0, pitch_deg=45.0, yaw_deg=60.0,
        M_gyro=M_gyro_true, b_gyro=b_gyro_true,
        M_acc=M_acc_true, b_acc=b_acc_true,
        M_mag=M_mag_true, b_mag=b_mag_true
    )

    # 3. 보정 적용
    calib_gyro = (W_gyro @ (meas_gyro - b_gyro).T).T
    calib_acc  = (W_acc  @ (meas_acc  - b_acc).T).T
    calib_mag  = (W_mag  @ (meas_mag  - b_mag).T).T

    # 4. 필터를 이용한 방향 추정
    print("\n[Tracking with Uncalibrated vs Calibrated Data]")
    q_init_calib = accel_mag_to_quaternion(calib_acc[0], calib_mag[0])
    q_init_uncalib = accel_mag_to_quaternion(meas_acc[0], meas_mag[0])

    comp_filter_calib = ComplementaryFilter(alpha=0.98, dt=dt, q_init=q_init_calib)
    est_quats_calib = np.array([comp_filter_calib.update(g, a, m) for g, a, m in zip(calib_gyro, calib_acc, calib_mag)])

    comp_filter_uncalib = ComplementaryFilter(alpha=0.98, dt=dt, q_init=q_init_uncalib)
    est_quats_uncalib = np.array([comp_filter_uncalib.update(g, a, m) for g, a, m in zip(meas_gyro, meas_acc, meas_mag)])

    # 5. 오차 계산
    angle_errors_calib = np.array([q_angle_error(gt, est) for gt, est in zip(gt_quats, est_quats_calib)])
    angle_errors_uncalib = np.array([q_angle_error(gt, est) for gt, est in zip(gt_quats, est_quats_uncalib)])
    
    euler_gt = np.array([quat_to_euler(q) for q in gt_quats])
    euler_est_calib = np.array([quat_to_euler(q) for q in est_quats_calib])
    euler_est_uncalib = np.array([quat_to_euler(q) for q in est_quats_uncalib])

    print("\n[Calculating Tracking Error (MSE/RMSE)]")
    mse_uncalib = np.mean(angle_errors_uncalib**2)
    mse_calib = np.mean(angle_errors_calib**2)
    rmse_uncalib = np.sqrt(mse_uncalib)
    rmse_calib = np.sqrt(mse_calib)
    print(f"   - Uncalibrated Angle MSE: {mse_uncalib:.3f} (RMSE: {rmse_uncalib:.3f} deg)")
    print(f"   - Calibrated Angle MSE:   {mse_calib:.3f} (RMSE: {rmse_calib:.3f} deg)")

    print("\n[Visualizing Static Object Drift...]")
    save_prefix = os.path.join(RESULT_DIR, "static_object")
    plot_static_tracking_compare(
        time_data, angle_errors_uncalib, angle_errors_calib, 
        euler_gt, euler_est_uncalib, euler_est_calib, save_prefix
    )
    print("완료되었습니다.")

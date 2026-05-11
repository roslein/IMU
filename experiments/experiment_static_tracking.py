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
def plot_static_tracking_compare(time_data, results, euler_gt, save_prefix):
    # 1. 각도 오차 비교 (2x2 Subplots)
    fig_err, axes_err = plt.subplots(2, 2, figsize=(12, 10))
    fig_err.suptitle('Static Tracking Error: Uncalibrated vs Calibrated by Alpha (\u03b1)', fontsize=16)
    
    alphas = list(results.keys())
    for i, a in enumerate(alphas):
        ax = axes_err[i // 2, i % 2]
        res = results[a]
        err_u, err_c = res['err_u'], res['err_c']
        
        ax.plot(time_data, err_u, color='gray', alpha=0.7, label='Uncalibrated')
        ax.plot(time_data, err_c, color='crimson', label='Calibrated')
        
        rmse_u = np.sqrt(np.mean(err_u**2))
        rmse_c = np.sqrt(np.mean(err_c**2))
        
        ax.set_title(f'Gyro Weight (\u03b1) = {a}')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Angle Error [deg]')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right', fontsize=8)
        
        ax.text(0.05, 0.80, f'Uncal RMSE: {rmse_u:.2f}\u00b0\nCal RMSE: {rmse_c:.2f}\u00b0',
                transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{save_prefix}_angle_error.png", dpi=300)

    # 2. 각 가중치별 오일러 각도 상세 비교 (각 Alpha마다 별도 Figure)
    labels = ['Roll (X)', 'Pitch (Y)', 'Yaw (Z)']
    color_cal = 'crimson'  # 보정 후 색상을 통일하여 가독성 향상
    color_uncal = 'gray'
    
    for a in alphas:
        res = results[a]
        e_u = np.array([quat_to_euler(q) for q in res['est_u']])
        e_c = np.array([quat_to_euler(q) for q in res['est_c']])
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f'Euler Angle Tracking Comparison: Alpha (\u03b1) = {a}', fontsize=14)
        
        for i in range(3):
            axes[i].plot(time_data, np.degrees(euler_gt[:, i]), color='black', linestyle='--', linewidth=2, label='Ground Truth')
            axes[i].plot(time_data, np.degrees(e_u[:, i]), color=color_uncal, alpha=0.5, label='Uncalibrated')
            axes[i].plot(time_data, np.degrees(e_c[:, i]), color=color_cal, alpha=0.8, label='Calibrated')
            
            axes[i].set_ylabel(f'{labels[i]} [deg]')
            axes[i].grid(True)
            if i == 0:
                axes[i].legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        axes[2].set_xlabel('Time [s]')
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_euler_comparison_{a}.png", dpi=300)
    
    # 모든 저장이 끝난 후 그래프 출력 (중간에 멈추지 않도록)
    print(f"\n[INFO] 모든 그래프가 {os.path.dirname(save_prefix)} 폴더에 저장되었습니다.")
    plt.show()

if __name__ == "__main__":
    RESULT_DIR = r"D:\바탕화면\CS-Study-Tracker\IMU\results"
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    print("=== [Static Object Tracking Experiment: Detailed Comparisons] ===")
    
    # 0. 공통 가상 센서 고유 오차(M, b) 정의
    M_acc_true = np.array([[1.05, 0.02, 0.01], [0.01, 0.98, 0.02], [0.02, 0.01, 1.02]])
    b_acc_true = np.array([0.05, -0.03, 0.1])
    M_mag_true = np.array([[1.1, 0.1, 0.0], [0.1, 0.9, -0.1], [0.0, -0.1, 1.2]])
    b_mag_true = np.array([0.3, -0.2, 0.1])
    M_gyro_true = np.array([[1.02, 0.01, -0.01], [0.01, 0.99, 0.02], [-0.02, 0.01, 1.05]])
    b_gyro_true = np.array([1.5, -0.8, 0.5])

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

    # 3. 보정 데이터 준비
    calib_gyro = (W_gyro @ (meas_gyro - b_gyro).T).T
    calib_acc  = (W_acc  @ (meas_acc  - b_acc).T).T
    calib_mag  = (W_mag  @ (meas_mag  - b_mag).T).T

    # 4. 필터를 이용한 방향 추정 (다양한 Alpha 비교)
    print("\n[Tracking with different Alpha values]")
    q_init_uncalib = accel_mag_to_quaternion(meas_acc[0], meas_mag[0])
    q_init_calib = accel_mag_to_quaternion(calib_acc[0], calib_mag[0])
    
    alphas = [0.0, 0.9, 0.98, 0.999]
    results = {}
    
    for a in alphas:
        # Uncalibrated Tracking
        cf_u = ComplementaryFilter(alpha=a, dt=dt, q_init=q_init_uncalib)
        est_u = np.array([cf_u.update(meas_gyro[i], meas_acc[i], meas_mag[i]) for i in range(len(time_data))])
        err_u = np.array([q_angle_error(gt_quats[i], est_u[i]) for i in range(len(time_data))])
        
        # Calibrated Tracking
        cf_c = ComplementaryFilter(alpha=a, dt=dt, q_init=q_init_calib)
        est_c = np.array([cf_c.update(calib_gyro[i], calib_acc[i], calib_mag[i]) for i in range(len(time_data))])
        err_c = np.array([q_angle_error(gt_quats[i], est_c[i]) for i in range(len(time_data))])
        
        results[a] = {
            'err_u': err_u, 'err_c': err_c,
            'est_u': est_u, 'est_c': est_c
        }
    
    # 5. 결과 시각화
    print("\n[Visualizing Detailed Comparisons...]")
    save_prefix = os.path.join(RESULT_DIR, "static_object")
    euler_gt = np.array([quat_to_euler(q) for q in gt_quats])
    
    plot_static_tracking_compare(time_data, results, euler_gt, save_prefix)
    
    print("\n[Final RMSE Summary]")
    for a in alphas:
        res = results[a]
        print(f" - Alpha {a:5}: Uncal RMSE {np.sqrt(np.mean(res['err_u']**2)):.4f} | Cal RMSE {np.sqrt(np.mean(res['err_c']**2)):.4f}")
        
    print("완료되었습니다.")

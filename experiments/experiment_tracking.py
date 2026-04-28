import numpy as np
import sys
import os

# 모듈 임포트를 위해 상위 디렉토리 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation.sensor_simulation import (
    sim_acc_6_position_static, 
    sim_mag_figure8_dynamic, 
    sim_gyro_static_for_bias, 
    sim_gyro_rate_table_for_M, 
    sim_imu_continuous_rotation
)
from calibration.calib_accelerometer import calibrate_acc_12param
from calibration.calib_magnetometer import calibrate_mag_ellipsoid
from calibration.calib_gyroscope import calibrate_gyroscope_full
from tracking.complementary_quaternion import ComplementaryFilter
from utils.quaternion_math import q_angle_error, quat_to_euler, accel_mag_to_quaternion
from utils.utils_visualization import (
    plot_tracking_angle_error,
    plot_tracking_euler_comparison,
    animate_quaternion_tracking
)
from simulation.sensor_simulation import sim_mag_multi_position_static

def run_ideal_tracking_experiment():
    print("=== [Ideal Zero-Noise] IMU Quaternion Tracking & Calibration Verification ===")
    print("이 실험은 센서의 노이즈(sigma)를 0으로 통제하여, 캘리브레이션 및 퓨전 알고리즘에")
    print("수학적/기하학적 결함이 없음을 증명하는 무결점 테스트입니다.\n")

    # 0. 공통 가상 센서 고유 오차(M, b) 정의
    M_acc_true = np.array([[1.05, 0.02, 0.01], [0.01, 0.98, 0.02], [0.02, 0.01, 1.02]])
    b_acc_true = np.array([0.05, -0.03, 0.1])
    
    M_mag_true = np.array([[1.1, 0.1, 0.0], [0.1, 0.9, -0.1], [0.0, -0.1, 1.2]])
    b_mag_true = np.array([0.3, -0.2, 0.1])
    
    M_gyro_true = np.array([[1.02, 0.01, -0.01], [0.01, 0.99, 0.02], [-0.02, 0.01, 1.05]])
    b_gyro_true = np.array([1.5, -0.8, 0.5])

    # 1. Accelerometer Calibration (Sigma = 0.0)
    print("1. Accelerometer Stage (Zero-Noise)...")
    _, raw_6pos = sim_acc_6_position_static(sigma=0.0, M_matrix=M_acc_true, b_vector=b_acc_true)
    W_acc, b_acc = calibrate_acc_12param(raw_6pos, n_samples_per_pos=100)

    # 2. Magnetometer Calibration (Sigma = 0.0)
    # (주의: calib_magnetometer.py에 대칭 행렬 강제(Symmetric Matrix) 업데이트가 적용되어 있어야 함)

    print("2. Magnetometer Calibration (Static Averaging + Symmetric Ellipsoid)...")
    _, raw_mag_multi = sim_mag_multi_position_static(
        n_positions=50, n_samples_per_pos=10, 
        M_matrix=M_mag_true, b_vector=b_mag_true
    )
    # 정적 평균화를 위해 n_samples_per_pos 전달
    W_mag, b_mag_est = calibrate_mag_ellipsoid(raw_mag_multi, n_samples_per_pos=10)

    # 3. Gyroscope Calibration (Sigma = 0.0)
    print("3. Gyroscope Stage (Zero-Noise)...")
    _, raw_gyro_static = sim_gyro_static_for_bias(sigma=0.0, M_matrix=M_gyro_true, b_vector=b_gyro_true)
    gt_gyro_rate, raw_gyro_rate = sim_gyro_rate_table_for_M(sigma=0.0, M_matrix=M_gyro_true, b_vector=b_gyro_true)
    sim_data_gyro = {'gyro_static': (None, raw_gyro_static), 'gyro_rate': (gt_gyro_rate, raw_gyro_rate)}
    W_gyro, b_gyro = calibrate_gyroscope_full(sim_data_gyro)

    # 4. Sensor Fusion 시뮬레이션 (Sigma = 0.0)
    print("\n4. Sensor Fusion Stage (Running 10s Zero-Noise Simulation)...")
    dt = 0.01
    time_data, gt_quats, meas_gyro, meas_acc, meas_mag = sim_imu_continuous_rotation(
        time_span=10.0, dt=dt, pitch_amp=1.6, # 극한의 짐벌 락 구역 포함
        sigma_gyro=0.0, sigma_acc=0.0, sigma_mag=0.0, # 노이즈 완전 제거
        M_gyro=M_gyro_true, b_gyro=b_gyro_true,
        M_acc=M_acc_true, b_acc=b_acc_true,
        M_mag=M_mag_true, b_mag=b_mag_true
    )

    # 도출된 파라미터로 측정 데이터 보정 역산
    calib_gyro = (W_gyro @ (meas_gyro - b_gyro).T).T
    calib_acc  = (W_acc  @ (meas_acc  - b_acc).T).T
    calib_mag  = (W_mag  @ (meas_mag  - b_mag).T).T

    # 5. 첫 프레임 측정값 기반 완벽한 초기 자세(q_init) 계산
    print("   - Initializing quaternion from first frame measurements...")
    q_init = accel_mag_to_quaternion(calib_acc[0], calib_mag[0])

    # 6. 상보 필터(Complementary Filter) 구동
    filter_alpha = 0.98
    comp_filter = ComplementaryFilter(alpha=filter_alpha, dt=dt, q_init=q_init)
    
    print("   - Running Complementary Filter...")
    est_quats = np.array([comp_filter.update(g, a, m) for g, a, m in zip(calib_gyro, calib_acc, calib_mag)])

    # 7. 성능 평가 및 오일러 각 변환 (억지 정렬 q_offset 제거됨)
    print("Calculating angle errors and Euler conversions...")
    angle_errors = np.array([q_angle_error(gt, est) for gt, est in zip(gt_quats, est_quats)])
    
    euler_gt = np.array([quat_to_euler(q) for q in gt_quats])
    euler_est = np.array([quat_to_euler(q) for q in est_quats])

    # 8. 시각화
    print("\n[1/3] Plotting Angle Error (\u03b8) with RMSE...")
    plot_tracking_angle_error(time_data, angle_errors, filter_alpha)

    print("\n[2/3] Plotting Euler Angle Comparison (Checking Gimbal Lock survival)...")
    plot_tracking_euler_comparison(time_data, euler_gt, euler_est, filter_alpha)

    print("\n[3/3] Running 3D Quaternion Tracking Animation...")
    animate_quaternion_tracking(time_data, gt_quats, est_quats, skip_frames=5)
    
    print("\nExperiment completed.")

if __name__ == "__main__":
    run_ideal_tracking_experiment()
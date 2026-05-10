import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 모듈 임포트 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation.sensor_simulation import sim_imu_continuous_rotation
from tracking.complementary_quaternion import ComplementaryFilter
from utils.quaternion_math import q_angle_error, quat_to_euler, accel_mag_to_quaternion
from utils.utils_visualization import (
    plot_tracking_angle_error,
    plot_tracking_euler_comparison,
    animate_quaternion_tracking
)

if __name__ == "__main__":
    RESULT_DIR = r"D:\바탕화면\CS-Study-Tracker\IMU\results"
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    exp_type = "uncalibrated"
    print("=== [Uncalibrated Sensor Tracking Experiment] ===")
    
    # 0. 공통 가상 센서 고유 오차(M, b) 정의
    M_acc_true = np.array([[1.05, 0.02, 0.01], [0.01, 0.98, 0.02], [0.02, 0.01, 1.02]])
    b_acc_true = np.array([0.05, -0.03, 0.1])
    M_mag_true = np.array([[1.1, 0.1, 0.0], [0.1, 0.9, -0.1], [0.0, -0.1, 1.2]])
    b_mag_true = np.array([0.3, -0.2, 0.1])
    M_gyro_true = np.array([[1.02, 0.01, -0.01], [0.01, 0.99, 0.02], [-0.02, 0.01, 1.05]])
    b_gyro_true = np.array([1.5, -0.8, 0.5])

    # 1. 동적 회전 시나리오 시뮬레이션 (왜곡된 데이터 생성)
    print("\n1. Running Dynamic Simulation (Generating Distorted Data)...")
    dt = 0.01
    time_data, gt_quats, meas_gyro, meas_acc, meas_mag = sim_imu_continuous_rotation(
        time_span=10.0, dt=dt, pitch_amp=1.6,
        M_gyro=M_gyro_true, b_gyro=b_gyro_true,
        M_acc=M_acc_true, b_acc=b_acc_true,
        M_mag=M_mag_true, b_mag=b_mag_true
    )

    # 2. 보정을 수행하지 않고 원본 데이터(Raw)로 바로 자세 추정 진행
    print("\n2. Tracking with Uncalibrated Data...")
    q_init = accel_mag_to_quaternion(meas_acc[0], meas_mag[0])

    comp_filter = ComplementaryFilter(alpha=0.98, dt=dt, q_init=q_init)
    est_quats = np.array([comp_filter.update(g, a, m) for g, a, m in zip(meas_gyro, meas_acc, meas_mag)])

    # 3. 에러 측정 및 오일러 각 변환
    print("\n3. Calculating Errors...")
    angle_errors = np.array([q_angle_error(gt, est) for gt, est in zip(gt_quats, est_quats)])
    euler_gt = np.array([quat_to_euler(q) for q in gt_quats])
    euler_est = np.array([quat_to_euler(q) for q in est_quats])

    # 4. 시각화
    print("\n[1/3] 보정 전 각도 오차 시각화 중...")
    plot_tracking_angle_error(time_data, angle_errors, 0.98, save_path=os.path.join(RESULT_DIR, f"{exp_type}_angle_error.png"))

    print("\n[2/3] 보정 전 오일러 각도 추이 비교...")
    plot_tracking_euler_comparison(time_data, euler_gt, euler_est, 0.98, save_path=os.path.join(RESULT_DIR, f"{exp_type}_euler_comparison.png"))

    print("\n[3/3] 보정 전 3D 쿼터니언 트래킹 애니메이션 실행...")
    animate_quaternion_tracking(time_data, gt_quats, est_quats, save_path=os.path.join(RESULT_DIR, f"{exp_type}_tracking_animation.gif"))
    
    print("\n실험이 완료되었습니다.")

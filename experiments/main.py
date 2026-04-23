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
from calibration.calib_accelerometer import calibrate_acc_ellipsoid, calibrate_acc_12param
from calibration.calib_magnetometer import calibrate_mag_ellipsoid
from calibration.calib_gyroscope import calibrate_gyroscope_full
from tracking.complementary_quaternion import ComplementaryFilter
from utils.utils_visualization import verify_calibration_pipeline, verify_gyro_calibration_pipeline

if __name__ == "__main__":
    print("=== [Full Pipeline Integration Test] ===")

    # 1. Accelerometer Calibration
    print("\n1. Accelerometer Stage...")
    _, raw_6pos = sim_acc_6_position_static()
    W_acc, b_acc = calibrate_acc_12param(raw_6pos, n_samples_per_pos=100)
    print(f"   - Accel Calibration Done. Bias: {b_acc}")

    # 2. Magnetometer Calibration
    print("\n2. Magnetometer Stage...")
    _, raw_mag = sim_mag_figure8_dynamic()
    W_mag, b_mag = calibrate_mag_ellipsoid(raw_mag)
    print(f"   - Mag Calibration Done. Bias: {b_mag}")

    # 3. Gyroscope Calibration
    print("\n3. Gyroscope Stage...")
    _, raw_gyro_static = sim_gyro_static_for_bias()
    gt_gyro_rate, raw_gyro_rate = sim_gyro_rate_table_for_M()
    sim_data_gyro = {'gyro_static': (None, raw_gyro_static), 'gyro_rate': (gt_gyro_rate, raw_gyro_rate)}
    W_gyro, b_gyro = calibrate_gyroscope_full(sim_data_gyro)
    print(f"   - Gyro Calibration Done. Bias: {b_gyro}")

    # 4. Sensor Fusion (Integration Test)
    print("\n4. Sensor Fusion Stage (Running 10s Simulation)...")
    dt = 0.01
    time_data, gt_quats, meas_gyro, meas_acc, meas_mag = sim_imu_continuous_rotation(
        time_span=10.0, dt=dt, pitch_amp=1.6 # 짐벌 락 구역 포함
    )

    comp_filter = ComplementaryFilter(alpha=0.98, dt=dt)
    est_quats = np.array([comp_filter.update(g, a, m) for g, a, m in zip(meas_gyro, meas_acc, meas_mag)])

    # 5. 결과 저장 (시각화 창 없이 자동 저장)
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 간단한 쿼터니언 트래킹 그래프 생성 및 저장
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    labels = ['q_w', 'q_x', 'q_y', 'q_z']
    for i in range(4):
        axes[i].plot(time_data, gt_quats[:, i], 'k--', label='GT')
        axes[i].plot(time_data, est_quats[:, i], 'b', alpha=0.7, label='Est')
        axes[i].set_ylabel(labels[i])
        axes[i].grid(True)
    axes[0].legend()
    plt.tight_layout()
    
    report_path = os.path.join(RESULTS_DIR, 'integration_test_report.png')
    plt.savefig(report_path)
    print(f"\n[Success] Full pipeline check completed. Result saved to: {report_path}")
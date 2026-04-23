import numpy as np
import sys
import os

# 모듈 임포트를 위해 상위 디렉토리 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation.sensor_simulation import sim_imu_continuous_rotation
from tracking.complementary_quaternion import ComplementaryFilter
from utils.quaternion_math import q_angle_error, q_align_offset, quat_to_euler, q_mult
from utils.utils_visualization import (
    plot_tracking_angle_error,
    plot_tracking_euler_comparison,
    animate_quaternion_tracking
)

def run_tracking_experiment():
    print("=== IMU Quaternion Tracking Performance Evaluation ===")
    
    # 1. 시뮬레이션 데이터 생성 (극한의 짐벌 락 구역 포함)
    dt = 0.01
    time_span = 10.0
    print("Generating simulation data...")
    time_data, gt_quats, meas_gyro, meas_acc, meas_mag = sim_imu_continuous_rotation(
        time_span=time_span, dt=dt, 
        pitch_amp=1.6 # 90도(1.57rad)를 넘겨 오일러 붕괴 유도
    )

    # 2. 상보 필터(Complementary Filter)로 쿼터니언 추정
    filter_alpha = 0.98
    comp_filter = ComplementaryFilter(alpha=filter_alpha, dt=dt)
    
    print("Running Complementary Filter...")
    est_quats = []
    for g, a, m in zip(meas_gyro, meas_acc, meas_mag):
        est_quats.append(comp_filter.update(g, a, m))
    est_quats = np.array(est_quats)

    # 3. 기준 좌표계 정렬 (t=0 시점의 오프셋 제거)
    print("Aligning coordinate frames (Offset cancellation)...")
    q_offset = q_align_offset(gt_quats[0], est_quats[0])
    
    # 도출된 오프셋을 전체 추정 궤적에 적용
    est_quats_aligned = np.array([q_mult(q_offset, q) for q in est_quats])

    # 4. 성능 지표(Metrics) 계산
    print("Calculating angle errors and Euler conversions...")
    angle_errors = np.array([q_angle_error(gt, est) for gt, est in zip(gt_quats, est_quats_aligned)])
    
    euler_gt = np.array([quat_to_euler(q) for q in gt_quats])
    euler_est = np.array([quat_to_euler(q) for q in est_quats_aligned])

    # 5. 시각화 호출
    print("\n[1/3] Plotting Angle Error (\u03b8) with RMSE...")
    plot_tracking_angle_error(time_data, angle_errors, filter_alpha)

    print("\n[2/3] Plotting Euler Angle Comparison (Checking Gimbal Lock survival)...")
    plot_tracking_euler_comparison(time_data, euler_gt, euler_est, filter_alpha)

    print("\n[3/3] Running 3D Quaternion Tracking Animation...")
    # 애니메이션이 창으로 뜰 것입니다. 창을 닫으면 프로그램이 종료됩니다.
    animate_quaternion_tracking(time_data, gt_quats, est_quats_aligned, skip_frames=5)
    
    print("\nExperiment completed.")

if __name__ == "__main__":
    run_tracking_experiment()
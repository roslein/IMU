import numpy as np
import sys
import os

# 모듈 임포트 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation.sensor_simulation import sim_mag_figure8_dynamic, sim_mag_multi_position_static
from calibration.calib_magnetometer import calibrate_mag_ellipsoid
from utils.utils_visualization import verify_calibration_pipeline

if __name__ == "__main__":
    RESULT_DIR = r"D:\바탕화면\CS-Study-Tracker\IMU\results"
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    print("=== [Magnetometer Calibration: Dynamic vs Static Experiment] ===")
    
    # 공통 가상 센서 고유 오차(M, b) 정의
    M_mag_true = np.array([[1.1, 0.1, 0.0], [0.1, 0.9, -0.1], [0.0, -0.1, 1.2]])
    b_mag_true = np.array([0.3, -0.2, 0.1])
    
    # (A) 기존 방법: 동적 8자 궤적 보정
    print("\n1. Running Dynamic (Figure-8) Calibration...")
    _, raw_mag_dyn = sim_mag_figure8_dynamic(n_samples=1000, M_matrix=M_mag_true, b_vector=b_mag_true)
    W_mag_dyn, b_mag_dyn = calibrate_mag_ellipsoid(raw_mag_dyn)
    print(f"   -> Dynamic Bias: {b_mag_dyn}")

    # (B) 개선 방법: 정적 다중 자세 장시간 측정 보정
    print("\n2. Running Static Multi-Position Calibration...")
    _, raw_mag_static = sim_mag_multi_position_static(
        n_positions=50, n_samples_per_pos=500, # 측정 시간 대폭 증가
        M_matrix=M_mag_true, b_vector=b_mag_true
    )
    W_mag_static, b_mag_static = calibrate_mag_ellipsoid(raw_mag_static)
    print(f"   -> Static Bias: {b_mag_static}")

    # 보정 성능 비교를 위한 깨끗한 궤적에 노이즈 없는 왜곡만 적용된 테스트 데이터
    print("\n3. Evaluating Performance...")
    _, test_raw_mag = sim_mag_multi_position_static(
        n_positions=100, n_samples_per_pos=1, sigma=0.0, 
        M_matrix=M_mag_true, b_vector=b_mag_true
    )
    
    dict_calib_mag = {
        'Dynamic Fig-8': (W_mag_dyn @ (test_raw_mag - b_mag_dyn).T).T,
        'Static Multi-Pose': (W_mag_static @ (test_raw_mag - b_mag_static).T).T
    }
    
    verify_calibration_pipeline(test_raw_mag, dict_calib_mag, target_norm=1.0, title='Magnetometer Calibration Performance')
    print("완료되었습니다.")

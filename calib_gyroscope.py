import numpy as np

def calibrate_gyro_bias(raw_static):
    """
    정지 상태 데이터를 이용하여 제로 레이트 바이어스를 계산합니다.
    """
    # 원리 적용
    return np.mean(raw_static, axis=0)

def calibrate_gyro_matrix(gyro_meas, gyro_gt, b_est):
    """
    Rate Table 데이터와 GT를 이용하여 혼합 행렬 M의 역행렬 W를 도출합니다.
    수식: (gyro_meas - b_est) = gyro_gt @ M.T
    """
    # 1. 바이어스 제거된 측정값 준비
    y = gyro_meas - b_est
    
    # 2. 선형 최소제곱법 풀이 (y = gyro_gt @ M_T)
    # np.linalg.lstsq는 A @ x = B 형태를 풀이하므로 A=gyro_gt, B=y 대입
    M_T, residuals, rank, s = np.linalg.lstsq(gyro_gt, y, rcond=None)
    
    # 3. M_est 도출 및 역행렬(보정 행렬) 계산
    M_est = M_T.T
    W_est = np.linalg.inv(M_est)
    
    return W_est

def calibrate_gyroscope_full(sim_data):
    """
    정지 데이터와 Rate Table 데이터를 모두 사용하여 최종 파라미터를 도출합니다.
    """
    # 데이터 분리 (sensor_simulation.py 시나리오 대응)
    gt_static, raw_static = sim_data['gyro_static']
    gt_rate, raw_rate = sim_data['gyro_rate']
    
    # Step 1: Bias 도출
    b_est = calibrate_gyro_bias(raw_static)
    
    # Step 2: Mixing Matrix(W) 도출
    W_est = calibrate_gyro_matrix(raw_rate, gt_rate, b_est)
    
    return W_est, b_est
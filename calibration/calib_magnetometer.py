import numpy as np
from scipy.optimize import least_squares

def preprocess_static_samples(raw_data, n_samples_per_pos):
    """여러 프레임의 정지 데이터를 평균 내어 노이즈를 1차 상쇄합니다."""
    n_positions = len(raw_data) // n_samples_per_pos
    reshaped = raw_data.reshape((n_positions, n_samples_per_pos, 3))
    return np.mean(reshaped, axis=1)

def calibrate_mag_dynamic(raw_dynamic, target_norm=1.0):
    """
    동적으로 수집된 8자 궤적(Figure-8) 데이터를 바탕으로 지자기 센서를 보정합니다.
    대칭 행렬 제약을 사용하여 Soft-iron 왜곡을 보정합니다.
    """
    def residuals(p, d):
        b = p[:3] # 바이어스 3개 (Hard-iron)
        
        # 6개의 변수만으로 대칭 행렬(W) 구성 (Soft-iron)
        W = np.array([
            [p[3], p[4], p[5]],
            [p[4], p[6], p[7]],
            [p[5], p[7], p[8]]
        ])
        
        cal = (W @ (d - b).T).T
        return np.linalg.norm(cal, axis=1) - target_norm
    
    # 초기값 세팅: bias(평균) + 6개의 독립 변수 (단위 행렬 형태)
    mean_b = np.mean(raw_dynamic, axis=0)
    p0 = np.concatenate([mean_b, [1.0, 0.0, 0.0, 1.0, 0.0, 1.0]])
    
    res = least_squares(residuals, p0, args=(raw_dynamic,))
    
    p = res.x
    W_est = np.array([
        [p[3], p[4], p[5]],
        [p[4], p[6], p[7]],
        [p[5], p[7], p[8]]
    ])
    b_est = p[:3]
    
    return W_est, b_est

def calibrate_mag_static(raw_multi, target_norm=1.0, n_samples_per_pos=10):
    """
    정적 다중 자세(Static Multi-Pose) 데이터를 평균화하여 정밀 보정을 수행합니다.
    """
    data_avg = preprocess_static_samples(raw_multi, n_samples_per_pos)
    
    def residuals(p, d):
        b = p[:3]
        W = np.array([
            [p[3], p[4], p[5]],
            [p[4], p[6], p[7]],
            [p[5], p[7], p[8]]
        ])
        cal = (W @ (d - b).T).T
        return np.linalg.norm(cal, axis=1) - target_norm
    
    mean_b = np.mean(data_avg, axis=0)
    p0 = np.concatenate([mean_b, [1.0, 0.0, 0.0, 1.0, 0.0, 1.0]])
    
    res = least_squares(residuals, p0, args=(data_avg,))
    
    p = res.x
    W_est = np.array([
        [p[3], p[4], p[5]],
        [p[4], p[6], p[7]],
        [p[5], p[7], p[8]]
    ])
    b_est = p[:3]
    
    return W_est, b_est

# 하위 호환성을 위해 기본 함수명 유지 (기본은 정적 보정 사용)
def calibrate_mag_ellipsoid(raw_data, target_norm=1.0, n_samples_per_pos=10):
    if n_samples_per_pos > 1:
        return calibrate_mag_static(raw_data, target_norm, n_samples_per_pos)
    else:
        return calibrate_mag_dynamic(raw_data, target_norm)


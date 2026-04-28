import numpy as np
from scipy.optimize import least_squares

'''
# --- [알고리즘] 지자기 센서 타원체 피팅 (Hard/Soft-iron 보정) ---
def calibrate_mag_ellipsoid(raw_dynamic, target_norm=1.0):
    """
    동적으로 수집된 8자 궤적 데이터를 바탕으로 지자기 센서의 Hard/Soft-iron 왜곡을 보정합니다.
    가속도계와 달리 정적(Static) 평균화 전처리 없이 연속된 시계열 데이터를 그대로 최적화에 사용합니다.
    """
    
    # 1. 오차 정의(Nested Function): 측정값에서 bias를 빼고 W를 곱해 구의 반지름(target_norm)과의 차이 계산
    def residuals(p, d):
        b = p[:3]
        W = p[3:].reshape((3,3))
        # cal = W * (d - b)
        cal = (W @ (d - b).T).T
        return np.linalg.norm(cal, axis=1) - target_norm
    
    # 2. 파라미터 초기화: 바이어스는 데이터의 평균값으로, 행렬은 단위 행렬로 시작
    p0 = np.concatenate([np.mean(raw_dynamic, axis=0), np.eye(3).flatten()])
    
    # 3. 비선형 최소제곱법 최적화 실행
    # 8자 궤적 전체 데이터(raw_dynamic)를 그대로 입력으로 전달
    res = least_squares(residuals, p0, args=(raw_dynamic,))
    
    # 4. 결과 추출 및 형태 복원
    W_est = res.x[3:].reshape((3,3))
    b_est = res.x[:3]
    
    return W_est, b_est


def calibrate_mag_ellipsoid(raw_dynamic, target_norm=1.0):
    def residuals(p, d):
        b = p[:3] # 바이어스 3개
        
        # 🚨 [핵심] 6개의 변수만으로 대칭 행렬(W) 강제 조립 🚨
        W = np.array([
            [p[3], p[4], p[5]],
            [p[4], p[6], p[7]], # p[4] 재사용 (W12 = W21)
            [p[5], p[7], p[8]]  # p[5], p[7] 재사용 (W13 = W31, W23 = W32)
        ])
        
        cal = (W @ (d - b).T).T
        return np.linalg.norm(cal, axis=1) - target_norm
    
    # ----------------------------------------------------
    # 초기값 세팅: bias(3개) + 6개의 독립 변수 = 총 9개짜리 배열
    # 초기 대칭 행렬 형태: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] (단위 행렬)
    # p0 = [b_x, b_y, b_z,  1.0, 0.0, 0.0,  1.0, 0.0,  1.0]
    # ----------------------------------------------------
    mean_b = np.mean(raw_dynamic, axis=0)
    p0 = np.concatenate([mean_b, [1.0, 0.0, 0.0, 1.0, 0.0, 1.0]])
    
    # 비선형 최적화 실행 (컴퓨터는 6개의 p 변수만 조작 가능)
    res = least_squares(residuals, p0, args=(raw_dynamic,))
    
    # 결과 추출 후 최종 W 도출
    p = res.x
    W_est = np.array([
        [p[3], p[4], p[5]],
        [p[4], p[6], p[7]],
        [p[5], p[7], p[8]]
    ])
    b_est = p[:3]
    
    return W_est, b_est
'''
import numpy as np
from scipy.optimize import least_squares

def preprocess_static_samples(raw_data, n_samples_per_pos):
    """여러 프레임의 정지 데이터를 평균 내어 노이즈를 1차 상쇄합니다."""
    n_positions = len(raw_data) // n_samples_per_pos
    reshaped = raw_data.reshape((n_positions, n_samples_per_pos, 3))
    return np.mean(reshaped, axis=1)

def calibrate_mag_ellipsoid(raw_multi, target_norm=1.0, n_samples_per_pos=10):
    # 🚨 [추가된 로직] 노이즈 상쇄를 위한 정적 평균화
    data_avg = preprocess_static_samples(raw_multi, n_samples_per_pos)
    
    def residuals(p, d):
        b = p[:3]
        # 6변수 대칭 행렬 제약 (회전 꼼수 원천 차단)
        W = np.array([
            [p[3], p[4], p[5]],
            [p[4], p[6], p[7]],
            [p[5], p[7], p[8]]
        ])
        cal = (W @ (d - b).T).T
        return np.linalg.norm(cal, axis=1) - target_norm
    
    # 파라미터 초기화 (평균화된 데이터를 기준으로 중심점 도출)
    mean_b = np.mean(data_avg, axis=0)
    p0 = np.concatenate([mean_b, [1.0, 0.0, 0.0, 1.0, 0.0, 1.0]])
    
    # 비선형 최적화 (평균화된 data_avg 사용)
    res = least_squares(residuals, p0, args=(data_avg,))
    
    p = res.x
    W_est = np.array([
        [p[3], p[4], p[5]],
        [p[4], p[6], p[7]],
        [p[5], p[7], p[8]]
    ])
    b_est = p[:3]
    
    return W_est, b_est

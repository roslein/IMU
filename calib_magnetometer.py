import numpy as np
from scipy.optimize import least_squares

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
    
    # 2. 파라미터 초기화: 바이어스는 0, 행렬은 단위 행렬로 시작하여 1차원 배열로 평탄화(flatten)
    p0 = np.concatenate([np.zeros(3), np.eye(3).flatten()])
    
    # 3. 비선형 최소제곱법 최적화 실행
    # 8자 궤적 전체 데이터(raw_dynamic)를 그대로 입력으로 전달
    res = least_squares(residuals, p0, args=(raw_dynamic,))
    
    # 4. 결과 추출 및 형태 복원
    W_est = res.x[3:].reshape((3,3))
    b_est = res.x[:3]
    
    return W_est, b_est

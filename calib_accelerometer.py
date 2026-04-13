import numpy as np
from scipy.optimize import least_squares


# --- [전처리 함수] 여러 샘플을 하나의 평균 점으로 압축 ---
def preprocess_static_samples(raw_data, n_samples_per_pos):
    # (Total_Samples, 3) -> (Positions, n_samples_per_pos, 3) -> (Positions, 3)
    n_positions = len(raw_data) // n_samples_per_pos
    reshaped = raw_data.reshape((n_positions, n_samples_per_pos, 3))
    return np.mean(reshaped, axis=1)

# --- [알고리즘 1] 다중 자세 타원체 피팅 ---
def calibrate_acc_ellipsoid(raw_multi, n_samples_per_pos):
    # 1. 전처리 (노이즈 제거를 위한 평균화)
    data_avg = preprocess_static_samples(raw_multi, n_samples_per_pos)
    
    # 2. 오차 정의(Nested Function) -> 측정값 - bias 후 W 곱해서 1(이론값 = 중력가속도)에 빼주기
    def residuals(p, d):
        b, W = p[:3], p[3:].reshape((3,3))
        cal = (W @ (d - b).T).T
        return np.linalg.norm(cal, axis=1) - 1.0
    
    # 가중치와 bias 초기화 -> least square는 1차원 숫자 리스트만 받을 수 있어서 한 변수에 쑤셔넣고 flatten
    p0 = np.concatenate([np.zeros(3), np.eye(3).flatten()])
    # 3.scipy의 함수 사용 -> 오차 계산법과 시작점(p0) 그리고 데이터를 넘겨주고 최적화 -> 내부적으로 무한 루프를 돌면서 함수를 수백번 호출하며 W,b를 수정
    res = least_squares(residuals, p0, args=(data_avg,))
    return res.x[3:].reshape((3,3)), res.x[:3]

# --- [알고리즘 2] 12-파라미터 반복 보정 ---
def calibrate_acc_12param(raw_6pos, n_samples_per_pos, max_iter=20):
    # 1. 6면 평균 데이터 획득 (반드시 6행 3열이어야 함)
    d = preprocess_static_samples(raw_6pos, n_samples_per_pos)
    
    W_est, b_est = np.eye(3), np.zeros(3)
    alpha, beta = 0.0, 0.0 # 초기 환경(기울기) 가정
    
    # 보정 파라미터와 바닥 기울기 정확도를 서로를 반복해서 변화시키면서 최적화
    for _ in range(max_iter):
        # 2. 기울기가 반영된 이상적인 6면 중력 벡터 (NED 기준)
        g_ref = np.array([
            [1, 0, -np.sin(beta)], [-1, 0, np.sin(beta)],
            [0, 1, np.sin(alpha)], [0, -1, -np.sin(alpha)],
            [np.sin(beta), -np.sin(alpha), 1], [-np.sin(beta), np.sin(alpha), -1]
        ])
        
        # 3. 선형 회귀로 파라미터 갱신 (W^-1와 b를 동시에 구함)
        A = np.hstack([g_ref, np.ones((6, 1))])
        sol = np.linalg.lstsq(A, d, rcond=None)[0]
        M_T, b_new = sol[:3, :], sol[3, :]
        W_new = np.linalg.inv(M_T.T)
        
        # 4. 현재 보정된 값에서 잔여 기울기 역산 (arcsin)
        d_cal = (W_new @ (d - b_new).T).T
        
        # 수정됨: d_cal[2, 2]에서 - 1.0 제거 (기대값이 0이므로)
        alpha = np.arcsin(np.clip(d_cal[2, 2], -1.0, 1.0)) 
        beta = np.arcsin(np.clip(d_cal[4, 0], -1.0, 1.0))
        
        # 수렴 조건
        if np.allclose(b_est, b_new, atol=1e-7): break
        W_est, b_est = W_new, b_new
        
    return W_est, b_est
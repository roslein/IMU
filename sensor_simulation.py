import numpy as np

def apply_distortion(gt_data, M_matrix, b_vector, sigma):
	# 샘플을 행단위로 저장해서 gt_data @ M.T + b
    # 전치 행렬 성질을 이용한 행렬 연산 최적화: (AB)^T = B^T A^T
    measured = gt_data @ M_matrix.T + b_vector
    noise = np.random.normal(0, sigma, size=gt_data.shape)
    return measured + noise

# ==========================================
# 1. 자이로스코프 (Gyroscope) 시나리오
# ==========================================
def sim_gyro_static_for_bias(n_samples=1000, sigma=0.1, b_vector=None):
    # 완벽한 정지 상태 (각속도 0)
    gt_static = np.zeros((n_samples, 3))
    M_gyro = np.array([[1.02, 0.01, -0.01], [0.01, 0.99, 0.02], [-0.02, 0.01, 1.05]])
    b_gyro = np.array(b_vector) if b_vector is not None else np.array([1.5, -0.8, 0.5]) 
    return gt_static, apply_distortion(gt_static, M_gyro, b_gyro, sigma)

def sim_gyro_rate_table_for_M(rate_dps=100.0, n_samples_per_axis=100, sigma=0.5, b_vector=None):
    # 각 축(X, Y, Z)별로 알려진 각속도로 회전하는 테이블 모사
    gt_x = np.tile([rate_dps, 0.0, 0.0], (n_samples_per_axis, 1))
    gt_y = np.tile([0.0, rate_dps, 0.0], (n_samples_per_axis, 1))
    gt_z = np.tile([0.0, 0.0, rate_dps], (n_samples_per_axis, 1))
    gt_rate = np.vstack([gt_x, gt_y, gt_z])
    
    M_gyro = np.array([[1.02, 0.01, -0.01], [0.01, 0.99, 0.02], [-0.02, 0.01, 1.05]])
    b_gyro = np.array(b_vector) if b_vector is not None else np.array([1.5, -0.8, 0.5]) 
    return gt_rate, apply_distortion(gt_rate, M_gyro, b_gyro, sigma)

# ==========================================
# 2. 가속도계 (Accelerometer) 시나리오
# ==========================================
#[[(큐브 형태의 지그(Jig)에 센서를 고정하고 각 면을 측정하는 모습): 가속도계 6면 캘리브레이션 큐브]]
def sim_acc_6_position_static(n_samples_per_pos=100, sigma=0.01, b_vector=None):
    # 6개의 직교하는 방향으로 센서를 뒤집어가며 정지시킨 상태의 이론적 결과 (1g에 대한 up 벡터)
    axes = [
        [ 1.0,  0.0,  0.0], [-1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0], [ 0.0, -1.0,  0.0],
        [ 0.0,  0.0,  1.0], [ 0.0,  0.0, -1.0]
    ]
    # np.tile: 측정값을 각 pose마다 n_samples_per_pos씩 측정(여러 번 샘플링해서 떨림,노이즈 제거)
    # np.vstack: 한면당 (n_samples_per_pos,3)을 6번 수직으로 쌓아서 (n_samples_per_pos*6,3)
    gt_6pos = np.vstack([np.tile(ax, (n_samples_per_pos, 1)) for ax in axes])
    
    M_acc = np.array([[1.05, 0.02, 0.01], [0.01, 0.98, 0.02], [0.02, 0.01, 1.02]])
    b_acc = np.array(b_vector) if b_vector is not None else np.array([0.05, -0.03, 0.1])
    return gt_6pos, apply_distortion(gt_6pos, M_acc, b_acc, sigma)

def sim_acc_multi_position_static(n_positions=50, n_samples_per_pos=10, sigma=0.01, b_vector=None):
    # 무작위의 다양한 각도로 기울인 뒤 짧게 정지하는 상황 모사 (반지름 1g인 구면 좌표계 활용)
    theta = np.random.uniform(0, np.pi, n_positions)
    phi = np.random.uniform(0, 2*np.pi, n_positions)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    base_vectors = np.vstack([x, y, z]).T
    # [자세A, 자세A, 자세A] -> np.tile은 [자세A, 자세B, 자세C, 자세A, 자세B, 자세C] & np.repeat는 [자세A, 자세A, 자세A, 자세B, 자세B, 자세B]
    gt_multi = np.repeat(base_vectors, n_samples_per_pos, axis=0)
    
    M_acc = np.array([[1.05, 0.02, 0.01], [0.01, 0.98, 0.02], [0.02, 0.01, 1.02]])
    b_acc = np.array(b_vector) if b_vector is not None else np.array([0.05, -0.03, 0.1])
    return gt_multi, apply_distortion(gt_multi, M_acc, b_acc, sigma)

# ==========================================
# 3. 지자기 센서 (Magnetometer) 시나리오
# ==========================================
def sim_mag_figure8_dynamic(n_samples=1000, sigma=0.02, b_vector=None):
    # 허공에서 연속적으로 8자를 그리는 동적(Dynamic) 궤적
    t = np.linspace(0, 2*np.pi, n_samples)
    # 평면상의 8자 궤적은 리사주(Lissajous) 곡선으로 만들고, 휴대폰을 앞뒤로 뒤집는 동작은 sin(t)로 회전(tilt)을 준 것이다.
    gt_dynamic = np.vstack([np.cos(t), np.sin(t*2), np.sin(t)]).T
    # 단위 벡터로 정규화(자기장 크기 항상 일정하다고 가정)
    gt_dynamic = gt_dynamic / np.linalg.norm(gt_dynamic, axis=1, keepdims=True)
    
    M_mag = np.array([[1.1, 0.1, 0.0], [0.1, 0.9, -0.1], [0.0, -0.1, 1.2]])
    b_mag = np.array(b_vector) if b_vector is not None else np.array([0.3, -0.2, 0.1])
    return gt_dynamic, apply_distortion(gt_dynamic, M_mag, b_mag, sigma)
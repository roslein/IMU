import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

def apply_distortion(gt_data, M_matrix, b_vector, sigma):
	# 샘플을 행단위로 저장해서 gt_data @ M.T + b
    # 전치 행렬 성질을 이용한 행렬 연산 최적화: (AB)^T = B^T A^T
    measured = gt_data @ M_matrix.T + b_vector
    noise = np.random.normal(0, sigma, size=gt_data.shape)
    return measured + noise

# ==========================================
# 1. 자이로스코프 (Gyroscope) 시나리오
# ==========================================
def sim_gyro_static_for_bias(n_samples=1000, sigma=0.1, M_matrix=None, b_vector=None):
    # 완벽한 정지 상태 (각속도 0)
    gt_static = np.zeros((n_samples, 3))
    M_gyro = np.array(M_matrix) if M_matrix is not None else np.array([[1.02, 0.01, -0.01], [0.01, 0.99, 0.02], [-0.02, 0.01, 1.05]])
    b_gyro = np.array(b_vector) if b_vector is not None else np.array([1.5, -0.8, 0.5]) 
    return gt_static, apply_distortion(gt_static, M_gyro, b_gyro, sigma)

def sim_gyro_rate_table_for_M(rate_dps=100.0, n_samples_per_axis=100, sigma=0.5, M_matrix=None, b_vector=None):
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
def sim_acc_6_position_static(n_samples_per_pos=100, sigma=0.01, M_matrix=None, b_vector=None):
    # 6개의 직교하는 방향으로 센서를 뒤집어가며 정지시킨 상태의 이론적 결과 (1g에 대한 up 벡터)
    axes = [
        [ 1.0,  0.0,  0.0], [-1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0], [ 0.0, -1.0,  0.0],
        [ 0.0,  0.0,  1.0], [ 0.0,  0.0, -1.0]
    ]
    # np.tile: 측정값을 각 pose마다 n_samples_per_pos씩 측정(여러 번 샘플링해서 떨림,노이즈 제거)
    # np.vstack: 한면당 (n_samples_per_pos,3)을 6번 수직으로 쌓아서 (n_samples_per_pos*6,3)
    gt_6pos = np.vstack([np.tile(ax, (n_samples_per_pos, 1)) for ax in axes])
    
    M_acc = np.array(M_matrix) if M_matrix is not None else np.array([[1.05, 0.02, 0.01], [0.01, 0.98, 0.02], [0.02, 0.01, 1.02]])
    b_acc = np.array(b_vector) if b_vector is not None else np.array([0.05, -0.03, 0.1])
    return gt_6pos, apply_distortion(gt_6pos, M_acc, b_acc, sigma)

def sim_acc_multi_position_static(n_positions=50, n_samples_per_pos=10, sigma=0.01, M_matrix=None, b_vector=None):
    # 무작위의 다양한 각도로 기울인 뒤 짧게 정지하는 상황 모사 (반지름 1g인 구면 좌표계 활용)
    theta = np.random.uniform(0, np.pi, n_positions)
    phi = np.random.uniform(0, 2*np.pi, n_positions)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    base_vectors = np.vstack([x, y, z]).T
    # [자세A, 자세A, 자세A] -> np.tile은 [자세A, 자세B, 자세C, 자세A, 자세B, 자세C] & np.repeat는 [자세A, 자세A, 자세A, 자세B, 자세B, 자세B]
    gt_multi = np.repeat(base_vectors, n_samples_per_pos, axis=0)
    
    M_acc = np.array(M_matrix) if M_matrix is not None else np.array([[1.05, 0.02, 0.01], [0.01, 0.98, 0.02], [0.02, 0.01, 1.02]])
    b_acc = np.array(b_vector) if b_vector is not None else np.array([0.05, -0.03, 0.1])
    return gt_multi, apply_distortion(gt_multi, M_acc, b_acc, sigma)

# ==========================================
# 3. 지자기 센서 (Magnetometer) 시나리오
# ==========================================
def sim_mag_figure8_dynamic(n_samples=1000, sigma=0.02, M_matrix=None, b_vector=None):
    # 허공에서 연속적으로 8자를 그리는 동적(Dynamic) 궤적
    t = np.linspace(0, 2*np.pi, n_samples)
    # 평면상의 8자 궤적은 리사주(Lissajous) 곡선으로 만들고, 휴대폰을 앞뒤로 뒤집는 동작은 sin(t)로 회전(tilt)을 준 것이다.
    gt_dynamic = np.vstack([np.cos(t), np.sin(t*2), np.sin(t)]).T
    # 단위 벡터로 정규화(자기장 크기 항상 일정하다고 가정)
    gt_dynamic = gt_dynamic / np.linalg.norm(gt_dynamic, axis=1, keepdims=True)
    
    M_mag = np.array(M_matrix) if M_matrix is not None else np.array([[1.1, 0.1, 0.0], [0.1, 0.9, -0.1], [0.0, -0.1, 1.2]])
    b_mag = np.array(b_vector) if b_vector is not None else np.array([0.3, -0.2, 0.1])
    return gt_dynamic, apply_distortion(gt_dynamic, M_mag, b_mag, sigma)

def sim_mag_multi_position_static(n_positions=50, n_samples_per_pos=10, sigma=0.02, M_matrix=None, b_vector=None):
    """
    정적 평균화(Static Averaging)와 구면 전체의 고른 커버리지를 위해 
    다양한 무작위 3D 각도로 센서를 멈춰서 지자기 데이터를 수집하는 상황을 모사합니다.
    """
    # 3D 구면상에 고르게 퍼진 타겟 벡터(자세) 생성
    theta = np.random.uniform(0, np.pi, n_positions)
    phi = np.random.uniform(0, 2*np.pi, n_positions)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    base_vectors = np.vstack([x, y, z]).T
    
    # 각 자세마다 n_samples_per_pos 만큼 정지하여 데이터 수집 (노이즈 상쇄용)
    gt_multi = np.repeat(base_vectors, n_samples_per_pos, axis=0)
    
    M_mag = np.array(M_matrix) if M_matrix is not None else np.array([[1.1, 0.1, 0.0], [0.1, 0.9, -0.1], [0.0, -0.1, 1.2]])
    b_mag = np.array(b_vector) if b_vector is not None else np.array([0.3, -0.2, 0.1])
    
    return gt_multi, apply_distortion(gt_multi, M_mag, b_mag, sigma)


# ==========================================
# 4. 센서 퓨전 테스트용 동적 연속 회전 시나리오
# ==========================================
def sim_imu_continuous_rotation(time_span=10.0, dt=0.01, 
                                roll_amp=0.5, roll_freq=0.1,
                                pitch_amp=0.3, pitch_freq=0.15,
                                yaw_amp=0.8, yaw_freq=0.05,
                                sigma_gyro=0.01, sigma_acc=0.02, sigma_mag=0.05,
                                M_gyro=None, b_gyro=None,
                                M_acc=None, b_acc=None,
                                M_mag=None, b_mag=None):
    """
    시간에 따라 Roll, Pitch, Yaw가 부드럽게 변하는 3D 궤적을 만들고,
    이에 따른 이상적인 자이로, 가속도, 지자기 센서 데이터를 시뮬레이션합니다.
    """
    time = np.arange(0, time_span, dt)
    n_samples = len(time)

    # 1. 동적 궤적 생성 (임의의 부드러운 사인파 궤적)
    # [파라미터화의 필요성]: 험지(고주파 진동), 우주(저주파) 등 다양한 환경에서 필터의 강건성을 테스트하기 위해 진폭과 주파수를 파라미터로 개방합니다.
    # [짐벌 락 안전성]: 시뮬레이터는 오일러 각을 '생성'하여 행렬로 순방향 변환(정방향 기하학)만 하므로 역산 붕괴인 짐벌 락(Gimbal Lock)이 발생하지 않습니다.
    # 따라서 Pitch 진폭이 90도(1.57rad)를 넘어 극한으로 치솟아도 데이터는 멀쩡하게 생성되며,
    # 오히려 이 진폭 폭주 극한 데이터를 통해 우리의 쿼터니언 필터가 짐벌 락 구역을 발작 없이 이겨내는 것을 완벽히 증명할 수 있습니다.
    roll = roll_amp * np.sin(2 * np.pi * roll_freq * time)
    pitch = pitch_amp * np.cos(2 * np.pi * pitch_freq * time)
    yaw = yaw_amp * np.sin(2 * np.pi * yaw_freq * time)

    gt_quats = []
    raw_gyro = np.zeros((n_samples, 3))
    raw_acc = np.zeros((n_samples, 3))
    raw_mag = np.zeros((n_samples, 3))

    # 2. 생성한 동적 궤적에 따른 이상적인 센서 데이터 생성
    # NED 기준 절대 벡터
    # 가속도계는 중력의 반작용 힘을 느끼므로, Down(+Z)의 반대인 Up(-Z) 방향을 측정함
    ned_up = np.array([0.0, 0.0, -1.0]) 
    # 지자기는 북쪽(+X)을 향하되 약간 아래를 향하는 Dip angle 포함
    ned_north = np.array([1.0, 0.0, 0.5]) 
    ned_north = ned_north / np.linalg.norm(ned_north)

    for i in range(n_samples):
        # 오일러 각을 Ground Truth 쿼터니언으로 변환 (scipy 이용)
        r = R_scipy.from_euler('xyz', [roll[i], pitch[i], yaw[i]])
        q = r.as_quat() # scipy는 [x, y, z, w] 순서
        q_gt = np.array([q[3], q[0], q[1], q[2]]) # 우리 시스템의 [w, x, y, z]로 맞춤
        gt_quats.append(q_gt)

        # 자이로스코프: 이전 자세와의 차이(Delta)를 통해 각속도 추출
        if i > 0:
            r_prev = R_scipy.from_euler('xyz', [roll[i-1], pitch[i-1], yaw[i-1]])
            # $$R_{prev} \times R_{delta} = R_{current}$$이므로 $$R_{delta} = R_{prev}^{-1} \times R_{current}$$
            r_delta = r_prev.inv() * r
            axis_angle = r_delta.as_rotvec() # 회전 행렬 -> 축-각도 형태 반환
            raw_gyro[i] = axis_angle / dt  # 각도/시간 = 각속도
        else:
            raw_gyro[i] = np.array([0.0, 0.0, 0.0])

        # 가속도/지자기: 글로벌 NED 벡터를 현재 센서(Body) 프레임으로 역회전(R^T)
        raw_acc[i] = r.inv().apply(ned_up)
        raw_mag[i] = r.inv().apply(ned_north)

    # 3. 생성된 완벽한 데이터에 기존 apply_distortion 함수로 노이즈/편향 추가
    M_g = M_gyro if M_gyro is not None else np.eye(3)
    b_g = b_gyro if b_gyro is not None else np.zeros(3)
    
    M_a = M_acc if M_acc is not None else np.eye(3)
    b_a = b_acc if b_acc is not None else np.zeros(3)
    
    M_m = M_mag if M_mag is not None else np.eye(3)
    b_m = b_mag if b_mag is not None else np.zeros(3)
    # 테스트용이므로 편향(b)은 0으로 두고 노이즈(sigma)만 적용합니다. (원하면 캘리브레이션 전후 비교를 위해 편향 추가 가능)
    meas_gyro = apply_distortion(raw_gyro, M_g, b_g, sigma_gyro)
    meas_acc = apply_distortion(raw_acc, M_a, b_a, sigma_acc)
    meas_mag = apply_distortion(raw_mag, M_m, b_m, sigma_mag)

    return time, np.array(gt_quats), meas_gyro, meas_acc, meas_mag
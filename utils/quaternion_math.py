import numpy as np

def q_normalize(q):
    """
    쿼터니언을 단위 쿼터니언으로 정규화합니다.
    q: [w, x, y, z] 형태의 numpy array
    """
    norm = np.linalg.norm(q)
    if norm == 0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm

def q_mult(q1, q2):
    """
    두 쿼터니언의 해밀턴 곱(Hamilton Product)을 계산합니다.
    q1, q2: [w, x, y, z] 형태의 numpy array
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def q_conj(q):
    """
    쿼터니언의 켤레(Conjugate)를 반환합니다. 역회전에 사용됩니다.
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])

def get_shortest_arc_quaternion(v1, v2):
    """
    두 3D 벡터 v1에서 v2로 회전하는 최단 경로 쿼터니언을 구합니다. (Method 2)
    행렬(DCM) 연산을 완전히 배제하고 내적과 외적 사칙연산만으로 쿼터니언을 다이렉트로 도출하는 최적화 기법입니다.
    
    [수학적 원리 (로드리게스 대통합의 역과정)]
    1. q_raw = (1 + v1·v2) + (v1 × v2)
    2. v1·v2 = cos(θ), v1 × v2 = n*sin(θ) 이므로:
       q_raw = (1 + cos(θ)) + n*sin(θ)
    3. 삼각함수 반각 공식 (1+cos(θ) = 2cos²(θ/2), sin(θ) = 2sin(θ/2)cos(θ/2)) 적용:
       q_raw = 2cos(θ/2) * [ cos(θ/2) + n*sin(θ/2) ]
    4. 괄호 안의 식은 완벽한 회전 쿼터니언(e^(θ/2*n))과 일치합니다.
       따라서 q_raw를 정규화(Normalize)하면 앞의 스칼라 덩어리(2cos(θ/2))가 약분되어 날아가고, 
       온전한 회전 쿼터니언 q만 우아하게 남게 됩니다.
       
    180도 회전(반대 방향)일 때의 특이점은 병렬 축 외적 검사를 통해 예외 처리합니다.
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    dot = np.dot(v1, v2)
    
    # 두 벡터가 거의 반대 방향일 때 (180도 회전 특이점)
    if dot < -0.999999:
        # v1과 직교하는 임의의 축을 찾습니다.
        cross = np.cross([1, 0, 0], v1)
        if np.linalg.norm(cross) < 1e-6:
            cross = np.cross([0, 1, 0], v1)
        cross = cross / np.linalg.norm(cross)
        return np.array([0.0, cross[0], cross[1], cross[2]]) # 180도 회전 쿼터니언 (w=0)
    
    # [일반적인 경우: 수식과 코드 1:1 매핑]
    
    # 1. (u x v) 외적 계산: 회전축(n) * sin(θ)에 해당하는 벡터부(x, y, z) 성분을 도출합니다.
    cross = np.cross(v1, v2)
    
    # 2. q_raw = (1 + u·v) + (u × v) 조합하기:
    # v1, v2가 단위 벡터이므로 ||v1||*||v2|| = 1 이 되어, 스칼라부(w)는 1.0 + dot이 됩니다.
    # 이때 만들어진 q는 아직 2cos(θ/2) * [cos(θ/2) + n*sin(θ/2)] 형태의 스칼라가 붙어있는 q_raw 입니다.
    q = np.array([1.0 + dot, cross[0], cross[1], cross[2]])
    
    # 3. 정규화 (Normalization) 및 스칼라 약분:
    # 자기 자신의 크기(Norm)로 나누는 q_normalize를 거치는 순간, 
    # q_raw에 붙어있던 스칼라 덩어리 2cos(θ/2)가 완벽하게 약분되어 날아가고 순수한 단위 회전 쿼터니언만 남습니다.
    return q_normalize(q)

def dcm_to_quaternion(R):
    """
    3x3 회전 행렬(DCM, Direction Cosine Matrix)을 쿼터니언으로 변환합니다.
    """
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return q_normalize(np.array([w, x, y, z]))

def accel_mag_to_quaternion(acc_meas, mag_meas):
    """
    가속도(Gravity)와 지자기(Magnetic North) 측정값을 이용해
    고정된 NED 좌표계 대비 센서의 절대 자세(Quaternion)를 계산합니다. (TRIAD 알고리즘 기반)
    """
    # 1. Down 벡터 (가속도는 중력의 반발력을 측정하여 Up 방향을 향하므로, -부호를 붙여 실제 Down 방향 복원)
    D = -acc_meas / np.linalg.norm(acc_meas)
    
    # 2. East 벡터 (지자기 벡터와 Down 벡터의 외적)
    mag_norm = mag_meas / np.linalg.norm(mag_meas)
    E = np.cross(D, mag_norm)
    E = E / np.linalg.norm(E)
    
    # 3. North 벡터 (East와 Down의 외적을 통해 직교하는 수평성분만 담긴 North 방향 도출)
    N = np.cross(E, D)
    
    # 4. 회전 행렬 구성 (Body to NED) -> 변환 후 쿼터니언으로 도출
    # 열 벡터로 N, E, D를 배치하여 DCM을 만듭니다.
    R = np.column_stack((N, E, D))
    
    return dcm_to_quaternion(R)

def slerp(q0, q1, alpha):
    """
    두 쿼터니언 사이를 구면 선형 보간(SLERP)합니다.
    alpha: 0.0 이면 q0, 1.0 이면 q1에 가깝습니다.
    """
    # 두 쿼터니언 내적 (코사인 각도)
    dot = np.dot(q0, q1)
    
    # 최단 경로를 위해 내적이 음수면 한 쿼터니언을 뒤집음
    if dot < 0.0:
        q1 = -q1
        dot = -dot
        
    # 두 쿼터니언이 너무 가까우면 선형 보간(LERP) 수행
    if dot > 0.9995:
        result = q0 + alpha * (q1 - q0)
        return q_normalize(result)
        
    # SLERP 수식 적용
    theta_0 = np.arccos(dot)
    theta = theta_0 * alpha
    
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return q_normalize((s0 * q0) + (s1 * q1))
import numpy as np
import sys
import os

# 이전 단계에서 만든 쿼터니언 수학 모듈 임포트
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.quaternion_math import q_mult, accel_mag_to_quaternion, slerp

class ComplementaryFilter:
    def __init__(self, alpha=0.98, dt=0.01, q_init=None): # q_init 파라미터 추가
        self.alpha = alpha
        self.dt = dt
        
        # 외부에서 초기 자세를 넘겨주면 그것을 사용하고, 없으면 기본값(0도) 사용
        if q_init is not None:
            self.q = q_init
        else:
            self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def update(self, gyro, acc, mag):
        """
        매 센서 측정 스텝마다 자세(쿼터니언)를 업데이트합니다.
        """
        # ----------------------------------------------------
        # 1. 자이로스코프를 이용한 상태 예측 (Dead Reckoning)
        # ----------------------------------------------------
        gyro_norm = np.linalg.norm(gyro)
        if gyro_norm > 0:
            angle = gyro_norm * self.dt
            axis = gyro / gyro_norm
            
            # 노트 3.5.1: 회전 쿼터니언은 목표 각도를 반각(theta/2)으로 쪼개서 지수에 올림
            q_delta = np.array([
                np.cos(angle / 2.0),
                axis[0] * np.sin(angle / 2.0),
                axis[1] * np.sin(angle / 2.0),
                axis[2] * np.sin(angle / 2.0)
            ])
            # 이전 자세에 델타 회전 곱하기 (새로운 예측 자세)
            q_gyro = q_mult(self.q, q_delta)
        else:
            q_gyro = self.q

        # ----------------------------------------------------
        # 2. 가속도계와 지자기를 이용한 절대 방위 보정 (AM 쿼터니언)
        # ----------------------------------------------------
        try:
            # utils/quaternion_math.py 에 구현한 NED 기반 절대 자세 도출
            q_am = accel_mag_to_quaternion(acc, mag)
            
            # ----------------------------------------------------
            # 3. 상보 필터 (구면 선형 보간 - SLERP)
            # ----------------------------------------------------
            # alpha가 1.0에 가까우면 q_gyro를 강하게 신뢰하고, q_am은 드리프트 방지용 닻(Anchor) 역할만 함
            self.q = slerp(q_am, q_gyro, self.alpha)
            
        except np.linalg.LinAlgError:
            # 특이점(예: 자유낙하 상태 등)으로 AM 계산 실패 시 이번 스텝은 자이로만 믿음
            self.q = q_gyro
            
        return self.q
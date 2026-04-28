import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R_scipy

def calculate_mse(data_calibrated, target_norm=1.0):
    norms = np.linalg.norm(data_calibrated, axis=1)
    mse_val = np.mean((norms - target_norm)**2)
    return mse_val

def plot_calibration_results(data_raw_test, dict_calibrated_data, 
                             title='3D Sensor Calibration Verification', 
                             unit='Normalized'):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 기준이 되는 반지름 1의 이상적인 구(Sphere) 와이어프레임 생성
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 15)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.2, label='Ideal Unit Sphere')
    
    # 보정 전 왜곡된 데이터 (Raw Data) 산점도 플롯
    ax.scatter(data_raw_test[:, 0], data_raw_test[:, 1], data_raw_test[:, 2], 
               color='red', alpha=0.5, s=10, label='Raw Distorted Data')
    
    colors = ['blue', 'green', 'purple']
    markers = ['o', '^', 's']
    
    # 보정 알고리즘별 복원 데이터 산점도 오버레이 플롯
    for idx, (algo_name, data_cal) in enumerate(dict_calibrated_data.items()):
        ax.scatter(data_cal[:, 0], data_cal[:, 1], data_cal[:, 2], 
                   color=colors[idx % len(colors)], marker=markers[idx % len(markers)], 
                   alpha=0.6, s=15, label=f'Calibrated ({algo_name})')
        
    # 매개변수로 받은 제목과 단위 적용
    ax.set_title(title)
    ax.set_xlabel(f'X axis ({unit})')
    ax.set_ylabel(f'Y axis ({unit})')
    ax.set_zlabel(f'Z axis ({unit})')
    
    ax.set_box_aspect([1,1,1])
    ax.legend()
    plt.show()

def verify_calibration_pipeline(data_raw_test, dict_calibrated_data, target_norm=1.0, title='3D Sensor Calibration Verification', unit='Normalized'):
    print(f"\n[{title}]")
    print(f"Raw Test Data MSE: {calculate_mse(data_raw_test, target_norm):.6f}")
    
    for algo_name, data_cal in dict_calibrated_data.items():
        print(f"Algorithm ({algo_name}) MSE: {calculate_mse(data_cal, target_norm):.6f}")
        
    plot_calibration_results(data_raw_test, dict_calibrated_data, title=title, unit=unit)

# utils_visualization.py 에 추가할 코드

def calculate_mse_gt(data_calibrated, data_gt):
    # 구 표면(Norm)이 아닌 참값(GT) 벡터와의 직접적인 거리 오차 제곱 평균 계산
    diff = np.linalg.norm(data_calibrated - data_gt, axis=1)
    mse_val = np.mean(diff**2)
    return mse_val

def plot_gyro_calibration_results(data_gt, data_raw, dict_calibrated_data, 
                                  title='3D Gyroscope Calibration Verification', 
                                  unit='dps'):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Ground Truth 데이터 플롯 (목표로 하는 정확한 회전축과 속도)
    ax.scatter(data_gt[:, 0], data_gt[:, 1], data_gt[:, 2], 
               color='black', marker='*', s=100, label='Ground Truth (Rate Table)')
    
    # 2. 보정 전 왜곡된 데이터 (Raw Data) 산점도 플롯
    ax.scatter(data_raw[:, 0], data_raw[:, 1], data_raw[:, 2], 
               color='red', alpha=0.5, s=15, label='Raw Distorted Data')
    
    colors = ['blue', 'green', 'purple']
    markers = ['o', '^', 's']
    
    # 3. 보정 알고리즘별 복원 데이터 산점도 오버레이 플롯
    for idx, (algo_name, data_cal) in enumerate(dict_calibrated_data.items()):
        ax.scatter(data_cal[:, 0], data_cal[:, 1], data_cal[:, 2], 
                   color=colors[idx % len(colors)], marker=markers[idx % len(markers)], 
                   alpha=0.6, s=15, label=f'Calibrated ({algo_name})')
        
    ax.set_title(title)
    ax.set_xlabel(f'X axis ({unit})')
    ax.set_ylabel(f'Y axis ({unit})')
    ax.set_zlabel(f'Z axis ({unit})')

    ax.set_box_aspect([1, 1, 1])
    
    ax.legend()
    plt.show()

def verify_gyro_calibration_pipeline(data_gt, data_raw, dict_calibrated_data, 
                                     title='3D Gyroscope Calibration Verification', 
                                     unit='dps'):
    # 정량적 MSE 평가 (GT와의 거리 기준)
    print(f"Gyroscope Raw Data MSE (vs GT): {calculate_mse_gt(data_raw, data_gt):.6f}")
    
    for algo_name, data_cal in dict_calibrated_data.items():
        print(f"Gyroscope Calibrated ({algo_name}) MSE (vs GT): {calculate_mse_gt(data_cal, data_gt):.6f}")
    
    # 정성적 3D 시각화
    plot_gyro_calibration_results(data_gt, data_raw, dict_calibrated_data, title, unit)

# ==========================================
# [Step 2] 2D 성능 평가 그래프 유틸리티
# ==========================================

def plot_tracking_angle_error(time_data, angle_errors, alpha_val,save_path=None):
    """
    시간에 따른 두 쿼터니언 사이의 절대 각도 오차(Degree)를 시각화합니다.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(time_data, angle_errors, color='crimson', label='Angle Error (\u03b8)')
    
    # 평균 제곱근 오차(RMSE) 계산 및 표시
    rmse = np.sqrt(np.mean(angle_errors**2))
    
    plt.title(f'Quaternion Tracking Angle Error (Filter \u03b1={alpha_val})')
    plt.xlabel('Time [s]')
    plt.ylabel('Angle Error [deg]')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 그래프 내부에 RMSE 텍스트 박스 추가
    plt.text(0.02, 0.90, f'RMSE: {rmse:.3f}\u00b0', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"   > Euler Comparison plot saved to: {save_path}")
    plt.show()

def plot_tracking_euler_comparison(time_data, euler_gt, euler_est, alpha_val,save_path=None):
    """
    Roll, Pitch, Yaw 각각의 추이를 Ground Truth와 비교합니다.
    (짐벌 락 구역에서의 오일러 플립 현상을 확인하기 좋은 그래프입니다.)
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f'Euler Angle Tracking Comparison (\u03b1={alpha_val})', fontsize=16)
    
    labels = ['Roll (X)', 'Pitch (Y)', 'Yaw (Z)']
    colors_gt = ['black', 'black', 'black']
    colors_est = ['red', 'green', 'blue']
    
    for i in range(3):
        # 오일러 각도(Radian)를 Degree로 변환하여 플롯
        axes[i].plot(time_data, np.degrees(euler_gt[:, i]), 
                     label='Ground Truth', color=colors_gt[i], linestyle='--', linewidth=2)
        axes[i].plot(time_data, np.degrees(euler_est[:, i]), 
                     label='Estimated', color=colors_est[i], alpha=0.7, linewidth=2)
        
        axes[i].set_ylabel(f'{labels[i]} [deg]')
        axes[i].grid(True)
        if i == 0:
            axes[i].legend(loc='upper right')
            
    axes[2].set_xlabel('Time [s]')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"   > Euler Comparison plot saved to: {save_path}")
    plt.show()

# ==========================================
# [Step 3] 3D 동적 시각화 애니메이션
# ==========================================

def animate_quaternion_tracking(time_data, gt_quats, est_quats, skip_frames=5,save_path=None):
    """
    Matplotlib을 이용하여 GT와 Estimated 쿼터니언의 3D 회전을 실시간으로 나란히 비교합니다.
    skip_frames: 렌더링 속도를 위해 건너뛸 프레임 수 (기본 5)
    """
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle('3D Quaternion Tracking Animation', fontsize=16)
    
    # 좌우로 분할된 3D 캔버스 생성
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # 렌더링할 축의 기준 벡터 (X, Y, Z)
    base_axes = np.eye(3) 
    colors = ['r', 'g', 'b'] # X=Red, Y=Green, Z=Blue
    
    # 화면 갱신용 빈 리스트
    lines_gt = [ax1.plot([0], [0], [0], color=c, linewidth=3)[0] for c in colors]
    lines_est = [ax2.plot([0], [0], [0], color=c, linewidth=3)[0] for c in colors]
    
    def setup_ax(ax, title):
        ax.set_title(title)
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # 시점 고정 (살짝 비스듬하게)
        ax.view_init(elev=30, azim=45)
        ax.set_box_aspect([1,1,1])

    setup_ax(ax1, 'Ground Truth')
    setup_ax(ax2, 'Estimated (Filter)')
    
    time_text = fig.text(0.5, 0.05, '', ha='center', fontsize=12)

    def update(frame):
        idx = frame * skip_frames
        if idx >= len(time_data):
            idx = len(time_data) - 1
            
        # 현재 프레임의 쿼터니언 가져오기 ([w, x, y, z] -> [x, y, z, w] for scipy)
        q_gt = gt_quats[idx]
        q_est = est_quats[idx]
        
        rot_gt = R_scipy.from_quat([q_gt[1], q_gt[2], q_gt[3], q_gt[0]])
        rot_est = R_scipy.from_quat([q_est[1], q_est[2], q_est[3], q_est[0]])
        
        # 기준 축(base_axes)을 쿼터니언으로 회전시킴
        rotated_gt = rot_gt.apply(base_axes)
        rotated_est = rot_est.apply(base_axes)
        
        # 선 그리기 업데이트
        for i in range(3):
            lines_gt[i].set_data([0, rotated_gt[i, 0]], [0, rotated_gt[i, 1]])
            lines_gt[i].set_3d_properties([0, rotated_gt[i, 2]])
            
            lines_est[i].set_data([0, rotated_est[i, 0]], [0, rotated_est[i, 1]])
            lines_est[i].set_3d_properties([0, rotated_est[i, 2]])
            
        time_text.set_text(f'Time: {time_data[idx]:.2f} s')
        
        # update 함수는 튜플을 반환해야 함
        return lines_gt + lines_est + [time_text]

    # 전체 프레임 수 계산
    total_frames = len(time_data) // skip_frames
    
    # 애니메이션 실행 (interval은 프레임간 지연 시간 ms)
    ani = animation.FuncAnimation(fig, update, frames=total_frames, 
                                  interval=20, blit=False, repeat=False)
    if save_path:
        print(f"   > Saving animation to GIF (this may take a moment)...")
        # GIF 저장을 위해 pillow 라이브러리가 필요합니다 (pip install pillow)
        ani.save(save_path, writer='pillow', fps=int(1/(time_data[1]-time_data[0])/skip_frames))
        print(f"   > Animation saved to: {save_path}")
    plt.show()
    return ani
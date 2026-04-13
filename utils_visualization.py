import numpy as np
import matplotlib.pyplot as plt


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
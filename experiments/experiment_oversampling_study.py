import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from simulation.sensor_simulation import (
    sim_acc_6_position_static,
    sim_acc_multi_position_static,
    sim_mag_figure8_dynamic,
    sim_mag_multi_position_static,
    sim_imu_static_pose
)
from calibration.calib_accelerometer import calibrate_acc_ellipsoid, calibrate_acc_12param
from calibration.calib_magnetometer import calibrate_mag_dynamic, calibrate_mag_static
from utils.quaternion_math import q_angle_error, accel_mag_to_quaternion

def compute_mse_norm(data, target_norm=1.0):
    norms = np.linalg.norm(data, axis=1)
    return np.mean((norms - target_norm)**2)

def run_master_oversampling_study(sample_counts, sigma_mag=0.1, sigma_acc=0.02, n_trials=20):
    results = {
        'acc_ell': [], 'acc_12p': [],
        'mag_dyn': [], 'mag_sta': [],
        'tracking_rmse': []
    }
    
    n_pos = 20
    b_true = np.array([0.5, -0.3, 0.2])
    M_true = np.array([[1.1, 0.1, 0.0], [0.1, 0.9, -0.1], [0.0, -0.1, 1.2]])
    
    # 평가용 Noiseless Test Data (공통)
    _, test_acc = sim_acc_multi_position_static(n_positions=100, n_samples_per_pos=1, sigma=0.0, M_matrix=M_true, b_vector=b_true)
    _, test_mag = sim_mag_multi_position_static(n_positions=100, n_samples_per_pos=1, sigma=0.0, M_matrix=M_true, b_vector=b_true)

    for s in sample_counts:
        t_acc_ell, t_acc_12p, t_mag_dyn, t_mag_sta, t_track = [], [], [], [], []
        
        for _ in range(n_trials):
            # Accel Train
            # 1. Ellipsoid 모델은 타원체를 피팅하기 위해 다양한 각도(20개)가 필요함 (6개 점만 주면 무한 해(Degeneracy) 발생)
            _, train_acc_ell = sim_acc_multi_position_static(n_positions=n_pos, n_samples_per_pos=s, sigma=sigma_acc, M_matrix=M_true, b_vector=b_true)
            W_ae, b_ae = calibrate_acc_ellipsoid(train_acc_ell, n_samples_per_pos=s)
            
            # 2. 12-Param 모델은 태생적으로 6-face(직교하는 6면) 정지 자세 데이터를 가정하고 만들어진 알고리즘임
            _, train_acc_12p = sim_acc_6_position_static(n_samples_per_pos=s, sigma=sigma_acc, M_matrix=M_true, b_vector=b_true)
            W_a12, b_a12 = calibrate_acc_12param(train_acc_12p, n_samples_per_pos=s)
            
            # Mag Train
            _, train_mag_dyn = sim_mag_figure8_dynamic(n_samples=1000, sigma=sigma_mag, M_matrix=M_true, b_vector=b_true)
            _, train_mag_sta = sim_mag_multi_position_static(n_positions=n_pos, n_samples_per_pos=s, sigma=sigma_mag, M_matrix=M_true, b_vector=b_true)
            W_md, b_md = calibrate_mag_dynamic(train_mag_dyn)
            W_ms, b_ms = calibrate_mag_static(train_mag_sta, n_samples_per_pos=s)
            
            # MSE 측정
            t_acc_ell.append(compute_mse_norm((W_ae @ (test_acc - b_ae).T).T, 1.0))
            t_acc_12p.append(compute_mse_norm((W_a12 @ (test_acc - b_a12).T).T, 1.0))
            t_mag_dyn.append(compute_mse_norm((W_md @ (test_mag - b_md).T).T, 1.0))
            t_mag_sta.append(compute_mse_norm((W_ms @ (test_mag - b_ms).T).T, 1.0))
            
            # Tracking 평가 (ONLY Acc + Mag, No Gyro)
            _, gt_q, _, m_acc, m_mag = sim_imu_static_pose(3.0, 0.1, sigma_acc=sigma_acc, sigma_mag=sigma_mag, 
                                                            M_acc=M_true, b_acc=b_true, M_mag=M_true, b_mag=b_true)
            cal_acc = (W_a12 @ (m_acc - b_a12).T).T
            cal_mag = (W_ms @ (m_mag - b_ms).T).T
            
            errs = [q_angle_error(gt_q[i], accel_mag_to_quaternion(cal_acc[i], cal_mag[i])) for i in range(len(cal_acc))]
            t_track.append(np.sqrt(np.mean(np.array(errs)**2)))
            
        results['acc_ell'].append(np.mean(t_acc_ell))
        results['acc_12p'].append(np.mean(t_acc_12p))
        results['mag_dyn'].append(np.mean(t_mag_dyn))
        results['mag_sta'].append(np.mean(t_mag_sta))
        results['tracking_rmse'].append(np.mean(t_track))
        print(f"S:{s:4d} | AccEll MSE:{results['acc_ell'][-1]:.6f} | Acc12p MSE:{results['acc_12p'][-1]:.6f} | MagSta MSE:{results['mag_sta'][-1]:.6f} | Track RMSE:{results['tracking_rmse'][-1]:.2f}deg")
        
    return results

def plot_master_study(sample_counts, results, filename):
    fig, axes = plt.subplots(3, 1, figsize=(10, 16))
    
    # Plot 1: Accelerometer MSE
    axes[0].loglog(sample_counts, results['acc_ell'], 'r--o', label='Ellipsoid (N_pos=20)')
    axes[0].loglog(sample_counts, results['acc_12p'], 'r-s', label='12-Param (N_pos=6)')
    axes[0].set_title('Accelerometer Calibration: Effect of Oversampling')
    axes[0].set_ylabel('MSE (Noiseless Test)')
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    # Plot 2: Magnetometer MSE
    axes[1].loglog(sample_counts, results['mag_dyn'], 'b--', label='Dynamic (Fixed 1000 total)')
    axes[1].loglog(sample_counts, results['mag_sta'], 'b-o', label='Static (N_pos=20)')
    axes[1].set_title('Magnetometer Calibration: Effect of Oversampling')
    axes[1].set_ylabel('MSE (Noiseless Test)')
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    # Plot 3: Tracking RMSE (TRIAD Only)
    axes[2].plot(sample_counts, results['tracking_rmse'], 'g-^', label='TRIAD (Acc+Mag Only)')
    axes[2].set_title('Static Tracking Stability (No Gyro)')
    axes[2].set_xlabel('Samples per Position (S)')
    axes[2].set_ylabel('Orientation RMSE [deg]')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    # 그래프 아래에 콘솔 출력 내용과 동일한 데이터 요약표 텍스트 추가
    summary_text = "[ Data Summary ]\n"
    for i, s in enumerate(sample_counts):
        summary_text += f"S:{s:<4d} | AccEll MSE:{results['acc_ell'][i]:.6f} | Acc12p MSE:{results['acc_12p'][i]:.6f} | MagSta MSE:{results['mag_sta'][i]:.6f} | Track RMSE:{results['tracking_rmse'][i]:.2f}deg\n"
    
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
    fig.text(0.5, 0.02, summary_text.strip(), ha='center', va='bottom', fontsize=10, family='monospace', bbox=props)

    plt.tight_layout(rect=[0, 0.15, 1, 1]) # 텍스트가 들어갈 하단 공간(15%) 확보
    plt.savefig(filename, dpi=300)
    plt.show()

if __name__ == "__main__":
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("=== [Master Oversampling Study: Acc/Mag/Tracking] ===")
    sample_counts = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
    results = run_master_oversampling_study(sample_counts, n_trials=15)
    
    plot_master_study(sample_counts, results, os.path.join(RESULTS_DIR, 'master_oversampling_study.png'))
    print(f"\n[Study Complete] Results saved to results/master_oversampling_study.png")

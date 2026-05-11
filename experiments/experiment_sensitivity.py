import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import simulation functions
from simulation.sensor_simulation import (
    sim_acc_6_position_static, 
    sim_acc_multi_position_static,
    sim_mag_figure8_dynamic,
    sim_gyro_static_for_bias, 
    sim_gyro_rate_table_for_M,
    sim_mag_multi_position_static
)

# Import calibration algorithms
from calibration.calib_accelerometer import calibrate_acc_ellipsoid, calibrate_acc_12param, preprocess_static_samples as preprocess_acc
from calibration.calib_magnetometer import calibrate_mag_dynamic, calibrate_mag_static, preprocess_static_samples as preprocess_mag
from calibration.calib_gyroscope import calibrate_gyroscope_full

# Import visualization tools
from utils.utils_visualization import verify_calibration_pipeline, verify_gyro_calibration_pipeline

def compute_mse_norm(calibrated_data, target_norm=1.0):
    norms = np.linalg.norm(calibrated_data, axis=1)
    return np.mean((norms - target_norm)**2)

def compute_mse_direct(calibrated_data, gt_data):
    return np.mean((calibrated_data - gt_data)**2)

def evaluate_sensors(sigma, b_vector):
    """
    Generate data and evaluate MSE for all sensors given sigma and b_vector.
    """
    results = {}
    
    # ---------------------------
    # 1. Accelerometer (MSE to norm 1.0)
    # ---------------------------
    # 공정성 확보: 6 포지션 * 50 샘플 = 300 총 샘플로 두 알고리즘 통일
    N_ACC_POS = 6
    N_ACC_SAMP = 50
    _, raw_6pos_train = sim_acc_6_position_static(n_samples_per_pos=N_ACC_SAMP, sigma=sigma, b_vector=b_vector)
    
    # Test Data: 노이즈가 포함된 상태로 평가 (실제 환경 모사)
    _, raw_acc_test = sim_acc_multi_position_static(n_positions=30, n_samples_per_pos=10, sigma=sigma, b_vector=b_vector)
    
    W_acc_ell, b_acc_ell = calibrate_acc_ellipsoid(raw_6pos_train, n_samples_per_pos=N_ACC_SAMP)
    W_acc_12p, b_acc_12p = calibrate_acc_12param(raw_6pos_train, n_samples_per_pos=N_ACC_SAMP)
    
    del_ell = (W_acc_ell @ (raw_acc_test - b_acc_ell).T).T
    del_12p = (W_acc_12p @ (raw_acc_test - b_acc_12p).T).T
    
    results['Acc_Ellipsoid'] = compute_mse_norm(del_ell, 1.0)
    results['Acc_12Param'] = compute_mse_norm(del_12p, 1.0)

    # ---------------------------
    # 2. Magnetometer Comparison (Dynamic vs Static)
    # ---------------------------
    # 정적 평균화의 장점을 극대화하기 위해 포지션 당 샘플 수를 대폭 늘림 (Oversampling 효과 확인)
    N_MAG_TOTAL_DYN = 1000
    N_MAG_POS = 20
    N_MAG_SAMP = 500
    
    # Train Data
    _, raw_mag_dyn = sim_mag_figure8_dynamic(n_samples=N_MAG_TOTAL_DYN, sigma=sigma, b_vector=b_vector)
    _, raw_mag_sta = sim_mag_multi_position_static(n_positions=N_MAG_POS, n_samples_per_pos=N_MAG_SAMP, sigma=sigma, b_vector=b_vector)
    
    # Test Data: 노이즈 포함
    _, raw_mag_test = sim_mag_multi_position_static(n_positions=30, n_samples_per_pos=10, sigma=sigma, b_vector=b_vector)

    # Dynamic Cal (Figure-8)
    W_mag_d, b_mag_d = calibrate_mag_dynamic(raw_mag_dyn)
    # Static Cal (Multi-Pose)
    W_mag_s, b_mag_s = calibrate_mag_static(raw_mag_sta, n_samples_per_pos=N_MAG_SAMP)

    res_mag_d = (W_mag_d @ (raw_mag_test - b_mag_d).T).T
    res_mag_s = (W_mag_s @ (raw_mag_test - b_mag_s).T).T

    results['Mag_Dynamic'] = compute_mse_norm(res_mag_d, 1.0)
    results['Mag_Static'] = compute_mse_norm(res_mag_s, 1.0)

    # ---------------------------
    # 3. Gyroscope (MSE to Ground Truth)
    # ---------------------------
    _, raw_gyro_static = sim_gyro_static_for_bias(n_samples=1000, sigma=sigma, b_vector=b_vector)
    gt_gyro_rate, raw_gyro_rate = sim_gyro_rate_table_for_M(rate_dps=100.0, n_samples_per_axis=100, sigma=sigma, b_vector=b_vector)
    
    sim_data_gyro = {
        'gyro_static': (None, raw_gyro_static), 
        'gyro_rate': (gt_gyro_rate, raw_gyro_rate)
    }
    W_gyro, bg = calibrate_gyroscope_full(sim_data_gyro)
    del_gyro = (W_gyro @ (raw_gyro_rate - bg).T).T
    results['Gyro_LeastSquares'] = compute_mse_direct(del_gyro, gt_gyro_rate)

    return results

def run_sigma_sweep(sigmas, n_trials=100):
    print(f"Running Sigma Sweep (Bias = 0) ... {n_trials} trials per step")
    b_zero = np.array([0.0, 0.0, 0.0])
    keys = ['Acc_Ellipsoid', 'Acc_12Param', 'Mag_Dynamic', 'Mag_Static', 'Gyro_LeastSquares']
    avg_results = {k: [] for k in keys}
    
    for i, s in enumerate(sigmas):
        print(f"  Step {i+1}/{len(sigmas)}: Sigma = {s:.3f} ...", end=" ", flush=True)
        trials = {k: [] for k in keys}
        for _ in range(n_trials):
            res = evaluate_sensors(sigma=s, b_vector=b_zero)
            for k in res: trials[k].append(res[k])
        
        for k in avg_results:
            avg_results[k].append(np.mean(trials[k]))
        print("Done.")
            
    return avg_results

def run_bias_sweep(bias_mags, n_trials=100):
    print(f"Running Bias Sweep (Sigma = 0) ... {n_trials} trials per step")
    base_bias = np.array([1.0, 1.0, 1.0])
    keys = ['Acc_Ellipsoid', 'Acc_12Param', 'Mag_Dynamic', 'Mag_Static', 'Gyro_LeastSquares']
    avg_results = {k: [] for k in keys}
    
    for i, bm in enumerate(bias_mags):
        print(f"  Step {i+1}/{len(bias_mags)}: Bias Mag = {bm:.3f} ...", end=" ", flush=True)
        b_vec = base_bias * bm
        trials = {k: [] for k in keys}
        for _ in range(n_trials):
            res = evaluate_sensors(sigma=0.0, b_vector=b_vec)
            for k in res: trials[k].append(res[k])
            
        for k in avg_results:
            avg_results[k].append(np.mean(trials[k]))
        print("Done.")
            
    return avg_results

def plot_results(x_vals, results, xlabel_base, title_prefix, n_trials, filename=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{title_prefix} vs Error (MSE) - Avg of {n_trials} trials', fontsize=16)

    # Sensor specific units
    units = {'Acc': 'g', 'Mag': 'normalized', 'Gyro': 'deg/s'}

    # 1. Accelerometer
    axes[0].plot(x_vals, results['Acc_Ellipsoid'], marker='o', label='Ellipsoid [Pos:6, Samp:50, Tot:300]')
    axes[0].plot(x_vals, results['Acc_12Param'], marker='x', label='12-Param [Pos:6, Samp:50, Tot:300]')
    axes[0].set_title('Accelerometer Calibration')
    axes[0].set_ylabel(f'MSE (${units["Acc"]}^2$)')
    axes[0].set_xlabel(f'{xlabel_base} ({units["Acc"]})')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # 2. Magnetometer
    axes[1].plot(x_vals, results['Mag_Dynamic'], marker='o', linestyle='--', color='blue', label='Dynamic(Fig-8) [Tot:1000]')
    axes[1].plot(x_vals, results['Mag_Static'], marker='s', color='green', label='Static(Multi) [Pos:20, Samp:500, Tot:10000]')
    axes[1].set_title('Magnetometer Calibration Comparison')
    axes[1].set_ylabel(f'MSE (${units["Mag"]}^2$)')
    axes[1].set_xlabel(f'{xlabel_base} ({units["Mag"]})')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # 3. Gyroscope
    axes[2].plot(x_vals, results['Gyro_LeastSquares'], marker='o', color='purple', label='Least Squares [Tot:300]')
    axes[2].set_title('Gyroscope Calibration')
    axes[2].set_ylabel(f'MSE (${units["Gyro"]}^2$)')
    axes[2].set_xlabel(f'{xlabel_base} ({units["Gyro"]})')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()

def demonstrate_3d_calibration(results_dir):
    """
    본 실험(Sweep) 전에 각 센서별 3D 구면 피팅(Sphere Fitting) 과정을 시각적으로 1회 보여줍니다.
    보정에 사용된 학습 데이터(Train Data)가 구 표면을 어떻게 덮고 있는지 확인할 수 있습니다.
    """
    print("=== [3D Visualization Demonstration] ===")
    sigma = 0.1
    b_vector = np.array([0.5, -0.3, 0.2])
    
    # 1. Accelerometer 3D (Train Data 시각화)
    print(" -> Rendering Accelerometer 3D Sphere (Train Data)...")
    _, raw_6pos = sim_acc_6_position_static(n_samples_per_pos=50, sigma=sigma, b_vector=b_vector)
    
    # 50개의 샘플을 평균낸 6개의 깨끗한 점 추출 (실제 알고리즘이 피팅하는 대상)
    raw_6pos_avg = preprocess_acc(raw_6pos, 50)
    
    W_acc_ell, b_acc_ell = calibrate_acc_ellipsoid(raw_6pos, n_samples_per_pos=50)
    W_acc_12p, b_acc_12p = calibrate_acc_12param(raw_6pos, n_samples_per_pos=50)
    
    dict_acc = {
        'Ellipsoid': (W_acc_ell @ (raw_6pos_avg - b_acc_ell).T).T,
        '12-Param': (W_acc_12p @ (raw_6pos_avg - b_acc_12p).T).T
    }
    verify_calibration_pipeline(raw_6pos_avg, dict_acc, title='Accelerometer 3D Calibration (6-Pos Averaged Data)', unit='g', 
                                filename=os.path.join(results_dir, '3d_acc_6pos_train.png'))
    
    # 2. Magnetometer 3D (Train Data 별도 시각화)
    print(" -> Rendering Magnetometer 3D Sphere (Dynamic Train Data)...")
    _, raw_mag_dyn = sim_mag_figure8_dynamic(n_samples=1000, sigma=sigma, b_vector=b_vector)
    W_mag_d, b_mag_d = calibrate_mag_dynamic(raw_mag_dyn)
    
    dict_mag_dyn = {
        'Dynamic (Fig-8)': (W_mag_d @ (raw_mag_dyn - b_mag_d).T).T
    }
    verify_calibration_pipeline(raw_mag_dyn, dict_mag_dyn, title='Magnetometer 3D Calibration (Fig-8 Train Data)', unit='Norm',
                                filename=os.path.join(results_dir, '3d_mag_fig8_train.png'))
    
    print(" -> Rendering Magnetometer 3D Sphere (Static Train Data)...")
    # 정적 평균화 장점 확인: 포지션 20개, 샘플 500개 (총 10,000개)
    _, raw_mag_sta = sim_mag_multi_position_static(n_positions=20, n_samples_per_pos=500, sigma=sigma, b_vector=b_vector)
    
    # 500개의 샘플을 평균낸 20개의 깨끗한 점 추출
    raw_mag_sta_avg = preprocess_mag(raw_mag_sta, 500)
    
    W_mag_s, b_mag_s = calibrate_mag_static(raw_mag_sta, n_samples_per_pos=500)
    
    dict_mag_sta = {
        'Static (Multi)': (W_mag_s @ (raw_mag_sta_avg - b_mag_s).T).T
    }
    verify_calibration_pipeline(raw_mag_sta_avg, dict_mag_sta, title='Magnetometer 3D Calibration (Multi-Pose Averaged Data)', unit='Norm',
                                filename=os.path.join(results_dir, '3d_mag_multipose_train.png'))
    
    # 3. Gyroscope 3D
    print(" -> Rendering Gyroscope 3D Error...")
    _, raw_gyro_static = sim_gyro_static_for_bias(n_samples=1000, sigma=sigma, b_vector=b_vector)
    gt_gyro_rate, raw_gyro_rate = sim_gyro_rate_table_for_M(rate_dps=100.0, n_samples_per_axis=100, sigma=sigma, b_vector=b_vector)
    
    sim_data_gyro = {'gyro_static': (None, raw_gyro_static), 'gyro_rate': (gt_gyro_rate, raw_gyro_rate)}
    W_gyro, bg = calibrate_gyroscope_full(sim_data_gyro)
    
    dict_gyro = {
        'Least Squares': (W_gyro @ (raw_gyro_rate - bg).T).T
    }
    verify_gyro_calibration_pipeline(gt_gyro_rate, raw_gyro_rate, dict_gyro, title='Gyroscope 3D Calibration',
                                     filename=os.path.join(results_dir, '3d_gyro_calibration.png'))
    print("=== [Demonstration Complete] ===\n")

if __name__ == "__main__":
    # Ensure results directory exists
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 0. 3D 시각화 데모 실행 (수백번의 Sweep 전 1회 시각적 확인)
    demonstrate_3d_calibration(RESULTS_DIR)
    
    # Test Parameters
    n_trials = 30 # Reduced for speed, increase for precision
    
    # 1. Sigma Sweep
    sigmas = np.linspace(0.0, 0.5, 8)
    res_sigma = run_sigma_sweep(sigmas, n_trials=n_trials)
    plot_results(sigmas, res_sigma, 'Noise Sigma', 'Noise Sigma', n_trials, 
                 filename=os.path.join(RESULTS_DIR, 'sigma_sweep_results.png'))
    
    # 2. Bias Sweep
    bias_mags = np.linspace(0.0, 5.0, 8)
    res_bias = run_bias_sweep(bias_mags, n_trials=n_trials)
    plot_results(bias_mags, res_bias, 'Bias Magnitude', 'Bias Magnitude', n_trials, 
                 filename=os.path.join(RESULTS_DIR, 'bias_sweep_results.png'))


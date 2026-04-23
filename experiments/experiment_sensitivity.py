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
    sim_gyro_rate_table_for_M
)

# Import calibration algorithms
from calibration.calib_accelerometer import calibrate_acc_ellipsoid, calibrate_acc_12param
from calibration.calib_magnetometer import calibrate_mag_ellipsoid
from calibration.calib_gyroscope import calibrate_gyroscope_full

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
    # Train
    _, raw_6pos_train = sim_acc_6_position_static(n_samples_per_pos=100, sigma=sigma, b_vector=b_vector)
    _, raw_multi_train = sim_acc_multi_position_static(n_positions=50, n_samples_per_pos=10, sigma=sigma, b_vector=b_vector)
    # Test
    _, raw_multi_test = sim_acc_multi_position_static(n_positions=30, n_samples_per_pos=10, sigma=sigma, b_vector=b_vector)
    
    # Calibrate
    W_acc_ell, b_acc_ell = calibrate_acc_ellipsoid(raw_multi_train, n_samples_per_pos=10)
    W_acc_12p, b_acc_12p = calibrate_acc_12param(raw_6pos_train, n_samples_per_pos=100)
    
    del_ell = (W_acc_ell @ (raw_multi_test - b_acc_ell).T).T
    del_12p = (W_acc_12p @ (raw_multi_test - b_acc_12p).T).T
    
    results['Acc_Ellipsoid'] = compute_mse_norm(del_ell, 1.0)
    results['Acc_12Param'] = compute_mse_norm(del_12p, 1.0)

    # ---------------------------
    # 2. Magnetometer (MSE to norm 1.0)
    # ---------------------------
    _, raw_mag = sim_mag_figure8_dynamic(n_samples=1000, sigma=sigma, b_vector=b_vector)
    W_mag, b_mag = calibrate_mag_ellipsoid(raw_mag, target_norm=1.0)
    del_mag = (W_mag @ (raw_mag - b_mag).T).T
    results['Mag_Ellipsoid'] = compute_mse_norm(del_mag, 1.0)

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
    avg_results = {k: [] for k in ['Acc_Ellipsoid', 'Acc_12Param', 'Mag_Ellipsoid', 'Gyro_LeastSquares']}
    
    for s in sigmas:
        trials = {k: [] for k in avg_results.keys()}
        for _ in range(n_trials):
            res = evaluate_sensors(sigma=s, b_vector=b_zero)
            for k in res: trials[k].append(res[k])
        
        for k in avg_results:
            avg_results[k].append(np.mean(trials[k]))
            
    return avg_results

def run_bias_sweep(bias_mags, n_trials=100):
    print(f"Running Bias Sweep (Sigma = 0) ... {n_trials} trials per step")
    base_bias = np.array([1.0, 1.0, 1.0])
    avg_results = {k: [] for k in ['Acc_Ellipsoid', 'Acc_12Param', 'Mag_Ellipsoid', 'Gyro_LeastSquares']}
    
    for bm in bias_mags:
        b_vec = base_bias * bm
        trials = {k: [] for k in avg_results.keys()}
        for _ in range(n_trials):
            res = evaluate_sensors(sigma=0.0, b_vector=b_vec)
            for k in res: trials[k].append(res[k])
            
        for k in avg_results:
            avg_results[k].append(np.mean(trials[k]))
            
    return avg_results

def plot_results(x_vals, results, xlabel_name, title_prefix, filename=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'{title_prefix} vs Error (MSE)', fontsize=16)

    # 1. Accelerometer
    axes[0].plot(x_vals, results['Acc_Ellipsoid'], marker='o', label='Ellipsoid')
    axes[0].plot(x_vals, results['Acc_12Param'], marker='x', label='12-Param')
    axes[0].set_title('Accelerometer Calibration')
    axes[0].set_ylabel('MSE (norm distance)')
    axes[0].set_xlabel(xlabel_name)
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # 2. Magnetometer
    axes[1].plot(x_vals, results['Mag_Ellipsoid'], marker='o', color='green', label='Ellipsoid')
    axes[1].set_title('Magnetometer Calibration')
    axes[1].set_ylabel('MSE (norm distance)')
    axes[1].set_xlabel(xlabel_name)
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # 3. Gyroscope
    axes[2].plot(x_vals, results['Gyro_LeastSquares'], marker='o', color='purple', label='Least Squares')
    axes[2].set_title('Gyroscope Calibration')
    axes[2].set_ylabel('MSE (Direct GT diff)')
    axes[2].set_xlabel(xlabel_name)
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)

if __name__ == "__main__":
    # Test Parameters
    n_trials = 5 # reduced for faster test execution, adjust to 100 or 1000 later.
    
    # 1. Sigma Sweep
    sigmas = np.linspace(0.0, 0.5, 10)
    res_sigma = run_sigma_sweep(sigmas, n_trials=n_trials)
    plot_results(sigmas, res_sigma, 'Noise Sigma', 'Noise Sigma', filename='sigma_sweep_results.png')
    
    # 2. Bias Sweep
    bias_mags = np.linspace(0.0, 5.0, 10)
    res_bias = run_bias_sweep(bias_mags, n_trials=n_trials)
    plot_results(bias_mags, res_bias, 'Bias Magnitude (base=[1,1,1])', 'Bias Magnitude', filename='bias_sweep_results.png')

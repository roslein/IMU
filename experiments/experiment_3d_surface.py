import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from simulation.sensor_simulation import sim_mag_multi_position_static
from calibration.calib_magnetometer import calibrate_mag_static

def compute_mse_norm(data, target_norm=1.0):
    norms = np.linalg.norm(data, axis=1)
    return np.mean((norms - target_norm)**2)

def run_3d_surface_study(sample_counts, sigma_levels, n_trials=5):
    """
    X: 샘플 수 (Oversampling)
    Y: 노이즈 크기 (Sigma)
    Z: 다중 평가 지표 (Acc 2종, Mag 2종, Tracking 4조합)
    """
    n_pos = 20
    b_true = np.array([0.5, -0.3, 0.2])
    M_true = np.array([[1.1, 0.1, 0.0], [0.1, 0.9, -0.1], [0.0, -0.1, 1.2]])
    
    from simulation.sensor_simulation import sim_acc_multi_position_static, sim_acc_6_position_static, sim_mag_multi_position_static, sim_mag_figure8_dynamic, sim_imu_static_pose
    from calibration.calib_accelerometer import calibrate_acc_ellipsoid, calibrate_acc_12param
    from calibration.calib_magnetometer import calibrate_mag_dynamic, calibrate_mag_static
    from utils.quaternion_math import q_angle_error, accel_mag_to_quaternion
    
    _, test_acc = sim_acc_multi_position_static(n_positions=100, n_samples_per_pos=1, sigma=0.0, M_matrix=M_true, b_vector=b_true)
    _, test_mag = sim_mag_multi_position_static(n_positions=100, n_samples_per_pos=1, sigma=0.0, M_matrix=M_true, b_vector=b_true)

    X, Y = np.meshgrid(sample_counts, sigma_levels)
    Z_acc_ell = np.zeros_like(X, dtype=float)
    Z_acc_12p = np.zeros_like(X, dtype=float)
    Z_mag_dyn = np.zeros_like(X, dtype=float)
    Z_mag_sta = np.zeros_like(X, dtype=float)
    Z_trk_ED = np.zeros_like(X, dtype=float)
    Z_trk_ES = np.zeros_like(X, dtype=float)
    Z_trk_12D = np.zeros_like(X, dtype=float)
    Z_trk_12S = np.zeros_like(X, dtype=float)

    total_steps = len(sample_counts) * len(sigma_levels)
    step = 0
    log_messages = []
    
    header = "=== [Ultimate 3D Surface Study: 8x Combinations] ==="
    print(header)
    log_messages.append(header)

    for i, sigma in enumerate(sigma_levels):
        for j, s in enumerate(sample_counts):
            t_ae, t_a12, t_md, t_ms = [], [], [], []
            t_ED, t_ES, t_12D, t_12S = [], [], [], []
            for _ in range(n_trials):
                # 1. Accel Calib
                _, train_acc_ell = sim_acc_multi_position_static(n_positions=n_pos, n_samples_per_pos=s, sigma=sigma, M_matrix=M_true, b_vector=b_true)
                W_ae, b_ae = calibrate_acc_ellipsoid(train_acc_ell, n_samples_per_pos=s)
                t_ae.append(compute_mse_norm((W_ae @ (test_acc - b_ae).T).T, 1.0))
                
                _, train_acc_12p = sim_acc_6_position_static(n_samples_per_pos=s, sigma=sigma, M_matrix=M_true, b_vector=b_true)
                W_a12, b_a12 = calibrate_acc_12param(train_acc_12p, n_samples_per_pos=s)
                t_a12.append(compute_mse_norm((W_a12 @ (test_acc - b_a12).T).T, 1.0))
                
                # 2. Mag Calib
                _, train_mag_dyn = sim_mag_figure8_dynamic(n_samples=1000, sigma=sigma, M_matrix=M_true, b_vector=b_true)
                W_md, b_md = calibrate_mag_dynamic(train_mag_dyn)
                t_md.append(compute_mse_norm((W_md @ (test_mag - b_md).T).T, 1.0))
                
                _, train_mag_sta = sim_mag_multi_position_static(n_positions=n_pos, n_samples_per_pos=s, sigma=sigma, M_matrix=M_true, b_vector=b_true)
                W_ms, b_ms = calibrate_mag_static(train_mag_sta, n_samples_per_pos=s)
                t_ms.append(compute_mse_norm((W_ms @ (test_mag - b_ms).T).T, 1.0))
                
                # 3. Tracking (4 Combinations)
                _, gt_q, _, m_acc, m_mag = sim_imu_static_pose(3.0, 0.1, sigma_acc=sigma, sigma_mag=sigma, M_acc=M_true, b_acc=b_true, M_mag=M_true, b_mag=b_true)
                
                cal_acc_ell = (W_ae @ (m_acc - b_ae).T).T
                cal_acc_12p = (W_a12 @ (m_acc - b_a12).T).T
                cal_mag_dyn = (W_md @ (m_mag - b_md).T).T
                cal_mag_sta = (W_ms @ (m_mag - b_ms).T).T
                
                errs_ED = [q_angle_error(gt_q[idx], accel_mag_to_quaternion(cal_acc_ell[idx], cal_mag_dyn[idx])) for idx in range(len(m_acc))]
                errs_ES = [q_angle_error(gt_q[idx], accel_mag_to_quaternion(cal_acc_ell[idx], cal_mag_sta[idx])) for idx in range(len(m_acc))]
                errs_12D = [q_angle_error(gt_q[idx], accel_mag_to_quaternion(cal_acc_12p[idx], cal_mag_dyn[idx])) for idx in range(len(m_acc))]
                errs_12S = [q_angle_error(gt_q[idx], accel_mag_to_quaternion(cal_acc_12p[idx], cal_mag_sta[idx])) for idx in range(len(m_acc))]
                
                t_ED.append(np.sqrt(np.mean(np.array(errs_ED)**2)))
                t_ES.append(np.sqrt(np.mean(np.array(errs_ES)**2)))
                t_12D.append(np.sqrt(np.mean(np.array(errs_12D)**2)))
                t_12S.append(np.sqrt(np.mean(np.array(errs_12S)**2)))
                
            Z_acc_ell[i, j] = np.mean(t_ae)
            Z_acc_12p[i, j] = np.mean(t_a12)
            Z_mag_dyn[i, j] = np.mean(t_md)
            Z_mag_sta[i, j] = np.mean(t_ms)
            Z_trk_ED[i, j] = np.mean(t_ED)
            Z_trk_ES[i, j] = np.mean(t_ES)
            Z_trk_12D[i, j] = np.mean(t_12D)
            Z_trk_12S[i, j] = np.mean(t_12S)
            step += 1
            msg = f"[{step:2d}/{total_steps}] Sig:{sigma:.2f}, S:{s:<4d} | AccMSE(Ell:{Z_acc_ell[i,j]:.5f}, 12p:{Z_acc_12p[i,j]:.5f}) | MagMSE(Dyn:{Z_mag_dyn[i,j]:.5f}, Sta:{Z_mag_sta[i,j]:.5f}) | TrkRMSE(ED:{Z_trk_ED[i,j]:.1f}, ES:{Z_trk_ES[i,j]:.1f}, 12D:{Z_trk_12D[i,j]:.1f}, 12S:{Z_trk_12S[i,j]:.1f})"
            print(msg, flush=True)
            log_messages.append(msg)

    return X, Y, Z_acc_ell, Z_acc_12p, Z_mag_dyn, Z_mag_sta, Z_trk_ED, Z_trk_ES, Z_trk_12D, Z_trk_12S, log_messages

def plot_3d_surfaces(X, Y, Z_acc_ell, Z_acc_12p, Z_mag_dyn, Z_mag_sta, Z_trk_ED, Z_trk_ES, Z_trk_12D, Z_trk_12S, sample_counts, out_dir):
    X_log = np.log10(X)
    xticks = np.log10(sample_counts)
    xlabels = [str(int(s)) for s in sample_counts]
    
    def setup_ax(ax, title, Z, cmap):
        surf = ax.plot_surface(X_log, Y, Z, cmap=cmap, edgecolor='k', alpha=0.9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Samples/Pos (log10)')
        ax.set_ylabel('Noise (Sigma)')
        ax.set_zlabel('MSE / Error')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.view_init(elev=25, azim=-45)
        return surf

    # 1. Accelerometer Figure
    fig_acc = plt.figure(figsize=(14, 6))
    ax1 = fig_acc.add_subplot(121, projection='3d')
    setup_ax(ax1, 'Accel MSE: Ellipsoid (20-Pos)', Z_acc_ell, 'Reds')
    ax2 = fig_acc.add_subplot(122, projection='3d')
    setup_ax(ax2, 'Accel MSE: 12-Param (6-Face)', Z_acc_12p, 'Reds')
    fig_acc.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.15)
    fig_acc.savefig(os.path.join(out_dir, '3d_surface_accel.png'), dpi=300)
    
    # 2. Magnetometer Figure
    fig_mag = plt.figure(figsize=(14, 6))
    ax3 = fig_mag.add_subplot(121, projection='3d')
    setup_ax(ax3, 'Mag MSE: Dynamic (Fixed 1000)', Z_mag_dyn, 'Blues')
    ax4 = fig_mag.add_subplot(122, projection='3d')
    setup_ax(ax4, 'Mag MSE: Static (20-Pos)', Z_mag_sta, 'Blues')
    fig_mag.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.15)
    fig_mag.savefig(os.path.join(out_dir, '3d_surface_magnetometer.png'), dpi=300)

    # 3. Tracking Figure
    fig_trk = plt.figure(figsize=(14, 12))
    ax5 = fig_trk.add_subplot(221, projection='3d')
    setup_ax(ax5, 'Track RMSE (Acc:Ell + Mag:Dyn)', Z_trk_ED, 'Greens')
    ax6 = fig_trk.add_subplot(222, projection='3d')
    setup_ax(ax6, 'Track RMSE (Acc:Ell + Mag:Sta)', Z_trk_ES, 'Greens')
    ax7 = fig_trk.add_subplot(223, projection='3d')
    setup_ax(ax7, 'Track RMSE (Acc:12p + Mag:Dyn)', Z_trk_12D, 'Greens')
    ax8 = fig_trk.add_subplot(224, projection='3d')
    setup_ax(ax8, 'Track RMSE (Acc:12p + Mag:Sta)', Z_trk_12S, 'Greens')
    fig_trk.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.15, hspace=0.25)
    fig_trk.savefig(os.path.join(out_dir, '3d_surface_tracking.png'), dpi=300)
    
    plt.show()

if __name__ == "__main__":
    import os
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    sample_counts = np.array([1, 5, 10, 50, 100, 500, 1000])
    sigma_levels = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    
    res = run_3d_surface_study(sample_counts, sigma_levels, n_trials=5)
    X, Y, Z_acc_ell, Z_acc_12p, Z_mag_dyn, Z_mag_sta, Z_trk_ED, Z_trk_ES, Z_trk_12D, Z_trk_12S, log_messages = res
    
    log_path = os.path.join(RESULTS_DIR, '3d_surface_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(log_messages))
    
    plot_3d_surfaces(X, Y, Z_acc_ell, Z_acc_12p, Z_mag_dyn, Z_mag_sta, Z_trk_ED, Z_trk_ES, Z_trk_12D, Z_trk_12S, sample_counts, RESULTS_DIR)
    
    print("\n[Study Complete] Saved files to results/ directory:")
    print("- 3d_surface_log.txt (Terminal values safely saved here!)")
    print("- 3d_surface_accel.png")
    print("- 3d_surface_magnetometer.png")
    print("- 3d_surface_tracking.png")

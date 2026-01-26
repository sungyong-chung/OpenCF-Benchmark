import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
import os
import time
import json

# --- CONFIGURATION ---
BASE_DIR = '/Users/sungyongchung/Desktop/OpenCF-Eval'
DATA_DIR = f'{BASE_DIR}/benchmark_data'
SUBMISSION_DIR = f'{BASE_DIR}/submissions'
PARAM_DIR = f'{BASE_DIR}/baselines/params'

os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(PARAM_DIR, exist_ok=True)

# History is provided up to 2.9s. Prediction starts at 3.0s.
OBSERVATION_WINDOW = 2.9


# --- MODEL DEFINITIONS ---
def idm_acceleration(v, delta_v, s, params):
    v0, T, a_max, b, s0, delta = params
    s_star = s0 + v * T + (v * delta_v) / (2 * np.sqrt(a_max * b))
    s = max(s, 0.1)
    raw = a_max * (1 - (v / v0) ** delta - (s_star / s) ** 2)
    return float(np.clip(raw, -10, 5))


def sidm_acceleration(v, delta_v, s, params, noise=1.0):
    v0, T, a_max, b, s0, delta, sigma = params
    det_a = idm_acceleration(v, delta_v, s, [v0, T, a_max, b, s0, delta])
    stoch_a = det_a + sigma * np.random.normal() * noise
    return float(np.clip(stoch_a, -10, 5))


def van_arem_acceleration(v, delta_v, s, a_p, params):
    ka, kv, kd, t_system, v_int, r_min, d_p, k, d = params
    delta_v_rel = -1 * delta_v
    a_ref_v = k * (v_int - v)
    r_safe = (v ** 2) / (2 * (1 / d_p - 1 / d)) if d != d_p else 0.0
    r_system = t_system * v
    r_ref = max(r_safe, r_system, r_min)
    a_ref_d = ka * a_p + kv * delta_v_rel + kd * (s - r_ref)
    a = min(a_ref_v, a_ref_d)
    return float(np.clip(a, -10, 5))


def fvdm_cth_acceleration(v, delta_v, s, params):
    delta_v_rel = -1 * delta_v
    K1n, K2n, s0n, Tn, Vmaxn = params
    Vsn = 0.0 if s <= s0n else min(Vmaxn, (s - s0n) / Tn) if s <= s0n + Tn * Vmaxn else Vmaxn
    a = K1n * (Vsn - v) + K2n * delta_v_rel
    return float(np.clip(a, -10, 5))


def fvdm_sigmoid_acceleration(v, delta_v, s, params):
    delta_v_rel = -1 * delta_v
    K1n, K2n, s0n, Tn, Vmaxn = params
    Vsn = 0.0 if s <= s0n else (Vmaxn / 2) * (
                1 - np.cos(np.pi * (s - s0n) / (Tn * Vmaxn))) if s <= s0n + Tn * Vmaxn else Vmaxn
    a = K1n * (Vsn - v) + K2n * delta_v_rel
    return float(np.clip(a, -10, 5))


def gipps_acceleration(v, v_lead, s, params):
    amaxn, bn, taun, thetan, s0n, Vmaxn, bhat = params
    A = v + 2.5 * amaxn * taun * (1 - v / Vmaxn) * np.sqrt(0.025 + v / Vmaxn)
    term = bn ** 2 * (taun / 2 + thetan) ** 2 + bn * (2 * (s - s0n) - taun * v + (v_lead ** 2) / bhat)
    if term < 0: term = 0
    B = -bn * (taun / 2 + thetan) + np.sqrt(term)
    v_next = min(A, B)
    a = (v_next - v) / taun if taun > 0 else 0.0
    return float(np.clip(a, -10, 5))


# --- SIMULATION CORE ---
def simulate_trajectory(leader_speed, leader_pos, leader_accel,
                        initial_speed, initial_pos, dt, params, model_name):
    """
    Simulates trajectory.
    Note: The input arrays (leader_*) should start at time t-1 so that
    v_sim[0] corresponds to the initial state at t-1, and v_sim[1] is the first prediction at t.
    """
    n = len(leader_speed)
    v_sim = np.zeros(n)
    x_sim = np.zeros(n)
    a_sim = np.zeros(n)

    v_sim[0] = initial_speed
    x_sim[0] = initial_pos

    for t in range(1, n):
        v = v_sim[t - 1]
        x = x_sim[t - 1]

        l_v = leader_speed[t - 1]
        l_x = leader_pos[t - 1]
        l_a = leader_accel[t - 1] if leader_accel is not None else 0.0

        s = max(l_x - x, 0.1)
        delta_v = v - l_v

        if model_name == 'IDM':
            a = idm_acceleration(v, delta_v, s, params)
        elif model_name == 'SIDM':
            a = sidm_acceleration(v, delta_v, s, params, noise=1.0)
        elif model_name == 'VAN_AREM':
            a = van_arem_acceleration(v, delta_v, s, l_a, params)
        elif model_name == 'FVDM_CTH':
            a = fvdm_cth_acceleration(v, delta_v, s, params)
        elif model_name == 'FVDM_SIGMOID':
            a = fvdm_sigmoid_acceleration(v, delta_v, s, params)
        elif model_name == 'GIPPS':
            a = gipps_acceleration(v, l_v, s, params)
        else:
            a = 0

        v_new = max(v + a * dt, 0)
        v_sim[t] = v_new
        x_sim[t] = x + (v + v_new) / 2 * dt
        a_sim[t - 1] = a

    a_sim[n - 1] = a_sim[n - 2]  # Repeat last accel
    return x_sim, v_sim, a_sim


def calibration_objective(params, df, dt, model_name, target):
    total_sq_error = 0
    total_count = 0

    for cf_id, group in df.groupby('CF_pair_id'):
        group = group.sort_values('Time')

        sim_s, sim_v, sim_a = simulate_trajectory(
            group['leader_speed'].values,
            group['leader_dist'].values,
            group['leader_acceleration'].values,
            group['follower_speed'].values[0],
            group['follower_dist'].values[0],
            dt, params, model_name
        )

        if target == 's':
            total_sq_error += np.sum((sim_s - group['follower_dist'].values) ** 2)
        elif target == 'v':
            total_sq_error += np.sum((sim_v - group['follower_speed'].values) ** 2)
        elif target == 'a':
            total_sq_error += np.sum((sim_a - group['follower_acceleration'].values) ** 2)

        total_count += len(group)

    return np.sqrt(total_sq_error / total_count)


# --- CONFIG ---
BOUNDS = {
    'IDM': [(5, 50), (0.5, 3.0), (0.1, 5.0), (0.1, 10.0), (0.5, 10.0), (1.0, 10.0)],
    'SIDM': [(5, 50), (0.5, 3.0), (0.1, 5.0), (0.1, 10.0), (0.5, 10.0), (1.0, 10.0), (0.01, 2.0)],
    'VAN_AREM': [(0.1, 5.0), (0.1, 5.0), (0.1, 5.0), (0.5, 3.0), (5, 50), (0.1, 5.0), (0.1, 10.0), (0.1, 1.0),
                 (0.1, 10.0)],
    'FVDM_CTH': [(0.1, 5.0), (0.1, 5.0), (0.1, 10.0), (0.5, 3.0), (5, 50)],
    'FVDM_SIGMOID': [(0.1, 5.0), (0.1, 5.0), (0.1, 10.0), (0.5, 3.0), (5, 50)],
    'GIPPS': [(0.5, 3.0), (1.0, 4.0), (0.1, 1.5), (0.3, 1.0), (0.1, 10.0), (5, 50), (2.0, 5.0)]
}

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Loading datasets...")
    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    test_truth_df = pd.read_csv(f"{DATA_DIR}/test_ground_truth.csv")

    # Subsample for calibration speed
    train_pair_ids = train_df['CF_pair_id'].unique()
    sampled_ids = np.random.choice(train_pair_ids, min(500, len(train_pair_ids)), replace=False)
    calib_df = train_df[train_df['CF_pair_id'].isin(sampled_ids)].copy()

    models = ['IDM', 'SIDM', 'VAN_AREM', 'FVDM_CTH', 'FVDM_SIGMOID', 'GIPPS']
    targets = ['v', 's', 'a']

    param_registry = {}

    for model_name in models:
        for target in targets:
            sub_name = f"{model_name}_RMSE_{target}"
            print(f"\nProcessing: {sub_name}")

            # 1. Calibrate
            start_t = time.time()
            result = differential_evolution(
                calibration_objective,
                bounds=BOUNDS[model_name],
                args=(calib_df, 0.1, model_name, target),
                strategy='best1bin', popsize=15, maxiter=50, tol=0.01, mutation=(0.5, 1.0), recombination=0.7,
                disp=True, seed=42
            )
            best_params = result.x
            print(f"Calibration Done ({time.time() - start_t:.1f}s). Params: {best_params}")
            param_registry[sub_name] = best_params.tolist()

            # 2. Generate Submission
            # For SIDM, we generate 6 samples. For others, just 1.
            num_samples = 6 if model_name == 'SIDM' else 1
            submission_rows = []

            for cf_id, group in test_truth_df.groupby('CF_pair_id'):
                group = group.sort_values('Time').reset_index(drop=True)

                # Identify Simulation Start (First point > 2.9s)
                mask_future = group['Time'] > OBSERVATION_WINDOW
                start_indices = group.index[mask_future].tolist()

                sim_start_idx = start_indices[0] if start_indices else len(group)

                # Get History (Ground Truth)
                history_times = group['Time'].iloc[:sim_start_idx].values
                history_x = group['follower_dist'].iloc[:sim_start_idx].values
                history_v = group['follower_speed'].iloc[:sim_start_idx].values
                history_a = group['follower_acceleration'].iloc[:sim_start_idx].values

                # If there is a future to simulate
                if sim_start_idx < len(group) and sim_start_idx > 0:
                    # Prepare Input: Include the last history point (t=2.9) to compute first future step (t=3.0)
                    input_start_idx = sim_start_idx - 1

                    leader_v_input = group['leader_speed'].values[input_start_idx:]
                    leader_x_input = group['leader_dist'].values[input_start_idx:]
                    leader_a_input = group['leader_acceleration'].values[input_start_idx:]

                    init_v = history_v[-1]
                    init_x = history_x[-1]

                    future_times = group['Time'].iloc[sim_start_idx:].values
                else:
                    # Edge case (shouldn't happen with 10s min duration): No future
                    leader_v_input = []
                    future_times = []

                # Loop for Multiple Samples
                for sample_id in range(num_samples):

                    if len(leader_v_input) > 0:
                        x_s, v_s, a_s = simulate_trajectory(
                            leader_v_input, leader_x_input, leader_a_input,
                            init_v, init_x, 0.1, best_params, model_name
                        )
                        # Slice off the first element (which is just the initial condition at t=2.9)
                        x_future = x_s[1:]
                        v_future = v_s[1:]
                        a_future = a_s[1:]
                    else:
                        x_future, v_future, a_future = [], [], []

                    # Combine History + Future
                    full_x = np.concatenate([history_x, x_future])
                    full_v = np.concatenate([history_v, v_future])
                    full_a = np.concatenate([history_a, a_future])
                    full_times = group['Time'].values


                    for t, x, v, a in zip(full_times, full_x, full_v, full_a):
                        submission_rows.append([cf_id, sample_id, t, x, v, a])

            # Convert to DataFrame
            cols = ['CF_pair_id', 'sample_id', 'Time', 'follower_dist', 'follower_speed', 'follower_acceleration']
            sub_df = pd.DataFrame(submission_rows, columns=cols)

            filename = f"{SUBMISSION_DIR}/{sub_name}.csv"
            sub_df.to_csv(filename, index=False)
            print(f"Saved: {filename} (Rows: {len(sub_df)})")

    # Save Params
    with open(f"{PARAM_DIR}/baseline_parameters.json", "w") as f:
        json.dump(param_registry, f, indent=4)
    print("\n✅ All Baselines Generated.")
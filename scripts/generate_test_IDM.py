"""
Generate a test submission using the simple (deterministic) Intelligent Driver Model (IDM).
Output: submissions/test_IDM.csv and submissions/test_IDM.json
"""
import pandas as pd
import numpy as np
import os
import json

# --- CONFIGURATION ---
DATA_DIR = "benchmark_data"
SUBMISSION_DIR = "submissions"
PARAM_DIR = "baselines/params"
OBSERVATION_WINDOW = 2.9  # Prediction starts at 3.0s
PARAM_FILE = f"{PARAM_DIR}/baseline_parameters.json"

# Simple IDM default params: v0, T, a_max, b, s0, delta (used if no baseline file)
DEFAULT_IDM_PARAMS = [30.0, 1.5, 2.0, 1.5, 2.0, 4.0]


def idm_acceleration(v, delta_v, s, params):
    v0, T, a_max, b, s0, delta = params
    s_star = s0 + v * T + (v * delta_v) / (2 * np.sqrt(a_max * b))
    s = max(s, 0.1)
    raw = a_max * (1 - (v / v0) ** delta - (s_star / s) ** 2)
    return float(np.clip(raw, -10, 5))


def simulate_idm_trajectory(leader_speed, leader_pos, leader_accel, initial_speed, initial_pos, dt, params):
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
        s = max(l_x - x, 0.1)
        delta_v = v - l_v
        a = idm_acceleration(v, delta_v, s, params)
        v_new = max(v + a * dt, 0)
        v_sim[t] = v_new
        x_sim[t] = x + (v + v_new) / 2 * dt
        a_sim[t - 1] = a
    a_sim[n - 1] = a_sim[n - 2]
    return x_sim, v_sim, a_sim


def main():
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Load params: prefer IDM_RMSE_v from baseline, else defaults
    params = DEFAULT_IDM_PARAMS
    if os.path.exists(PARAM_FILE):
        try:
            with open(PARAM_FILE, "r") as f:
                reg = json.load(f)
            if "IDM_RMSE_v" in reg:
                params = reg["IDM_RMSE_v"]
                print(f"Using IDM params from {PARAM_FILE}")
        except Exception as e:
            print(f"Using default IDM params ({e})")
    else:
        print("Using default IDM params")

    # Per README: use test_input.csv (included in repo). Fallback: test_ground_truth.csv (CI/maintainer).
    test_path = f"{DATA_DIR}/test_input.csv"
    if not os.path.exists(test_path):
        test_path = f"{DATA_DIR}/test_ground_truth.csv"
    if not os.path.exists(test_path):
        print(f"Error: neither {DATA_DIR}/test_input.csv nor test_ground_truth.csv found. Cannot generate submission.")
        return
    print(f"Using input: {test_path}")

    test_truth_df = pd.read_csv(test_path)
    submission_rows = []

    for cf_id, group in test_truth_df.groupby("CF_pair_id"):
        group = group.sort_values("Time").reset_index(drop=True)
        mask_future = group["Time"] > OBSERVATION_WINDOW
        if not mask_future.any():
            continue
        start_idx_in_group = group.index[mask_future][0]
        input_start_idx = start_idx_in_group - 1
        if input_start_idx < 0:
            continue

        leader_v_input = group["leader_speed"].values[input_start_idx:]
        leader_x_input = group["leader_dist"].values[input_start_idx:]
        leader_a_input = group["leader_acceleration"].values[input_start_idx:]
        init_v = group["follower_speed"].values[input_start_idx]
        init_x = group["follower_dist"].values[input_start_idx]
        future_times = group["Time"].values[start_idx_in_group:]

        x_s, v_s, a_s = simulate_idm_trajectory(
            leader_v_input, leader_x_input, leader_a_input,
            init_v, init_x, 0.1, params
        )
        x_future = x_s[1:]
        v_future = v_s[1:]
        a_future = a_s[1:]
        min_len = min(len(future_times), len(x_future))
        for i in range(min_len):
            submission_rows.append([
                cf_id, 0, future_times[i], x_future[i], v_future[i], a_future[i]
            ])

    cols = ["CF_pair_id", "sample_id", "Time", "follower_dist", "follower_speed", "follower_acceleration"]
    sub_df = pd.DataFrame(submission_rows, columns=cols)
    sub_df = sub_df[sub_df["Time"] > OBSERVATION_WINDOW]

    csv_path = f"{SUBMISSION_DIR}/test_IDM.csv"
    sub_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} (rows: {len(sub_df)})")

    meta = {
        "description": "Test submission using simple (deterministic) Intelligent Driver Model (IDM).",
        "calibration": "Uses baseline IDM_RMSE_v parameters if available, else default IDM parameters.",
        "assumptions": "",
        "paper_link": ""
    }
    json_path = f"{SUBMISSION_DIR}/test_IDM.json"
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"Saved: {json_path}")
    print("Done. Run: python scripts/evaluate_submission.py submissions/test_IDM.csv")


if __name__ == "__main__":
    main()

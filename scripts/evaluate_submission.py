import pandas as pd
import numpy as np
import pickle
import os
import argparse
import json
from scipy.stats import mannwhitneyu
from sklearn.metrics import mean_squared_error

# PATHS
GT_PATH = 'benchmark_data/test_ground_truth.csv'
REF_MODEL_PATH = 'benchmark_data/reference_model.pkl'


class OpenCFBenchmarkEvaluator:
    def __init__(self, gt_path=GT_PATH, ref_model_path=REF_MODEL_PATH):
        self.gt_df = pd.read_csv(gt_path)
        with open(ref_model_path, 'rb') as f:
            self.ref_model = pickle.load(f)

        self.bins = self.ref_model['bins']
        self.trans_probs = self.ref_model['trans_probs']

        self.gt_probs = self._compute_trajectory_probs(self.gt_df, is_ground_truth=True)

    def _get_bin_indices(self, rel_v, spacing, foll_v):
        idx_r = np.clip(np.digitize(rel_v, self.bins['rel_v']) - 1, 0, len(self.bins['rel_v']) - 2)
        idx_s = np.clip(np.digitize(spacing, self.bins['spacing']) - 1, 0, len(self.bins['spacing']) - 2)
        idx_f = np.clip(np.digitize(foll_v, self.bins['foll_v']) - 1, 0, len(self.bins['foll_v']) - 2)
        return idx_r, idx_s, idx_f

    def _compute_trajectory_probs(self, df, is_ground_truth=False):
        probs = []
        if is_ground_truth:
            l_spd_col, l_dst_col = 'leader_speed', 'leader_dist'
            f_spd_col, f_dst_col = 'follower_speed', 'follower_dist'
        else:
            l_spd_col, l_dst_col = 'leader_speed_gt', 'leader_dist_gt'
            f_spd_col, f_dst_col = 'follower_speed', 'follower_dist'

        groups = df.groupby(['CF_pair_id', 'sample_id']) if 'sample_id' in df.columns else df.groupby('CF_pair_id')

        for _, group in groups:
            group = group.sort_values('Time')
            l_speed = group[l_spd_col].values
            f_speed = group[f_spd_col].values
            l_dist = group[l_dst_col].values
            f_dist = group[f_dst_col].values

            rel_v = l_speed - f_speed
            spacing = l_dist - f_dist

            if len(rel_v) < 2: continue

            log_probs = []
            idx_r, idx_s, idx_f = self._get_bin_indices(rel_v, spacing, f_speed)

            for t in range(len(rel_v) - 1):
                curr = (int(idx_r[t]), int(idx_s[t]), int(idx_f[t]))
                nxt = (int(idx_r[t + 1]), int(idx_s[t + 1]), int(idx_f[t + 1]))

                if curr in self.trans_probs and nxt in self.trans_probs[curr]:
                    val = self.trans_probs[curr][nxt]
                    log_probs.append(np.log(val) if val > 0 else np.log(1e-9))
                else:
                    log_probs.append(np.log(1e-9))

            if log_probs:
                probs.append(np.exp(np.mean(log_probs)))

        return probs

    def validate_format(self, df, filename):
        """Strictly validates the submission format before evaluation."""

        # 1. Required Columns
        required_cols = {'CF_pair_id', 'Time', 'follower_dist', 'follower_speed', 'follower_acceleration'}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"❌ REJECTED {filename}: Missing columns: {missing}")

        # 2. Check for NaN values in critical columns
        if df[list(required_cols)].isna().any().any():
            raise ValueError(f"❌ REJECTED {filename}: Contains NaN (empty) values in required columns.")

        # 3. Check Time format (Must contain prediction horizon)
        # We expect at least SOME data >= 3.0s
        if not (df['Time'] >= 3.0).any():
            raise ValueError(
                f"❌ REJECTED {filename}: No prediction data found (Time >= 3.0s). Did you only submit the input history?")

        return True

    def evaluate(self, submission_path):
        filename = os.path.basename(submission_path)
        try:
            sub_df = pd.read_csv(submission_path)
        except Exception:
            raise ValueError(f"❌ REJECTED {filename}: File is not a valid CSV.")

        # --- 1. VALIDATION ---
        self.validate_format(sub_df, filename)

        # --- 2. PRE-PROCESSING ---
        if 'sample_id' not in sub_df.columns:
            sub_df['sample_id'] = 0

        # Disregard extra stochastic samples (Keep 0-5)
        sub_df = sub_df[sub_df['sample_id'].between(0, 5)]

        # Filter for Prediction Horizon
        sub_df = sub_df[sub_df['Time'] >= 3.0]

        if sub_df.empty:
            raise ValueError(f"❌ REJECTED {filename}: No valid data remains after filtering for Time >= 3.0s.")

        # --- 3. MERGE & EVAL ---
        gt_subset = self.gt_df[['CF_pair_id', 'Time', 'leader_dist', 'leader_speed', 'follower_dist', 'follower_speed',
                                'follower_acceleration']].copy()
        gt_subset.rename(columns={
            'leader_dist': 'leader_dist_gt', 'leader_speed': 'leader_speed_gt',
            'follower_dist': 'follower_dist_gt', 'follower_speed': 'follower_speed_gt',
            'follower_acceleration': 'follower_acceleration_gt'
        }, inplace=True)

        eval_df = pd.merge(sub_df, gt_subset, on=['CF_pair_id', 'Time'], how='inner')
        if eval_df.empty:
            raise ValueError(f"❌ REJECTED {filename}: IDs do not match Ground Truth. Check your 'CF_pair_id'.")

        # Metrics Calculation
        sub_probs = self._compute_trajectory_probs(eval_df, is_ground_truth=False)
        try:
            _, p_value = mannwhitneyu(self.gt_probs, sub_probs, alternative='two-sided')
        except:
            p_value = 0.0

        avg_trans_prob = np.mean(sub_probs) if sub_probs else 0.0

        # One-Step
        t3_df = eval_df[(eval_df['Time'].round(1) == 3.0) & (eval_df['sample_id'] == 0)]
        if not t3_df.empty:
            rmse_v = np.sqrt(mean_squared_error(t3_df['follower_speed_gt'], t3_df['follower_speed']))
            rmse_s = np.sqrt(mean_squared_error(t3_df['follower_dist_gt'], t3_df['follower_dist']))
            rmse_a = np.sqrt(mean_squared_error(t3_df['follower_acceleration_gt'], t3_df['follower_acceleration']))
        else:
            rmse_v, rmse_s, rmse_a = 0.0, 0.0, 0.0

        # Open-Loop
        collisions, pairs = 0, 0
        ade_list, fde_list = [], []

        for pid, pair_data in eval_df.groupby('CF_pair_id'):
            pairs += 1
            pair_collided = False
            sample_ades, sample_fdes = [], []

            for _, sample_grp in pair_data.groupby('sample_id'):
                sample_grp = sample_grp.sort_values('Time')
                if np.any(sample_grp['follower_dist'] > sample_grp['leader_dist_gt']):
                    pair_collided = True
                diff = np.abs(sample_grp['follower_dist'] - sample_grp['follower_dist_gt'])
                sample_ades.append(diff.mean())
                sample_fdes.append(diff.iloc[-1])

            if pair_collided: collisions += 1
            if sample_ades:
                ade_list.append(np.min(sample_ades))
                fde_list.append(np.min(sample_fdes))

        return {
            "Model": filename.replace('.csv', ''),
            "MW Test (p)": float(f"{p_value:.4f}"),
            "Avg Trans Prob": float(f"{avg_trans_prob:.4f}"),
            "RMSE (v)": float(f"{rmse_v:.3f}"),
            "RMSE (s)": float(f"{rmse_s:.3f}"),
            "RMSE (a)": float(f"{rmse_a:.3f}"),
            "Collision Rate (%)": float(f"{(collisions / pairs) * 100:.2f}") if pairs > 0 else 0.0,
            "minADE": float(f"{np.mean(ade_list):.3f}") if ade_list else 0.0,
            "minFDE": float(f"{np.mean(fde_list):.3f}") if fde_list else 0.0
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path")
    args = parser.parse_args()
    print(json.dumps(OpenCFBenchmarkEvaluator().evaluate(args.file_path), indent=4))
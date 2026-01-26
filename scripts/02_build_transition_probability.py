import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm

# --- CONFIG ---
BASE_DIR = '/Users/sungyongchung/Desktop/OpenCF-Eval'
INPUT_FILE = f'{BASE_DIR}/benchmark_data/train.csv'
OUTPUT_DIR = f'{BASE_DIR}/benchmark_data'
OUTPUT_FILE = f'{OUTPUT_DIR}/reference_model.pkl'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# FIXED GRID SETTINGS
BINS = {
    'rel_v': np.arange(-10, 10 + 1, 1.0),  # Range: -10 to 10 m/s
    'spacing': np.arange(0, 45 + 1, 1.0),  # Range: 0 to 45 m
    'foll_v': np.arange(0, 20 + 1, 1.0)  # Range: 0 to 20 m/s
}


def get_bin_indices(df):
    """
    Maps continuous variables to 0-based grid indices.
    """
    # Calculate Features
    rel_v = df['leader_speed'] - df['follower_speed']
    spacing = df['leader_dist'] - df['follower_dist']
    foll_v = df['follower_speed']

    # Digitize (Returns 1-based index)
    idx_r = np.digitize(rel_v, BINS['rel_v']) - 1
    idx_r = np.clip(idx_r, 0, len(BINS['rel_v']) - 2)

    idx_s = np.digitize(spacing, BINS['spacing']) - 1
    idx_s = np.clip(idx_s, 0, len(BINS['spacing']) - 2)

    idx_f = np.digitize(foll_v, BINS['foll_v']) - 1
    idx_f = np.clip(idx_f, 0, len(BINS['foll_v']) - 2)

    return idx_r, idx_s, idx_f


def build_reference_model():
    print(f"Loading training data: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    # 1. Discretize States
    print("Discretizing states...")
    df['bin_r'], df['bin_s'], df['bin_f'] = get_bin_indices(df)

    # 2. Prepare Transition Pairs
    print("Building transition matrix...")

    # Create 'Next State' columns
    df['next_bin_r'] = df.groupby('CF_pair_id')['bin_r'].shift(-1)
    df['next_bin_s'] = df.groupby('CF_pair_id')['bin_s'].shift(-1)
    df['next_bin_f'] = df.groupby('CF_pair_id')['bin_f'].shift(-1)

    # Drop rows where 'next' is NaN (end of trajectories)
    df = df.dropna(subset=['next_bin_r', 'next_bin_s', 'next_bin_f'])

    # Convert next bins to int (pandas makes them float after shift)
    df['next_bin_r'] = df['next_bin_r'].astype(int)
    df['next_bin_s'] = df['next_bin_s'].astype(int)
    df['next_bin_f'] = df['next_bin_f'].astype(int)

    # 3. Count Transitions
    # Key: (r, s, f) -> Value: { (next_r, next_s, next_f): count }
    transition_counts = {}

    # Group by [Current State, Next State] and count
    transition_groups = df.groupby(['bin_r', 'bin_s', 'bin_f', 'next_bin_r', 'next_bin_s', 'next_bin_f']).size()

    print(f"Aggregating {len(transition_groups)} unique transitions...")

    for (r, s, f, nr, ns, nf), count in transition_groups.items():
        state = (r, s, f)
        next_state = (nr, ns, nf)

        if state not in transition_counts:
            transition_counts[state] = {}

        transition_counts[state][next_state] = count

    # 4. Normalize to Probabilities
    transition_probs = {}

    for state, next_states in transition_counts.items():
        total_count = sum(next_states.values())
        transition_probs[state] = {
            ns: count / total_count
            for ns, count in next_states.items()
        }

    # 5. Save Artifacts
    model_artifacts = {
        'bins': BINS,
        'trans_probs': transition_probs,
        'description': "Fixed Grid Reference Model (-10~10, 0~45, 0~20)"
    }

    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(model_artifacts, f)

    print(f"✅ Reference Model saved to: {OUTPUT_FILE}")
    print(f"   Total Active States: {len(transition_probs)}")


if __name__ == "__main__":
    build_reference_model()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# --- CONFIG ---
BASE_DIR = '/Users/sungyongchung/Desktop/OpenCF-Eval'
INPUT_FILE = f'{BASE_DIR}/benchmark_data/train.csv'
OUTPUT_DIR = f'{BASE_DIR}/benchmark_data/plots'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# FIXED GRID SETTINGS
BINS = {
    'rel_v': np.arange(-10, 10 + 1, 1.0),  # Indices: -10 is 0, -5 is 5, 0 is 10, 5 is 15
    'spacing': np.arange(0, 45 + 1, 1.0),  # Indices: 10 is 10, 30 is 30
    'foll_v': np.arange(0, 20 + 1, 1.0)  # Indices: 5 is 5, 15 is 15
}


def get_bin_indices(df):
    """Maps continuous variables to 0-based grid indices."""
    rel_v = df['leader_speed'] - df['follower_speed']
    spacing = df['leader_dist'] - df['follower_dist']
    foll_v = df['follower_speed']

    idx_r = np.clip(np.digitize(rel_v, BINS['rel_v']) - 1, 0, len(BINS['rel_v']) - 2)
    idx_s = np.clip(np.digitize(spacing, BINS['spacing']) - 1, 0, len(BINS['spacing']) - 2)
    idx_f = np.clip(np.digitize(foll_v, BINS['foll_v']) - 1, 0, len(BINS['foll_v']) - 2)

    return idx_r, idx_s, idx_f


def get_bin_label(r_idx, s_idx, f_idx):
    """Returns a readable string for the bin ranges."""
    r_val = f"{BINS['rel_v'][r_idx]:.0f} to {BINS['rel_v'][r_idx + 1]:.0f}"
    s_val = f"{BINS['spacing'][s_idx]:.0f} to {BINS['spacing'][s_idx + 1]:.0f}"
    f_val = f"{BINS['foll_v'][f_idx]:.0f} to {BINS['foll_v'][f_idx + 1]:.0f}"
    return f"RelV: {r_val} m/s\nSpacing: {s_val} m\nSpeed: {f_val} m/s"


def plot_distributions():
    print(f"Loading data: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    print("Binning data...")
    df['bin_r'], df['bin_s'], df['bin_f'] = get_bin_indices(df)

    # Group accelerations by State
    state_groups = df.groupby(['bin_r', 'bin_s', 'bin_f'])['follower_acceleration']

    # --- FILTER CANDIDATES ---
    # high-density ranges:
    # RelV: -5 to 5  => Indices 5 to 15
    # Spacing: 10 to 30 => Indices 10 to 30
    # Speed: 5 to 15 => Indices 5 to 15

    candidates = []

    for state_key, group in state_groups:
        r_idx, s_idx, f_idx = state_key

        # Check range constraints
        if not (5 <= r_idx <= 15): continue
        if not (10 <= s_idx <= 30): continue
        if not (5 <= f_idx <= 15): continue

        if len(group) > 50:
            candidates.append(state_key)

    print(f"Found {len(candidates)} candidate states in the high-density region.")

    # --- RANDOM SAMPLING ---
    if len(candidates) < 6:
        print("Warning: Not enough states meet the criteria. Plotting all found.")
        selected_states = candidates
    else:
        selected_states = random.sample(candidates, 6)

    # --- PLOT ---
    print(f"Plotting {len(selected_states)} distributions...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, state_key in enumerate(selected_states):
        ax = axes[i]
        accels = state_groups.get_group(state_key)

        # Plot Histogram
        ax.hist(accels, bins=30, color='#4C72B0', edgecolor='white', alpha=0.8, density=True)

        # Plot Mean/Median
        mean_val = accels.mean()
        median_val = accels.median()
        ax.axvline(mean_val, color='#C44E52', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='#55A868', linestyle='-', linewidth=2, label=f'Median: {median_val:.2f}')

        # Formatting
        label_text = get_bin_label(*state_key)
        ax.set_title(label_text, fontsize=14)
        if i >= 3: ax.set_xlabel("Acceleration ($m/s^2$)", fontsize=12)
        if i % 3 == 0: ax.set_ylabel("Density", fontsize=12)
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(-4, 4)  # Fix x-axis to keep plots comparable

    plt.tight_layout()
    save_path = f"{OUTPUT_DIR}/acceleration_distributions.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Plots saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    plot_distributions()
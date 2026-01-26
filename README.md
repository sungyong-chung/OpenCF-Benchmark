# OpenCF Benchmark: Data-Driven Car-Following Evaluation

[**Sungyong Chung**](https://scholar.google.com/citations?user=7jX4Aw8AAAAJ&hl=en&oi=ao) &nbsp;•&nbsp; [**Alireza Talebpour**](https://scholar.google.com/citations?user=lW4PAysAAAAJ&hl=en) &nbsp;•&nbsp; [**Yanlin Zhang**](https://scholar.google.com/citations?user=qhO_nfcAAAAJ&hl=en)

[![Leaderboard](https://img.shields.io/badge/Leaderboard-Live-brightgreen)](https://sungyong-chung.github.io/OpenCF-Benchmark/)

**OpenCF-Benchmark** is an open-source evaluation framework for car-following models.

It provides a standardized testbed to benchmark car-following models against real-world data derived from the **Waymo Open Motion Dataset (WOMD)**.

🔗 **[View the Live Leaderboard](https://sungyong-chung.github.io/OpenCF-Benchmark/)**

---

## 📂 Dataset Access

This benchmark utilizes data derived from the **Waymo Open Motion Dataset (WOMD)**.

* **Training Data (`train.csv`):** [**DOWNLOAD HERE (Google Drive)**](https://drive.google.com/file/d/1JCrPkS9FUDC4OihimvV7UpYNfeVIBx4E/view?usp=share_link)
    * *Contains processed leader-follower pairs for model calibration/training.*
    * *You may use this data to train your model, but it is not required for evaluation.*
* **Test Input (`benchmark_data/test_input.csv`):** Included in this repo.
    * *Contains the first 2.9s of history for each test pair. Your model must predict the future from t=3.0s onwards.*

### ⚠️ A Note on Data Processing
To maintain the integrity of the benchmark and prevent overfitting to specific filtering criteria, **we do not disclose the exact logic used to extract car-following pairs from WOMD.**

However, please note:
1.  **Identical Processing:** The Training Set and Test Set were generated using the **exact same** filtering, extraction, and smoothing pipeline.
2.  **Consistency:** Models trained on the provided `train.csv` will encounter a statistically similar distribution in the test set.

---

## 🏆 How to Participate

We use an automated **"Evaluation-as-a-Service"** workflow. You do not need to run the evaluation scripts yourself; GitHub Actions will do it for you.

### 1. Prepare Your Model
1.  Download `benchmark_data/test_input.csv`.
2.  Run your model to generate trajectories for all pairs.
3.  **Format:** Your output CSV must strictly follow this format:

| Column | Description |
| :--- | :--- |
| `CF_pair_id` | ID matching the test set |
| `sample_id` | `0` for deterministic, `0-N` for stochastic samples |
| `Time` | Time in seconds (must align with ground truth, e.g., 0.1s steps) |
| `follower_dist` | Longitudinal position (meters) |
| `follower_speed` | Speed (m/s) |
| `follower_acceleration`| Acceleration (m/s²) |

*See `benchmark_data/submission_template.csv` for an example.*

### 2. Submit Your Results
1.  **Fork** this repository.
2.  Place your results CSV file into the `submissions/` folder.
    * *Naming convention: `ModelName_Variant.csv` (e.g., `MyModel_LSTM.csv`).*
3.  Open a **Pull Request (PR)** to the `main` branch of this repository.

### 3. Automated Evaluation
* Once your PR is opened, our **Automated Judge** (GitHub Actions) will run immediately.
* It will evaluate your submission against the hidden Ground Truth.
* **Check the PR comments:** The bot will post your scores (RMSE, Collision Rate, etc.) and Pass/Fail status.
* If your submission is valid, we will merge it, and you will appear on the **Leaderboard Website**.

---

## 📊 Evaluation Metrics

We evaluate models on three dimensions:

### 1. Transition Dynamics (Turing Test)
We compare the **Geometric Mean Probability (GMP)** of your generated trajectories against the Ground Truth using a **Mann-Whitney U Test**.
* **PASS:** p-value > 0.05 (Your model's behavior is statistically indistinguishable from real human driving).
* **FAIL:** p-value < 0.05.

### 2. One-Step Prediction (Short-term)
Measures accuracy at exactly `t = 3.0s` (the first predicted step).
* **RMSE (v):** Root Mean Square Error of Speed.
* **RMSE (s):** Root Mean Square Error of Spacing.
* **RMSE (a):** Root Mean Square Error of Acceleration.

### 3. Open-Loop Prediction (Long-term)
Measures consistency over the full trajectory horizon.
* **minADE:** Minimum Average Displacement Error (over K samples).
* **minFDE:** Minimum Final Displacement Error (at the last timestep).
* **Collision Rate:** Percentage of test pairs where the follower collides with the leader (`follower_dist > leader_dist`).

---

## 📂 Repository Structure

```text
OpenCF-Benchmark/
├── benchmark_data/           # Test inputs and Reference Models
│   ├── reference_model.pkl   # Transition Matrix for the "Judge"
│   ├── test_input.csv        # Input data for your model
│   └── submission_template.csv
├── submissions/              # Community Submissions (CSVs go here)
├── scripts/                  # Evaluation Logic (Python)
├── docs/                     # Leaderboard Website Source
└── README.md                 # This file
```

## Citation

If you use this benchmark or dataset please cite the following work.

```bibtex
@article{chung2025characterizing,
  title={Characterizing Lane Changing Behavior in Mixed Traffic},
  author={Chung, Sungyong and Talebpour, Alireza and Hamdar, Samer H},
  journal={arXiv preprint arXiv:2512.07219},
  year={2025}
}
```
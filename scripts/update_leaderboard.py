import os
import json
import glob
import zipfile
import getpass
import numpy as np
from datetime import datetime
from evaluate_submission import OpenCFBenchmarkEvaluator

SUBMISSIONS_DIR = 'submissions'
OUTPUT_FILE = 'docs/leaderboard_data.json'
META_FILE = 'docs/metadata.json'
BENCHMARK_DIR = 'benchmark_data'
GT_FILENAME = 'test_ground_truth.csv'
ZIP_FILENAME = 'test_ground_truth.zip'

GT_PATH = os.path.join(BENCHMARK_DIR, GT_FILENAME)
ZIP_PATH = os.path.join(BENCHMARK_DIR, ZIP_FILENAME)


def ensure_ground_truth_exists():
    if os.path.exists(GT_PATH):
        return True, False

    if not os.path.exists(ZIP_PATH):
        print(f"❌ Error: Could not find {GT_FILENAME} or {ZIP_FILENAME}.")
        return False, False

    print(f"🔒 Ground Truth is locked. Attempting to extract from {ZIP_FILENAME}...")
    password = os.environ.get('GT_PASSWORD')
    if not password:
        try:
            password = getpass.getpass(prompt='Enter GT_PASSWORD: ')
        except:
            return False, False

    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
            zf.extractall(path=BENCHMARK_DIR, pwd=bytes(password, 'utf-8'))
        return True, True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False, False


def cleanup_ground_truth():
    if os.path.exists(GT_PATH):
        os.remove(GT_PATH)


def main():
    print("--- UPDATING LEADERBOARD ---")
    ready, was_extracted = ensure_ground_truth_exists()
    if not ready:
        exit(1)

    try:
        evaluator = OpenCFBenchmarkEvaluator()

        # 1. Calculate & Save Ground Truth Stats (Metadata)
        gt_gmp = np.mean(evaluator.gt_probs) if evaluator.gt_probs else 0.0
        with open(META_FILE, 'w') as f:
            json.dump({"gt_gmp": gt_gmp}, f)
        print(f"📊 Ground Truth GMP calculated: {gt_gmp:.4f}")

        # 2. Load Existing Dates (to persist them)
        model_dates = {}
        if os.path.exists(OUTPUT_FILE):
            try:
                with open(OUTPUT_FILE, 'r') as f:
                    old_data = json.load(f)
                    for row in old_data:
                        if 'Model' in row and 'Date' in row:
                            model_dates[row['Model']] = row['Date']
            except:
                print("⚠️ Could not read old leaderboard data. Starting fresh.")

        results = []
        csv_files = glob.glob(os.path.join(SUBMISSIONS_DIR, '*.csv'))

        for file_path in csv_files:
            print(f"Processing {file_path}...")
            try:
                res = evaluator.evaluate(file_path)

                # 3. Handle Date Assignment
                model_name = res['Model']
                if model_name in model_dates:
                    res['Date'] = model_dates[model_name]  # Keep original date
                else:
                    res['Date'] = datetime.now().strftime('%Y-%m-%d')  # New submission

                results.append(res)
            except Exception as e:
                print(f"⚠️ FAILED {file_path}: {e}")

        results.sort(key=lambda x: x['RMSE (v)'])

        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"✅ Leaderboard updated with {len(results)} entries.")

    finally:
        if was_extracted:
            cleanup_ground_truth()


if __name__ == "__main__":
    main()
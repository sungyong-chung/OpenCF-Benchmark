import os
import json
import glob
import zipfile
import getpass
from datetime import datetime
from evaluate_submission import OpenCFBenchmarkEvaluator

SUBMISSIONS_DIR = 'submissions'
OUTPUT_FILE = 'docs/leaderboard_data.json'
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

        # Load Existing Dates
        model_dates = {}
        if os.path.exists(OUTPUT_FILE):
            try:
                with open(OUTPUT_FILE, 'r') as f:
                    old_data = json.load(f)
                    for row in old_data:
                        if 'Model' in row and 'Date' in row:
                            model_dates[row['Model']] = row['Date']
            except:
                print("⚠️ Could not read old data.")

        results = []
        csv_files = glob.glob(os.path.join(SUBMISSIONS_DIR, '*.csv'))

        for file_path in csv_files:
            print(f"Processing {file_path}...")
            try:
                res = evaluator.evaluate(file_path)

                # 1. Assign Date
                model_name = res['Model']
                if model_name in model_dates:
                    res['Date'] = model_dates[model_name]
                else:
                    res['Date'] = datetime.now().strftime('%Y-%m-%d')

                # 2. Load Metadata (JSON) if it exists
                json_path = file_path.replace('.csv', '.json')
                meta = {
                    "description": "No description provided.",
                    "calibration": "No calibration details provided.",
                    "assumptions": "No assumptions provided.",
                    "paper_link": ""
                }

                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as jf:
                            user_meta = json.load(jf)
                            # Merge safely
                            meta.update({k: v for k, v in user_meta.items() if v})
                    except Exception as e:
                        print(f"⚠️ Error reading JSON for {model_name}: {e}")

                res['meta'] = meta
                results.append(res)
            except Exception as e:
                print(f"⚠️ FAILED {file_path}: {e}")

        # Sort by proximity to Ground Truth (0.2572). Smallest difference is best.
        GT_PROB = 0.2572
        results.sort(key=lambda x: abs(x.get('Avg Trans Prob', 0) - GT_PROB))

        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"✅ Leaderboard updated with {len(results)} entries.")

    finally:
        if was_extracted:
            cleanup_ground_truth()


if __name__ == "__main__":
    main()
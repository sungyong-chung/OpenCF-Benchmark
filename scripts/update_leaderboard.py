import os
import json
import glob
import zipfile
import getpass
from evaluate_submission import OpenCFBenchmarkEvaluator

SUBMISSIONS_DIR = 'submissions'
OUTPUT_FILE = 'docs/leaderboard_data.json'
BENCHMARK_DIR = 'benchmark_data'
GT_FILENAME = 'test_ground_truth.csv'
ZIP_FILENAME = 'ground_truth.zip'

GT_PATH = os.path.join(BENCHMARK_DIR, GT_FILENAME)
ZIP_PATH = os.path.join(BENCHMARK_DIR, ZIP_FILENAME)


def ensure_ground_truth_exists():
    """
    Ensures the Ground Truth CSV is available.
    If missing, it attempts to unzip 'ground_truth.zip'.
    Returns: Boolean (True if file is ready, False if failed)
    """
    if os.path.exists(GT_PATH):
        return True, False

    if not os.path.exists(ZIP_PATH):
        print(f"❌ Error: Could not find {GT_FILENAME} or {ZIP_FILENAME}.")
        return False, False

    print(f"🔒 Ground Truth is locked. Attempting to extract from {ZIP_FILENAME}...")

    # Try to get password from Environment (GitHub Action) or Prompt (Local)
    password = os.environ.get('GT_PASSWORD')
    if not password:
        print("🔑 Password required to unlock Ground Truth.")
        try:
            password = getpass.getpass(prompt='Enter GT_PASSWORD: ')
        except Exception as e:
            print(f"Error reading password: {e}")
            return False, False

    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
            zf.extractall(path=BENCHMARK_DIR, pwd=bytes(password, 'utf-8'))

        if os.path.exists(GT_PATH):
            print("✅ Successfully unlocked Ground Truth.")
            return True, True
        else:
            print("❌ Extraction finished but CSV not found. Check zip content.")
            return False, False
    except RuntimeError as e:
        print(f"❌ Wrong password or unzip error: {e}")
        return False, False
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        return False, False


def cleanup_ground_truth():
    """Deletes the extracted CSV to maintain security."""
    if os.path.exists(GT_PATH):
        os.remove(GT_PATH)
        print("🔒 Re-locked Ground Truth (deleted temporary CSV).")


def main():
    print("--- UPDATING LEADERBOARD ---")

    # 1. Ensure Data is Ready
    ready, was_extracted = ensure_ground_truth_exists()
    if not ready:
        print("❌ Aborting: Cannot evaluate without Ground Truth.")
        exit(1)

    try:
        # 2. Run Evaluation
        # Note: OpenCFBenchmarkEvaluator defaults to looking for BENCHMARK_DIR/test_ground_truth.csv
        evaluator = OpenCFBenchmarkEvaluator()
        results = []

        csv_files = glob.glob(os.path.join(SUBMISSIONS_DIR, '*.csv'))

        if not csv_files:
            print("⚠️ No submission files found.")

        for file_path in csv_files:
            print(f"Processing {file_path}...")
            try:
                res = evaluator.evaluate(file_path)
                results.append(res)
            except Exception as e:
                print(f"⚠️ FAILED {file_path}: {e}")

        # 3. Sort Results (Lowest RMSE wins)
        results.sort(key=lambda x: x['RMSE (v)'])

        # 4. Save to JSON
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"✅ Leaderboard updated with {len(results)} entries.")

    finally:
        if was_extracted:
            cleanup_ground_truth()


if __name__ == "__main__":
    main()
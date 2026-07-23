"""
download_data.py
----------------
Helper script to download the IBM Telco Customer Churn dataset.

Method 1 (automatic): Uses the Kaggle API if credentials are configured.
Method 2 (manual):    Prints instructions for manual download.

Usage
-----
    python download_data.py
"""

import os
import shutil
import sys

DATA_DIR      = os.path.join(os.path.dirname(__file__), "data")
TARGET_PATH   = os.path.join(DATA_DIR, "customer_churn.csv")
KAGGLE_SLUG   = "blastchar/telco-customer-churn"
KAGGLE_FILE   = "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def try_kaggle_api() -> bool:
    """
    Attempt to download using the kaggle Python package.

    Returns True if successful, False otherwise.
    """
    try:
        import kaggle  # noqa: F401
        print("Kaggle API found. Downloading dataset...")
        os.makedirs(DATA_DIR, exist_ok=True)

        import subprocess
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_SLUG,
             "--unzip", "-p", DATA_DIR],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Kaggle CLI error: {result.stderr}")
            return False

        # Rename to the expected filename if necessary
        downloaded = os.path.join(DATA_DIR, KAGGLE_FILE)
        if os.path.exists(downloaded) and downloaded != TARGET_PATH:
            shutil.move(downloaded, TARGET_PATH)
            print(f"Renamed to: {TARGET_PATH}")

        if os.path.exists(TARGET_PATH):
            size_kb = os.path.getsize(TARGET_PATH) / 1024
            print(f"✅  Dataset ready: {TARGET_PATH} ({size_kb:.1f} KB)")
            return True
        else:
            return False

    except ImportError:
        return False
    except Exception as exc:
        print(f"Kaggle API download failed: {exc}")
        return False


def print_manual_instructions() -> None:
    """Print clear manual download instructions."""
    print("\n" + "=" * 60)
    print("  MANUAL DATASET DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("""
  1. Go to:
     https://www.kaggle.com/datasets/blastchar/telco-customer-churn

  2. Click 'Download' (you need a free Kaggle account).

  3. Extract the ZIP file.

  4. Rename the CSV file to:
       customer_churn.csv

  5. Place it in the data/ folder:
       Customer-Churn-Prediction/data/customer_churn.csv

  Alternatively, set up the Kaggle API:
  - Install: pip install kaggle
  - Place kaggle.json in ~/.kaggle/
  - Then re-run: python download_data.py
""")
    print("=" * 60)


def main():
    if os.path.exists(TARGET_PATH):
        size_kb = os.path.getsize(TARGET_PATH) / 1024
        print(f"✅  Dataset already present: {TARGET_PATH} ({size_kb:.1f} KB)")
        return

    print("Dataset not found. Attempting automatic download via Kaggle API...")

    if not try_kaggle_api():
        print_manual_instructions()
        sys.exit(1)


if __name__ == "__main__":
    main()

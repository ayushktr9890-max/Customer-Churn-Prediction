"""
download_data.py
----------------
Check that the local IBM Telco Customer Churn dataset is available.

Usage
-----
    python download_data.py
"""

import os
import sys

DATA_PATH = os.path.join(os.path.dirname(__file__), "customer_churn.csv")


def main() -> None:
    if os.path.exists(DATA_PATH):
        size_kb = os.path.getsize(DATA_PATH) / 1024
        print(f"✅  Local dataset ready: {DATA_PATH} ({size_kb:.1f} KB)")
        return

    print(f"Dataset not found: {DATA_PATH}")
    print("Add your CSV file to the project root and name it customer_churn.csv.")
    sys.exit(1)


if __name__ == "__main__":
    main()

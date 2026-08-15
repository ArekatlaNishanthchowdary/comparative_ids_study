"""
Prep script for CIC-IDS2017 dataset.
Reads all 8 per-day raw CSV files from data/cicids2017/raw/*.csv,
strips column name whitespace, concatenates them, and saves to data/cicids2017/all_days.csv.
"""

from pathlib import Path
import glob
import pandas as pd

RAW_DIR = Path("data/cicids2017/raw")
OUTPUT_CSV = Path("data/cicids2017/all_days.csv")


def main():
    raw_files = sorted(glob.glob(str(RAW_DIR / "*.csv")))
    if not raw_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

    print(f"Found {len(raw_files)} per-day CSV files in {RAW_DIR}:")
    dfs = []
    for f in raw_files:
        print(f"  Reading {f}...")
        df = pd.read_csv(f, encoding="utf-8", encoding_errors="replace")
        # Strip leading/trailing whitespace from column names
        df.columns = df.columns.str.strip()
        # Clean non-ASCII chars in string columns if any
        if "Label" in df.columns:
            df["Label"] = df["Label"].astype(str)

        dfs.append(df)

    print("Concatenating per-day files...")
    all_df = pd.concat(dfs, ignore_index=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing concatenated dataset to {OUTPUT_CSV} (shape: {all_df.shape})...")
    all_df.to_csv(OUTPUT_CSV, index=False)
    print("Done preparing CIC-IDS2017.")


if __name__ == "__main__":
    main()

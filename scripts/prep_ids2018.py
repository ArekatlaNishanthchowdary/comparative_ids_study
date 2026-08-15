"""
scripts/prep_ids2018.py
=======================
Concatenates all CSE-CIC-IDS2018 per-day CICFlowMeter CSVs into one clean
file at data/ids2018/all_days.csv.

Capping strategy (NIDS-realistic):
  - Keep ALL Benign rows encountered (up to MAX_BENIGN cap, default 500k)
  - Cap total Attack rows at MAX_ATTACK (default 200k), sampled proportionally
    across attack types so no single family dominates.
  -> Resulting imbalance: Benign >> Attack (attack is the minority)

What it does:
  - Strips leading/trailing whitespace from ALL column names.
  - Drops mid-file header sentinel rows (where Label == "Label").
  - Replaces inf / -inf with NaN.
  - Writes shuffled result to data/ids2018/all_days.csv

Usage:
    python scripts/prep_ids2018.py
"""

import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import pandas as pd
import numpy as np

from config import DATASETS, RANDOM_STATE, MAX_BENIGN_ROWS, MAX_ATTACK_ROWS

cfg        = DATASETS["ids2018"]
RAW_DIR    = cfg["raw_dir"]
OUT_PATH   = cfg["train_path"]
LABEL_COL  = cfg["label_col"]
NORMAL_LBL = cfg["normal_label"]

# ── Capping: keep Benign as the majority class ─────────────────────────────
MAX_BENIGN = MAX_BENIGN_ROWS   # max benign rows  (~majority class)
MAX_ATTACK = MAX_ATTACK_ROWS   # max attack rows  (~minority class)
# Resulting ratio: up to ~1:2.5  (attack fraction ≈ 29%)
# ──────────────────────────────────────────────────────────────────────────

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

csv_files = sorted(RAW_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSVs found in {RAW_DIR}")

print(f"Found {len(csv_files)} CSV files in {RAW_DIR}")
print(f"Cap: <={MAX_BENIGN:,} Benign rows, <={MAX_ATTACK:,} Attack rows\n")

benign_chunks = []
attack_chunks = []
benign_count  = 0

for fp in csv_files:
    print(f"  Reading {fp.name}  ({fp.stat().st_size / 1e6:.0f} MB) ...")
    for chunk in pd.read_csv(fp, chunksize=100_000, low_memory=False,
                              on_bad_lines="skip"):
        chunk.columns = chunk.columns.str.strip()
        if LABEL_COL not in chunk.columns:
            continue
        # Drop repeated-header rows
        chunk = chunk[chunk[LABEL_COL].astype(str).str.strip() != LABEL_COL]
        chunk[LABEL_COL] = chunk[LABEL_COL].astype(str).str.strip()

        # Replace inf
        num_cols = chunk.select_dtypes(include=[np.number]).columns
        chunk[num_cols] = chunk[num_cols].replace([np.inf, -np.inf], np.nan)

        is_benign = chunk[LABEL_COL] == NORMAL_LBL
        atk = chunk[~is_benign]
        ben = chunk[is_benign]

        # Collect benign up to cap
        remaining_ben = MAX_BENIGN - benign_count
        if remaining_ben > 0:
            take = ben.iloc[:remaining_ben]
            benign_chunks.append(take)
            benign_count += len(take)

        # Always collect attacks (will subsample later)
        if len(atk):
            attack_chunks.append(atk)

# Subsample attacks proportionally across all attack families
print(f"\nTotal Benign collected: {benign_count:,}")
attack_df  = pd.concat(attack_chunks, ignore_index=True)
print(f"Total Attack collected: {len(attack_df):,}")
print(f"Attack distribution (before cap):\n{attack_df[LABEL_COL].value_counts().to_string()}\n")

if len(attack_df) > MAX_ATTACK:
    # Stratified subsample to preserve family proportions
    frac = MAX_ATTACK / len(attack_df)
    sampled = []
    for lbl, grp in attack_df.groupby(LABEL_COL):
        n = max(1, int(round(len(grp) * frac)))
        sampled.append(grp.sample(n=min(n, len(grp)), random_state=RANDOM_STATE))
    attack_df = pd.concat(sampled, ignore_index=True)
    print(f"Attack distribution (after cap to ~{MAX_ATTACK:,}):")
    print(attack_df[LABEL_COL].value_counts().to_string())
    print()

benign_df = pd.concat(benign_chunks, ignore_index=True)
full_df   = pd.concat([benign_df, attack_df], ignore_index=True).sample(
                frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

attack_frac = (full_df[LABEL_COL] != NORMAL_LBL).mean()
benign_frac = 1 - attack_frac
print(f"Final dataset: {len(full_df):,} rows")
print(f"  Benign : {int(benign_frac * len(full_df)):,}  ({benign_frac:.1%})")
print(f"  Attack : {int(attack_frac * len(full_df)):,}  ({attack_frac:.1%})")
print(f"  Imbalance ratio ~ 1:{benign_frac/attack_frac:.1f}  (attack is minority)\n")

full_df.to_csv(OUT_PATH, index=False)
print(f"Saved -> {OUT_PATH}")

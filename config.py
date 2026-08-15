"""
Central configuration. Edit DATASET_PATHS to point at your local CSV files.
Nothing here is dataset-agnostic magic -- you must tell it where the raw
files are and what the label column is called for each dataset.
"""

from pathlib import Path

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Datasets
# ---------------------------------------------------------------------------
# label_col:      name of the ground-truth column in the raw CSV
# normal_label:   the label value that means "benign / normal" (maps to 0)
#                 everything else maps to 1 (attack)
# drop_cols:      columns to drop before modeling
# raw_dir:        (CIC datasets) folder containing per-day raw CSVs
# train_path /
# test_path:      (NSL-KDD) pre-split files; test_path=None → split from train
DATASETS = {
    "nslkdd": {
        "train_path":    Path("data/nslkdd/KDDTrain+.csv"),
        "test_path":     Path("data/nslkdd/KDDTest+.csv"),
        "label_col":     "label",
        "normal_label":  "normal",   # anything != "normal" → attack
        "drop_cols":     ["difficulty"],
        "raw_dir":       None,
    },
    "ids2018": {
        # CSE-CIC-IDS2018 — 10 per-day CICFlowMeter CSVs
        # Label col is "Label"; benign rows are labelled "Benign"
        "raw_dir":       Path("CSE-CIC-IDS2018"),
        "train_path":    Path("data/ids2018/all_days.csv"),
        "test_path":     None,       # will be split 80/20 from train_path
        "label_col":     "Label",
        "normal_label":  "Benign",
        # 02-20-2018.csv (only) ships 4 extra ID columns the other 9 days
        # don't have (Flow ID/Src IP/Src Port/Dst IP); left in place they'd
        # be NaN for every other day and get mean/mode-imputed from Feb-20's
        # values alone, leaking a fabricated IP/port into 9/10 of the rows.
        "drop_cols":     ["Timestamp", "Flow ID", "Src IP", "Src Port", "Dst IP"],
    },
    "ddos2019": {
        # CIC-DDoS2019 — CSVs split across two day-folders (01-12 and 03-11)
        # Label col is " Label" (leading space — stripped during prep)
        # Benign rows are labelled "BENIGN"
        "raw_dir":       Path("CIC-DDoS2019"),
        "train_path":    Path("data/ddos2019/all_days.csv"),
        "test_path":     None,
        "label_col":     "Label",
        "normal_label":  "BENIGN",
        "drop_cols":     ["Unnamed: 0", "Flow ID", " Source IP",
                          " Destination IP", " Timestamp",
                          "SimillarHTTP", " Inbound"],
    },
}

TEST_SIZE = 0.2   # fraction held out when no pre-split test file exists

# Row caps applied by scripts/prep_ids2018.py and scripts/prep_ddos2019.py
# while concatenating the raw per-day CIC CSVs (keeps memory manageable and
# keeps benign as the majority class before the final imbalance is fixed).
MAX_BENIGN_ROWS = 500_000
MAX_ATTACK_ROWS = 200_000

# ---------------------------------------------------------------------------
# 2. Balancing techniques (the independent variable)
# ---------------------------------------------------------------------------
BALANCING_MATRIX = {
    "baseline":  ["raw", "ros"],
    "ml_based":  ["smote", "adasyn"],
    "ml_hybrid": ["smote_tomek", "smote_enn"],
    "dl_based":  ["vae_oversample", "gan_oversample"],
    "dl_hybrid": ["gan_tomek", "vae_smote"],
}

ALL_BALANCERS = [b for group in BALANCING_MATRIX.values() for b in group]

# ---------------------------------------------------------------------------
# 3. Feature selection
# ---------------------------------------------------------------------------
FS_CONFIG = {
    "pearson_threshold":   0.02,   # drop features with |corr(feature, target)| below this
    "redundancy_threshold": 0.95,  # drop one of a pair correlated above this
    "hoa_population":      10,
    "hoa_iterations":      15,
    "hoa_alpha":           0.9,    # weight on accuracy vs. feature-count penalty
}

# ---------------------------------------------------------------------------
# 4. Classifiers (fixed across all runs)
# ---------------------------------------------------------------------------
CLASSIFIER_NAMES = ["DT", "RF", "ET", "XGB", "LGBM", "LR", "DNN"]

# ---------------------------------------------------------------------------
# 5. Output
# ---------------------------------------------------------------------------
# Anchored to this file's own directory (not the caller's cwd) so results
# always land in comparative_ids_study/results/ regardless of whether
# main.py is invoked from here or from the parent dir (where it must be run
# from for the dataset paths above to resolve).
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_CSV = RESULTS_DIR / "results.csv"

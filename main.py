"""
Loops the full experiment matrix (datasets x balancing techniques x
classifiers) and writes results/results.csv.

Usage:
    python main.py --datasets nslkdd ids2018 ddos2019
    python main.py --datasets nslkdd --quick        # subsample for a fast smoke test
    python main.py --datasets nslkdd --balancers smote adasyn raw
"""

from __future__ import annotations
import argparse
import pandas as pd

from config import DATASETS, BALANCING_MATRIX, RESULTS_DIR, RESULTS_CSV, RANDOM_STATE
from data_loader import load_dataset
from pipeline import run_combo


def balancer_to_category(name: str) -> str:
    for cat, names in BALANCING_MATRIX.items():
        if name in names:
            return cat
    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                         choices=list(DATASETS.keys()))
    parser.add_argument("--balancers", nargs="+", default=None,
                         help="subset of balancing techniques to run (default: all)")
    parser.add_argument("--quick", action="store_true",
                         help="subsample training data to 5k rows for a fast smoke test")
    args = parser.parse_args()

    balancers = args.balancers or [b for group in BALANCING_MATRIX.values() for b in group]

    RESULTS_DIR.mkdir(exist_ok=True)
    all_rows = []

    for ds_name in args.datasets:
        print(f"\n=== Loading {ds_name} ===")
        X_train, X_test, y_train, y_test = load_dataset(ds_name)

        if args.quick:
            X_train = X_train.sample(min(5000, len(X_train)), random_state=RANDOM_STATE)
            y_train = y_train.loc[X_train.index]
            X_train, y_train = X_train.reset_index(drop=True), y_train.reset_index(drop=True)

        for balancer_name in balancers:
            category = balancer_to_category(balancer_name)
            print(f"  -> {ds_name} / {category} / {balancer_name}")
            try:
                rows = run_combo(ds_name, balancer_name, category,
                                  X_train, y_train, X_test, y_test)
                all_rows.extend(rows)
            except Exception as e:
                print(f"     FAILED ({e.__class__.__name__}: {e})")

            # write incrementally so a long run isn't lost on a crash
            pd.DataFrame(all_rows).to_csv(RESULTS_CSV, index=False)

    print(f"\nDone. Results written to {RESULTS_CSV}")


if __name__ == "__main__":
    main()

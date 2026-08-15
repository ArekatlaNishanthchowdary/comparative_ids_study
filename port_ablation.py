"""
port_ablation.py
=================
Leakage-check ablation for CIC-DDoS2019 referenced in paper.tex Section 4.1.3 /
5.1 (Threats to Validity): does the RAW+RandomForest result depend on
memorizing Source Port / Destination Port / Protocol, or does it reflect
genuine flow-level separability of volumetric DDoS traffic?

Compares the existing RAW+RF baseline (with port features, from
results/results.csv) against the same pipeline with Source Port,
Destination Port, and Protocol dropped before feature selection.

Usage (run from the parent "Conference" directory so config.py's relative
data paths resolve -- same working directory main.py was originally run
from):
    python comparative_ids_study/port_ablation.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from data_loader import load_dataset
from feature_selection import select_features
from models import get_classifiers, evaluate

RESULTS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "results.csv")

PORT_COLS = ["Source Port", "Destination Port", "Protocol"]

print("=== Loading CIC-DDoS2019 (RAW, no balancing) ===")
X_train, X_test, y_train, y_test = load_dataset("ddos2019")

dropped = [c for c in PORT_COLS if c in X_train.columns]
print(f"Dropping port-identifier columns: {dropped}")
X_train_noport = X_train.drop(columns=dropped)
X_test_noport = X_test.drop(columns=dropped)

print("\n=== Feature selection (with ports removed) ===")
features = select_features(X_train_noport, y_train)
print(f"Selected {len(features)}/{X_train_noport.shape[1]} features: {features}")

X_train_fs = X_train_noport[features]
X_test_fs = X_test_noport[features]

print("\n=== Training Random Forest probe (no port features) ===")
clfs = get_classifiers(input_dim=len(features))
rf = clfs["RF"]
metrics = evaluate(rf, X_train_fs, y_train, X_test_fs, y_test)

print("\n=== Results: RAW + RandomForest, CIC-DDoS2019 ===")
baseline_row = pd.read_csv(RESULTS_CSV).query(
    "dataset == 'ddos2019' and balancer == 'raw' and classifier == 'RF'"
)
if baseline_row.empty:
    raise RuntimeError(
        "No RAW+RF/ddos2019 row found in results/results.csv -- run main.py first "
        "so there's a with-ports baseline to compare against."
    )
with_ports = baseline_row.iloc[0]

print(f"{'Metric':12s} {'With Ports':>12s} {'Without Ports':>14s} {'Delta':>10s}")
for k in ["accuracy", "precision", "recall", "f1", "auc"]:
    wo = metrics[k]
    w = with_ports[k]
    print(f"{k:12s} {w:12.4f} {wo:14.4f} {wo-w:+10.4f}")

print(f"\nn_features_selected: with_ports={with_ports['n_features_selected']}  without_ports={len(features)}")
print(f"train_time_sec (without ports): {metrics['train_time_sec']:.4f}")

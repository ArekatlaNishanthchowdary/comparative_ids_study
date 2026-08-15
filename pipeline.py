"""
Runs a single (dataset, balancing technique) combination end-to-end:
balance -> select features -> train+score every classifier in the suite.
Returns a list of result rows (one per classifier), ready to append to the
master results table.
"""

from __future__ import annotations
import pandas as pd

from balancing import balance
from feature_selection import select_features
from models import get_classifiers, evaluate


def run_combo(dataset_name: str, balancer_name: str, category: str,
              X_train, y_train, X_test, y_test) -> list[dict]:
    X_bal, y_bal, bal_time = balance(balancer_name, X_train, y_train)

    features = select_features(X_bal, y_bal)
    X_bal_fs = X_bal[features]
    X_test_fs = X_test[features]

    rows = []
    classifiers = get_classifiers(input_dim=len(features))
    for clf_name, clf in classifiers.items():
        metrics = evaluate(clf, X_bal_fs, y_bal, X_test_fs, y_test)
        rows.append({
            "dataset": dataset_name,
            "category": category,
            "balancer": balancer_name,
            "classifier": clf_name,
            "n_features_selected": len(features),
            "balancing_time_sec": bal_time,
            **metrics,
        })
    return rows

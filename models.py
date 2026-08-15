"""
The fixed classifier suite. Every (dataset, balancing technique, feature
subset) combination is scored with all seven of these, with the same
hyperparameters every time -- so any accuracy differences trace back to the
data pipeline, not to per-run tuning.
"""

from __future__ import annotations
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from config import RANDOM_STATE

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# XGBoost's device="cuda" is unambiguous (CUDA is NVIDIA-only, unlike
# LightGBM's OpenCL "gpu" device which can silently land on an integrated
# GPU instead). Probe once at import time and fall back to CPU if this
# build/machine can't actually use it.
XGB_DEVICE = "cpu"
if XGB_AVAILABLE:
    try:
        XGBClassifier(n_estimators=2, device="cuda", verbosity=0).fit(
            np.zeros((10, 2)), np.array([0, 1] * 5)
        )
        XGB_DEVICE = "cuda"
    except Exception:
        XGB_DEVICE = "cpu"

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except (ImportError, OSError):
    TORCH_AVAILABLE = False


def _build_classifiers():
    clfs = {
        "DT": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RF": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "ET": ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "LR": LogisticRegression(max_iter=500),
    }
    if XGB_AVAILABLE:
        clfs["XGB"] = XGBClassifier(
            n_estimators=200, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
            device=XGB_DEVICE,
        )
    if LGBM_AVAILABLE:
        clfs["LGBM"] = LGBMClassifier(n_estimators=200, random_state=RANDOM_STATE, verbosity=-1)
    return clfs


class _TorchDNN:
    """Small feed-forward DNN wrapped in an sklearn-like fit/predict interface."""

    def __init__(self, input_dim, epochs=30, batch_size=256, lr=1e-3):
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        ).to(DEVICE)

    def fit(self, X, y):
        torch.manual_seed(RANDOM_STATE)
        X_t = torch.tensor(np.asarray(X), dtype=torch.float32, device=DEVICE)
        y_t = torch.tensor(np.asarray(y), dtype=torch.float32, device=DEVICE).view(-1, 1)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        bce = nn.BCELoss()
        n = X_t.shape[0]
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                out = self.model(X_t[idx])
                loss = bce(out, y_t[idx])
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, X):
        X_t = torch.tensor(np.asarray(X), dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            p = self.model(X_t).cpu().numpy().flatten()
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def get_classifiers(input_dim: int):
    clfs = _build_classifiers()
    if TORCH_AVAILABLE:
        clfs["DNN"] = _TorchDNN(input_dim)
    return clfs


def evaluate(clf, X_train, y_train, X_test, y_test) -> dict:
    start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = clf.predict(X_test)
    try:
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = float("nan")

    return {
        "accuracy": accuracy_score(y_test, y_pred) * 100,
        "precision": precision_score(y_test, y_pred, zero_division=0) * 100,
        "recall": recall_score(y_test, y_pred, zero_division=0) * 100,
        "f1": f1_score(y_test, y_pred, zero_division=0) * 100,
        "auc": auc,
        "train_time_sec": train_time,
    }

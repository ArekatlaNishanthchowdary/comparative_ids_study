"""
Two-stage feature selection, held constant across every (dataset, balancing)
combination so that FS is not a confound in the comparison:

  1. Pearson filter: drop weakly-target-correlated features, then drop
     redundant (highly inter-correlated) features.
  2. Wrapper: binary Hippopotamus Optimization Algorithm (HOA), same
     algorithm as the PC-HO paper, re-implemented generically here. Fitness
     is measured with a fast surrogate classifier (Logistic Regression) on a
     held-out slice of the *balanced training data* -- the wrapper never
     touches the test set.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from config import FS_CONFIG, RANDOM_STATE


def pearson_filter(X: pd.DataFrame, y: pd.Series) -> list[str]:
    corr_with_target = X.apply(lambda col: abs(np.corrcoef(col, y)[0, 1])
                                if col.std() > 0 else 0.0)
    kept = corr_with_target[corr_with_target >= FS_CONFIG["pearson_threshold"]].index.tolist()
    if not kept:  # guard against degenerate thresholds wiping everything out
        kept = corr_with_target.sort_values(ascending=False).index[: max(5, X.shape[1] // 2)].tolist()

    if len(kept) <= 1:
        return kept

    corr_matrix = X[kept].corr().abs()
    to_drop = set()
    for i, fi in enumerate(kept):
        if fi in to_drop:
            continue
        for fj in kept[i + 1:]:
            if fj in to_drop:
                continue
            if corr_matrix.loc[fi, fj] >= FS_CONFIG["redundancy_threshold"]:
                to_drop.add(fj)
    return [f for f in kept if f not in to_drop]


def _fitness(mask: np.ndarray, X_tr, y_tr, X_val, y_val, alpha: float) -> float:
    if mask.sum() == 0:
        return 0.0
    cols = X_tr.columns[mask.astype(bool)]
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_tr[cols], y_tr)
    acc = accuracy_score(y_val, clf.predict(X_val[cols]))
    penalty = mask.sum() / len(mask)
    return FS_CONFIG["hoa_alpha"] * acc + (1 - FS_CONFIG["hoa_alpha"]) * (1 - penalty)


def hoa_wrapper(X: pd.DataFrame, y: pd.Series, candidate_features: list[str]) -> list[str]:
    """Binary Hippopotamus Optimization for feature subset selection.
    Mirrors Algorithm 1 of the PC-HO paper: exploration + exploitation update
    on binary position vectors, transfer-function binarization, fitness =
    alpha*accuracy + (1-alpha)*(1 - feature_ratio).
    """
    rng = np.random.RandomState(RANDOM_STATE)
    m = len(candidate_features)
    if m == 0:
        return []

    X_sub = X[candidate_features]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_sub, y, test_size=0.3, random_state=RANDOM_STATE,
        stratify=y if y.nunique() > 1 else None,
    )

    pop_size = FS_CONFIG["hoa_population"]
    iters = FS_CONFIG["hoa_iterations"]
    alpha = FS_CONFIG["hoa_alpha"]

    population = rng.randint(0, 2, size=(pop_size, m)).astype(float)
    fitness = np.array([_fitness(ind, X_tr, y_tr, X_val, y_val, alpha) for ind in population])
    best_idx = fitness.argmax()
    best = population[best_idx].copy()
    best_fit = fitness[best_idx]

    for _ in range(iters):
        for i in range(pop_size):
            r1, r2 = rng.rand(), rng.rand()
            # Position update (continuous relaxation of Algorithm 1, Step 4)
            continuous = population[i] + r1 * (best - r2 * population[i])
            # Exploitation nudge (Step 5) toward two random peers half the time
            if rng.rand() < 0.5:
                j, k = rng.randint(0, pop_size, 2)
                r4 = rng.rand()
                continuous = continuous + r4 * (population[j] - population[k])
            # Transfer function -> binary
            probs = 1 / (1 + np.exp(-continuous))
            new_ind = (probs > rng.rand(m)).astype(float)

            new_fit = _fitness(new_ind, X_tr, y_tr, X_val, y_val, alpha)
            if new_fit >= fitness[i]:
                population[i] = new_ind
                fitness[i] = new_fit

        gen_best_idx = fitness.argmax()
        if fitness[gen_best_idx] > best_fit:
            best = population[gen_best_idx].copy()
            best_fit = fitness[gen_best_idx]

    if best.sum() == 0:  # never return an empty feature set
        best[rng.randint(0, m)] = 1

    return [f for f, keep in zip(candidate_features, best) if keep]


def select_features(X: pd.DataFrame, y: pd.Series) -> list[str]:
    filtered = pearson_filter(X, y)
    return hoa_wrapper(X, y, filtered)

"""
Per-dataset loading, cleaning, and binary-label construction.

Every loader returns (X_train, X_test, y_train, y_test) as pandas
DataFrames/Series with:
  - duplicates removed
  - missing values mean-imputed (numeric) / mode-imputed (categorical),
    fit on train only and applied to test (no test-set stats leak in)
  - categorical columns label-encoded
  - numeric columns standard-scaled (fit on train only)
  - label mapped to {0, 1}  with 1 = attack
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import DATASETS, TEST_SIZE, RANDOM_STATE


def _binarize_label(series: pd.Series, cfg: dict) -> pd.Series:
    """Map normal_label -> 0, everything else -> 1 (attack)."""
    return (series.astype(str).str.strip() != str(cfg["normal_label"])).astype(int)


def _basic_clean(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Structural cleaning only (no statistics computed, so nothing here can
    leak between train/test): strip columns, drop unwanted columns, drop
    duplicates, replace inf with NaN, coerce numeric columns. Missing-value
    imputation is a separate step (see `_impute`) so it can be fit on train
    only.
    """
    df = df.copy()

    # Strip whitespace from column names (CIC datasets have leading spaces)
    df.columns = df.columns.str.strip()

    df = df.drop(columns=[c.strip() for c in cfg.get("drop_cols", [])
                           if c.strip() in df.columns], errors="ignore")
    df = df.drop_duplicates()

    # Replace inf (common in CICFlowMeter flow-rate columns) with NaN
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    label_col = cfg["label_col"].strip()
    for col in df.columns:
        if col == label_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _impute(df: pd.DataFrame, cfg: dict, fill_values: dict | None = None):
    """Mean-impute numeric / mode-impute categorical columns.

    If `fill_values` is None, the fill values are computed from `df` itself
    (call this on the training set). Otherwise the given values are applied
    as-is (call this on the test set) -- so test-set statistics never leak
    into the imputation, matching how the scaler is fit on train only.

    Returns (imputed_df, fill_values) so the caller can reuse the values.
    """
    df = df.copy()
    label_col = cfg["label_col"].strip()
    compute = fill_values is None
    if compute:
        fill_values = {}

    for col in df.columns:
        if col == label_col:
            continue
        if compute:
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_values[col] = df[col].mean()
            else:
                mode = df[col].mode(dropna=True)
                fill_values[col] = mode.iloc[0] if not mode.empty else "unknown"
        df[col] = df[col].fillna(fill_values[col])
    return df, fill_values


def _encode_and_scale(train: pd.DataFrame, test: pd.DataFrame, label_col: str):
    label_col = label_col.strip()
    y_train = train[label_col]
    y_test  = test[label_col]
    X_train = train.drop(columns=[label_col])
    X_test  = test.drop(columns=[label_col])

    cat_cols = [c for c in X_train.columns
                if not pd.api.types.is_numeric_dtype(X_train[c])]
    for c in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train[c], X_test[c]], axis=0).astype(str)
        le.fit(combined)
        X_train[c] = le.transform(X_train[c].astype(str))
        X_test[c]  = le.transform(X_test[c].astype(str))

    num_cols = [c for c in X_train.columns if c not in cat_cols]
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    # Fill NaNs from zero-variance / constant columns with 0
    X_train = X_train.fillna(0.0)
    X_test  = X_test.fillna(0.0)

    return X_train, X_test, y_train, y_test


def load_dataset(name: str):
    """
    Load, clean, encode, and split dataset `name`.
    Works for all three datasets:
      - nslkdd   : pre-split train/test CSVs
      - ids2018  : single concatenated CSV, split 80/20
      - ddos2019 : single concatenated CSV, split 80/20
    """
    cfg       = DATASETS[name]
    label_col = cfg["label_col"].strip()

    train_df = pd.read_csv(cfg["train_path"], low_memory=False)

    # Strip column names (CIC datasets)
    train_df.columns = train_df.columns.str.strip()
    # Make sure label values are also stripped
    train_df[label_col] = train_df[label_col].astype(str).str.strip()

    if cfg["test_path"] is not None:
        test_df = pd.read_csv(cfg["test_path"], low_memory=False)
        test_df.columns = test_df.columns.str.strip()
        test_df[label_col] = test_df[label_col].astype(str).str.strip()

        train_df = _basic_clean(train_df, cfg)
        test_df  = _basic_clean(test_df, cfg)
    else:
        full = _basic_clean(train_df, cfg)
        strat = _binarize_label(full[label_col], cfg)
        train_df, test_df = train_test_split(
            full, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strat
        )

    # Impute missing values using train-set statistics only, then apply the
    # same fill values to test (avoids leaking test-set means/modes in).
    train_df, fill_values = _impute(train_df, cfg)
    test_df, _ = _impute(test_df, cfg, fill_values=fill_values)

    train_df[label_col] = _binarize_label(train_df[label_col], cfg)
    test_df[label_col]  = _binarize_label(test_df[label_col],  cfg)

    X_train, X_test, y_train, y_test = _encode_and_scale(train_df, test_df, label_col)

    attack_rate = y_train.mean()
    print(f"  [{name}] train={X_train.shape}  test={X_test.shape}  "
          f"attack_rate={attack_rate:.3f}  "
          f"imbalance~1:{(1-attack_rate)/attack_rate:.1f}" if attack_rate > 0 else "")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "nslkdd"
    Xtr, Xte, ytr, yte = load_dataset(name)
    print(f"{name}: train={Xtr.shape}, test={Xte.shape}, "
          f"train_attack={ytr.mean():.3f}, test_attack={yte.mean():.3f}")

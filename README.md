# Comparative Study of Data-Balancing Strategies for Intrusion Detection

Code and results for *"A Comparative Study of Data-Balancing Strategies for Intrusion
Detection Across Heterogeneous Network Datasets"* (ICMLDE, Procedia Computer Science).

The question: **does class balancing actually help network intrusion detection, and does
the answer depend on how imbalanced the dataset is?** We hold everything else fixed —
same preprocessing, same feature-selection pipeline, same seven classifiers, same seed —
and vary only the balancing strategy, across three datasets with very different imbalance
profiles.

**210 experiments** = 3 datasets × 10 balancing strategies × 7 classifiers.

---

## Headline result

Best Macro-F1 per dataset, and how much balancing bought over doing nothing (`raw`):

| Dataset | Imbalance | Best strategy | Macro-F1 | vs. RAW |
|---|---|---|---|---|
| NSL-KDD | 1 : 1.15 (mild) | GAN-Oversample + LGBM | 80.55 | **+2.76 pp** |
| CSE-CIC-IDS2018 | 1 : 3.47 (severe) | ROS + LGBM | 98.20 | +0.12 pp |
| CIC-DDoS2019 | 1 : 0.54 (high-volume) | ROS + DT | 99.995 | +0.05 pp |

The counter-intuitive part: the **mildly** imbalanced dataset gained the most, and the
severely imbalanced one barely moved. Where classes are already separable in feature
space, balancing has nothing left to fix — and plain Random Oversampling matches or beats
every GAN/VAE method at a fraction of the cost. A Friedman test across all ten strategies
gives χ² = 48.72 (p < 10⁻⁶), but Kendall's W = 0.258 says the *ranking* is only weakly
consistent across datasets. Pick your balancer per dataset, not from a paper's leaderboard.

---

## What's compared

**Balancing strategies** (`config.py: BALANCING_MATRIX`)

| Category | Strategies |
|---|---|
| baseline | `raw` (none), `ros` |
| ml_based | `smote`, `adasyn` |
| ml_hybrid | `smote_tomek`, `smote_enn` |
| dl_based | `vae_oversample`, `gan_oversample` |
| dl_hybrid | `gan_tomek`, `vae_smote` |

**Classifiers** — DT, RF, ET, XGB, LGBM, LR, DNN. Fixed hyperparameters throughout, no
tuning and no cross-validation, so every difference in the results table traces back to the
data pipeline rather than to per-run luck.

**Feature selection** — PC-HO, two stages, rerun independently for each balanced
distribution: a Pearson filter (drop |corr| < 0.02 with the target, then drop one of any
pair correlated ≥ 0.95) followed by a binary Hippopotamus Optimization wrapper (10
individuals × 15 iterations) scoring subsets with `0.9·accuracy + 0.1·(1 − feature ratio)`
on a 70/30 stratified split of the *balanced training data*. The test set is never touched.

---

## Repository layout

```
main.py                 experiment driver — loops the full matrix, writes results/results.csv
config.py               datasets, balancer matrix, FS hyperparameters, paths  ← edit this first
data_loader.py          load, dedupe, impute, encode, scale, binarize labels (train-fit only)
balancing.py            unified balance(name, X, y) -> (X_bal, y_bal, seconds)
feature_selection.py    Pearson filter + binary HOA wrapper
models.py               the seven-classifier suite and metric computation
pipeline.py             one (dataset, balancer) combo end-to-end
stat_tests.py           Friedman + Nemenyi + critical-difference diagram
generate_figures.py     the 10 result figures at 300 DPI
port_ablation.py        leakage check: does CIC-DDoS2019 depend on port memorization?
smoke_test.py           synthetic end-to-end check — no real datasets needed
scripts/prep_*.py       raw-CSV preprocessing per dataset
results/results.csv     all 210 runs
results/stat_tests.csv  Friedman/Nemenyi output
```

---

## Running it

```bash
pip install -r requirements.txt

# sanity check on synthetic data — takes seconds, needs no downloads
python smoke_test.py
```

Then get the data. **Datasets are not in this repo** (size and licensing); download them
and preprocess:

| Dataset | Source |
|---|---|
| NSL-KDD | https://www.unb.ca/cic/datasets/nsl.html |
| CSE-CIC-IDS2018 | https://www.unb.ca/cic/datasets/ids-2018.html |
| CIC-DDoS2019 | https://www.unb.ca/cic/datasets/ddos-2019.html |

```bash
python scripts/prep_nslkdd.py      # -> data/nslkdd/KDDTrain+.csv, KDDTest+.csv
python scripts/prep_ids2018.py     # -> data/ids2018/all_days.csv
python scripts/prep_ddos2019.py    # -> data/ddos2019/all_days.csv
```

Point `DATASETS` in `config.py` at wherever your raw files actually landed, then:

```bash
python main.py --datasets nslkdd ids2018 ddos2019   # the full matrix
python main.py --datasets nslkdd --quick            # 5k-row subsample, fast
python main.py --datasets nslkdd --balancers raw smote adasyn

python stat_tests.py        # -> results/stat_tests.csv, images/cd_diagram.png
python generate_figures.py  # -> images/*.png
```

`main.py` rewrites `results.csv` after every combination, so a crash eight hours into a run
doesn't cost you the first seven.

> **Working directory matters.** `config.py`'s dataset paths are relative to the *caller's*
> cwd, while results are always written to `comparative_ids_study/results/`. Run `main.py`
> and `port_ablation.py` from whichever directory makes your `data/` paths resolve.

---

## Reproducibility notes

- `RANDOM_STATE = 42` everywhere — splits, resamplers, HOA, and classifiers.
- NSL-KDD uses its official KDDTrain+/KDDTest+ partition. The two CIC datasets get an
  80/20 stratified hold-out (`TEST_SIZE`).
- Imputation, encoding, and scaling are fit on training data only.
- Balancing and feature selection touch training data only; the test set is transformed,
  never fitted on.
- CIC datasets are row-capped during prep (`MAX_BENIGN_ROWS = 500k`, `MAX_ATTACK_ROWS =
  200k`) to keep memory bounded while preserving benign as the majority class.
- All runs were CPU-only. GAN/VAE oversamplers are small, dependency-light PyTorch models
  built for a fair and fast comparison — not to compete with CTGAN/SDV on sample fidelity.

## `results.csv` schema

`dataset`, `category`, `balancer`, `classifier`, `n_features_selected`,
`balancing_time_sec`, `accuracy`, `precision`, `recall`, `f1`, `auc`, `train_time_sec` —
one row per (dataset, balancer, classifier) triple. Metrics are percentages; `f1` is
Macro-F1.

## Citation

Paper under review at ICMLDE (Procedia Computer Science). Citation details will be added on
acceptance.

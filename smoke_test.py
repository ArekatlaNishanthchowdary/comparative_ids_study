"""
End-to-end smoke test on synthetic tabular data (no real IDS CSVs needed).
Validates: balancing -> feature selection -> classifier suite -> metrics,
for one technique per category, so a broken combo fails loudly before a
multi-hour real run.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from balancing import balance, TORCH_AVAILABLE
from feature_selection import select_features
from models import get_classifiers, evaluate

np.random.seed(42)
X, y = make_classification(
    n_samples=1500, n_features=20, n_informative=10, n_redundant=4,
    weights=[0.85, 0.15], random_state=42,
)
X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
y = pd.Series(y, name="label")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

techniques = ["raw", "smote", "adasyn", "smote_tomek", "smote_enn"]
if TORCH_AVAILABLE:
    techniques += ["vae_oversample", "gan_oversample", "gan_tomek", "vae_smote"]
else:
    print("[smoke_test] torch unavailable in this environment -- skipping DL-based/"
          "DL-hybrid techniques (they will still run wherever torch works).")

for tech in techniques:
    print(f"\n--- {tech} ---")
    X_bal, y_bal, t = balance(tech, X_train, y_train)
    print(f"balanced shape={X_bal.shape}, class balance={y_bal.mean():.3f}, time={t:.2f}s")

    feats = select_features(X_bal, y_bal)
    print(f"selected {len(feats)}/{X_bal.shape[1]} features")

    clfs = get_classifiers(input_dim=len(feats))
    for name, clf in clfs.items():
        m = evaluate(clf, X_bal[feats], y_bal, X_test[feats], y_test)
        print(f"  {name:5s} acc={m['accuracy']:.1f} f1={m['f1']:.1f} "
              f"time={m['train_time_sec']:.2f}s")

print("\nSMOKE TEST PASSED")

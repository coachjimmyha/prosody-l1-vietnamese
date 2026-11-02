"""hyperparameter_sensitivity.py — sensitivity of the validation PCC to the Random Forest hyperparameters
(Section 4.4; Appendix B.13).

The hyperparameters of the canonical model (200 trees, maximum depth 8, minimum leaf size 10) were fixed
in advance. This script checks how much the validation PCC would change over a grid of alternatives,
using exactly the canonical 70/15/15 split of final_evaluation_v3.py (seed 42). Only the training and
validation sets are used; the test set is not touched, so the check cannot leak into the headline result.

Grid: trees {100, 200, 500} x maximum depth {4, 6, 8, 12, unlimited} x minimum leaf size {1, 5, 10, 20}.

Run:  python scripts/hyperparameter_sensitivity.py
"""
import csv
import itertools
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from paths import OUTPUTS
from final_evaluation_v3 import FEATURE_COLS, SPEECHOCEAN_CSV

TREES = [100, 200, 500]
DEPTHS = [4, 6, 8, 12, None]
LEAVES = [1, 5, 10, 20]
CHOSEN = (200, 8, 10)
OUT_CSV = OUTPUTS / "hyperparameter_sensitivity.csv"


def main():
    rows = []
    with open(SPEECHOCEAN_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({c: float(r[c]) for c in FEATURE_COLS + ["prosodic_score"]})
            except (KeyError, ValueError):
                continue
    X = np.array([[r[c] for c in FEATURE_COLS] for r in rows])
    y = np.array([r["prosodic_score"] for r in rows])
    X_tv, _X_test, y_tv, _y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tv, y_tv, test_size=0.176, random_state=42)
    print(f"Hyperparameter sensitivity — train {len(y_tr)}, validation {len(y_val)} (test set not used)")

    results = []
    for n, d, leaf in itertools.product(TREES, DEPTHS, LEAVES):
        rf = RandomForestRegressor(n_estimators=n, max_depth=d, min_samples_leaf=leaf,
                                   random_state=42, n_jobs=-1).fit(X_tr, y_tr)
        pcc = float(stats.pearsonr(rf.predict(X_val), y_val)[0])
        results.append((n, d, leaf, pcc))

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n_estimators", "max_depth", "min_samples_leaf", "val_pcc"])
        for n, d, leaf, pcc in results:
            w.writerow([n, "None" if d is None else d, leaf, f"{pcc:.4f}"])

    print(f"\nValidation PCC by maximum depth x minimum leaf size (200 trees):")
    print("  depth \\ leaf " + "".join(f"{leaf:>8}" for leaf in LEAVES))
    for d in DEPTHS:
        vals = [p for n, dd, leaf, p in results if n == 200 and dd == d]
        print(f"  {str(d):>12} " + "".join(f"{v:8.3f}" for v in vals))
    print("\nValidation PCC by number of trees (depth 8, leaf 10): " +
          ", ".join(f"{n}: {p:.3f}" for n, d, leaf, p in results if d == 8 and leaf == 10))

    pccs = np.array([p for *_, p in results])
    best = max(results, key=lambda t: t[3])
    chosen = next(p for n, d, leaf, p in results if (n, d, leaf) == CHOSEN)
    print(f"\nAll {len(results)} settings: validation PCC {pccs.min():.3f}–{pccs.max():.3f}")
    print(f"Best: trees={best[0]}, depth={best[1]}, leaf={best[2]}  PCC={best[3]:.3f}")
    print(f"Chosen a priori (200, 8, 10): PCC={chosen:.3f}  (best − chosen = {best[3] - chosen:.3f})")
    print(f"-> saved outputs/{OUT_CSV.name}")


if __name__ == "__main__":
    main()

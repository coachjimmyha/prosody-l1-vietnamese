"""feature_correlations.py — pairwise Spearman correlations among the fourteen model features on the
Speechocean762 training table (Section 4.3.2; Appendix B.14).

Reports the feature pairs with |rho| >= 0.5, which matter for reading permutation importance (correlated
features share importance; Section 2.3.2), and saves the full matrix to outputs/feature_correlations.csv.

Run:  python scripts/feature_correlations.py
"""
import csv
import itertools
import numpy as np
from scipy import stats

from paths import OUTPUTS
from final_evaluation_v3 import FEATURE_COLS, SPEECHOCEAN_CSV

OUT_CSV = OUTPUTS / "feature_correlations.csv"


def main():
    rows = []
    with open(SPEECHOCEAN_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows.append([float(r[c]) for c in FEATURE_COLS])
            except (KeyError, ValueError):
                continue
    X = np.array(rows)
    rho = stats.spearmanr(X).correlation
    print(f"Spearman correlations among the 14 features (n = {len(X)} utterances)")
    pairs = sorted(((abs(rho[i, j]), FEATURE_COLS[i], FEATURE_COLS[j], rho[i, j])
                    for i, j in itertools.combinations(range(len(FEATURE_COLS)), 2)), reverse=True)
    print("Pairs with |rho| >= 0.5:")
    for a, fi, fj, r in pairs:
        if a >= 0.5:
            print(f"  {fi:22} {fj:22} rho = {r:+.2f}")
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature"] + FEATURE_COLS)
        for i, c in enumerate(FEATURE_COLS):
            w.writerow([c] + [f"{rho[i, j]:.3f}" for j in range(len(FEATURE_COLS))])
    print(f"-> saved outputs/{OUT_CSV.name}")


if __name__ == "__main__":
    main()

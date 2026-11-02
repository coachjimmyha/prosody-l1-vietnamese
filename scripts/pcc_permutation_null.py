"""pcc_permutation_null.py — shuffled-label null for RQ1 (Section 4.5; Figure 5.2a; Appendix B.2).

The training labels are shuffled, the Random Forest is refitted with the same settings and split,
and the PCC is recomputed on the unchanged test set; repeated 1,000 times (seed 42).
Empirical one-sided p = (k + 1) / (N + 1), where k = number of null PCCs >= the observed PCC.
Output: outputs/pcc_null_1000.csv

Run:  python scripts/pcc_permutation_null.py      (about 10–60 min, depending on the machine)
"""
import csv
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

from paths import OUTPUTS as OUT
FEAT = ["mean_duration_ratio","std_duration_ratio","max_duration_ratio","mean_f0_ratio",
        "std_f0_ratio","mean_intensity_diff","std_intensity_diff","n_multisyllabic_words",
        "f0_range_hz","speaking_rate","silence_proportion","npvi","percent_v","varcov"]

rows = []
for r in csv.DictReader(open(OUT/"speechocean762_full_per_utt_v3.csv", encoding="utf-8")):
    try: rows.append({c: float(r[c]) for c in FEAT+["prosodic_score"]})
    except (KeyError, ValueError): pass
X = np.array([[r[c] for c in FEAT] for r in rows])
y = np.array([r["prosodic_score"] for r in rows])

def rf(): return RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=10,
                                       random_state=42, n_jobs=-1)

Xtv, Xte, ytv, yte = train_test_split(X, y, test_size=0.15, random_state=42)
Xtr, Xval, ytr, yval = train_test_split(Xtv, ytv, test_size=0.176, random_state=42)
m = rf(); m.fit(Xtr, ytr); real = pearsonr(yte, m.predict(Xte))[0]
print(f"real test PCC = {real:.4f}")

rng = np.random.default_rng(42); N = 1000; null = np.empty(N); ys = ytr.copy()
for i in range(N):
    rng.shuffle(ys)
    mn = rf(); mn.fit(Xtr, ys); null[i] = pearsonr(yte, mn.predict(Xte))[0]
p = (np.sum(null >= real) + 1) / (N + 1)
print(f"null max = {null.max():.3f}   mean = {null.mean():.3f}   "
      f"n >= real = {int(np.sum(null >= real))}/{N}   empirical p = {p:.4f}")
np.savetxt(OUT/"pcc_null_1000.csv", null, delimiter=",")

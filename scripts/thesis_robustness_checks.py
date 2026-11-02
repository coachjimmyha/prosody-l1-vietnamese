"""thesis_robustness_checks.py — ablation, baselines, permutation importance and bootstrap CI.

All checks use the canonical split of final_evaluation_v3.py (1,789 / 383 / 384, seed 42):
  1. quick 10-run shuffled-label check (the 1,000-run null is pcc_permutation_null.py)
  2. feature-group ablation: nested 8 -> 11 -> 14 features, one group removed at a time, and
     single-feature and linear baselines (Section 4.5; Figure 4.2; Appendix B.4)
  3. permutation importance of all 14 features on the test set (R² drop, 30 repeats)
     -> outputs/permutation_importance_v3.csv (Figures 5.4 and 5.5; Appendix B.5)
  4. bootstrap 95% CI of the test PCC (2,000 resamples; Table 5.1)
  5. speaker-level summary of the predicted scores (native vs Vietnamese, n = 4 per group)

Run:  python scripts/thesis_robustness_checks.py
"""
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

from paths import OUTPUTS as OUT
TRAIN = OUT / "speechocean762_full_per_utt_v3.csv"
FEAT = ["mean_duration_ratio","std_duration_ratio","max_duration_ratio","mean_f0_ratio","std_f0_ratio",
        "mean_intensity_diff","std_intensity_diff","n_multisyllabic_words","f0_range_hz","speaking_rate",
        "silence_proportion","npvi","percent_v","varcov"]

def rf():
    return RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=10,
                                 random_state=42, n_jobs=-1)

# ---- load training data ----
X, y = [], []
for r in csv.DictReader(open(TRAIN, encoding="utf-8")):
    try: X.append([float(r[c]) for c in FEAT]); y.append(float(r["prosodic_score"]))
    except (KeyError, ValueError): pass
X, y = np.array(X), np.array(y)
# canonical split — identical to final_evaluation_v3.py
Xtv, Xte, ytv, yte = train_test_split(X, y, test_size=0.15, random_state=42)
Xtr, Xval, ytr, yval = train_test_split(Xtv, ytv, test_size=0.176, random_state=42)
print(f"split: train {len(ytr)} / val {len(yval)} / test {len(yte)}")

print("="*72); print("1. SHUFFLE-LABEL NULL"); print("="*72)
m = rf(); m.fit(Xtr, ytr); real = pearsonr(m.predict(Xte), yte)[0]
rng = np.random.default_rng(42); nulls = []
for _ in range(10):
    yp = ytr.copy(); rng.shuffle(yp)
    mn = rf(); mn.fit(Xtr, yp); nulls.append(pearsonr(mn.predict(Xte), yte)[0])
nulls = np.array(nulls)
print(f"  real test PCC = {real:+.3f}   null mean = {nulls.mean():+.3f} (SD {nulls.std():.3f})")
print(f"  real PCC is {(real-nulls.mean())/nulls.std():.1f} SD above the null mean")

print("\n"+"="*72); print("2. ABLATION (same split, same RF settings)"); print("="*72)
GROUPS = {"stress": list(range(0, 8)), "utterance": [8, 9, 10], "rhythm": [11, 12, 13]}
def pcc_cols(cols):
    mm = rf(); mm.fit(Xtr[:, cols], ytr); return pearsonr(mm.predict(Xte[:, cols]), yte)[0]
print("  nested (order: stress -> utterance -> rhythm)")
for lab, k in [("stress only (8)", 8), ("+ utterance-level (11)", 11), ("+ rhythm = full (14)", 14)]:
    print(f"    {lab:26} test PCC = {pcc_cols(list(range(k))):+.3f}")
print("  leave-one-group-out / single group")
for g, cols in GROUPS.items():
    rest = [i for i in range(14) if i not in cols]
    print(f"    drop {g:10} ({len(rest):2} feats)  PCC = {pcc_cols(rest):+.3f}    only {g:10} PCC = {pcc_cols(cols):+.3f}")
i_rate = FEAT.index("speaking_rate")
print("  baselines")
print(f"    speaking_rate raw value vs score        PCC = {pearsonr(Xte[:, i_rate], yte)[0]:+.3f}")
print(f"    RF on speaking_rate only                PCC = {pcc_cols([i_rate]):+.3f}")
lr = LinearRegression().fit(Xtr, ytr)
print(f"    linear regression, 14 features          PCC = {pearsonr(lr.predict(Xte), yte)[0]:+.3f}")
print(f"    RF without speaking_rate (13 feats)     PCC = {pcc_cols([i for i in range(14) if i != i_rate]):+.3f}")

print("\n"+"="*72); print("3. PERMUTATION IMPORTANCE (unbiased, held-out test)"); print("="*72)
pi = permutation_importance(m, Xte, yte, n_repeats=30, random_state=42, scoring="r2")
with open(OUT / "permutation_importance_v3.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["feature", "importance_mean_r2_drop", "importance_std"])
    for i in np.argsort(pi.importances_mean)[::-1]:
        w.writerow([FEAT[i], round(float(pi.importances_mean[i]), 5), round(float(pi.importances_std[i]), 5)])
        print(f"  {FEAT[i]:22} {pi.importances_mean[i]:+.4f} ± {pi.importances_std[i]:.4f}")
print("  -> saved outputs/permutation_importance_v3.csv")
print("  (NB: final_feature_importance_v3_eval.csv is impurity/MDI importance, NOT permutation importance)")

print("\n"+"="*72); print("4. BOOTSTRAP 95% CI on test PCC"); print("="*72)
pred = m.predict(Xte); rng = np.random.default_rng(42); bs = []
for _ in range(2000):
    s = rng.integers(0, len(yte), len(yte))
    if len(np.unique(yte[s])) > 1: bs.append(pearsonr(pred[s], yte[s])[0])
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"  test PCC = {real:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")

print("\n"+"="*72); print("5. SPEAKER-LEVEL CROSS-L1 on PREDICTED SCORE"); print("="*72)
mfull = rf(); mfull.fit(X, y)   # train on ALL Speechocean762
def load_X(p):
    S, M = [], []
    for r in csv.DictReader(open(p, encoding="utf-8")):
        S.append(r["speaker"]); M.append([float(r[c]) for c in FEAT])
    return np.array(S), np.array(M)
def cohens_d(a, b):
    a, b = np.asarray(a,float), np.asarray(b,float)
    pp = np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2); return (a.mean()-b.mean())/pp if pp>0 else 0.0
def spk_means(S, p):
    g = defaultdict(list)
    for s, v in zip(S, p): g[s].append(v)
    return {s: float(np.mean(v)) for s, v in g.items()}
try:
    Sn, Xn = load_X(OUT/"X14_native_fresh.csv"); Sv, Xv = load_X(OUT/"X14_vn_fresh.csv")
    nm = spk_means(Sn, mfull.predict(Xn)); vm = spk_means(Sv, mfull.predict(Xv))
    nv = np.array([nm[s] for s in sorted(nm)]); vv = np.array([vm[s] for s in sorted(vm)])
    d_utt = cohens_d(mfull.predict(Xn), mfull.predict(Xv)); d_spk = cohens_d(nv, vv)
    rng = np.random.default_rng(42); ds = []
    for _ in range(5000):
        ds.append(cohens_d(nv[rng.integers(0,len(nv),len(nv))], vv[rng.integers(0,len(vv),len(vv))]))
    ds = np.array([x for x in ds if np.isfinite(x)]); lo, hi = np.percentile(ds, [2.5, 97.5])
    print(f"  native predicted-score mean {nv.mean():.3f} vs VN {vv.mean():.3f} (gap {nv.mean()-vv.mean():+.3f})")
    print(f"  d utterance-level = {d_utt:+.3f} (pseudoreplicated)")
    print(f"  d speaker-level   = {d_spk:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]  ({'clears 0' if lo>0 else 'crosses 0'})")
    print("  NOTE: this bootstrap resamples 4 speaker means and ignores model randomness; it is NOT the")
    print("  inferential test for RQ2. The canonical test is the 1000-run fake-model null in")
    print("  transfer_reality_check_v2.py (p ~ 0.38 -> inconclusive); content-matched check: rq2_matched_content.py.")
except FileNotFoundError:
    print("  (skip) X14_native_fresh.csv / X14_vn_fresh.csv not found in outputs.")

print("\nDone.")

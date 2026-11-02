"""reduction_stats.py — function-word reduction statistics (Section 5.3; Figure 5.7; Appendix B.8).

Reads outputs/reduction_full.csv (20 speakers: 5 groups x 4) and prints the group means,
native vs Vietnamese and native vs pooled-L2 Cohen's d with Mann–Whitney p, the exact permutation p
(70 relabellings), 5,000-resample speaker bootstrap CIs, the correlation between reduction and
speaking rate across the 20 speakers, and the per-learner z-scores against the native reference.
Output: outputs/reduction_stats_summary.csv (read by Figures 5.3 and 5.5)

Run:  python scripts/reduction_stats.py
"""
import csv, itertools
import numpy as np
from collections import defaultdict
from scipy.stats import mannwhitneyu, pearsonr
from pathlib import Path

from paths import OUTPUTS
CSV = OUTPUTS / "reduction_full.csv"
RNG = np.random.RandomState(42)

def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if sp2 <= 0:
        return np.nan            # guard: identical resample -> undefined d (skip in bootstrap)
    return (a.mean() - b.mean()) / np.sqrt(sp2)

def boot_ci(a, b, n=5000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ds = []
    for _ in range(n):
        da = cohens_d(RNG.choice(a, len(a), replace=True), RNG.choice(b, len(b), replace=True))
        if np.isfinite(da):
            ds.append(da)
    return np.percentile(ds, [2.5, 97.5])

def exact_perm_p(a, b):
    """Two-sided exact permutation p on the difference of means (all relabellings)."""
    v = np.r_[a, b]; n = len(a); obs = abs(np.mean(a) - np.mean(b)); hit = tot = 0
    for idx in itertools.combinations(range(len(v)), n):
        m = np.zeros(len(v), bool); m[list(idx)] = True
        tot += 1; hit += abs(v[m].mean() - v[~m].mean()) >= obs - 1e-12
    return hit / tot

def main():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    red = defaultdict(list); rate = defaultdict(list); names = defaultdict(list)
    for r in rows:
        red[r["group"]].append(float(r["reduction"]))
        rate[r["group"]].append(float(r["rate"]))
        names[r["group"]].append(r["speaker"])

    print("=== speaker-level group means (n=4 each) ===")
    for g in ["Native", "Korean", "Spanish", "Vietnamese", "Chinese"]:
        print(f"  {g:12} {np.mean(red[g]):.3f}   {[round(x,3) for x in red[g]]}")

    nat, vn = red["Native"], red["Vietnamese"]
    pooledL2 = red["Vietnamese"] + red["Korean"] + red["Spanish"] + red["Chinese"]

    print("\n=== effect sizes (speaker-level) ===")
    lo, hi = boot_ci(nat, vn)
    print(f"  native-vs-VN         d = {cohens_d(nat,vn):+.3f}   MWU p = {mannwhitneyu(nat,vn).pvalue:.4f}"
          f"   boot 95% CI [{lo:+.2f}, {hi:+.2f}]  {'no cross 0' if hi<0 else 'crosses 0'}")
    pex = exact_perm_p(nat, vn)
    print(f"  native-vs-VN         exact permutation p = {pex:.3f}  (70 relabellings; minimum attainable 0.029)")
    with open(CSV.parent / "reduction_stats_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["feature", "d_native_minus_vn", "boot_lo", "boot_hi", "exact_perm_p", "native_mean", "vn_mean"])
        w.writerow(["reduction", round(cohens_d(nat, vn), 4), round(lo, 4), round(hi, 4), round(pex, 4),
                    round(float(np.mean(nat)), 4), round(float(np.mean(vn)), 4)])
    lo2, hi2 = boot_ci(nat, pooledL2)
    print(f"  native-vs-pooledL2   d = {cohens_d(nat,pooledL2):+.3f}   MWU p = {mannwhitneyu(nat,pooledL2).pvalue:.4f}"
          f"   boot 95% CI [{lo2:+.2f}, {hi2:+.2f}]  {'no cross 0' if hi2<0 else 'crosses 0'}")

    allred = [float(r["reduction"]) for r in rows]
    allrate = [float(r["rate"]) for r in rows]
    rr, pp = pearsonr(allred, allrate)
    print(f"\n  reduction <-> speaking rate:  r = {rr:+.3f}  p = {pp:.4f}  (n={len(allred)} speakers)")

    nm, nsd = np.mean(nat), np.std(nat, ddof=1)
    print(f"\n=== per-VN-speaker reduction z (vs native mean {nm:.3f}, SD {nsd:.3f}) ===")
    for name, val in zip(names["Vietnamese"], vn):
        print(f"  {name:8} reduction={val:.3f}  z={(val-nm)/nsd:+.2f}")

if __name__ == "__main__":
    main()

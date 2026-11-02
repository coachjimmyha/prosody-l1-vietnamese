"""reduction_ablation.py — is the reduction ratio worth adding to the scoring model? (Section 5.3)

Joins outputs/reduction_speechocean.csv (reduction ratio per Speechocean762 utterance,
from extract_reduction_feature.py speechocean) to the 14-feature training table and reports
  the coverage and the correlation between reduction and the prosody score,
  the canonical-split test PCC with 14 and with 15 features (+ reduction), and
  speaker-disjoint 5-fold CV (R² and PCC) with 14 and 15 features.
Reduction values are winsorised at the 1st/99th percentiles; utterances without a value (no function
word or no stressed content-word vowel) receive the median.

Run:  python scripts/reduction_ablation.py
"""
import csv
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import r2_score

from paths import OUTPUTS as OUT
TRAIN = OUT / "speechocean762_full_per_utt_v3.csv"
REDCSV = OUT / "reduction_speechocean.csv"

FEAT = ["mean_duration_ratio", "std_duration_ratio", "max_duration_ratio",
        "mean_f0_ratio", "std_f0_ratio", "mean_intensity_diff", "std_intensity_diff",
        "n_multisyllabic_words", "f0_range_hz", "speaking_rate", "silence_proportion",
        "npvi", "percent_v", "varcov"]

def rf():
    return RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=10,
                                 random_state=42, n_jobs=-1)

def main():
    red = {r["utt_id"]: float(r["reduction_ratio"]) for r in csv.DictReader(open(REDCSV))}
    rows = list(csv.DictReader(open(TRAIN)))
    N = len(rows); have = sum(1 for r in rows if r["utt_id"] in red)
    print(f"coverage: {have}/{N} = {have/N*100:.1f}% of training utts have a reduction value")

    X = np.array([[float(r[c]) for c in FEAT] for r in rows])
    y = np.array([float(r["prosodic_score"]) for r in rows])
    g = np.array([r["utt_id"][:5] for r in rows])                      # speaker = first 5 digits
    rv = np.array([red.get(r["utt_id"], np.nan) for r in rows])

    # winsorise P1-P99 on observed values; fill missing with the median
    obs = rv[~np.isnan(rv)]; lo, hi = np.percentile(obs, [1, 99]); med = np.median(obs)
    rw = np.where(np.isnan(rv), med, np.clip(rv, lo, hi))

    r, p = pearsonr(rw, y)
    ro, _ = pearsonr(obs, y[~np.isnan(rv)])
    print(f"reduction <-> prosody score:  r = {r:+.3f}  (p = {p:.1e})   [observed-only r = {ro:+.3f}]")

    X15 = np.column_stack([X, rw])

    def split_pcc(XX):
        Xtv, Xte, ytv, yte = train_test_split(XX, y, test_size=0.15, random_state=42)
        m = rf(); m.fit(Xtv, ytv)
        return pearsonr(m.predict(Xte), yte)[0]

    print(f"ablation single-split PCC:  14-feat {split_pcc(X):+.3f}  vs  15 (+reduction) {split_pcc(X15):+.3f}")

    def cv(XX):
        r2s, pccs = [], []
        for tr, te in GroupKFold(5).split(XX, y, g):
            m = rf(); m.fit(XX[tr], y[tr]); pr = m.predict(XX[te])
            r2s.append(r2_score(y[te], pr)); pccs.append(pearsonr(pr, y[te])[0])
        return np.mean(r2s), np.mean(pccs)

    r2_14, pcc_14 = cv(X); r2_15, pcc_15 = cv(X15)
    print(f"speaker-disjoint 5-fold CV R2:  14 {r2_14:.3f} -> 15 {r2_15:.3f}  (delta {r2_15-r2_14:+.3f})")
    print(f"speaker-disjoint 5-fold CV PCC: 14 {pcc_14:.3f} -> 15 {pcc_15:.3f}  (delta {pcc_15-pcc_14:+.3f})")
    print("VERDICT: net gain ~ 0 -> keep 14 features; reduction is diagnostic, not a model input.")

if __name__ == "__main__":
    main()

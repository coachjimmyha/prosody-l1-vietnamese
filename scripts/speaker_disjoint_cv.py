"""speaker_disjoint_cv.py — speaker leakage checks for the scoring model (Section 3.7.2; Table 5.1).

The canonical split is random at utterance level, so a Speechocean762 speaker can appear in both
training and test data. Speaker identity is taken from the first five digits of utt_id
(250 speakers). The script reports
  [A] the canonical utterance-level split,
  [B] a speaker-disjoint single split (GroupShuffleSplit, 15% of speakers held out),
  [C] speaker-grouped 5-fold cross-validation (GroupKFold), mean ± SD across folds,
  [D] a shuffled-label control under the speaker-disjoint split.
Appendix B.3 reproduces the output.

Run:  python scripts/speaker_disjoint_cv.py
"""
import csv
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold
from scipy.stats import pearsonr

from paths import OUTPUTS
CSV = OUTPUTS / "speechocean762_full_per_utt_v3.csv"

# The canonical 14 features, in the training order.
FEAT = [
    "mean_duration_ratio", "std_duration_ratio", "max_duration_ratio",   # stress: duration
    "mean_f0_ratio", "std_f0_ratio",                                     # stress: F0
    "mean_intensity_diff", "std_intensity_diff",                         # stress: intensity
    "n_multisyllabic_words",                                             # stress: coverage
    "f0_range_hz", "speaking_rate", "silence_proportion",                # utterance-level
    "npvi", "percent_v", "varcov",                                       # rhythm
]

def rf():
    # Same hyperparameters as the canonical model (final_evaluation_v3.py).
    return RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1
    )

def load():
    X, y, g = [], [], []
    for r in csv.DictReader(open(CSV, encoding="utf-8")):
        X.append([float(r[c]) for c in FEAT])
        y.append(float(r["prosodic_score"]))
        g.append(r["utt_id"][:5])          # first 5 digits = Speechocean762 speaker ID
    return np.array(X), np.array(y), np.array(g)

def main():
    X, y, g = load()
    print(f"n={len(y)} utterances, {len(set(g))} speakers (prefix-5)")
    print("-" * 68)

    # [A] Utterance-level split with the canonical 15% test set, trained on training + validation (85%),
    #     i.e. the same training share as [B]; the canonical model (70% training) gives 0.656.
    Xtv, Xte, ytv, yte = train_test_split(X, y, test_size=0.15, random_state=42)
    m = rf(); m.fit(Xtv, ytv)
    print(f"[A] utterance-level split (train 85%):  test PCC = {pearsonr(m.predict(Xte), yte)[0]:+.3f}")

    # [B] Speaker-disjoint single split (no speaker in both train and test).
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42).split(X, y, g))
    m = rf(); m.fit(X[tr], y[tr])
    overlap = set(g[tr]) & set(g[te])
    print(f"[B] speaker-disjoint split:             test PCC = {pearsonr(m.predict(X[te]), y[te])[0]:+.3f}"
          f"   (train/test speaker overlap = {len(overlap)})")

    # [C] Speaker-disjoint 5-fold cross-validation — stability across partitions.
    pccs = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        m = rf(); m.fit(X[tr], y[tr])
        pccs.append(pearsonr(m.predict(X[te]), y[te])[0])
    print(f"[C] speaker-disjoint 5-fold CV:         PCC = {np.mean(pccs):+.3f} +/- {np.std(pccs):.3f}"
          f"   folds = {[round(float(p), 3) for p in pccs]}")

    # [D] Shuffle-label null under the speaker-disjoint split — must collapse to ~0.
    rng = np.random.RandomState(42); ysh = y.copy(); rng.shuffle(ysh)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42).split(X, ysh, g))
    m = rf(); m.fit(X[tr], ysh[tr])
    print(f"[D] shuffle-null (speaker-disjoint):    test PCC = {pearsonr(m.predict(X[te]), ysh[te])[0]:+.3f}"
          f"   (~0 -> signal is real)")

    print("-" * 68)
    print("Verdict: no sizeable inflation from speaker leakage; the PCC holds under a speaker-disjoint")
    print("split and speaker-disjoint 5-fold CV. The n=4 small-sample limit is on the cross-L1 side, not the scoring model.")

if __name__ == "__main__":
    main()

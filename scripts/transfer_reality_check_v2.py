"""transfer_reality_check_v2.py — fake-model null for RQ2 (Section 3.7.2; Figure 5.2b; Appendix B.6).

1. Real model: the Random Forest is trained on all 2,556 Speechocean762 utterances with the true
   prosody scores and applied to the native and Vietnamese utterances.
2. Fake models: the training scores are shuffled, the same model is refitted and applied to the same
   utterances; repeated 1,000 times (seed 42).
3. For every model, Cohen's d between native and Vietnamese predictions is computed at utterance
   level and at speaker level (4 vs 4 speaker means).
Empirical two-sided p = (k + 1) / (N + 1), k = number of fake models with |d| >= the real |d|.
Output: outputs/transfer_fake_null_1000.csv

Run:  python scripts/transfer_reality_check_v2.py      (about 10–60 min, depending on the machine)
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from paths import OUTPUTS as OUT
FEAT = ["mean_duration_ratio", "std_duration_ratio", "max_duration_ratio", "mean_f0_ratio",
        "std_f0_ratio", "mean_intensity_diff", "std_intensity_diff", "n_multisyllabic_words",
        "f0_range_hz", "speaking_rate", "silence_proportion", "npvi", "percent_v", "varcov"]
N_PERM = 1000
SEED = 42


def load_train(p):
    X, y = [], []
    for r in csv.DictReader(open(p, encoding="utf-8")):
        try:
            X.append([float(r[c]) for c in FEAT]); y.append(float(r["prosodic_score"]))
        except (KeyError, ValueError):
            pass
    return np.array(X), np.array(y)


def load_X(p):
    S, X = [], []
    for r in csv.DictReader(open(p, encoding="utf-8")):
        S.append(r["speaker"]); X.append([float(r[c]) for c in FEAT])
    return np.array(S), np.array(X)


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / pp if pp > 0 else 0.0


def spk_means(S, p):
    g = defaultdict(list)
    for s, v in zip(S, p):
        g[s].append(v)
    return np.array([np.mean(g[s]) for s in sorted(g)])


def rf():
    return RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=10,
                                 random_state=SEED, n_jobs=-1)


X, y = load_train(OUT / "speechocean762_full_per_utt_v3.csv")
Sn, Xn = load_X(OUT / "X14_native_fresh.csv")
Sv, Xv = load_X(OUT / "X14_vn_fresh.csv")
print(f"Training: {len(y)} Speechocean762 utterances | native {len(Xn)} utts "
      f"({len(set(Sn))} spk) | VN {len(Xv)} utts ({len(set(Sv))} spk)")

m = rf(); m.fit(X, y)
pn, pv = m.predict(Xn), m.predict(Xv)
real_utt = cohens_d(pn, pv)
real_spk = cohens_d(spk_means(Sn, pn), spk_means(Sv, pv))
print(f"REAL model: utterance d = {real_utt:+.3f} | speaker d = {real_spk:+.3f}")

rng = np.random.default_rng(SEED)
rows = []
for i in range(N_PERM):
    ys = y.copy(); rng.shuffle(ys)
    mf = rf(); mf.fit(X, ys)
    fn, fv = mf.predict(Xn), mf.predict(Xv)
    rows.append((i + 1, cohens_d(fn, fv), cohens_d(spk_means(Sn, fn), spk_means(Sv, fv))))
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{N_PERM} fake models done", flush=True)

fu = np.array([r[1] for r in rows]); fs = np.array([r[2] for r in rows])
k_u = int(np.sum(np.abs(fu) >= abs(real_utt))); k_s = int(np.sum(np.abs(fs) >= abs(real_spk)))
p_u = (k_u + 1) / (N_PERM + 1); p_s = (k_s + 1) / (N_PERM + 1)

with open(OUT / "transfer_fake_null_1000.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["run", "fake_d_utterance", "fake_d_speaker"])
    w.writerows([(r[0], f"{r[1]:.6f}", f"{r[2]:.6f}") for r in rows])

print("=" * 72)
print(f"N = {N_PERM} fake models (shuffled Speechocean762 labels), seed {SEED}")
print(f"Utterance level: real |d| = {abs(real_utt):.3f}; fake |d| mean {np.abs(fu).mean():.3f}, "
      f"max {np.abs(fu).max():.3f}; k = {k_u}; empirical p = {p_u:.3f}")
print(f"Speaker level  : real |d| = {abs(real_spk):.3f}; fake |d| mean {np.abs(fs).mean():.3f}, "
      f"max {np.abs(fs).max():.3f}; k = {k_s}; empirical p = {p_s:.3f}")
print("Verdict:", "INCONCLUSIVE - fake models separate the groups as much as the real model"
      if p_s >= 0.05 else "real separation exceeds the fake-model null")
print(f"Saved: {OUT / 'transfer_fake_null_1000.csv'}")

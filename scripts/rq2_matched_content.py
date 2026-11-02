"""rq2_matched_content.py — content-matched sensitivity check for RQ2 (Sections 5.2 and 5.6; Appendix B.6).

The native matrix covers prompts arctic_a0001–a0200 (722 utterances), whereas the Vietnamese matrix
covers all prompts (4,129 utterances). This script restricts the Vietnamese set to the prompts present
in the native set and repeats transfer_reality_check_v2.py exactly (same real model, same 1,000 fake
models, same random sequence).
Output: outputs/rq2_matched_fake_null_1000.csv

Run:  python scripts/rq2_matched_content.py
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
N_PERM, SEED = 1000, 42


def load_train(p):
    X, y = [], []
    for r in csv.DictReader(open(p, encoding="utf-8")):
        try:
            X.append([float(r[c]) for c in FEAT]); y.append(float(r["prosodic_score"]))
        except (KeyError, ValueError):
            pass
    return np.array(X), np.array(y)


def load_X(p, keep=None):
    S, U, X = [], [], []
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if keep is not None and r["utt_id"] not in keep:
            continue
        S.append(r["speaker"]); U.append(r["utt_id"]); X.append([float(r[c]) for c in FEAT])
    return np.array(S), U, np.array(X)


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
Sn, Un, Xn = load_X(OUT / "X14_native_fresh.csv")
Sv, Uv, Xv = load_X(OUT / "X14_vn_fresh.csv", keep=set(Un))
print(f"native {len(Xn)} utts ({min(Un)}–{max(Un)}); VN matched {len(Xv)} utts "
      f"({', '.join(f'{s} {n}' for s, n in zip(*np.unique(Sv, return_counts=True)))})")

m = rf(); m.fit(X, y)
pn, pv = m.predict(Xn), m.predict(Xv)
gap = spk_means(Sn, pn).mean() - spk_means(Sv, pv).mean()
real_utt = cohens_d(pn, pv); real_spk = cohens_d(spk_means(Sn, pn), spk_means(Sv, pv))
print(f"REAL model (matched): raw gap {gap:+.3f} points | utterance d = {real_utt:+.3f} | speaker d = {real_spk:+.3f}")

rng = np.random.default_rng(SEED); rows = []
for i in range(N_PERM):
    ys = y.copy(); rng.shuffle(ys)
    mf = rf(); mf.fit(X, ys)
    fn, fv = mf.predict(Xn), mf.predict(Xv)
    rows.append((i + 1, cohens_d(fn, fv), cohens_d(spk_means(Sn, fn), spk_means(Sv, fv))))
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{N_PERM}", flush=True)
fu = np.array([r[1] for r in rows]); fs = np.array([r[2] for r in rows])
k_u = int(np.sum(np.abs(fu) >= abs(real_utt))); k_s = int(np.sum(np.abs(fs) >= abs(real_spk)))
with open(OUT / "rq2_matched_fake_null_1000.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["run", "fake_d_utterance", "fake_d_speaker"])
    w.writerows([(r[0], f"{r[1]:.6f}", f"{r[2]:.6f}") for r in rows])
print("=" * 72)
print(f"MATCHED CONTENT, N = {N_PERM} fake models, seed {SEED}")
print(f"Utterance level: real |d| = {abs(real_utt):.3f}; k = {k_u}; p = {(k_u + 1) / (N_PERM + 1):.3f}")
print(f"Speaker level  : real |d| = {abs(real_spk):.3f}; k = {k_s}; p = {(k_s + 1) / (N_PERM + 1):.3f}")
print("Direction native > VN:", "yes" if real_spk > 0 else "no")

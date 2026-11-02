"""fig_two_nulls.py — Figure 5.2: permutation nulls for RQ1 and RQ2.
(a) shuffled-label null (outputs/pcc_null_1000.csv, pcc_permutation_null.py);
(b) fake-model null at speaker level (outputs/transfer_fake_null_1000.csv,
    transfer_reality_check_v2.py).
The observed values are recomputed with the same models and splits.
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from paths import OUTPUTS as OUT, FIGURES as FIG
FIG.mkdir(exist_ok=True)
FEAT = ["mean_duration_ratio", "std_duration_ratio", "max_duration_ratio", "mean_f0_ratio",
        "std_f0_ratio", "mean_intensity_diff", "std_intensity_diff", "n_multisyllabic_words",
        "f0_range_hz", "speaking_rate", "silence_proportion", "npvi", "percent_v", "varcov"]


def rf():
    return RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=10,
                                 random_state=42, n_jobs=-1)


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / pp if pp > 0 else 0.0


# ---- data
X, y = [], []
for r in csv.DictReader(open(OUT / "speechocean762_full_per_utt_v3.csv", encoding="utf-8")):
    try:
        X.append([float(r[c]) for c in FEAT]); y.append(float(r["prosodic_score"]))
    except (KeyError, ValueError):
        pass
X, y = np.array(X), np.array(y)


def load_X(p):
    S, Z = [], []
    for r in csv.DictReader(open(p, encoding="utf-8")):
        S.append(r["speaker"]); Z.append([float(r[c]) for c in FEAT])
    return np.array(S), np.array(Z)


def spk_means(S, p):
    g = defaultdict(list)
    for s, v in zip(S, p):
        g[s].append(v)
    return np.array([np.mean(g[s]) for s in sorted(g)])


# ---- observed RQ1 PCC (same split as pcc_permutation_null.py)
Xtv, Xte, ytv, yte = train_test_split(X, y, test_size=0.15, random_state=42)
Xtr, _, ytr, _ = train_test_split(Xtv, ytv, test_size=0.176, random_state=42)
m = rf(); m.fit(Xtr, ytr); real_pcc = pearsonr(yte, m.predict(Xte))[0]
null_pcc = np.loadtxt(OUT / "pcc_null_1000.csv", delimiter=",")
k1 = int(np.sum(null_pcc >= real_pcc)); n1 = len(null_pcc); p1 = (k1 + 1) / (n1 + 1)

# ---- observed RQ2 speaker-level |d| (same model as transfer_reality_check_v2.py)
Sn, Xn = load_X(OUT / "X14_native_fresh.csv"); Sv, Xv = load_X(OUT / "X14_vn_fresh.csv")
m2 = rf(); m2.fit(X, y)
real_d = abs(cohens_d(spk_means(Sn, m2.predict(Xn)), spk_means(Sv, m2.predict(Xv))))
fake = np.abs(np.array([float(r["fake_d_speaker"]) for r in
                        csv.DictReader(open(OUT / "transfer_fake_null_1000.csv", encoding="utf-8"))]))
k2 = int(np.sum(fake >= real_d)); n2 = len(fake); p2 = (k2 + 1) / (n2 + 1)

print(f"RQ1: PCC {real_pcc:.3f}; {k1}/{n1} nulls >= observed; p = {p1:.4f}")
print(f"RQ2: |d| {real_d:.2f}; {k2}/{n2} fakes >= observed; p = {p2:.3f}")

# ---- plot
INK, MUTED, GRID, BAR = "#1f2328", "#59636e", "#e6e8eb", "#b6bec8"
C1, C2 = "#1f3a5f", "#b8612a"
plt.rcParams.update({"font.family": "serif", "font.size": 8.5})
fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0))

ax = axes[0]
ax.hist(null_pcc, bins=30, color=BAR, edgecolor="white", linewidth=1)
ax.axvline(real_pcc, color=C1, linewidth=1.6)
ax.text(real_pcc - 0.02, ax.get_ylim()[1] * 0.97, f"observed\nPCC = {real_pcc:.3f}",
        ha="right", va="top", color=INK, fontsize=8.5)
ax.text(real_pcc - 0.02, ax.get_ylim()[1] * 0.72,
        f"{k1}/{n1} ≥ observed\n$p$ < 0.001" if k1 == 0 else
        f"{k1}/{n1} ≥ observed\n$p$ = {p1:.3f}",
        ha="right", va="top", color=MUTED, fontsize=8)
ax.set_title("(a) RQ1: shuffle-label null", color=INK, fontsize=9, loc="left")
ax.set_xlabel("Held-out test PCC")
ax.set_ylabel("Number of permutations")

ax = axes[1]
xmax = max(fake.max(), real_d) * 1.02
ax.hist(fake, bins=np.linspace(0, xmax, 36), color=BAR, edgecolor="white", linewidth=1)
ax.axvline(real_d, color=C2, linewidth=1.6)
ax.text(0.98, 0.97, f"observed |$d$| = {real_d:.2f}", transform=ax.transAxes,
        ha="right", va="top", color=INK, fontsize=8.5)
ax.text(0.98, 0.84, f"{k2}/{n2} ≥ observed\n$p$ = {p2:.2f}", transform=ax.transAxes,
        ha="right", va="top", color=MUTED, fontsize=8)
ax.set_title("(b) RQ2: fake-model null", color=INK, fontsize=9, loc="left")
ax.set_xlabel("Speaker-level |$d$|, native vs Vietnamese")
ax.set_ylabel("Number of fake models")

for ax in axes:
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#8c959f")
    ax.tick_params(colors=MUTED)

fig.tight_layout()
out = FIG / "Figure_5.2_permutation_nulls.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print("Saved:", out)

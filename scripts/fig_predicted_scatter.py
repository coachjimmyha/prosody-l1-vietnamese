"""fig_predicted_scatter.py — Figure 5.1: predicted versus human prosody score on the held-out test set
(canonical model and split, as in final_evaluation_v3.py).
"""
import os; from paths import REPO_ROOT; os.chdir(REPO_ROOT); os.makedirs("figures", exist_ok=True)
import csv
import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from fig_style import setup, clean, BLUE, ORANGE, INK

FEAT = ["mean_duration_ratio", "std_duration_ratio", "max_duration_ratio", "mean_f0_ratio", "std_f0_ratio",
        "mean_intensity_diff", "std_intensity_diff", "n_multisyllabic_words", "f0_range_hz", "speaking_rate",
        "silence_proportion", "npvi", "percent_v", "varcov"]
rows = list(csv.DictReader(open("outputs/speechocean762_full_per_utt_v3.csv", encoding="utf-8")))
X = np.array([[float(r[c]) for c in FEAT] for r in rows]); y = np.array([float(r["prosodic_score"]) for r in rows])
Xtv, Xte, ytv, yte = train_test_split(X, y, test_size=0.15, random_state=42)
Xtr, _, ytr, _ = train_test_split(Xtv, ytv, test_size=0.176, random_state=42)
m = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1).fit(Xtr, ytr)
pred = m.predict(Xte)
pcc = pearsonr(pred, yte)[0]; rmse = mean_squared_error(yte, pred) ** 0.5; r2 = r2_score(yte, pred)

setup()
fig, ax = plt.subplots(figsize=(5.8, 5.6))
rng = np.random.RandomState(0); yj = yte + rng.uniform(-0.15, 0.15, len(yte))   # jitter: scores are integers
ax.scatter(yj, pred, s=16, alpha=0.35, color=BLUE, edgecolors="none", zorder=3)
lims = [min(yte.min(), pred.min()) - 0.3, 10.3]
ax.plot(lims, lims, color=ORANGE, ls="--", lw=1.4, zorder=4, label="Perfect agreement")
b, a = np.polyfit(yte, pred, 1); xs = np.array(lims)
ax.plot(xs, a + b * xs, color=INK, lw=1.6, zorder=5, label="Least-squares fit")
ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
ax.set_xlabel("Human prosody score (0–10; jittered)"); ax.set_ylabel("Predicted prosody score")
ax.text(0.04, 0.96, f"PCC = {pcc:.3f}\nRMSE = {rmse:.2f}\nR² = {r2:.2f}\nn = {len(yte)}", transform=ax.transAxes,
        va="top", fontsize=10, color=INK, bbox=dict(boxstyle="round", fc="white", ec="#d3dae2"))
ax.legend(loc="lower right", fontsize=9, frameon=False)
clean(ax, "both")
fig.tight_layout()
out = "figures/Figure_5.1_predicted_scatter.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"saved {out}  PCC={pcc:.3f} RMSE={rmse:.2f} R2={r2:.2f}")

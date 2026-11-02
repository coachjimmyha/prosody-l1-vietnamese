"""fig_shap_beeswarm_v2.py — Figure 5.6: SHAP beeswarm for the Random Forest trained on all 2,556
Speechocean762 utterances (TreeExplainer; base value = training mean prosody score).
"""
import os as _os; from paths import REPO_ROOT; _os.chdir(REPO_ROOT); _os.makedirs("figures", exist_ok=True)
import csv, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestRegressor
import shap

from fig_style import setup, BLUE, ORANGE, INK, MUTED
setup()
HOUSE = LinearSegmentedColormap.from_list("house", [BLUE, "#b9c2cc", ORANGE])  # low->high feature value

from pathlib import Path as _P
BASE_DIR = "outputs"
FEAT=["mean_duration_ratio","std_duration_ratio","max_duration_ratio","mean_f0_ratio","std_f0_ratio",
 "mean_intensity_diff","std_intensity_diff","n_multisyllabic_words","f0_range_hz","speaking_rate",
 "silence_proportion","npvi","percent_v","varcov"]
from fig_style import NICE  # shared feature labels (intensity = dB difference)
labels=[NICE[f] for f in FEAT]

def load(p, cols, tgt=None):
    X=[]; y=[]
    for r in csv.DictReader(open(p, encoding="utf-8")):
        try:
            X.append([float(r[c]) for c in cols])
            if tgt: y.append(float(r[tgt]))
        except (KeyError, ValueError): pass
    return np.array(X), (np.array(y) if tgt else None)

Xtr, ytr = load(BASE_DIR+"/speechocean762_full_per_utt_v3.csv", FEAT, "prosodic_score")
m = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1)
m.fit(Xtr, ytr)
expl = shap.TreeExplainer(m)
sv = expl.shap_values(Xtr)
base = float(np.ravel(expl.expected_value)[0])
print("base value =", round(base,3), "| n =", len(ytr))

fig = plt.figure(figsize=(6.3, 4.6))   # native print size: 16 cm wide
shap.summary_plot(sv, Xtr, feature_names=labels, show=False, cmap=HOUSE, rng=np.random.default_rng(42),
                  plot_size=None, color_bar=True, sort=True)
ax = plt.gca()
ax.set_xlabel("SHAP value (impact on predicted prosody score)", fontsize=9, color=INK)
for lab in ax.get_yticklabels(): lab.set_color(INK); lab.set_fontsize(8.5)
ax.tick_params(axis="x", labelsize=8.5)
ax.axvline(0, color="#94a0ad", lw=0.8, zorder=0)
cb = fig.axes[-1]
cb.set_ylabel("Feature value", fontsize=8.5, color=INK)
cb.tick_params(labelsize=8.5)

plt.tight_layout()
ax.tick_params(axis="x", labelsize=8.5)
for lab in ax.get_xticklabels(): lab.set_fontsize(8.5)
out="figures/Figure_5.6_shap_beeswarm.png"
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white"); print("saved", out)

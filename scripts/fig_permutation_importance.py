"""fig_permutation_importance.py — Figure 5.4: permutation importance of the 14 features on the held-out
test set (R² drop, 30 repeats; error bars = SD over repeats).
Reads outputs/permutation_importance_v3.csv (thesis_robustness_checks.py).
"""
import os; from paths import REPO_ROOT; os.chdir(REPO_ROOT); os.makedirs("figures", exist_ok=True)
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from fig_style import setup, clean, NICE, FAMILY, FAMILY_COLOUR, INK

rows = list(csv.DictReader(open("outputs/permutation_importance_v3.csv", encoding="utf-8")))
rows.sort(key=lambda r: -float(r["importance_mean_r2_drop"]))
feat = [r["feature"] for r in rows]
val = np.array([float(r["importance_mean_r2_drop"]) for r in rows])
sd = np.array([float(r["importance_std"]) for r in rows])

setup()
fig, ax = plt.subplots(figsize=(6.3, 4.9))
ys = np.arange(len(feat))[::-1]
ax.barh(ys, val, height=0.66, color=[FAMILY_COLOUR[FAMILY[f]] for f in feat], edgecolor="white", linewidth=2)
ax.errorbar(val, ys, xerr=sd, fmt="none", ecolor="#3b4650", elinewidth=1, capsize=2.5)
for y, v, s in zip(ys, val, sd):
    ax.text(max(v + s, 0) + 0.01, y, f"{v:.3f}", va="center", fontsize=8.5, color=INK)
ax.set_yticks(ys); ax.set_yticklabels([NICE[f] for f in feat], fontsize=9, color=INK)
ax.set_xlabel("Permutation importance\n(drop in test R² when the feature is shuffled; 30 repeats)", fontsize=9)
ax.set_xlim(0, max(val + sd) * 1.12)
clean(ax, "x")
ax.legend(handles=[Patch(color=FAMILY_COLOUR[k], label=k) for k in ["Utterance-level", "Rhythm", "Stress"]],
          title="Feature group", loc="lower right", frameon=False, fontsize=8.5, title_fontsize=8.5)
fig.tight_layout()
out = "figures/Figure_5.4_permutation_importance.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white"); print("saved", out)

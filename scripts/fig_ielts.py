"""fig_ielts.py — Figure 1.1: mean IELTS band scores of Vietnamese candidates by skill, 2023–2024.
Values typed from IELTS (2024) Test statistics, https://ielts.org/researchers/our-research/test-statistics
"""
import os as _os; from paths import REPO_ROOT; _os.chdir(REPO_ROOT); _os.makedirs("figures", exist_ok=True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fig_style import setup, clean, BLUE, ORANGE, INK, MUTED
setup()
skills = ["Reading", "Listening", "Writing", "Speaking"]
score = [6.4, 6.3, 6.0, 5.7]
fig, ax = plt.subplots(figsize=(6.3, 2.6))
y = list(range(len(skills)))[::-1]
cols = [BLUE, BLUE, BLUE, ORANGE]
ax.barh(y, score, height=0.6, color=cols, edgecolor="white", linewidth=2)
for yi, s in zip(y, score):
    ax.text(s + 0.08, yi, f"{s:.1f}", va="center", ha="left", fontsize=9, color=INK, weight="bold")
ax.set_yticks(y); ax.set_yticklabels(skills, fontsize=9, color=INK)
ax.set_xlim(0, 9); ax.set_xticks(range(0, 10))
ax.set_xlabel("Mean band score (IELTS 0–9 scale)", fontsize=9)
clean(ax, grid_axis="x")
fig.text(0.01, 0.01, "Source: IELTS (2024), Test statistics.", fontsize=7.8, color=MUTED, style="italic")
plt.tight_layout(rect=(0, 0.06, 1, 1))
out = "figures/Figure_1.1_ielts.png"
fig.savefig(out, dpi=300, facecolor="white"); print("saved", out)

"""fig_field_map.py — Figure 2.2: pronunciation-assessment studies by year and degree of interpretability.
Positions follow the author's classification in Chapter 2; small vertical offsets within a band only
separate overlapping labels. This study is placed at 2026.
"""
import os as _os; from paths import REPO_ROOT; _os.chdir(REPO_ROOT); _os.makedirs("figures", exist_ok=True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from fig_style import setup, clean, BLUE, ORANGE, AQUA, INK, MUTED
setup()
# (label, year, y, kind, vn_focus, label dx, label dy, ha)
P = [("Witt and Young (GOP)", 2000.0, 1.13, "seg", False, 0, 0.09, "center"),
     ("Chen and Wang", 2010.0, 1.25, "sup", False, 0, 0.09, "center"),
     ("Ferrer et al.", 2015.0, 1.00, "sup", False, 0, 0.09, "center"),
     ("Kang and Johnson", 2018.0, 2.00, "sup", False, 0, 0.1, "center"),
     ("Bamdev et al.", 2023.0, 1.87, "sup", False, -0.35, 0, "right"),
     ("Cai et al.", 2025.0, 1.80, "sup", False, 0.35, -0.05, "left"),
     ("Jahanbin", 2025.0, 1.60, "sup", False, 0.35, -0.05, "left"),
     ("Phung et al.", 2022.0, 0.50, "seg", True, 0, 0.12, "center"),
     ("Li et al.", 2018.0, 0.30, "sup", False, -0.35, 0, "right"),
     ("Ruan et al.", 2019.0, 0.15, "sup", False, -0.35, 0, "right"),
     ("Lin et al.", 2021.0, 0.25, "sup", False, 0, 0.09, "center"),
     ("Nguyen and Tran", 2024.3, 0.20, "seg", True, 0.4, 0.04, "left"),
     ("Yan et al.", 2024.0, 0.08, "sup", False, -0.3, 0.05, "right"),
     ("Gong et al. (GOPT)", 2022.0, 0.05, "sup", False, -0.35, -0.03, "right"),
     ("Chao and Chen", 2025.3, 0.00, "sup", False, 0, -0.12, "center")]
fig, ax = plt.subplots(figsize=(6.3, 4.4))
ax.add_patch(plt.Rectangle((2016.6, 1.4), 13.4, 1.05, fc="#e8f6f0", ec="none", zorder=0))
ax.text(2016.9, 2.33, "interpretable and suprasegmental\n(target region)", fontsize=8, color=INK, style="italic", va="center")
for lab, x, y, k, vn, dx, dy, ha in P:
    c = BLUE if k == "sup" else ORANGE
    if vn:
        ax.scatter(x, y, s=150, facecolor="none", edgecolor=AQUA, linewidth=1.8, zorder=3)
    ax.scatter(x, y, s=42, color=c, edgecolor="white", linewidth=0.8, zorder=4)
    ax.text(x + dx, y + dy, lab, fontsize=8, color=INK, ha=ha, va="center", zorder=5)
ax.scatter(2026, 2.05, s=260, marker="*", color=AQUA, edgecolor=INK, linewidth=0.8, zorder=6)
ax.text(2026.5, 2.05, "This study", fontsize=9, weight="bold", color=INK, ha="left", va="center")
ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["black-box", "partial", "white-box"], fontsize=9, color=INK)
ax.set_ylim(-0.25, 2.5); ax.set_xlim(1998, 2030); ax.set_xticks(range(2000, 2030, 5))
ax.set_xlabel("Year of publication", fontsize=9); ax.set_ylabel("Interpretability", fontsize=9)
clean(ax, grid_axis="y")
leg = [Line2D([], [], marker="o", ls="", color=BLUE, label="Suprasegmental"),
       Line2D([], [], marker="o", ls="", color=ORANGE, label="Segmental"),
       Line2D([], [], marker="o", ls="", mfc="none", mec=AQUA, mew=1.8, ms=10, label="L1-Vietnamese focus"),
       Line2D([], [], marker="*", ls="", color=AQUA, mec=INK, ms=13, label="This study")]
ax.legend(handles=leg, loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=8, frameon=True, edgecolor="#d3dae2")
plt.tight_layout()
out = "figures/Figure_2.2_field_map.png"
fig.savefig(out, dpi=300, facecolor="white"); print("saved", out)

"""fig_research_gap.py — Figure 2.4: the research gap as the intersection of three requirements
(diagram, studies from Table 2.3). Drawn at its print width (12.5 cm).
"""
import os as _os; from paths import REPO_ROOT; _os.chdir(REPO_ROOT); _os.makedirs("figures", exist_ok=True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from fig_style import setup, BLUE, ORANGE, AQUA, INK, MUTED
setup()
FW, FH = 4.92, 4.68                                  # 12.5 cm x 11.9 cm
fig = plt.figure(figsize=(FW, FH)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.set_aspect("equal")
R, D = 1.36, 1.46                                    # radius, horizontal distance between A and B
cx, ya, yc = FW / 2, 2.95, 1.68
A, B, C = (cx - D / 2, ya), (cx + D / 2, ya), (cx, yc)
for (x, y), col in [(A, BLUE), (B, AQUA), (C, ORANGE)]:
    ax.add_patch(Circle((x, y), R, fc=col, ec=col, alpha=0.13, lw=0))
    ax.add_patch(Circle((x, y), R, fc="none", ec=col, alpha=0.9, lw=1.2))
LAB = dict(ha="center", va="center", fontsize=9.5, weight="bold", color=INK, linespacing=1.15)
ax.text(0.78, 4.50, "Automatic and\ninterpretable ML", **LAB)
ax.text(FW - 0.78, 4.50, "Prosody detection\nand localisation", **LAB)
ax.text(cx, 0.14, "L1-Vietnamese English", **LAB)
def t(x, y, head, body):
    ax.text(x, y, head, ha="center", va="bottom", fontsize=8, weight="bold", color=INK, linespacing=1.1)
    ax.text(x, y - 0.03, body, ha="center", va="top", fontsize=7.5, color=MUTED, linespacing=1.2)
t(1.04, 3.22, "general XAI", "Molnar (2019);\nLundberg and Lee\n(2017); Ribeiro\net al. (2016)")
t(3.86, 3.22, "automatic prosody,\nother L1s", "Li et al. (2018);\nLin et al. (2021);\nChen and Wang (2010);\nGong et al. (2022);\nYan et al. (2024)")
t(cx, 3.66, "interpretable\nscoring", "Kang and\nJohnson (2018);\nCai et al. (2025)")
t(1.63, 2.20, "segmental\nMDD (VN)", "Phung et al.\n(2022); Dang and\nDang Thi (2017)")
t(3.36, 2.20, "no\nautomation", "descriptive\nonly")
t(cx, 1.08, "descriptive VN prosody", "Nguyễn (2003); Nguyễn,\nIngram and Pensalfini (2008);\nMixdorff and Ingram (2009)")
ax.add_patch(Circle((cx, 2.48), 0.34, fc=INK, ec="white", lw=1.3, zorder=5))
ax.text(cx, 2.48, "THIS\nSTUDY", ha="center", va="center", fontsize=8.5, weight="bold", color="white", zorder=6)
out = "figures/Figure_2.4_research_gap.png"
fig.savefig(out, dpi=300, facecolor="white"); print("saved", out)

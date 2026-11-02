"""fig_lit_search.py — Figure 2.1: literature search and screening procedure (diagram, no data).
"""
import os as _os; from paths import REPO_ROOT; _os.chdir(REPO_ROOT); _os.makedirs("figures", exist_ok=True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from fig_style import setup, BLUE, ORANGE, AQUA, INK, MUTED
setup()
W, H = 6.3, 7.4
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
ax.set_xlim(0, W); ax.set_ylim(0, H)

def box(x, y, w, h, title, body, ec, fc="white", tc=INK, bc=INK, dark=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=INK if dark else fc, ec=INK if dark else ec, lw=1.4))
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="center", fontsize=10.5, weight="bold",
            color="white" if dark else tc)
    ax.text(x + w / 2, y + h - 0.42, body, ha="center", va="top", fontsize=8.6, linespacing=1.3,
            color="#e3e7ec" if dark else bc)

def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#8a96a3", lw=1.1, mutation_scale=10))

L, R, FW, HW, G = 0.25, 6.05, 5.8, 2.82, 0.16
box(L, 6.45, FW, 0.85, "Identification: sources",
    "Google Scholar · LJMU Discovery (Primo), federating\nIEEE Xplore · ProQuest · Taylor & Francis · Web of Science · arXiv", BLUE)
box(L, 5.0, HW, 1.1, "Vietnamese-focused searches",
    "to establish novelty\nL1-Vietnamese × corpus / rhythm (nPVI) /\nlexical stress / mispronunciation detection", BLUE)
box(L + HW + G, 5.0, HW, 1.1, "Method-focused searches",
    "to locate the foundations\nautomatic lexical-stress detection ·\nL2 rhythm metrics · interpretable ML", BLUE)
arrow(3.15, 6.45, 1.66, 6.1); arrow(3.15, 6.45, 4.64, 6.1)
box(L, 3.75, FW, 0.85, "Screening: four criteria per record",
    "task level (segmental vs prosodic) · dataset ·\nmethod (automatic vs descriptive) · L1-Vietnamese focus", BLUE)
arrow(1.66, 5.0, 2.6, 4.6); arrow(4.64, 5.0, 3.7, 4.6)
box(L, 2.55, FW, 0.85, "Exclusions",
    "records from unrelated fields sharing common Vietnamese surnames ·\nsearch stopped when consecutive pages returned no new relevant work", ORANGE, fc="#fdf1ea")
arrow(3.15, 3.75, 3.15, 3.4)
box(L, 1.1, HW, 1.1, "Vietnamese-focused: outcome",
    "a substantial descriptive and\nperceptual literature on Vietnamese-\naccented rhythm and stress, but no\nautomatic, interpretable system", AQUA, fc="#e8f6f0")
box(L + HW + G, 1.1, HW, 1.1, "Method-focused: outcome",
    "the interpretable-modelling and\nrhythm-metric foundations\non which this study builds", AQUA, fc="#e8f6f0")
arrow(3.15, 2.55, 1.66, 2.2); arrow(3.15, 2.55, 4.64, 2.2)
box(L, 0.05, FW, 0.8, "Identified gap",
    "the errors are well documented, but no interpretable automatic system\ndetects and localises them for L1-Vietnamese learners (Table 2.3)", INK, dark=True)
arrow(1.66, 1.1, 2.6, 0.85); arrow(4.64, 1.1, 3.7, 0.85)
out = "figures/Figure_2.1_lit_search.png"
fig.savefig(out, dpi=300, facecolor="white"); print("saved", out)

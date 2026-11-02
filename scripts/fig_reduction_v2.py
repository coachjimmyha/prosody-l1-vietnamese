"""fig_reduction_v2.py — Figure 5.7: function-word reduction ratio by L1 group (n = 4 speakers per group).
Reads outputs/reduction_full.csv; the annotated statistics are those of reduction_stats.py.
"""
import os as _os; from paths import REPO_ROOT; _os.chdir(REPO_ROOT); _os.makedirs("figures", exist_ok=True)
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from fig_style import setup, clean, AQUA, ORANGE, NEUTRAL, INK, MUTED, GRID
setup()

# ---- load data ----
csv_path = Path("outputs/reduction_full.csv")
if not csv_path.exists():
    csv_path = Path("outputs/reduction_full.csv")
rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
vals = defaultdict(list)
for r in rows:
    vals[r["group"]].append(float(r["reduction"]))

ORDER = ["Native","Korean","Spanish","Vietnamese","Chinese"]   # ascending mean
COL = {"Native":AQUA, "Vietnamese":ORANGE, "Korean":NEUTRAL, "Spanish":NEUTRAL, "Chinese":NEUTRAL}

fig, ax = plt.subplots(figsize=(6.3, 4.0))   # native print size: 16 cm wide
rng = np.random.default_rng(7)
native_mean = float(np.mean(vals["Native"]))

ax.axhline(native_mean, color=AQUA, lw=1.0, ls="--", zorder=1)
ax.text(4.42, native_mean, "native\nmean", va="center", ha="left", fontsize=8, color=INK, fontstyle="italic")

for i, g in enumerate(ORDER):
    v = np.array(vals[g]); c = COL[g]
    jit = (rng.random(len(v))-0.5)*0.22
    ax.scatter(i+jit, v, s=34, color=c, alpha=0.75, edgecolor="white", linewidth=0.8, zorder=3)
    m = float(v.mean())
    ax.plot([i-0.24, i+0.24], [m, m], color=INK if g not in ("Native","Vietnamese") else c, lw=2.4,
            solid_capstyle="round", zorder=4)
    ax.text(i+0.28, m + (0.018 if g == "Native" else 0), f"{m:.2f}", ha="left", va="center", fontsize=8.5,
            fontweight="bold", color=INK,
            bbox=dict(fc="white", ec="none", pad=0.6))

ax.set_xticks(range(5))
ax.set_xticklabels(ORDER, fontsize=9, color=INK)
ax.set_xlim(-0.5, 4.95)
ax.set_ylim(0.58, 1.10)
ax.set_ylabel("Function-word reduction ratio\n(lower = more reduced)", fontsize=9)
ax.text(0.0, 1.02, r"Native vs Vietnamese: $d$ = $-$3.58, exact $p$ = 0.029;   native vs pooled L2: $d$ = $-$2.44",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color=MUTED)
clean(ax, grid_axis="y")

leg=[Line2D([0],[0],marker="o",color="w",markerfacecolor=AQUA,markersize=7,label="Native (reference)"),
     Line2D([0],[0],marker="o",color="w",markerfacecolor=ORANGE,markersize=7,label="Vietnamese"),
     Line2D([0],[0],marker="o",color="w",markerfacecolor=NEUTRAL,markersize=7,label="Other L2"),
     Line2D([0],[0],color=INK,lw=2.4,label="Group mean")]
ax.legend(handles=leg, loc="upper left", fontsize=8, frameon=True, framealpha=0.95, edgecolor="#d3dae2")

plt.tight_layout()
out="figures/Figure_5.7_reduction.png"
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white"); print("saved", out)

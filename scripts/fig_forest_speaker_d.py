"""fig_forest_speaker_d.py — Figure 5.3: speaker-level native–Vietnamese effect sizes (n = 4 per group)
with 95% speaker bootstrap CIs and exact permutation p.
Reads outputs/speaker_level_tests.csv (speaker_bootstrap.py) and
reduction_stats_summary.csv (reduction_stats.py). Filled markers: exact p <= 0.05.
"""
import os; from paths import REPO_ROOT; os.chdir(REPO_ROOT); os.makedirs("figures", exist_ok=True)
import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from fig_style import setup, clean, BLUE, NEUTRAL, INK, MUTED

LABEL = {"reduction": "Function-word reduction", "speaking_rate": "Speaking rate",
         "duration_ratio": "Duration ratio (stress)", "silence_prop": "Silence proportion",
         "nPVI": "nPVI", "VarcoV": "VarcoV", "percent_V": "%V", "f0_range_st": "F0 range (semitones)"}
rows = {r["feature"]: r for f in ["outputs/speaker_level_tests.csv",
                                  "outputs/reduction_stats_summary.csv"]
        for r in csv.DictReader(open(f, encoding="utf-8"))}
order = sorted(LABEL, key=lambda k: -abs(float(rows[k]["d_native_minus_vn"])))

setup()
XLO, XHI = -8, 6
fig, ax = plt.subplots(figsize=(6.3, 4.5))
for i, k in enumerate(order):
    y = len(order) - i
    r = rows[k]; d, lo, hi, p = (float(r[c]) for c in ["d_native_minus_vn", "boot_lo", "boot_hi", "exact_perm_p"])
    sig = p <= 0.05
    col = BLUE if sig else NEUTRAL
    ax.plot([max(lo, XLO), min(hi, XHI)], [y, y], color=col, linewidth=2.2, solid_capstyle="round", zorder=2)
    if lo < XLO:
        ax.annotate("", xy=(XLO, y), xytext=(XLO + 0.6, y),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2), zorder=2)
        ax.text(XLO + 0.1, y - 0.3, f"lower bound {lo:.1f}".replace("-", "−"), fontsize=8, color=MUTED, style="italic", va="center")
    ax.plot(d, y, "o", markersize=7, zorder=3, markeredgewidth=2, markeredgecolor=col,
            markerfacecolor=col if sig else "white")
    ax.text(XHI + 0.3, y, f"d = {d:+.2f}\np = {p:.3f}".replace("-", "−"), va="center", fontsize=8, color=INK, linespacing=1.05)
ax.axvline(0, color=INK, linewidth=1.1, zorder=1)
ax.set_yticks(range(len(order), 0, -1)); ax.set_yticklabels([LABEL[k] for k in order], color=INK, fontsize=9)
ax.set_xlim(XLO, XHI); ax.set_ylim(0.4, len(order) + 0.6)
ax.set_xlabel("Cohen's d (native − Vietnamese), speaker level,\nwith 95% bootstrap CI", fontsize=8.5)
ax.text(XLO + 0.1, len(order) + 0.45, "← Vietnamese higher", fontsize=8, color=MUTED, style="italic", va="center")
ax.text(XHI - 0.1, len(order) + 0.45, "native higher →", fontsize=8, color=MUTED, style="italic", va="center", ha="right")
ax.text(XHI + 0.3, len(order) + 0.45, "exact p", fontsize=8, color=MUTED, va="center")
clean(ax, "x")
ax.legend(handles=[Line2D([], [], marker="o", color=BLUE, markerfacecolor=BLUE, lw=2.2, label="exact p ≤ 0.05"),
                   Line2D([], [], marker="o", color=NEUTRAL, markerfacecolor="white", markeredgewidth=2,
                          lw=2.2, label="exact p > 0.05")],
          loc="lower left", frameon=False, fontsize=8)
fig.tight_layout()
out = "figures/Figure_5.3_speaker_forest.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white"); print("saved", out)

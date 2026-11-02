"""fig_importance_vs_separation.py — Figure 5.5: predictive importance (a) versus native–Vietnamese
separation (b).
(a) permutation importance (permutation_importance_v3.csv); (b) speaker-level |Cohen's d| and exact p
(speaker_level_tests.csv, reduction_stats_summary.csv). Function-word reduction is not a model
feature. F0 range: model feature = f0_range_hz; separation = robust semitone measure.
"""
import os; from paths import REPO_ROOT; os.chdir(REPO_ROOT); os.makedirs("figures", exist_ok=True)
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from fig_style import setup, clean, BLUE, ORANGE, NEUTRAL, INK, MUTED

imp = {r["feature"]: float(r["importance_mean_r2_drop"])
       for r in csv.DictReader(open("outputs/permutation_importance_v3.csv", encoding="utf-8"))}
sep = {r["feature"]: r for f in ["outputs/speaker_level_tests.csv", "outputs/reduction_stats_summary.csv"]
       for r in csv.DictReader(open(f, encoding="utf-8"))}
# (label, importance key or None, separation key)
ROWS = [("Function-word reduction", None, "reduction"), ("Speaking rate", "speaking_rate", "speaking_rate"),
        ("Duration ratio (stress)", "mean_duration_ratio", "duration_ratio"),
        ("Silence proportion", "silence_proportion", "silence_prop"), ("nPVI", "npvi", "nPVI"),
        ("VarcoV", "varcov", "VarcoV"), ("F0 range", "f0_range_hz", "f0_range_st"), ("%V", "percent_v", "percent_V")]
ROWS.sort(key=lambda r: -abs(float(sep[r[2]]["d_native_minus_vn"])))   # largest separation first

setup()
fig, (aL, aR) = plt.subplots(1, 2, figsize=(6.3, 4.3), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1.15]})
ys = np.arange(len(ROWS))[::-1]
for y, (lab, ik, sk) in zip(ys, ROWS):
    if ik is None:
        aL.text(0.004, y, "not a model\nfeature", va="center", fontsize=8, color=MUTED, style="italic", linespacing=0.95)
    else:
        aL.barh(y, imp[ik], height=0.62, color=BLUE, edgecolor="white", linewidth=2)
        aL.text(imp[ik] + 0.01, y, f"{imp[ik]:.3f}", va="center", fontsize=8.5, color=INK)
    d, p = abs(float(sep[sk]["d_native_minus_vn"])), float(sep[sk]["exact_perm_p"])
    sig = p <= 0.05
    aR.barh(y, d, height=0.62, color=ORANGE if sig else "white", edgecolor=ORANGE if sig else NEUTRAL,
            linewidth=1.6)
    aR.text(d + 0.08, y, f"{d:.2f} (p = {p:.3f})", va="center", fontsize=8, color=INK)
aL.set_yticks(ys); aL.set_yticklabels([r[0] for r in ROWS], color=INK, fontsize=9)
aL.set_title("(a) Predicts the score", loc="left", fontsize=9.5, color=INK)
aR.set_title("(b) Separates native and Vietnamese", loc="left", fontsize=9.5, color=INK)
aL.set_xlabel("Permutation importance\n(drop in test R²)", fontsize=8.5)
aR.set_xlabel("|Cohen's d|, speaker level\n(n = 4 per group)", fontsize=8.5)
aL.set_xlim(0, 0.75); aR.set_xlim(0, 6.6)
for a in (aL, aR):
    clean(a, "x")
fig.legend(handles=[Patch(facecolor=ORANGE, edgecolor=ORANGE, label="(b) exact p ≤ 0.05"),
                    Patch(facecolor="white", edgecolor=NEUTRAL, label="(b) exact p > 0.05")],
           loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.6, 0.0))
fig.tight_layout(rect=(0, 0.06, 1, 1))
out = "figures/Figure_5.5_importance_vs_separation.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white"); print("saved", out)

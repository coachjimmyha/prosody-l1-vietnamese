"""fig_feature_overview.py — Figure 3.5: the fourteen features grouped into three families
(labels as in Table 3.2 and fig_style.NICE).
"""
import os as _os; from paths import REPO_ROOT; _os.chdir(REPO_ROOT); _os.makedirs("figures", exist_ok=True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from fig_style import setup, FAMILY_COLOUR, INK, MUTED
setup()

FAM = [
    ("Stress", "8 features", FAMILY_COLOUR["Stress"], [
        ("Duration ratio", "mean · SD · max"), ("F0 ratio", "mean · SD"),
        ("Intensity difference (dB)", "mean · SD"), ("Valid multisyllabic words", "count")]),
    ("Utterance-level", "3 features", FAMILY_COLOUR["Utterance-level"], [
        ("F0 range", "max − min F0 (Hz)"), ("Speaking rate", "vowels per second"),
        ("Silence proportion", "share of utterance in silence")]),
    ("Rhythm", "3 features", FAMILY_COLOUR["Rhythm"], [
        ("%V", "vocalic proportion"), ("VarcoV", "vowel-duration variability"),
        ("nPVI", "pairwise vowel variability")]),
]

fig = plt.figure(figsize=(6.3, 4.3)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
ax.set_xlim(0, 6.3); ax.set_ylim(0, 4.3)

def box(x, y, w, h, fc, ec, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=fc, ec=ec, lw=lw))

# top box
box(1.45, 3.78, 3.4, 0.42, INK, INK)
ax.text(3.15, 3.99, "14 interpretable acoustic features", ha="center", va="center",
        color="white", fontsize=9.5, weight="bold")

colw, gap, x0 = 1.95, 0.12, 0.115
for i, (name, n, col, feats) in enumerate(FAM):
    x = x0 + i * (colw + gap); cx = x + colw / 2
    ax.annotate("", xy=(cx, 3.38), xytext=(3.15, 3.78),
                arrowprops=dict(arrowstyle="-|>", color="#9aa4ae", lw=0.9, mutation_scale=8))
    box(x, 2.92, colw, 0.46, "white", col, lw=1.6)
    ax.text(cx, 3.20, name, ha="center", va="center", fontsize=9.5, weight="bold", color=INK)
    ax.text(cx, 3.03, n, ha="center", va="center", fontsize=8, color=MUTED)
    for k, (f, sub) in enumerate(feats):
        y = 2.30 - k * 0.62
        box(x, y, colw, 0.52, "white", "#d3dae2", lw=0.8)
        ax.add_patch(plt.Rectangle((x, y), 0.06, 0.52, fc=col, ec="none"))
        ax.text(x + 0.14, y + 0.34, f, ha="left", va="center", fontsize=8.5, color=INK)
        ax.text(x + 0.14, y + 0.15, sub, ha="left", va="center", fontsize=8, color=MUTED, style="italic")
ax.text(x0, 0.30, "Stress features (3 + 2 + 2 + 1 = 8) are computed per multisyllabic word, then summarised over the utterance.",
        ha="left", va="center", fontsize=7.8, color=MUTED, style="italic")
out = "figures/Figure_3.5_feature_overview.png"
fig.savefig(out, dpi=300, facecolor="white"); print("saved", out)

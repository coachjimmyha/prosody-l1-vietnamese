"""fig_pipeline.py — Figure 3.1: overall pipeline of the system (diagram). Drawn at print width (16 cm);
the side notes match Chapter 3 and Section 4.3 (sample rates, alignment, the 14 features, data
cleaning and EDA, the 1,789 / 383 / 384 split, validation and evaluation).
"""
import os as _os; from paths import REPO_ROOT; _os.chdir(REPO_ROOT); _os.makedirs("figures", exist_ok=True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse
from fig_style import setup, BLUE, AQUA, ORANGE, INK, MUTED, NEUTRAL
setup()

FW = 6.3
X0, BW = 0.62, 2.75                       # main column: left edge, width
CX = X0 + BW / 2
NX = X0 + BW + 0.32                        # side notes start
BH, GAP = 0.50, 0.18
TINT = {BLUE: "#eaf2fc", AQUA: "#e6f6ef", ORANGE: "#fdeee7"}

STEPS = [  # (title, subtitle, colour, note title, note body)
    ("Data ingestion", "three public corpora", BLUE, "corpora",
     "Speechocean762 (training; Mandarin L1)\nL2-ARCTIC (4 Vietnamese speakers)\nCMU ARCTIC (4 native speakers)"),
    ("Audio (as distributed)", "mono, no resampling", BLUE, "native sample rates kept",
     "Speechocean762, CMU ARCTIC: 16 kHz\nL2-ARCTIC: 44.1 kHz\nMFA aligns at 16 kHz"),
    ("Forced alignment (MFA)", "english_us_arpa", BLUE, "word and phone boundaries",
     "Speechocean762, CMU ARCTIC: aligned\nby the author; L2-ARCTIC: TextGrids\nsupplied with the corpus"),
    ("Feature extraction", "Parselmouth (Praat)", BLUE, "14 interpretable features",
     "8 lexical stress, 3 utterance-level,\n3 rhythm"),
    ("Data cleaning and EDA", "filtering and exploratory analysis", BLUE, "data quality and distributions",
     "Speechocean762: 5,000 → 2,556 utterances;\nno missing values; training and target\ncorpora compared"),
    ("Data split", "70 / 15 / 15", AQUA, "Speechocean762",
     "train 1,789 / validation 383 / test 384\n(2,556 utterances)"),
    ("Model training", "Random Forest regressor", AQUA, "validation",
     "held-out test; speaker-grouped 5-fold CV;\nspeaker-disjoint split; shuffled-label null"),
    ("Cross-L1 transfer", "applied to Vietnamese and native speech", AQUA,
     "evaluation", "PCC, RMSE, R², Cohen’s $d$, bootstrap\nCIs, fake-model and exact permutation tests"),
    ("Model interpretation", None, ORANGE, None, None),
]
n = len(STEPS)
OV = 0.42                                                  # oval height
FH = 0.12 + OV + GAP + n * (BH + GAP) + 0.62 + GAP + BH + GAP + OV + 0.12
fig = plt.figure(figsize=(FW, FH)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
ax.set_xlim(0, FW); ax.set_ylim(0, FH)

ARW = dict(arrowstyle="-|>", color=NEUTRAL, lw=1.0, mutation_scale=9, shrinkA=0, shrinkB=0)
def arrow(x1, y1, x2, y2): ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=ARW)
def box(x, y, w, h, col, title, sub=None, bold_fill=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.07",
                                fc=TINT[col], ec=col, lw=1.3))
    if sub:
        ax.text(x + w / 2, y + (h * 0.66 if '\n' not in sub else h * 0.74), title, ha="center", va="center", fontsize=10, weight="bold", color=INK)
        ax.text(x + w / 2, y + (h * 0.28 if '\n' not in sub else h * 0.33), sub, ha="center", va="center", fontsize=8.5, color=MUTED, linespacing=1.15)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=10, weight="bold", color=INK)
def oval(x, y, text):
    ax.add_patch(Ellipse((x, y), 1.25, OV, fc=INK, ec="none"))
    ax.text(x, y, text, ha="center", va="center", fontsize=10, weight="bold", color="white")

def note(my, nt, nb):
    ax.plot([X0 + BW, NX - 0.06], [my, my], ls=(0, (1.5, 1.5)), color=NEUTRAL, lw=0.9)
    nlines = nb.count("\n") + 1
    top = my + 0.08 + 0.075 * (nlines - 1)
    ax.text(NX, top, nt, ha="left", va="bottom", fontsize=8.5, weight="bold", color=INK)
    ax.text(NX, top - 0.03, nb, ha="left", va="top", fontsize=8, color=MUTED, linespacing=1.3)

y = FH - 0.12 - OV / 2
oval(CX, y, "Start"); arrow(CX, y - OV / 2, CX, y - OV / 2 - GAP); y = y - OV / 2 - GAP
tops = []
for i, (title, sub, col, nt, nb) in enumerate(STEPS):
    box(X0, y - BH, BW, BH, col, title, sub); tops.append((y, col))
    if nt:
        note(y - BH / 2, nt, nb)
    yb = y - BH
    if i < n - 1:
        arrow(CX, yb, CX, yb - GAP)
    y = yb - GAP
# global / local branches
bw2, bh2 = 1.3, 0.72
for k, (t, s) in enumerate([("Global", "permutation\nimportance"), ("Local", "SHAP values\nper feature")]):
    bx = X0 + k * (BW - bw2)
    arrow(CX, y + GAP, bx + bw2 / 2, y)
    box(bx, y - bh2, bw2, bh2, ORANGE, t, s)
    arrow(bx + bw2 / 2, y - bh2, CX + (k - 0.5) * 0.9, y - bh2 - GAP)
y = y - bh2 - GAP
box(X0, y - BH, BW, BH, ORANGE, "Error localisation →\nactionable learner feedback")
note(y - BH / 2, "per-learner profile",
     "z-scores against the native reference;\ndirect measures incl. function-word\nreduction and robust F0 range")
arrow(CX, y - BH, CX, y - BH - GAP); y = y - BH - GAP
oval(CX, y - OV / 2, "End")
# stage brackets on the left
def bracket(ytop, ybot, col, label):
    ax.plot([0.42, 0.34, 0.34, 0.42], [ytop, ytop, ybot, ybot], color=col, lw=1.6, solid_capstyle="round")
    ax.text(0.2, (ytop + ybot) / 2, label, rotation=90, ha="center", va="center", fontsize=9, weight="bold", color=col)
bracket(tops[0][0], tops[4][0] - BH, BLUE, "Data preparation")
bracket(tops[5][0], tops[7][0] - BH, AQUA, "Modelling")
bracket(tops[8][0], y + GAP, ORANGE, "Interpretation and feedback")
out = "figures/Figure_3.1_pipeline.png"
fig.savefig(out, dpi=300, facecolor="white"); print("saved", out, f"{FW} x {FH:.2f} in")

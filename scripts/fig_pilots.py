"""fig_pilots.py — Figures 5.9–5.11: vowel centralisation, pitch-accent density and the five-group
comparison. Values are parsed from the logs written by vowel_centralization_pilot.py,
accent_density_full.py and contrast_full_pilot.py (logs/pilot_*.log); the exact permutation p
(70 relabellings of 4 + 4 speakers, two-sided) is computed here.
"""
import os, re, itertools
from paths import REPO_ROOT; os.chdir(REPO_ROOT); os.makedirs("figures", exist_ok=True)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from fig_style import setup, clean, AQUA, ORANGE, NEUTRAL, INK, MUTED, GRID
setup()

def two_group(log):
    txt = open(f"logs/{log}", encoding="utf-8").read()
    nat, vn = {}, {}
    for spk, grp, val in re.findall(r"^(\w+)\s+(native|VN)\s+([\d.]+)", txt, re.M):
        (nat if grp == "native" else vn)[spk] = float(val)
    d = float(re.search(r"\bd\b[^=\n]*=\s*([+-]?[\d.]+)", txt).group(1))
    ci = [float(x) for x in re.search(r"CI[^\[]*\[([+-]?[\d.]+),\s*([+-]?[\d.]+)\]", txt).groups()]
    return nat, vn, d, ci

def exact_p(a, b):
    allv = np.array(list(a) + list(b)); obs = abs(np.mean(a) - np.mean(b)); hit = tot = 0
    for idx in itertools.combinations(range(8), 4):
        m = np.zeros(8, bool); m[list(idx)] = True
        hit += abs(allv[m].mean() - allv[~m].mean()) >= obs - 1e-12; tot += 1
    return hit / tot

def dotplot(log, ylab, out, ref=None, reftxt=None):
    nat, vn, d, ci = two_group(log)
    p = exact_p(list(nat.values()), list(vn.values()))
    fig, ax = plt.subplots(figsize=(6.3, 3.7))
    if ref is not None:
        ax.axhline(ref, color=NEUTRAL, lw=0.9, ls="--", zorder=1)
        ax.text(1.98, ref, reftxt, ha="right", va="top", fontsize=8, color=MUTED, style="italic")
    for x, (grp, col, name) in enumerate([(nat, AQUA, "Native (CMU ARCTIC)"), (vn, ORANGE, "Vietnamese (L2-ARCTIC)")]):
        vals = sorted(grp.items(), key=lambda kv: kv[1]); m = np.mean(list(grp.values()))
        ax.plot([x - 0.14, x + 0.22], [m, m], color=col, lw=2.4, solid_capstyle="round", zorder=3)
        ax.text(x + 0.27, m, f"mean {m:.2f}", va="center", fontsize=9, color=INK, weight="bold")
        span = max(max(nat.values()), max(vn.values())) - min(min(nat.values()), min(vn.values()))
        ly = [v for _, v in vals]
        for k in range(1, len(ly)):                       # keep labels at least ~8 pt apart
            ly[k] = max(ly[k], ly[k - 1] + 0.055 * span)
        for i, ((spk, v), yl) in enumerate(zip(vals, ly)):
            xx = x - 0.12 + 0.08 * i
            ax.scatter(xx, v, s=46, color=col, edgecolor="white", lw=1.2, zorder=4)
            ax.text(x - 0.17, yl, spk, ha="right", va="center", fontsize=8, color=MUTED,
                    bbox=dict(fc="white", ec="none", pad=0.6), zorder=5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Native (CMU ARCTIC)", "Vietnamese (L2-ARCTIC)"], fontsize=9, color=INK)
    ax.set_xlim(-0.6, 2.0); ax.set_ylabel(ylab, fontsize=9)
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + (hi - lo) * 0.18)
    ax.text(0.5, 0.97, f"Cohen’s $d$ = {d:+.2f}".replace("-", "−") + f"   ·   95% bootstrap CI [{ci[0]:+.2f}, {ci[1]:+.2f}]".replace("-", "−")
            + f"   ·   exact $p$ = {p:.3f}", transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=INK)
    clean(ax, "y"); ax.spines["left"].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=300, facecolor="white"); plt.close(fig)
    print("saved", out, f"d={d:+.2f} CI={ci} exact p={p:.3f}")

dotplot("pilot_vowel_centralisation.log", "Vowel centralisation ratio\n(lower = more reduced)",
        "figures/Figure_5.9_vowel_centralisation.png", ref=1.0, reftxt="1.0 = no reduction")
dotplot("pilot_accent_density.log", "Syllables per pitch accent\n(lower = denser accenting)",
        "figures/Figure_5.10_accent_density.png")

# ---- Figure 5.11: five L1 groups, accent density vs vowel centralisation (group means) ----
txt = open("logs/pilot_typology_5groups.log", encoding="utf-8").read()
acc_blk, cen_blk = txt.split("VOWEL CENTRALIZATION")[0], txt.split("VOWEL CENTRALIZATION")[1]
gm = lambda blk: {g.capitalize(): float(v) for g, v in re.findall(r"^\s+(native|Vietnamese|Korean|Spanish|Chinese)\s+mean=([\d.]+)", blk, re.M)}
acc, cen = gm(acc_blk), gm(cen_blk)
COL = {"Native": AQUA, "Vietnamese": ORANGE}
OFF = {"Native": (9, 0, "left"), "Korean": (-9, 0, "right"), "Spanish": (-9, 0, "right"), "Vietnamese": (9, -2, "left"), "Chinese": (9, 2, "left")}
fig, ax = plt.subplots(figsize=(6.3, 4.0))
ax.axvline(acc["Native"], color=AQUA, lw=0.9, ls="--", zorder=1); ax.axhline(cen["Native"], color=AQUA, lw=0.9, ls="--", zorder=1)
for g in ["Korean", "Spanish", "Chinese", "Vietnamese", "Native"]:
    c = COL.get(g, NEUTRAL)
    ax.scatter(acc[g], cen[g], s=90, color=c, edgecolor="white", lw=1.4, zorder=4)
    dx, dy, ha = OFF[g]
    ax.annotate(g, (acc[g], cen[g]), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                fontsize=9, color=INK, weight="bold" if g in COL else "normal",
                bbox=dict(fc="white", ec="none", pad=0.6), zorder=5)
ax.text(acc["Native"] + 0.012, 1.015, "native level", fontsize=8, color=MUTED, style="italic", va="top")
ax.set_xlim(1.40, 2.32); ax.set_ylim(0.81, 1.02)
ax.set_xlabel("Accent density: syllables per pitch accent (lower = denser)", fontsize=9)
ax.set_ylabel("Vowel centralisation ratio\n(lower = more reduced)", fontsize=9)
leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=AQUA, markersize=8, label="Native (reference)"),
       Line2D([0], [0], marker="o", color="w", markerfacecolor=ORANGE, markersize=8, label="Vietnamese"),
       Line2D([0], [0], marker="o", color="w", markerfacecolor=NEUTRAL, markersize=8, label="Other L2 groups")]
ax.legend(handles=leg, frameon=False, fontsize=8.5, loc="upper right")
clean(ax, "both")
fig.tight_layout(); out = "figures/Figure_5.11_typology.png"; fig.savefig(out, dpi=300, facecolor="white"); plt.close(fig)
print("saved", out, acc, cen)

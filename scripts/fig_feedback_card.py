"""fig_feedback_card.py — Figure 5.8: per-learner feedback output (default learner TLV).
z-scores, roles and flags follow learner_error_profiles.py (same native reference and flag rule).
The model score is the learner's mean predicted prosody score from the Random Forest trained on all
2,556 Speechocean762 utterances.
Usage:  python scripts/fig_feedback_card.py [SPEAKER]
"""
import os, sys, io, contextlib, csv
from paths import REPO_ROOT; os.chdir(REPO_ROOT); os.makedirs("figures", exist_ok=True)
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from fig_style import setup, BLUE, ORANGE, INK, MUTED, NEUTRAL

SPK = sys.argv[1] if len(sys.argv) > 1 else "TLV"
with contextlib.redirect_stdout(io.StringIO()):
    import learner_error_profiles as lep          # runs the profile computation silently

FEAT = ["mean_duration_ratio", "std_duration_ratio", "max_duration_ratio", "mean_f0_ratio", "std_f0_ratio",
        "mean_intensity_diff", "std_intensity_diff", "n_multisyllabic_words", "f0_range_hz", "speaking_rate",
        "silence_proportion", "npvi", "percent_v", "varcov"]
tr = list(csv.DictReader(open("outputs/speechocean762_full_per_utt_v3.csv", encoding="utf-8")))
m = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1)
m.fit([[float(r[c]) for c in FEAT] for r in tr], [float(r["prosodic_score"]) for r in tr])
vx = [r for r in csv.DictReader(open("outputs/X14_vn_fresh.csv", encoding="utf-8")) if r["speaker"] == SPK]
score = float(np.mean(m.predict([[float(r[c]) for c in FEAT] for r in vx])))

ACTION = {
    "reduction": ("Function-word reduction", "function words kept almost full",
                  "Reduce function words to a schwa: “to / for / of” short and weak."),
    "speaking_rate": ("Speaking rate", "slower than the native reference",
                      "Speak in connected stress groups; let unstressed syllables run quickly."),
    "mean_duration_ratio": ("Stress duration contrast", "stressed and unstressed vowels too similar in length",
                            "Lengthen the stressed vowel and shorten the unstressed one: to-DAY."),
    "silence_proportion": ("Silence / phrasing", "more pausing than the native reference",
                           "Group words into phrases; reduce pauses between words."),
}
flags, quiet = [], []
for name, role, errdir, _ in lep.FEATS:
    if name not in lep.vn[SPK]:
        continue
    mu, sd = lep.ref[name]; z = (lep.vn[SPK][name] - mu) / sd
    flagged = role == "TEACH" and ((errdir == "high" and z >= 1) or (errdir == "low" and z <= -1))
    (flags if flagged else quiet).append((name, z))
flags.sort(key=lambda t: (t[0] != "reduction", -abs(t[1])))

setup()
import textwrap
W = lambda t, n: "\n".join(textwrap.wrap(t, n, break_on_hyphens=False))
BOX = 1.62                                   # height of each flagged-dimension box
H = 3.6 + (BOX + 0.25) * len(flags) + 2.2
fig = plt.figure(figsize=(6.3, H * 0.62)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")  # native print size: 16 cm wide
ax.set_xlim(0, 10); ax.set_ylim(0, H)
y = H - 0.25
ax.add_patch(FancyBboxPatch((0.3, y - 1.25), 9.4, 1.25, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#14283a", ec="none"))
ax.text(0.6, y - 0.45, "Per-learner prosody feedback", color="white", fontsize=14, weight="bold", va="center")
ax.text(0.6, y - 0.92, f"Learner {SPK} · L2-ARCTIC (Vietnamese subset)", color="#c9d3dc", fontsize=10, va="center")
y -= 1.85
ax.text(0.4, y, "Model prosody score:", fontsize=11, color=INK, va="center")
ax.text(4.1, y, f"{score:.1f} / 10", fontsize=15, color=BLUE, weight="bold", va="center")
y -= 0.65
ax.text(0.4, y, "The score alone does not say what to practise. Flagged dimensions:",
        fontsize=10, color=MUTED, style="italic", va="center")
y -= 0.45
for i, (name, z) in enumerate(flags):
    title, what, act = ACTION[name]
    top = y - 0.1; h = BOX
    fc = "#fdf1ea" if name == "reduction" else "white"
    ax.add_patch(FancyBboxPatch((0.3, top - h), 9.4, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec="#cfd6dd", lw=1))
    ax.add_patch(Rectangle((0.3, top - h), 0.1, h, fc=ORANGE, ec="none"))
    ax.text(0.65, top - 0.35, title, fontsize=12, weight="bold", color=INK, va="center")
    ax.text(9.5, top - 0.35, "PRIORITY" if name == "reduction" else "PRACTISE", fontsize=9.5, weight="bold",
            color=ORANGE, ha="right", va="center")
    ax.text(0.65, top - 0.75, f"{what} — z = {z:+.1f}".replace("-", "−").replace("−−", "—"), fontsize=9.5,
            color=MUTED, style="italic", va="center")
    ax.text(0.65, top - 1.05, W("→ " + act, 76), fontsize=10, color=INK, va="top", linespacing=1.3)
    y = top - h - 0.25
qtxt = ", ".join(f"{lep_name} (z = {z:+.1f})".replace("-", "−") for lep_name, z in
                 [({"f0_range_st": "F0 range", "npvi": "nPVI", "varcov": "VarcoV", "mean_duration_ratio": "duration contrast",
                    "silence_proportion": "silence", "speaking_rate": "speaking rate", "reduction": "reduction"}[n], z) for n, z in quiet])
ax.add_patch(FancyBboxPatch((0.3, y - 1.8), 9.4, 1.8, boxstyle="round,pad=0.02,rounding_size=0.06", fc="#f2f4f6", ec="none"))
ax.text(0.6, y - 0.35, "Not flagged", fontsize=11, weight="bold", color=INK, va="center")
ax.text(0.6, y - 0.75, qtxt, fontsize=9.5, color=MUTED, va="center")
ax.text(0.6, y - 1.1, "They do not separate native and Vietnamese speakers in this sample,\nso they are not teaching targets.",
        fontsize=9.5, color=MUTED, va="top", linespacing=1.3)
out = f"figures/Figure_5.8_feedback_{SPK}.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white"); print("saved", out, f"score {score:.2f};", ", ".join(f"{k} z={float(v):+.2f}" for k, v in flags))

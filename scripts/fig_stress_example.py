"""fig_stress_example.py — Figure 3.6: worked example of the stress features for the word "today".
The signal and values are illustrative (stated in the caption); the definitions follow
stress_contrast.py (the definition used for all three corpora).
"""
import os as _os; from paths import REPO_ROOT; _os.chdir(REPO_ROOT); _os.makedirs("figures", exist_ok=True)
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from fig_style import setup, BLUE, ORANGE, AQUA, INK, MUTED
setup()
NAVY = "#3b4a5a"
fig = plt.figure(figsize=(6.3, 6.15)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
ax.set_xlim(0, 6.3); ax.set_ylim(0, 6.15)

# segment boundaries (x in inches) : T | AH0 | D | EY1
X0, X1 = 1.45, 6.0
seg = [("T", 1.45, 1.95), ("AH0", 1.95, 3.05), ("D", 3.05, 3.55), ("EY1", 3.55, 6.0)]

# ---- waveform (synthetic) ----
t = np.linspace(X0, X1, 3000)
amp = np.full_like(t, 0.04)
amp[(t > 1.95) & (t < 3.05)] = 0.22 * np.sin(np.pi * (t[(t > 1.95) & (t < 3.05)] - 1.95) / 1.10) + 0.03
amp[(t > 3.55)] = 0.42 * np.sin(np.pi * (t[(t > 3.55)] - 3.55) / 2.45) + 0.03
wave = amp * np.sin(2 * np.pi * 22 * t)
ax.plot(t, 5.05 + wave, color=NAVY, lw=0.5)
ax.text(0.15, 5.05, "Waveform", fontsize=8, color=MUTED, style="italic", va="center")
# F0 contour over vowels + peak markers
for (lab, a, b), base, pk in [(seg[1], 5.50, 5.62), (seg[3], 5.62, 5.85)]:
    xs = np.linspace(a + 0.2 * (b - a), b - 0.2 * (b - a), 40)
    ys = base + (pk - base) * np.sin(np.pi * (xs - xs[0]) / (xs[-1] - xs[0]))
    ax.plot(xs, ys, color=ORANGE, lw=1.6)
    i = np.argmax(ys); ax.plot(xs[i], ys[i], "o", color=ORANGE, ms=4)
ax.text(0.15, 5.65, "F0 (central 60%\nof each vowel)", fontsize=8, color=ORANGE, style="italic", va="center")
ax.text(4.95, 5.86, "peak F0", fontsize=8, color=ORANGE, ha="left", va="center")
ax.text(2.62, 5.66, "peak F0", fontsize=8, color=ORANGE, ha="left", va="center")

# ---- tiers ----
def tier(y, label, cells):
    ax.text(1.35, y + 0.13, label, fontsize=8, color=MUTED, style="italic", ha="right", va="center")
    for txt, a, b, hl in cells:
        ax.add_patch(Rectangle((a, y), b - a, 0.26, fc="#fbe4d8" if hl else "white", ec=NAVY, lw=0.8))
        ax.text((a + b) / 2, y + 0.13, txt, fontsize=8.5, ha="center", va="center",
                color=ORANGE if hl else INK, weight="bold" if hl else "normal")
tier(4.18, "word", [("today", 1.45, 6.0, False)])
tier(3.88, "syllable", [("to", 1.45, 3.05, False), ("DAY", 3.05, 6.0, True)])
tier(3.58, "phone (ARPABET)", [(l, a, b, l == "EY1") for l, a, b in seg])
ax.text(2.50, 3.42, "unstressed vowel (marker 0)", fontsize=7.8, color=MUTED, ha="center", va="center")
ax.text(4.78, 3.42, "primary-stressed vowel (marker 1)", fontsize=7.8, color=ORANGE, ha="center", va="center")

# ---- illustrative values ----
ax.text(0.15, 3.05, "Illustrative values for this word:", fontsize=8.5, color=INK, weight="bold")
rows = [("Vowel duration", "AH0 0.17 s", "EY1 0.37 s", "ratio = 0.37 ÷ 0.17 ≈ 2.2"),
        ("Peak F0", "AH0 150 Hz", "EY1 180 Hz", "ratio = 180 ÷ 150 = 1.2"),
        ("Mean intensity", "AH0 66 dB", "EY1 72 dB", "difference = 72 − 66 = +6 dB")]
for k, (a, b, c, d) in enumerate(rows):
    y = 2.72 - k * 0.27
    ax.text(0.30, y, a, fontsize=8.3, color=INK)
    ax.text(1.75, y, b, fontsize=8.3, color=NAVY)
    ax.text(2.75, y, c, fontsize=8.3, color=ORANGE)
    ax.text(3.80, y, d, fontsize=8.3, color=INK)

# ---- formula boxes ----
F = [("Duration ratio", "stressed-vowel duration ÷ mean unstressed-vowel duration", BLUE),
     ("F0 ratio", "stressed-vowel peak F0 ÷ mean unstressed-vowel peak F0", ORANGE),
     ("Intensity difference (dB)", "stressed-vowel intensity − mean unstressed-vowel intensity", AQUA)]
for k, (n, f, c) in enumerate(F):
    y = 1.52 - k * 0.50
    ax.add_patch(FancyBboxPatch((0.15, y), 6.0, 0.40, boxstyle="round,pad=0,rounding_size=0.05",
                                fc="white", ec=c, lw=1.3))
    ax.add_patch(Rectangle((0.15, y), 0.07, 0.40, fc=c, ec="none"))
    ax.text(0.35, y + 0.20, n, fontsize=8.5, weight="bold", color=INK, va="center")
    ax.text(2.10, y + 0.20, f, fontsize=8.2, color=INK, va="center")
ax.text(0.15, 0.22, "Each value is computed per multisyllabic word; in “today” the only unstressed vowel is AH0,\n"
        "while in longer words the unstressed vowels (marker 0) are averaged. Signal values are illustrative.",
        fontsize=7.8, color=MUTED, style="italic", va="center")
out = "figures/Figure_3.6_stress_example.png"
fig.savefig(out, dpi=300, facecolor="white"); print("saved", out)

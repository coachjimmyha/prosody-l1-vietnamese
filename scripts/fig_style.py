"""fig_style.py — shared colours, feature labels and axis style for the thesis figures.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# palette validated with the dataviz validator (light surface): blue / orange / aqua
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, NEUTRAL = "#22303a", "#5b6570", "#e3e7ec", "#9aa4ae"

NICE = {
    "mean_duration_ratio": "Duration ratio, mean", "std_duration_ratio": "Duration ratio, SD",
    "max_duration_ratio": "Duration ratio, max", "mean_f0_ratio": "F0 ratio, mean",
    "std_f0_ratio": "F0 ratio, SD", "mean_intensity_diff": "Intensity difference (dB), mean",
    "std_intensity_diff": "Intensity difference (dB), SD",
    "n_multisyllabic_words": "Valid multisyllabic words", "f0_range_hz": "F0 range",
    "speaking_rate": "Speaking rate", "silence_proportion": "Silence proportion",
    "npvi": "nPVI", "percent_v": "%V", "varcov": "VarcoV",
}
FAMILY = {f: "Stress" for f in ["mean_duration_ratio", "std_duration_ratio", "max_duration_ratio",
                                "mean_f0_ratio", "std_f0_ratio", "mean_intensity_diff",
                                "std_intensity_diff", "n_multisyllabic_words"]}
FAMILY.update({"f0_range_hz": "Utterance-level", "speaking_rate": "Utterance-level",
               "silence_proportion": "Utterance-level", "npvi": "Rhythm", "percent_v": "Rhythm",
               "varcov": "Rhythm"})
FAMILY_COLOUR = {"Utterance-level": BLUE, "Rhythm": ORANGE, "Stress": AQUA}


def setup():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
                         "axes.edgecolor": "#b8c0c8", "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})


def clean(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

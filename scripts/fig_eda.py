"""fig_eda.py — Figure 4.1: exploratory view of the training and target data (Section 4.3.2).
(a) Speechocean762 prosody scores; (b) valid multisyllabic words per utterance; (c) speaking rate and
(d) silence proportion in Speechocean762, CMU ARCTIC and the Vietnamese L2-ARCTIC subset
(dotted lines = medians).
"""
import os; from paths import REPO_ROOT; os.chdir(REPO_ROOT); os.makedirs("figures", exist_ok=True)
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from fig_style import setup, clean, BLUE, AQUA, ORANGE, INK, MUTED
setup()
so = pd.read_csv("outputs/speechocean762_full_per_utt_v3.csv")
nat = pd.read_csv("outputs/X14_native_fresh.csv")
vn = pd.read_csv("outputs/X14_vn_fresh.csv")
fig, axs = plt.subplots(2, 2, figsize=(6.3, 5.3), gridspec_kw=dict(hspace=0.62, wspace=0.28))
(a, b), (c, d) = axs
sc = so.prosodic_score.value_counts().reindex(range(11), fill_value=0)
a.bar(sc.index, sc.values, width=0.72, color=BLUE)
a.axvline(so.prosodic_score.mean(), color=INK, lw=1, ls="--")
a.text(so.prosodic_score.mean() - 0.3, sc.max() * 0.97, f"mean {so.prosodic_score.mean():.2f}", ha="right", va="top", fontsize=8, color=INK)
a.set_xticks(range(11)); a.set_xlabel("Prosody score (0–10)", fontsize=8.5); a.set_ylabel("Utterances", fontsize=8.5)
a.set_title("(a) Target: Speechocean762 prosody score", fontsize=8.5, loc="left", color=INK)
nm = so.n_multisyllabic_words.value_counts().sort_index()
b.bar(nm.index, nm.values, width=0.6, color=BLUE)
for x, v in nm.items():
    pct = 100 * v / len(so)
    b.text(x, v + 25, f"{pct:.0f}%" if pct >= 1 else f"{pct:.1f}%", ha="center", va="bottom", fontsize=7.5, color=MUTED)
b.set_xticks(list(nm.index)); b.set_xlabel("Valid multisyllabic words per utterance", fontsize=8.5)
b.set_ylim(0, nm.max() * 1.18); b.set_ylabel("Utterances", fontsize=8.5)
b.set_title("(b) Words behind the stress features", fontsize=8.5, loc="left", color=INK)
groups = [("Speechocean762 (training)", so, BLUE), ("CMU ARCTIC (native)", nat, AQUA), ("L2-ARCTIC (Vietnamese)", vn, ORANGE)]
def dens(ax, col, bins, xlabel, title):
    for name, df, colr in groups:
        h, e = np.histogram(df[col].dropna(), bins=bins, density=True)
        ax.step(e[:-1], h, where="post", color=colr, lw=1.6)
        ax.axvline(df[col].median(), color=colr, lw=0.9, ls=":")
    ax.set_xlabel(xlabel, fontsize=8.5); ax.set_ylabel("Density", fontsize=8.5); ax.set_yticks([])
    ax.set_title(title, fontsize=8.5, loc="left", color=INK)
dens(c, "speaking_rate", np.linspace(0, 7, 36), "Speaking rate (vowels per second)", "(c) Speaking rate by corpus")
dens(d, "silence_proportion", np.linspace(0, 0.9, 37), "Silence proportion", "(d) Silence proportion by corpus")
for ax in axs.flat: clean(ax, "y"); ax.tick_params(labelsize=8)
leg = [Line2D([0], [0], color=colr, lw=1.8, label=name) for name, _, colr in groups]
fig.legend(handles=leg, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.005))
fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.14)
out = "figures/Figure_4.1_eda.png"; fig.savefig(out, dpi=300, facecolor="white")
print("saved", out)
for col in ["speaking_rate", "silence_proportion"]:
    print(col, "medians SO/NAT/VN", [round(float(g[1][col].median()), 2) for g in groups])
print("native faster than 95% of training:", round((nat.speaking_rate > so.speaking_rate.quantile(0.95)).mean(), 3),
      "VN:", round((vn.speaking_rate > so.speaking_rate.quantile(0.95)).mean(), 3))
print("native silence below training minimum:", round((nat.silence_proportion < so.silence_proportion.min()).mean(), 3),
      "VN:", round((vn.silence_proportion < so.silence_proportion.min()).mean(), 3))
print("one multisyllabic word: SO/NAT/VN", [round(float((g[1].n_multisyllabic_words == 1).mean()), 3) for g in groups])

"""fig_ablation.py — Figure 4.2: feature-group ablation on the canonical split (1,789 / 383 / 384).
(a) nested models: stress -> + utterance-level -> + rhythm; (b) full model and one group removed,
with the speaking-rate-only baseline. Same Random Forest settings as final_evaluation_v3.py.
"""
import os; from paths import REPO_ROOT; os.chdir(REPO_ROOT); os.makedirs("figures", exist_ok=True)
import csv
import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from fig_style import setup, clean, BLUE, NEUTRAL, INK, MUTED

FEAT = ["mean_duration_ratio", "std_duration_ratio", "max_duration_ratio", "mean_f0_ratio",
        "std_f0_ratio", "mean_intensity_diff", "std_intensity_diff", "n_multisyllabic_words",
        "f0_range_hz", "speaking_rate", "silence_proportion", "npvi", "percent_v", "varcov"]
X, y = [], []
for r in csv.DictReader(open("outputs/speechocean762_full_per_utt_v3.csv", encoding="utf-8")):
    try:
        X.append([float(r[c]) for c in FEAT]); y.append(float(r["prosodic_score"]))
    except (KeyError, ValueError):
        pass
X, y = np.array(X), np.array(y)
Xtv, Xte, ytv, yte = train_test_split(X, y, test_size=0.15, random_state=42)
Xtr, _, ytr, _ = train_test_split(Xtv, ytv, test_size=0.176, random_state=42)


def pcc(cols):
    m = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1)
    m.fit(Xtr[:, cols], ytr); return pearsonr(m.predict(Xte[:, cols]), yte)[0]


S, U, R = list(range(8)), [8, 9, 10], [11, 12, 13]
nested = [("Stress\n(8)", pcc(S)), ("+ Utterance\n(11)", pcc(S + U)), ("+ Rhythm\n(14, full)", pcc(S + U + R))]
full = nested[-1][1]
rate_only = pcc([9])
drop = [("Full\n(14)", full), ("− Stress\n(6)", pcc(U + R)),
        ("− Rhythm\n(11)", pcc(S + U)), ("− Utterance\n(11)", pcc(S + R))]
for lab, v in nested + drop:
    print(f"{lab.replace(chr(10), ' '):28} {v:.3f}")
print(f"speaking rate only           {rate_only:.3f}")

setup()
fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.5), sharey=True, gridspec_kw={"width_ratios": [3, 4]})
for ax, rows, title in [(axes[0], nested, "(a) Nested: groups added in order"),
                        (axes[1], drop, "(b) One group removed from the full model")]:
    xs = np.arange(len(rows))
    cols = [BLUE if "full" in lab.lower() or "Full" in lab else NEUTRAL for lab, _ in rows]
    ax.bar(xs, [v for _, v in rows], width=0.6, color=cols, edgecolor="white", linewidth=2)
    for x, (_, v) in zip(xs, rows):
        if ax is axes[1] and abs(v - rate_only) < 0.06 and v < rate_only:   # avoid the baseline line
            ax.text(x, v - 0.015, f"{v:.3f}", ha="center", va="top", color="white", fontsize=8.5, zorder=4)
        else:
            ax.text(x, v + 0.012, f"{v:.3f}", ha="center", va="bottom", color=INK, fontsize=8.5, zorder=4)
    ax.set_xticks(xs); ax.set_xticklabels([lab for lab, _ in rows], fontsize=8.5, color=INK)
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    clean(ax, "y")
axes[1].axhline(rate_only, color=MUTED, linestyle="--", linewidth=1.2, zorder=1,
                label=f"Speaking rate only = {rate_only:.3f}")
axes[1].legend(loc="upper right", frameon=False, fontsize=8, labelcolor=INK)
axes[0].set_ylabel("Held-out test PCC (n = 384)")
axes[0].set_ylim(0, 0.75)
fig.tight_layout()
out = "figures/Figure_4.2_ablation.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white"); print("saved", out)

"""shap_analysis.py — SHAP (TreeExplainer) attributions for the Random Forest trained on all 2,556
Speechocean762 utterances (Section 3.8; Appendix B.10).

Prints the SHAP base value (the training mean of the prosody score) and writes three supplementary
plots to figures/supplementary/: a global beeswarm, the mean signed SHAP profile of each Vietnamese
learner, and a local explanation of one learner utterance. Figure 5.6 itself is drawn by
fig_shap_beeswarm_v2.py.

Run:  python scripts/shap_analysis.py
"""
import os as _os; from paths import REPO_ROOT; _os.chdir(REPO_ROOT); _os.makedirs("figures/supplementary", exist_ok=True)
import csv, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
import shap
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"]})
BASE="outputs"
FEAT=["mean_duration_ratio","std_duration_ratio","max_duration_ratio","mean_f0_ratio","std_f0_ratio",
 "mean_intensity_diff","std_intensity_diff","n_multisyllabic_words","f0_range_hz","speaking_rate",
 "silence_proportion","npvi","percent_v","varcov"]
NICE={"mean_duration_ratio":"Duration ratio (stress)","std_duration_ratio":"Duration ratio SD","max_duration_ratio":"Max duration ratio",
 "mean_f0_ratio":"F0 ratio (stress)","std_f0_ratio":"F0 ratio SD","mean_intensity_diff":"Intensity diff (stress)",
 "std_intensity_diff":"Intensity diff SD","n_multisyllabic_words":"# multisyllabic words","f0_range_hz":"F0 range",
 "speaking_rate":"Speaking rate","silence_proportion":"Silence proportion","npvi":"nPVI (rhythm)",
 "percent_v":"%V (rhythm)","varcov":"VarcoV (rhythm)"}
labels=[NICE[f] for f in FEAT]

def load(p,cols,tgt=None):
    X=[];y=[];S=[]
    for r in csv.DictReader(open(p,encoding="utf-8")):
        try:
            X.append([float(r[c]) for c in cols])
            if tgt: y.append(float(r[tgt]))
            if "speaker" in r: S.append(r["speaker"])
        except (KeyError,ValueError): pass
    return np.array(X), np.array(y) if tgt else None, S

Xtr,ytr,_=load(BASE+"/speechocean762_full_per_utt_v3.csv",FEAT,"prosodic_score")
m=RandomForestRegressor(n_estimators=200,max_depth=8,min_samples_leaf=10,random_state=42,n_jobs=-1)
m.fit(Xtr,ytr)
Xv,_,Sv=load(BASE+"/X14_vn_fresh.csv",FEAT)
print(f"Trained on {len(ytr)}; VN utts {len(Xv)}")

expl=shap.TreeExplainer(m)
sv_tr=expl.shap_values(Xtr)          # global (training dist)
sv_v =expl.shap_values(Xv)           # learner-facing
base=float(np.ravel(expl.expected_value)[0])
print(f"model base value (mean prosody score) = {float(base):.3f}")

# ---- 1. GLOBAL beeswarm ----
plt.figure()
shap.summary_plot(sv_tr, Xtr, feature_names=labels, show=False, max_display=10, color_bar_label="Feature value",
                  rng=np.random.default_rng(42))  # fixed jitter of overlapping points
plt.title("SHAP global — what drives the predicted prosody score", fontsize=12, color="#1e3a5f")
plt.tight_layout(); plt.savefig("figures/supplementary/SHAP_global_beeswarm.png",dpi=200,bbox_inches="tight"); plt.close()
print("saved figures/supplementary/SHAP_global_beeswarm.png")

# ---- 2. per-VN-speaker mean signed SHAP ----
g=defaultdict(list)
for row,s in zip(sv_v,Sv): g[s].append(row)
spk=sorted(g); meanshap={s:np.mean(g[s],axis=0) for s in spk}
# rank features by overall |mean shap| across VN
overall=np.mean([np.abs(meanshap[s]) for s in spk],axis=0)
order=np.argsort(overall)[::-1][:8]
fig,ax=plt.subplots(figsize=(8.6,5.0))
NAVY="#1e3a5f"; cols=["#b23a48","#2c6e8f","#2e7d5b","#c8763a"]
yy=np.arange(len(order)); w=0.2
for i,s in enumerate(spk):
    vals=[meanshap[s][j] for j in order]
    ax.barh(yy+(1.5-i)*w, vals, height=w, color=cols[i%4], label=s, zorder=3)
ax.axvline(0,color="#6b7681",lw=1)
ax.set_yticks(yy); ax.set_yticklabels([labels[j] for j in order],fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("Mean SHAP value (impact on predicted prosody score)  — negative = lowers the score",fontsize=9.5,color="#20262e")
ax.text(0,1.11,"Why the model scores each Vietnamese learner as it does",transform=ax.transAxes,fontsize=13.5,color=NAVY,fontweight="bold")
ax.text(0,1.04,"Per-learner mean SHAP per feature. Negative bars = the dimension pulls the predicted prosody score down.",transform=ax.transAxes,fontsize=9,color="#5b6570")
ax.legend(title="learner",loc="lower right",frameon=False,fontsize=9)
for s2 in ["top","right"]: ax.spines[s2].set_visible(False)
for s2 in ["left","bottom"]: ax.spines[s2].set_color("#c9d2dc")
ax.tick_params(colors="#5b6570",length=0); ax.set_axisbelow(True); ax.xaxis.grid(True,color="#c9d2dc",lw=0.6,alpha=0.5)
plt.subplots_adjust(top=0.83,left=0.28)
plt.savefig("figures/supplementary/SHAP_learner_profiles.png",dpi=200,bbox_inches="tight"); plt.close()
print("saved figures/supplementary/SHAP_learner_profiles.png")

# ---- 3. local example: lowest-predicted VN utterance ----
pred_v=m.predict(Xv)
i=int(np.argmin(pred_v))
print(f"\nLOCAL EXAMPLE — VN utt idx {i}, speaker {Sv[i]}, predicted score {pred_v[i]:.2f} (base {float(base):.2f})")
contrib=sorted(zip(labels,sv_v[i],Xv[i]),key=lambda t:-abs(t[1]))
for name,s,val in contrib[:6]:
    print(f"  {name:24} value={val:8.2f}  SHAP={s:+.3f}")

# ---- 4. CLEAN LOCAL waterfall for the lowest-predicted VN utterance ----
i=int(np.argmin(pred_v))
contribs=sorted(zip(labels,sv_v[i]),key=lambda t:-abs(t[1]))[:7]
names=[c[0] for c in contribs][::-1]; vals=[c[1] for c in contribs][::-1]
fig,ax=plt.subplots(figsize=(8.4,4.6))
colors=["#b23a48" if v<0 else "#2e7d5b" for v in vals]
yy=np.arange(len(names))
ax.barh(yy,vals,color=colors,zorder=3,height=0.6)
for y,v in zip(yy,vals):
    ax.text(v+(0.02 if v>=0 else -0.02),y,f"{v:+.2f}",va="center",ha="left" if v>=0 else "right",fontsize=9.5,
            color="#2e7d5b" if v>=0 else "#b23a48")
ax.axvline(0,color="#6b7681",lw=1)
ax.set_yticks(yy); ax.set_yticklabels(names,fontsize=10)
ax.set_xlabel("SHAP contribution to THIS utterance's predicted score",fontsize=9.5,color="#20262e")
ax.text(0,1.13,f"Local explanation — one learner utterance ({Sv[i]})",transform=ax.transAxes,fontsize=13.5,color="#1e3a5f",fontweight="bold")
ax.text(0,1.05,f"Model base {float(base):.2f} → predicted {pred_v[i]:.2f}. Red pulls the score down; green pushes it up. Slow speaking rate dominates.",
    transform=ax.transAxes,fontsize=9,color="#5b6570")
mn=min(vals); mx=max(vals); ax.set_xlim(mn-0.2,mx+0.2)
for s2 in ["top","right"]: ax.spines[s2].set_visible(False)
for s2 in ["left","bottom"]: ax.spines[s2].set_color("#c9d2dc")
ax.tick_params(colors="#5b6570",length=0); ax.set_axisbelow(True); ax.xaxis.grid(True,color="#c9d2dc",lw=0.6,alpha=0.5)
plt.subplots_adjust(top=0.82,left=0.28)
plt.savefig("figures/supplementary/SHAP_local_example.png",dpi=200,bbox_inches="tight"); plt.close()
print("saved figures/supplementary/SHAP_local_example.png")

"""contrast_full_pilot.py — five-group comparison of accent density and vowel centralisation
(Section 5.4.3; Table 5.3; Figure 5.11).

Native speakers and the Vietnamese, Korean, Spanish and Chinese L2-ARCTIC subsets (4 speakers each),
restricted to the 12 sentences recorded by all groups (arctic_a0001–a0012). Same measures as
accent_density_full.py and vowel_centralization_pilot.py. Prints group means, native vs group d, the
pooled-L2 comparison with a bootstrap CI, and tonal vs non-tonal L2. Needs the audio (see paths.py).

Run:  python scripts/contrast_full_pilot.py > logs/pilot_typology_5groups.log
"""
import glob, re, os
import numpy as np
import parselmouth
from scipy.signal import find_peaks
from paths import cmu_dir, l2_dir
GROUPS={
 "native":[("bdl",str(cmu_dir("bdl"))),("slt",str(cmu_dir("slt"))),("clb",str(cmu_dir("clb"))),("rms",str(cmu_dir("rms")))],
 "Vietnamese":[(s,str(l2_dir(s))) for s in ["HQTV","PNV","THV","TLV"]],
 "Korean":[(s,str(l2_dir(s))) for s in ["HKK","HJK","YDCK","YKWK"]],
 "Spanish":[(s,str(l2_dir(s))) for s in ["EBVS","ERMS","MBMPS","NJS"]],
 "Chinese":[(s,str(l2_dir(s))) for s in ["BWC","LXC","NCC","TXHC"]],
}
UTTS=[f"arctic_a{i:04d}" for i in range(1,13)]  # arctic_a0001–a0012, recorded by all five groups
VOW={"AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW","AX"}
FUNC=set(("a an the of to in for on at by with from as into about over under i you he she it we they me him her us them my your his its our their this that these those is are was were be been being am do does did have has had having will would can could shall should may might must and but or nor so yet not no if than then there here what which who whom whose when where why how").split())
def strip(l): return re.sub(r"\d+","",l).strip().upper()
def sdig(l):
    m=re.search(r"(\d)$",l); return int(m.group(1)) if m else None
def tiers(p):
    data={};cur=None;xmin=None
    for ln in open(p,encoding="utf-8",errors="ignore"):
        s=ln.strip();m=re.match(r'name = "(.*)"',s)
        if m:cur=m.group(1);data[cur]=[];xmin=None;continue
        if cur is None:continue
        a=re.match(r'xmin = ([\d.]+)',s);b=re.match(r'xmax = ([\d.]+)',s);t=re.match(r'text = "(.*)"',s)
        if a:xmin=float(a.group(1))
        elif b:xmax=float(b.group(1))
        elif t is not None:data[cur].append((xmin,xmax,t.group(1)))
    W=[data[k] for k in data if k.lower().startswith("word")];P=[data[k] for k in data if k.lower().startswith("phone")]
    return (W[0] if W else []),(P[0] if P else [])
def analyse(wav,tg):
    W,P=tiers(tg)
    if not W or not P: return None
    snd=parselmouth.Sound(wav);pit=snd.to_pitch(time_step=0.01,pitch_floor=75,pitch_ceiling=500)
    fm=snd.to_formant_burg(max_number_of_formants=5,maximum_formant=5500)
    # accent density
    f0=pit.selected_array['frequency'];v=f0[f0>0]
    nacc=None
    if len(v)>=5:
        med=np.median(v);st=12*np.log2(np.where(f0>0,f0,med)/med)
        nacc=len(find_peaks(st,prominence=2.0,distance=12)[0])
    # syllables + formants for centralization
    nsyl=0; vw=[]
    for wmin,wmax,wt in W:
        w=(wt or "").strip().lower()
        if not w or w in {"sil","sp","spn","<unk>",""}: continue
        ph=[p for p in P if p[0] is not None and p[0]>=wmin-0.005 and p[1]<=wmax+0.005]
        for a,b,lab in ph:
            if strip(lab) not in VOW: continue
            nsyl+=1
            mid=(a+b)/2; f1=fm.get_value_at_time(1,mid); f2=fm.get_value_at_time(2,mid)
            if f1 and f2 and not np.isnan(f1) and not np.isnan(f2):
                isf=w in FUNC; iss=(sdig(lab)==1 and not isf)
                if isf or iss: vw.append((isf,iss,f1,f2))
    if nacc is None or nsyl==0: return None
    return nacc, nsyl, vw
def cohens_d(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    sp=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
    return (a.mean()-b.mean())/sp if sp>0 else np.nan
def boot(nat,oth,n=5000):
    rng=np.random.RandomState(42);ds=[dd for _ in range(n) if np.isfinite(dd:=cohens_d(rng.choice(nat,len(nat),True),rng.choice(oth,len(oth),True)))]
    return np.percentile(ds,[2.5,97.5])
spk_acc={};spk_cen={}
for grp,spks in GROUPS.items():
    for spk,d in spks:
        accs=[];vwall=[]
        for u in UTTS:
            wav=f"{d}/wav/{u}.wav";tg=f"{d}/textgrid/{u}.TextGrid"
            if not(os.path.exists(wav) and os.path.exists(tg)):continue
            r=analyse(wav,tg)
            if r: accs.append(r[1]/max(r[0],1)); vwall+=r[2]
        spk_acc[(grp,spk)]=np.mean(accs) if accs else np.nan
        if len(vwall)>=10:
            F1=np.array([x[2] for x in vwall]);F2=np.array([x[3] for x in vwall])
            z1=(F1-F1.mean())/F1.std();z2=(F2-F2.mean())/F2.std();dist=np.sqrt(z1**2+z2**2)
            fd=np.array([dist[i] for i,x in enumerate(vwall) if x[0]]);sd_=np.array([dist[i] for i,x in enumerate(vwall) if x[1]])
            spk_cen[(grp,spk)]=fd.mean()/sd_.mean() if len(fd) and len(sd_) else np.nan
        else: spk_cen[(grp,spk)]=np.nan
def report(metric,name,lowdenser):
    print(f"\n===== {name} =====")
    gm={}
    for grp in GROUPS:
        vals=[metric[(grp,s)] for s,_ in [(s,0) for s in [sp for (g,sp) in metric if g==grp]]]
        vals=[metric[k] for k in metric if k[0]==grp and np.isfinite(metric[k])]
        gm[grp]=vals
        print(f"  {grp:11} mean={np.mean(vals):.3f}  speakers={[round(x,2) for x in vals]}")
    nat=gm["native"]
    for grp in ["Vietnamese","Korean","Spanish","Chinese"]:
        d=cohens_d(nat,gm[grp]);print(f"  native vs {grp:11} d={d:+.2f}")
    pooled=gm["Vietnamese"]+gm["Korean"]+gm["Spanish"]+gm["Chinese"]
    lo,hi=boot(nat,pooled)
    print(f"  native vs POOLED-L2  d={cohens_d(nat,pooled):+.2f}  boot95%CI=[{lo:+.2f},{hi:+.2f}] {'no cross0' if (lo>0 or hi<0) else 'crosses0'}")
    tonal=gm["Vietnamese"]+gm["Chinese"];nontonal=gm["Korean"]+gm["Spanish"]
    print(f"  tonal(VN+Chi) vs non-tonal(Kor+Spa)  d={cohens_d(tonal,nontonal):+.2f}")
report(spk_acc,"ACCENT DENSITY (syll/accent; lower=denser)",True)
report(spk_cen,"VOWEL CENTRALIZATION (func/stress dist; lower=more centralised)",True)
print("\n5 groups x 4 speakers, 12 matched ARCTIC sentences. Proxy. EXPLORATORY.")

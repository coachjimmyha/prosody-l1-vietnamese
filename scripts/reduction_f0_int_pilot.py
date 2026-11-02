"""reduction_f0_int_pilot.py — pitch- and intensity-based reduction (Section 5.4.1).

For function-word vowels versus primary-stressed content-word vowels (F0 and intensity sampled at
three interior points of each vowel): the ratio of mean F0 and the difference in mean intensity (dB),
per speaker over matched ARCTIC sentences (NUTT, default 12).
Prints speaker values, Cohen's d and speaker bootstrap CIs. Needs the audio (see paths.py).

Run:  python scripts/reduction_f0_int_pilot.py > logs/pilot_reduction_f0_intensity.log
"""
import glob, re, os
import numpy as np
import parselmouth
from paths import cmu_dir, l2_dir
SPKS={"bdl":("native",str(cmu_dir("bdl"))),
 "slt":("native",str(cmu_dir("slt"))),
 "clb":("native",str(cmu_dir("clb"))),
 "rms":("native",str(cmu_dir("rms"))),
 "HQTV":("VN",str(l2_dir("HQTV"))),
 "PNV":("VN",str(l2_dir("PNV"))),
 "THV":("VN",str(l2_dir("THV"))),
 "TLV":("VN",str(l2_dir("TLV")))}
VOW={"AA","AE","AH","AO","AW","AX","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW"}
FUNC=set(("a an the of to in for on at by with from as into about over under i you he she it we they me him her us them my your his its our their this that these those is are was were be been being am do does did have has had having will would can could shall should may might must and but or nor so yet not no if than then there here what which who whom whose when where why how").split())
def strip(l): return re.sub(r"\d+","",l).strip().upper()
def sd(l):
    m=re.search(r"(\d)$",l); return int(m.group(1)) if m else None
def tiers(path):
    data={};cur=None;xmin=None
    for ln in open(path,encoding="utf-8",errors="ignore"):
        s=ln.strip();m=re.match(r'name = "(.*)"',s)
        if m: cur=m.group(1);data[cur]=[];xmin=None;continue
        if cur is None: continue
        a=re.match(r'xmin = ([\d.]+)',s);b=re.match(r'xmax = ([\d.]+)',s);t=re.match(r'text = "(.*)"',s)
        if a: xmin=float(a.group(1))
        elif b: xmax=float(b.group(1))
        elif t is not None: data[cur].append((xmin,xmax,t.group(1)))
    W=[data[k] for k in data if k.lower().startswith("word")];P=[data[k] for k in data if k.lower().startswith("phone")]
    return (W[0] if W else []),(P[0] if P else [])
def meanval(obj,a,b,kind):
    ts=np.linspace(a,b,5)[1:-1];vals=[]
    for t in ts:
        try:
            v=obj.get_value(t) if kind=="int" else obj.get_value_at_time(t)
        except Exception: v=None
        if v is not None and not (isinstance(v,float) and np.isnan(v)): vals.append(v)
    return np.mean(vals) if vals else None
def utt_ratios(wav,tg):
    W,P=tiers(tg)
    if not W or not P: return None
    snd=parselmouth.Sound(wav);pit=snd.to_pitch(time_step=0.01,pitch_floor=75,pitch_ceiling=500);inten=snd.to_intensity()
    ff0,sf0,fi,si=[],[],[],[]
    for wmin,wmax,wt in W:
        w=(wt or "").strip().lower()
        if not w or w in {"sil","sp","spn","<unk>",""}: continue
        ph=[p for p in P if p[0] is not None and p[0]>=wmin-0.005 and p[1]<=wmax+0.005]
        for a,b,lab in ph:
            if strip(lab) not in VOW: continue
            f0=meanval(pit,a,b,"pit"); it=meanval(inten,a,b,"int")
            if w in FUNC:
                if f0: ff0.append(f0)
                if it is not None: fi.append(it)
            elif sd(lab)==1:
                if f0: sf0.append(f0)
                if it is not None: si.append(it)
    if not ff0 or not sf0 or not fi or not si: return None
    return np.mean(ff0)/np.mean(sf0), np.mean(fi)-np.mean(si)   # F0 ratio, intensity diff (dB)
def d(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    sp=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
    return (a.mean()-b.mean())/sp if sp>0 else np.nan
res={}
for spk,(grp,rel) in SPKS.items():
    d0=rel;f0r=[];idf=[]
    for wav in sorted(glob.glob(f"{d0}/wav/arctic_a00*.wav"))[:int(os.environ.get("NUTT","12"))]:
        uid=os.path.basename(wav)[:-4];tg=f"{d0}/textgrid/{uid}.TextGrid"
        if not os.path.exists(tg):continue
        r=utt_ratios(wav,tg)
        if r: f0r.append(r[0]);idf.append(r[1])
    res[spk]=(grp,np.mean(f0r),np.mean(idf),len(f0r))
print(f"{'spk':6}{'grp':8}{'F0 ratio':>10}{'Int diff dB':>13}{'utts':>6}")
print("-"*43)
for spk,(grp,f,i,n) in res.items(): print(f"{spk:6}{grp:8}{f:10.3f}{i:13.2f}{n:6}")
print("-"*43)
for idx,name,note in [(1,"F0 ratio (func/stress)","lower=func pitch lower=more reduced"),(2,"Intensity diff dB (func-stress)","more negative=func weaker=more reduced")]:
    nat=[res[s][idx] for s in res if res[s][0]=="native"];vn=[res[s][idx] for s in res if res[s][0]=="VN"]
    print(f"\n{name}: native={np.mean(nat):+.3f}  VN={np.mean(vn):+.3f}  d={d(nat,vn):+.2f}  ({note})")
    rng=np.random.RandomState(42);ds=[dd for _ in range(5000) if np.isfinite(dd:=d(rng.choice(nat,4,True),rng.choice(vn,4,True)))]
    lo,hi=np.percentile(ds,[2.5,97.5]);print(f"   bootstrap 95% CI d=[{lo:+.2f},{hi:+.2f}] {'no cross 0' if (lo>0 or hi<0) else 'crosses 0'}")
print("\nProxy pilot, 12 matched utts/spk (set NUTT to change), 4+4 speakers. EXPLORATORY.")

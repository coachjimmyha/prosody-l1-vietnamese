"""vowel_centralization_pilot.py — vowel-quality reduction (Section 5.4.1; Figure 5.9).

F1/F2 are measured at the vowel midpoint (Burg method) and Lobanov-normalised (z-scores) within
speaker. The centralisation ratio is the mean distance of function-word vowels from the speaker's
vowel-space centroid divided by that of primary-stressed content-word vowels (< 1 = function-word vowels more
central). Computed over the matched ARCTIC sentences of each speaker (NUTT, default 24). Prints speaker
values, Cohen's d and a speaker bootstrap CI. Needs the audio (see paths.py).

Run:  python scripts/vowel_centralization_pilot.py > logs/pilot_vowel_centralisation.log
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
VOW={"AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW","AX"}
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
def formants(wav,tg):
    """return list of (is_func, is_stress_content, F1, F2) per vowel"""
    W,P=tiers(tg)
    if not W or not P: return []
    snd=parselmouth.Sound(wav); fm=snd.to_formant_burg(max_number_of_formants=5, maximum_formant=5500)
    out=[]
    for wmin,wmax,wt in W:
        w=(wt or "").strip().lower()
        if not w or w in {"sil","sp","spn","<unk>",""}: continue
        ph=[p for p in P if p[0] is not None and p[0]>=wmin-0.005 and p[1]<=wmax+0.005]
        for a,b,lab in ph:
            if strip(lab) not in VOW: continue
            mid=(a+b)/2
            f1=fm.get_value_at_time(1,mid); f2=fm.get_value_at_time(2,mid)
            if f1 is None or f2 is None or np.isnan(f1) or np.isnan(f2): continue
            isf=w in FUNC; iss=(sd(lab)==1 and not isf)
            if isf or iss: out.append((isf,iss,f1,f2))
    return out
def d(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    sp=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
    return (a.mean()-b.mean())/sp if sp>0 else np.nan
res={}
for spk,(grp,rel) in SPKS.items():
    d0=rel; allv=[]
    for wav in sorted(glob.glob(f"{d0}/wav/arctic_a00*.wav"))[:int(os.environ.get("NUTT","24"))]:
        uid=os.path.basename(wav)[:-4];tg=f"{d0}/textgrid/{uid}.TextGrid"
        if os.path.exists(tg): allv+=formants(wav,tg)
    if len(allv)<10: res[spk]=(grp,np.nan);continue
    F1=np.array([v[2] for v in allv]); F2=np.array([v[3] for v in allv])
    # Lobanov z-score within speaker -> centroid at (0,0)
    z1=(F1-F1.mean())/F1.std(); z2=(F2-F2.mean())/F2.std()
    dist=np.sqrt(z1**2+z2**2)
    func_d=np.array([dist[i] for i,v in enumerate(allv) if v[0]])
    str_d =np.array([dist[i] for i,v in enumerate(allv) if v[1]])
    ratio=func_d.mean()/str_d.mean()   # <1 = function vowels MORE central than stressed = reduced
    res[spk]=(grp,ratio)
print(f"{'spk':6}{'grp':8}{'centralization ratio (func/stress dist)':>42}")
print("-"*56)
nat=[];vn=[]
for spk,(grp,r) in res.items():
    print(f"{spk:6}{grp:8}{r:>42.3f}")
    (nat if grp=='native' else vn).append(r)
print("-"*56)
print(f"native mean={np.mean(nat):.3f}  VN mean={np.mean(vn):.3f}  d={d(nat,vn):+.2f}")
print("(<1 = function vowels closer to centre = more reduced/centralised; native expected LOWER)")
rng=np.random.RandomState(42);ds=[dd for _ in range(5000) if np.isfinite(dd:=d(rng.choice(nat,4,True),rng.choice(vn,4,True)))]
lo,hi=np.percentile(ds,[2.5,97.5]);print(f"bootstrap 95% CI d=[{lo:+.2f},{hi:+.2f}] {'no cross 0' if (lo>0 or hi<0) else 'crosses 0'}")
print(f"\nProxy (Burg formants, Lobanov-normalised), {os.environ.get('NUTT','24')} matched utts/spk (set NUTT to change), 4+4 speakers. EXPLORATORY, formants noisy.")

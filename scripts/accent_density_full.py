"""accent_density_full.py — pitch-accent density, native vs Vietnamese (Section 5.4.2; Figure 5.10).

Accents are F0 peaks with a prominence of at least two semitones (F0 in semitones relative to the
utterance median, minimum spacing 120 ms); syllables are the vowels in the MFA phone tier. The measure is syllables per
accent (lower = denser accenting), averaged over the matched ARCTIC sentences of each speaker
(environment variable NUTT, default 24). Prints speaker values, Cohen's d and a speaker bootstrap CI.
Needs the audio (see paths.py).

Run:  python scripts/accent_density_full.py > logs/pilot_accent_density.log
"""
import glob, re, os
import numpy as np
import parselmouth
from scipy.signal import find_peaks
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
def nsyl(tg):
    inph=False;n=0
    for ln in open(tg,encoding="utf-8",errors="ignore"):
        s=ln.strip();m=re.match(r'name = "(.*)"',s)
        if m: inph=m.group(1).lower().startswith("phone");continue
        if inph:
            t=re.match(r'text = "(.*)"',s)
            if t and re.sub(r"\d+","",t.group(1)).strip().upper() in VOW: n+=1
    return n
def nacc(wav):
    snd=parselmouth.Sound(wav);p=snd.to_pitch(time_step=0.01,pitch_floor=75,pitch_ceiling=500)
    f0=p.selected_array['frequency'];v=f0[f0>0]
    if len(v)<5: return None
    med=np.median(v);st=12*np.log2(np.where(f0>0,f0,med)/med)
    pk,_=find_peaks(st,prominence=2.0,distance=12);return len(pk)
def d(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    sp=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
    return (a.mean()-b.mean())/sp if sp>0 else np.nan
spk_syllacc={}
for spk,(grp,dr) in SPKS.items():
    vals=[]
    for wav in sorted(glob.glob(f"{dr}/wav/arctic_a00*.wav"))[:int(os.environ.get("NUTT","24"))]:
        uid=os.path.basename(wav)[:-4];tg=f"{dr}/textgrid/{uid}.TextGrid"
        if not os.path.exists(tg):continue
        s=nsyl(tg);a=nacc(wav)
        if a is None or s==0:continue
        vals.append(s/max(a,1))   # syllables per accent
    spk_syllacc[spk]=(grp,np.mean(vals),len(vals))
print(f"{'spk':6}{'grp':8}{'syll/accent':>12}{'utts':>6}")
print("-"*34)
nat=[];vn=[]
for spk,(grp,m,n) in spk_syllacc.items():
    print(f"{spk:6}{grp:8}{m:12.2f}{n:6}")
    (nat if grp=='native' else vn).append(m)
print("-"*34)
print(f"native mean syll/accent = {np.mean(nat):.2f}  (speakers: {[round(x,2) for x in nat]})")
print(f"VN     mean syll/accent = {np.mean(vn):.2f}  (speakers: {[round(x,2) for x in vn]})")
print(f"Cohen's d (native vs VN, speaker-level) = {d(nat,vn):+.2f}")
# bootstrap CI on d
rng=np.random.RandomState(42);ds=[]
for _ in range(5000):
    na=rng.choice(nat,4,replace=True);vv=rng.choice(vn,4,replace=True);dd=d(na,vv)
    if np.isfinite(dd):ds.append(dd)
lo,hi=np.percentile(ds,[2.5,97.5])
print(f"bootstrap 95% CI on d = [{lo:+.2f}, {hi:+.2f}]  {'no cross 0' if lo>0 else 'crosses 0'}")
print("\nLOWER syll/accent = DENSER accenting. Mixdorff: native 4.47 vs VN 2.62 (VN denser).")
print(f"Proxy (F0 peaks prom>=2st), {os.environ.get('NUTT','24')} matched utts/spk (set NUTT to change), 4+4 speakers. EXPLORATORY.")

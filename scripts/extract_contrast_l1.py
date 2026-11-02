"""extract_contrast_l1.py — utterance-level and rhythm features for the Korean, Spanish and Chinese
L2-ARCTIC speakers.

Applies the feature definitions of extract_extra_features_native_vn.py and extract_rhythm_features.py
to the three comparison groups (four speakers each). These files are used only in the supplementary
comparisons (speaking rate in reduction_full.csv; Section 5.4.3).

Outputs (outputs/): extra_features_<l1>_per_utt.csv, rhythm_<l1>_per_utt.csv
                    for l1 in {korean, spanish, chinese}

Run:  python scripts/extract_contrast_l1.py
"""
import csv, re
import numpy as np
import parselmouth
import textgrid

from paths import OUTPUTS, L2ARCTIC_ROOT, l2_dir
OUTPUT_DIR = OUTPUTS
OUTPUT_DIR.mkdir(exist_ok=True)

# Comparison groups (Vietnamese and native speakers are handled by the main extraction scripts;
# the Arabic and Hindi subsets of L2-ARCTIC were not used in this study).
L1_GROUPS = {
    "korean":  ["HKK", "HJK", "YDCK", "YKWK"],   # non-tonal
    "spanish": ["EBVS", "ERMS", "MBMPS", "NJS"], # non-tonal
    "chinese": ["BWC", "LXC", "NCC", "TXHC"],    # tonal (Mandarin)
}

ARPABET_VOWELS = {"AA","AE","AH","AO","AW","AX","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW"}
SILENCE = {"sil","sp","spn","[silence]",""}
F0_MIN_HZ, F0_MAX_HZ = 75.0, 600.0

def strip_stress(label): return re.sub(r"\d+","",label).strip()
def get_tier(tg, name_starts):
    for t in tg.tiers:
        if t.name.strip().lower().startswith(name_starts): return t
    return None

# ===================== utterance-level features =====================
def extra_features(wav_path, tg_path):
    try:
        tg=textgrid.TextGrid.fromFile(str(tg_path)); sound=parselmouth.Sound(str(wav_path))
    except Exception: return None
    phones=get_tier(tg,"phone")
    if phones is None: return None
    total=tg.maxTime
    try:
        pitch_obj=sound.to_pitch_ac(pitch_floor=F0_MIN_HZ,pitch_ceiling=F0_MAX_HZ)
        if total<0.2: f0_range=0.0
        else:
            vals=[]
            for t in np.linspace(0.05,total-0.05,100):
                f0=pitch_obj.get_value_at_time(t)
                if f0 is not None and not np.isnan(f0) and F0_MIN_HZ<=f0<=F0_MAX_HZ: vals.append(f0)
            f0_range=float(np.max(vals)-np.min(vals)) if len(vals)>=3 else 0.0
    except Exception: f0_range=0.0
    vc=0; sil=0.0
    for iv in phones.intervals:
        lab=(iv.mark or "").strip(); d=iv.maxTime-iv.minTime
        if not lab or lab in SILENCE: sil+=d; continue
        if strip_stress(lab) in ARPABET_VOWELS: vc+=1
    return {"f0_range_hz":round(float(f0_range),2),
            "speaking_rate":round(vc/total if total>0 else 0.0,3),
            "silence_proportion":round(sil/total if total>0 else 0.0,3)}

# ===================== rhythm metrics (TextGrid only) =====================
def rhythm(tg_path):
    try: tg=textgrid.TextGrid.fromFile(str(tg_path))
    except Exception: return None
    phones=get_tier(tg,"phone")
    if phones is None: return None
    vd=[]; vs=0.0; cs=0.0
    for iv in phones.intervals:
        lab=(iv.mark or "").strip(); d=float(iv.maxTime-iv.minTime)
        if not lab or lab in SILENCE: continue
        if strip_stress(lab) in ARPABET_VOWELS: vd.append(d); vs+=d
        else: cs+=d
    if len(vd)<2: return None
    v=np.array(vd,float); denom=vs+cs
    return {"npvi":round(100.0*float(np.mean(np.abs(v[:-1]-v[1:])/((v[:-1]+v[1:])/2.0))),2),
            "percent_v":round(100.0*vs/denom if denom>0 else 0.0,2),
            "varcov":round(100.0*(v.std()/v.mean()) if v.mean()>0 else 0.0,2)}

# ===================== driver =====================
def write_csv(path, rows):
    if not rows: print(f"    [WARN] no rows -> {path.name}"); return
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"    saved {path.name} ({len(rows)} rows)")

def main():
    print("="*74); print("Contrast L1 feature extraction (L2-ARCTIC)"); print("="*74)
    if not L2ARCTIC_ROOT.exists():
        print(f"[ERROR] {L2ARCTIC_ROOT} not found. Download L2-ARCTIC speakers first."); return
    for l1, codes in L1_GROUPS.items():
        print(f"\n### {l1.upper()} — {codes}")
        extra_all=[]; rhythm_all=[]
        for code in codes:
            spk_dir=l2_dir(code)
            wav_dir=spk_dir/"wav"; tg_dir=spk_dir/"textgrid"
            if not wav_dir.exists() or not tg_dir.exists():
                print(f"  [{code}] not found (download it) — skipping"); continue
            ne=nr=0
            for wav in sorted(wav_dir.glob("*.wav")):
                tg=tg_dir/f"{wav.stem}.TextGrid"
                if not tg.exists(): continue
                ef=extra_features(wav,tg)
                if ef: extra_all.append({"speaker":code,"utt_id":wav.stem,**ef}); ne+=1
                rf=rhythm(tg)
                if rf: rhythm_all.append({"speaker":code,"utt_id":wav.stem,**rf}); nr+=1
            print(f"  [{code}] extra={ne} rhythm={nr}")
        write_csv(OUTPUT_DIR/f"extra_features_{l1}_per_utt.csv", extra_all)
        write_csv(OUTPUT_DIR/f"rhythm_{l1}_per_utt.csv", rhythm_all)
    print("\nDone.")

if __name__ == "__main__":
    main()

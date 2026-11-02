"""extract_reduction_feature.py — function-word reduction ratio (Section 3.4, Equation 4).

    RR = mean duration of the vowels in function words
         / mean duration of the primary-stressed vowels in content words

computed per utterance from the MFA phone and word tiers (durations only, no audio). Function words
are a fixed closed-class list (articles, prepositions, pronouns, auxiliaries, conjunctions). Lower
values mean more reduction.

Modes
  python scripts/extract_reduction_feature.py
      L2-ARCTIC and CMU ARCTIC speakers (20), first 300 recordings per speaker (arctic_a0001–a0300)
      -> reduction_<group>_per_utt.csv and reduction_full.csv
  python scripts/extract_reduction_feature.py native vn
      selected groups only (runs can be split; the speaker table is written once all five groups exist)
  python scripts/extract_reduction_feature.py speechocean
      all Speechocean762 TextGrids -> reduction_speechocean.csv (used by reduction_ablation.py)

reduction_full.csv holds one row per speaker: group, tonal, speaker, mean reduction ratio and mean
speaking rate (from the extra_features_<group>_per_utt.csv files).
"""
import csv, re, glob, os, sys
from pathlib import Path
import numpy as np

from paths import OUTPUTS, SPEECHOCEAN_TEXTGRIDS, cmu_dir, l2_dir
OUT = OUTPUTS; OUT.mkdir(exist_ok=True)

# group -> {speaker: textgrid_dir}; dictionary order = row order of reduction_full.csv
GROUPS = {
    "native":  {s: cmu_dir(s) / "textgrid" for s in ["bdl", "slt", "clb", "rms"]},
    "vn":      {s: l2_dir(s) / "textgrid" for s in ["HQTV", "PNV", "THV", "TLV"]},
    "korean":  {s: l2_dir(s) / "textgrid" for s in ["HKK", "HJK", "YDCK", "YKWK"]},
    "spanish": {s: l2_dir(s) / "textgrid" for s in ["EBVS", "ERMS", "MBMPS", "NJS"]},
    "chinese": {s: l2_dir(s) / "textgrid" for s in ["BWC", "LXC", "NCC", "TXHC"]},
}
GROUP_INFO = {"native": ("Native", "native"), "vn": ("Vietnamese", "tonal"),
              "korean": ("Korean", "nontonal"), "spanish": ("Spanish", "nontonal"),
              "chinese": ("Chinese", "tonal")}
N_PER_SPEAKER = 300          # first 300 TextGrids per speaker in sorted order (arctic_a0001–a0300)
SPEECHOCEAN_TG = SPEECHOCEAN_TEXTGRIDS

ARPABET_VOWELS = {"AA","AE","AH","AO","AW","AX","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW"}
# closed-class function words (articles, prepositions, pronouns, auxiliaries, conjunctions)
FUNCTION_WORDS = set((
    "a an the of to in for on at by with from as into about over under "
    "i you he she it we they me him her us them my your his its our their this that these those "
    "is are was were be been being am do does did have has had having will would can could shall should may might must "
    "and but or nor so yet not no if than then there here what which who whom whose when where why how"
).split())

def strip_stress(lab): return re.sub(r"\d+", "", lab).strip()
def stress_digit(lab):
    m = re.search(r"(\d)$", lab); return int(m.group(1)) if m else None

def read_tiers(path):
    """Minimal long-format Praat TextGrid parser -> (word_intervals, phone_intervals)."""
    data = {}; cur = None; xmin = None
    for ln in open(path, encoding="utf-8", errors="ignore"):
        s = ln.strip(); m = re.match(r'name = "(.*)"', s)
        if m: cur = m.group(1); data[cur] = []; xmin = None; continue
        if cur is None: continue
        a = re.match(r'xmin = ([\d.]+)', s); b = re.match(r'xmax = ([\d.]+)', s); t = re.match(r'text = "(.*)"', s)
        if a: xmin = float(a.group(1))
        elif b: xmax = float(b.group(1))
        elif t is not None: data[cur].append((xmin, xmax, t.group(1)))
    W = [data[k] for k in data if k.lower().startswith("word")]
    P = [data[k] for k in data if k.lower().startswith("phone")]
    return (W[0] if W else []), (P[0] if P else [])

def reduction_ratio(path):
    W, P = read_tiers(path)
    if not W or not P: return None
    func_durs, content_stressed_durs = [], []
    for wmin, wmax, wtext in W:
        w = (wtext or "").strip().lower()
        if not w or w in {"sil", "sp", "spn", "<unk>", ""}: continue
        phones = [p for p in P if p[0] is not None and p[0] >= wmin - 0.005 and p[1] <= wmax + 0.005]
        for a, b, lab in phones:
            if strip_stress(lab) not in ARPABET_VOWELS: continue
            dur_ms = (b - a) * 1000
            if w in FUNCTION_WORDS:
                func_durs.append(dur_ms)
            elif stress_digit(lab) == 1:               # primary-stressed vowel in a content word
                content_stressed_durs.append(dur_ms)
    if not func_durs or not content_stressed_durs:
        return None
    return float(np.mean(func_durs) / np.mean(content_stressed_durs))

def extract_group(group):
    rows = []
    for spk, tg_dir in GROUPS[group].items():
        files = sorted(glob.glob(str(Path(tg_dir) / "*.TextGrid")))[:N_PER_SPEAKER]
        if not files:
            print(f"  [{group}/{spk}] no TextGrids under {tg_dir} — skipped"); continue
        n = 0
        for tg in files:
            r = reduction_ratio(tg)
            if r is not None:
                rows.append({"speaker": spk, "utt_id": Path(tg).stem, "reduction_ratio": round(r, 4)})
                n += 1
        print(f"  [{group}/{spk}] {n} of {len(files)} utterances")
    out = OUT / f"reduction_{group}_per_utt.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["speaker", "utt_id", "reduction_ratio"]); w.writeheader(); w.writerows(rows)
    print(f"    saved {out.name}")


def build_speaker_table():
    """One row per speaker: mean reduction ratio and mean speaking rate -> reduction_full.csv."""
    parts = {g: OUT / f"reduction_{g}_per_utt.csv" for g in GROUPS}
    if not all(p.exists() for p in parts.values()):
        print("  reduction_full.csv not written (not all five groups extracted yet)"); return
    out_rows = []
    for g, spks in GROUPS.items():
        red, rate = {}, {}
        for r in csv.DictReader(open(parts[g], encoding="utf-8")):
            red.setdefault(r["speaker"], []).append(float(r["reduction_ratio"]))
        for r in csv.DictReader(open(OUT / f"extra_features_{g}_per_utt.csv", encoding="utf-8")):
            rate.setdefault(r["speaker"], []).append(float(r["speaking_rate"]))
        name, tonal = GROUP_INFO[g]
        for s in spks:
            out_rows.append([name, tonal, s, f"{np.mean(red[s]):.3f}", f"{np.mean(rate[s]):.3f}"])
    with open(OUT / "reduction_full.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["group", "tonal", "speaker", "reduction", "rate"]); w.writerows(out_rows)
    print("    saved reduction_full.csv (20 speakers)")


def extract_speechocean():
    files = sorted(glob.glob(str(SPEECHOCEAN_TG / "*.TextGrid")))
    rows = []
    for tg in files:
        r = reduction_ratio(tg)
        if r is not None:
            rows.append({"utt_id": Path(tg).stem, "reduction_ratio": round(r, 4)})
    out = OUT / "reduction_speechocean.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["utt_id", "reduction_ratio"]); w.writeheader(); w.writerows(rows)
    print(f"  Speechocean762: {len(rows)} of {len(files)} utterances -> {out.name}")


def main():
    args = [a.lower() for a in sys.argv[1:]]
    print("Function-word reduction ratio (TextGrid durations only)")
    if args == ["speechocean"]:
        extract_speechocean(); return
    for g in (args or list(GROUPS)):
        if g not in GROUPS:
            sys.exit(f"unknown group {g!r}; choose from {list(GROUPS)} or 'speechocean'")
        extract_group(g)
    build_speaker_table()


if __name__ == "__main__":
    main()

"""learner_error_profiles.py — per-learner error profiles (Section 3.9; Table 5.2; Appendix B.11).

Each speaker is reduced to one mean vector. Each Vietnamese learner's values are expressed as
z-scores against the native reference (mean and SD of the four native speaker means). A dimension is
flagged only if it has the TEACH role and deviates by at least one SD in the error direction:
  TEACH  dimensions on which the literature predicts a learner deficit and the Vietnamese group
         differs from the native speakers in that direction (function-word reduction, stress
         duration ratio, speaking rate, silence proportion)
  LOW    dimensions with no difference in a deficit direction (rhythm metrics, robust F0 range);
         reported, never flagged
Content is matched across groups: the utterance-level dimensions (stress duration ratio, speaking rate,
silence proportion, rhythm metrics) are averaged over the same prompts for every speaker
(arctic_a0001–a0200, the prompts covered by the native stress contrasts); function-word reduction uses
arctic_a0001–a0300 (reduction_full.csv) and the robust F0 range all recordings (Section 3.10).
Reads the CSVs in outputs/ (including reduction_full.csv and f0_range_robust_per_utt.csv).

Run:  python scripts/learner_error_profiles.py
"""
import csv
import re
from pathlib import Path
from collections import defaultdict
import numpy as np

from paths import OUTPUTS as OUT

# ---- feature registry: name -> (ROLE, error direction, teaching construct) ----
# error direction: 'high' = value above native is the error; 'low' = below; 'any' = |z|
FEATS = [
    ("reduction",           "TEACH",     "high",
        "Function-word reduction — reduce the/to/of/a to a short schwa (first priority)"),
    ("mean_duration_ratio", "TEACH",     "low",
        "Lexical stress (duration) — lengthen stressed vowels, shorten unstressed"),
    ("speaking_rate",       "TEACH",     "low",
        "Tempo/fluency — too slow (correlated with weak reduction across speakers, r=-0.68)"),
    ("silence_proportion",  "TEACH",     "high",
        "Phrasing — too much pausing; group words into smoother thought-units"),
    ("f0_range_st",         "LOW",       "any",
        "Pitch range (robust, semitones) — does not separate groups"),
    ("npvi",                "LOW",       "any",  "Rhythm (nPVI) — does not separate groups"),
    ("varcov",              "LOW",       "any",  "Rhythm (VarcoV) — does not separate groups"),
]
ROLE_ORDER = {"TEACH": 0, "SIGNATURE": 1, "LOW": 2}

# ------------------------- data loading -------------------------
MATCHED = re.compile(r"arctic_a0(0\d\d|1\d\d|200)$")   # arctic_a0001–a0200, same prompts for all speakers

def load_su(p, cols):
    d = {}
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try: d[(r["speaker"], r["utt_id"])] = {c: float(r[c]) for c in cols}
            except (KeyError, ValueError): pass
    return d

def build(perword, extra, rhythm):
    """per-utterance 14-dim vectors, keyed for later speaker-mean collapse"""
    g = defaultdict(list)
    with open(perword, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            u, s = r.get("utt_id"), r.get("speaker")
            if not u or not s: continue
            try:
                g[(s, u)].append(float(r["dur_ratio"]))
            except (KeyError, ValueError): pass
    rows, spk = [], []
    for (s, u), durs in g.items():
        if not MATCHED.match(u): continue
        if (s, u) not in extra or (s, u) not in rhythm: continue
        e, rh = extra[(s, u)], rhythm[(s, u)]
        rows.append({"mean_duration_ratio": float(np.mean(durs)),
                     "f0_range_hz": e["f0_range_hz"], "speaking_rate": e["speaking_rate"],
                     "silence_proportion": e["silence_proportion"],
                     "npvi": rh["npvi"], "varcov": rh["varcov"]})
        spk.append(s)
    return rows, spk

def speaker_means(rows, spk, keys):
    g = defaultdict(list)
    for row, s in zip(rows, spk): g[s].append([row[k] for k in keys])
    return {s: dict(zip(keys, np.array(v).mean(axis=0))) for s, v in g.items()}

KEYS = ["mean_duration_ratio", "f0_range_hz", "speaking_rate", "silence_proportion", "npvi", "varcov"]
ne = load_su(OUT / "extra_features_native_per_utt.csv", ["f0_range_hz", "speaking_rate", "silence_proportion"])
ve = load_su(OUT / "extra_features_vn_per_utt.csv",     ["f0_range_hz", "speaking_rate", "silence_proportion"])
nr = load_su(OUT / "rhythm_native_per_utt.csv", ["npvi", "percent_v", "varcov"])
vr = load_su(OUT / "rhythm_vn_per_utt.csv",     ["npvi", "percent_v", "varcov"])
nat = speaker_means(*build(OUT / "stress_contrast_native_per_word.csv", ne, nr), KEYS)
vn  = speaker_means(*build(OUT / "stress_contrast_vn_per_word.csv",     ve, vr), KEYS)

# ---- merge in the phrase-level reduction feature (speaker means) ----
red = defaultdict(dict)
for r in csv.DictReader(open(OUT / "reduction_full.csv")):
    red[r["group"]][r["speaker"]] = float(r["reduction"])
for s, v in red.get("Native", {}).items():
    if s in nat: nat[s]["reduction"] = v
for s, v in red.get("Vietnamese", {}).items():
    if s in vn: vn[s]["reduction"] = v

# ---- robust F0 range (speaker means over all prompts) ----
rob = defaultdict(list)
for r in csv.DictReader(open(OUT / "f0_range_robust_per_utt.csv", encoding="utf-8")):
    rob[r["speaker"]].append(float(r["f0_range_p5p95_st"]))
for d in (nat, vn):
    for s in d:
        if s in rob: d[s]["f0_range_st"] = float(np.mean(rob[s]))

# ---- native reference = mean & SD across the 4 native speaker means ----
ref = {}
for name, *_ in FEATS:
    vals = [nat[s][name] for s in nat if name in nat[s]]
    ref[name] = (float(np.mean(vals)), float(np.std(vals, ddof=1)))

# ------------------------- report -------------------------
print("=" * 92)
print("PER-LEARNER PROSODY ERROR PROFILE  —  Vietnamese learner vs native reference (z = SDs from native mean)")
print("=" * 92)
for s in sorted(vn):
    print(f"\n### Learner {s}")
    rows = []
    for name, role, errdir, teach in FEATS:
        if name not in vn[s]: continue
        mu, sd = ref[name]; z = (vn[s][name] - mu) / sd if sd > 0 else 0.0
        flagged = (role == "TEACH") and (
            (errdir == "high" and z >= 1) or (errdir == "low" and z <= -1) or (errdir == "any" and abs(z) >= 1))
        rows.append((ROLE_ORDER[role], -abs(z), name, vn[s][name], z, role, flagged, teach))
    for _, _, name, val, z, role, flagged, teach in sorted(rows):
        mark = "⚑" if flagged else " "
        tag = f"-> {teach}" if flagged else ("(signature)" if role == "SIGNATURE" else "")
        print(f"  {mark} [{role:9}] {name:20} yours={val:8.2f}  z={z:+5.2f}  {tag}")

print("\n" + "=" * 92)
print("ROLES: TEACH = deficit to practise (flagged when z deviates by >= 1 SD in the error direction).")
print("       LOW = no difference from native speakers in a deficit direction (rhythm metrics, robust F0 range); never flagged.")
print("Utterance-level dimensions use the same prompts for all speakers (arctic_a0001-a0200).")
print("The native reference has n = 4, so z-scores indicate direction and rank rather than exact size.")

# ------------------------- sensitivity to the prompt set (Section 5.6) -------------------------
# Speaking rate and silence over ALL prompts of every speaker (no stress-word filter); duration ratio
# over all Vietnamese prompts (the native stress contrasts exist for arctic_a0001–a0200 only).
def _all_prompt_means(path, col):
    g = defaultdict(list)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        try: g[r["speaker"]].append(float(r[col]))
        except (KeyError, ValueError): pass
    return {s: float(np.mean(v)) for s, v in g.items()}

print("\nSENSITIVITY — z over all prompts (rate, silence) and all Vietnamese prompts (duration ratio)")
for col, path_n, path_v, errdir in [
        ("speaking_rate", OUT / "extra_features_native_per_utt.csv", OUT / "extra_features_vn_per_utt.csv", "low"),
        ("silence_proportion", OUT / "extra_features_native_per_utt.csv", OUT / "extra_features_vn_per_utt.csv", "high"),
        ("dur_ratio", OUT / "stress_contrast_native_per_word.csv", OUT / "stress_contrast_vn_per_word.csv", "low")]:
    n = _all_prompt_means(path_n, col); v = _all_prompt_means(path_v, col)
    mu, sd = float(np.mean(list(n.values()))), float(np.std(list(n.values()), ddof=1))
    cells = []
    for s in sorted(v):
        z = (v[s] - mu) / sd
        flag = (errdir == "low" and z <= -1) or (errdir == "high" and z >= 1)
        cells.append(f"{s} {z:+.2f}{' *' if flag else '  '}")
    print(f"  {col:20} " + "   ".join(cells))
print("  (* = beyond 1 SD in the deficit direction)")

"""speaker_bootstrap.py — speaker-level native vs Vietnamese tests (Section 3.7.2; Figure 5.3;
Appendix B.7).

Each speaker is reduced to one mean value per feature (n = 4 per group), so utterances from the same
speaker are not treated as independent. For each feature the script reports
  Cohen's d (native − Vietnamese, pooled SD),
  a 95% bootstrap CI from 5,000 resamples of whole speakers (seed 42), and
  an exact two-sided permutation p over all C(8,4) = 70 relabellings of the speaker means
  (smallest attainable p = 2/70 = 0.029).
F0 range uses the robust semitone measure (f0_range_robust.py); the original max − min Hz measure is
reported in the last row ("f0_range_OLD") for comparison.
Both groups are compared on the same prompts: all prompts for the utterance-level and rhythm measures,
and arctic_a0001–a0200 for the stress duration ratio (the prompts covered by the native stress contrasts).
Output: outputs/speaker_level_tests.csv

Run:  python scripts/speaker_bootstrap.py
"""
import csv
from pathlib import Path
from collections import defaultdict
import itertools
import numpy as np

from paths import OUTPUTS as OUT
N_BOOT = 5000
SEED = 42

def load(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))

def per_speaker_mean(rows, col):
    g = defaultdict(list)
    for r in rows:
        try: g[r["speaker"]].append(float(r[col]))
        except (KeyError, ValueError): pass
    return {s: float(np.mean(v)) for s, v in g.items()}

def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    p = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / p if p > 0 else 0.0

RHY_N, RHY_V = load(OUT/"rhythm_native_per_utt.csv"), load(OUT/"rhythm_vn_per_utt.csv")
EX_N,  EX_V  = load(OUT/"extra_features_native_per_utt.csv"), load(OUT/"extra_features_vn_per_utt.csv")
SC_N,  SC_V  = load(OUT/"stress_contrast_native_per_word.csv"), load(OUT/"stress_contrast_vn_per_word.csv")
# native stress contrasts cover arctic_a0001–a0200 only, so the Vietnamese words are restricted to the same prompts
_PROMPTS = {f"arctic_a{i:04d}" for i in range(1, 201)}
SC_V = [r for r in SC_V if r["utt_id"] in _PROMPTS]

ROB = load(OUT/"f0_range_robust_per_utt.csv")
ROB_N = [r for r in ROB if r["group"] == "Native"]; ROB_V = [r for r in ROB if r["group"] == "Vietnamese"]

FEATURES = [
    ("speaking_rate",  EX_N, EX_V, "speaking_rate"),
    ("f0_range_st",    ROB_N, ROB_V, "f0_range_p5p95_st"),
    ("silence_prop",   EX_N, EX_V, "silence_proportion"),
    ("duration_ratio", SC_N, SC_V, "dur_ratio"),
    ("nPVI",           RHY_N, RHY_V, "npvi"),
    ("percent_V",      RHY_N, RHY_V, "percent_v"),
    ("VarcoV",         RHY_N, RHY_V, "varcov"),
    ("f0_range_OLD",   EX_N, EX_V, "f0_range_hz"),   # original max−min Hz (superseded)
]


def exact_perm_p(a, b):
    """Two-sided exact permutation p on the difference of means (all relabellings)."""
    v = np.r_[a, b]; n = len(a); obs = abs(np.mean(a) - np.mean(b)); hit = tot = 0
    for idx in itertools.combinations(range(len(v)), n):
        m = np.zeros(len(v), bool); m[list(idx)] = True
        tot += 1; hit += abs(v[m].mean() - v[~m].mean()) >= obs - 1e-12
    return hit / tot

rng = np.random.default_rng(SEED)

def boot_ci(nvals, vvals, n_boot=N_BOOT):
    nvals, vvals = np.asarray(nvals, float), np.asarray(vvals, float)
    ds = np.empty(n_boot)
    for b in range(n_boot):
        na = nvals[rng.integers(0, len(nvals), len(nvals))]
        vb = vvals[rng.integers(0, len(vvals), len(vvals))]
        ds[b] = cohens_d(na, vb)
    ds = ds[np.isfinite(ds)]
    return ds

print("="*94)
print(f"SPEAKER-LEVEL BOOTSTRAP  ({N_BOOT} resamples of whole speakers, n=4 per group)")
print("="*94)
print(f"{'feature':15} {'point d':>9} {'boot mean':>10} {'95% CI (speaker-level)':>26} {'crosses 0?':>12} {'exact p':>8}")
print("-"*94)
results = []
for name, nrows, vrows, col in FEATURES:
    nm = per_speaker_mean(nrows, col); vm = per_speaker_mean(vrows, col)
    nvals = [nm[s] for s in sorted(nm)]; vvals = [vm[s] for s in sorted(vm)]
    point = cohens_d(nvals, vvals)
    ds = boot_ci(nvals, vvals)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    crosses = "YES (n.s.)" if lo <= 0 <= hi else "no"
    pex = exact_perm_p(nvals, vvals)
    print(f"{name:15} {point:+9.3f} {ds.mean():+10.3f}   [{lo:+6.3f}, {hi:+6.3f}]{'':6} {crosses:>12} {pex:8.3f}")
    results.append([name, round(point, 4), round(lo, 4), round(hi, 4), round(pex, 4),
                    round(float(np.mean(nvals)), 4), round(float(np.mean(vvals)), 4)])

with open(OUT/"speaker_level_tests.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["feature", "d_native_minus_vn", "boot_lo", "boot_hi", "exact_perm_p", "native_mean", "vn_mean"])
    w.writerows(results)
print("-> saved outputs/speaker_level_tests.csv")

print("\nNotes:")
print("- The bootstrap resamples whole speakers (n = 4 per group), so the intervals are wide.")
print("- With n = 4 per group the percentile bootstrap is optimistic; the exact permutation p is")
print("  the test reported in the thesis (smallest attainable two-sided p = 2/70 = 0.029).")

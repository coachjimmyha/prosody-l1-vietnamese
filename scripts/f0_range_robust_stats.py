"""f0_range_robust_stats.py — speaker-level statistics for the two F0-range measures (Sections 3.10
and 5.3; Appendix B.9).

Reads outputs/f0_range_robust_per_utt.csv (from f0_range_robust.py). For the original measure
(max − min Hz, 75–600 Hz) and the robust measure (p5–p95 semitones, sex-specific range) it prints the
speaker means, native vs Vietnamese Cohen's d with the exact permutation p (70 relabellings) and a
5,000-resample speaker bootstrap CI, native vs pooled L2, tonal (Vietnamese + Chinese) vs non-tonal
(Korean + Spanish) L2 groups, and per-learner z-scores.

Run:  python scripts/f0_range_robust_stats.py
"""
import csv, itertools
from collections import defaultdict
from pathlib import Path
import numpy as np

from paths import OUTPUTS
IN = OUTPUTS / "f0_range_robust_per_utt.csv"
GROUPS = ["Native", "Vietnamese", "Korean", "Spanish", "Chinese"]
rows = list(csv.DictReader(open(IN, encoding="utf-8")))


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    p = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / p if p > 0 else 0.0


def exact_p(a, b):
    v = np.r_[a, b]; n = len(a); obs = abs(np.mean(a) - np.mean(b)); hit = tot = 0
    for idx in itertools.combinations(range(len(v)), n):
        m = np.zeros(len(v), bool); m[list(idx)] = True
        tot += 1; hit += abs(v[m].mean() - v[~m].mean()) >= obs - 1e-12
    return hit / tot


def boot_ci(a, b, n=5000, seed=42):
    rng = np.random.default_rng(seed); a, b = np.asarray(a), np.asarray(b)
    ds = [cohens_d(a[rng.integers(0, len(a), len(a))], b[rng.integers(0, len(b), len(b))]) for _ in range(n)]
    return np.percentile(ds, [2.5, 97.5])


spk_utts = defaultdict(set)
for r in rows:
    spk_utts[r["speaker"]].add(r["utt_id"])
common = set.intersection(*spk_utts.values())

for label, keep in [("ALL prompts", None), (f"MATCHED prompts (n = {len(common)} per speaker)", common)]:
    for col, name in [("old_maxmin_hz", "OLD  max − min (Hz), 75–600 Hz"),
                      ("f0_range_p5p95_st", "NEW  p5–p95 (semitones), sex-specific range")]:
        acc = defaultdict(list); meta = {}
        for r in rows:
            if keep is None or r["utt_id"] in keep:
                acc[r["speaker"]].append(float(r[col])); meta[r["speaker"]] = (r["group"], r["sex"])
        sm = {s: float(np.mean(v)) for s, v in acc.items()}
        G = {g: [sm[s] for s in sorted(sm) if meta[s][0] == g] for g in GROUPS}
        nat, vn = G["Native"], G["Vietnamese"]
        ton, non = G["Vietnamese"] + G["Chinese"], G["Korean"] + G["Spanish"]
        print("=" * 88); print(f"{label}  |  {name}"); print("=" * 88)
        for g in GROUPS:
            ss = [s for s in sorted(sm) if meta[s][0] == g]
            print(f"  {g:11} mean {np.mean(G[g]):8.2f}   " + "  ".join(f"{s}({meta[s][1]}) {sm[s]:.2f}" for s in ss))
        lo, hi = boot_ci(nat, vn)
        print(f"  native vs VN        d = {cohens_d(nat, vn):+.2f}   exact p = {exact_p(nat, vn):.3f}   boot 95% CI [{lo:+.2f}, {hi:+.2f}]")
        print(f"  native vs pooled L2 d = {cohens_d(nat, ton + non):+.2f}")
        print(f"  tonal vs non-tonal  d = {cohens_d(ton, non):+.2f}   ({np.mean(ton):.2f} vs {np.mean(non):.2f})   exact p = {exact_p(ton, non):.3f}")
        mu, sd = np.mean(nat), np.std(nat, ddof=1)
        print("  VN z vs native:     " + "  ".join(f"{s} {(sm[s] - mu) / sd:+.2f}" for s in sorted(sm) if meta[s][0] == "Vietnamese"))
        print()

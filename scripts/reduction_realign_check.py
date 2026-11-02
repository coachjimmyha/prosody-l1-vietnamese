"""reduction_realign_check.py — robustness check of the function-word reduction ratio against the
alignment source (Section 5.6; Appendix B.15).

In the main analysis the native TextGrids come from this study's MFA run (MFA 3.3.9, english_us_arpa),
whereas the Vietnamese TextGrids are those distributed with L2-ARCTIC (MFA 1.0.0). Here the four
Vietnamese speakers are re-aligned with the same MFA set-up as the native speakers (all 1,132
recordings; prep_l2arctic_vn_for_mfa.py) and the reduction ratio is recomputed with the same function
(extract_reduction_feature.reduction_ratio):
  (A) on arctic_a0001-a0300, the recordings of the main analysis (native values from reduction_full.csv);
  (B) on every prompt for which all eight speakers have a ratio, with the native ratios recomputed
      from the native TextGrids, so that both groups are compared on identical prompts and aligner.

Input:  data/l2arctic_vn_mfa/<SPK>/textgrid/*.TextGrid (MFA output); corpus and native TextGrids
Output: outputs/reduction_realign_per_utt.csv (built speaker by speaker; reused if present)
        outputs/reduction_realign_check.csv   (speaker means)
Run:    python scripts/reduction_realign_check.py
"""
import csv, glob
from pathlib import Path
import numpy as np

from paths import DATA_ROOT, OUTPUTS, L2_SPEAKERS, NATIVE_SPEAKERS, l2_dir, cmu_dir
from extract_reduction_feature import reduction_ratio
from reduction_stats import cohens_d, boot_ci, exact_perm_p

NEW_ROOT = DATA_ROOT / "l2arctic_vn_mfa"
PER_UTT = OUTPUTS / "reduction_realign_per_utt.csv"
SOURCES = ([("native_mfa", s, cmu_dir(s) / "textgrid") for s in NATIVE_SPEAKERS]
           + [("vn_corpus", s, l2_dir(s) / "textgrid") for s in L2_SPEAKERS["vn"]]
           + [("vn_mfa", s, NEW_ROOT / s / "textgrid") for s in L2_SPEAKERS["vn"]])


def build_per_utt():
    """Per-utterance ratios for every source and speaker; speakers already in the table are kept."""
    done = set()
    if PER_UTT.exists():
        done = {(r["source"], r["speaker"]) for r in csv.DictReader(open(PER_UTT, encoding="utf-8"))}
    new_file = not PER_UTT.exists()
    with open(PER_UTT, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["source", "speaker", "utt_id", "reduction_ratio"])
        for src, spk, tg_dir in SOURCES:
            if (src, spk) in done:
                continue
            files = sorted(glob.glob(str(Path(tg_dir) / "*.TextGrid")))
            n = 0
            for tg in files:
                r = reduction_ratio(tg)
                if r is not None:
                    w.writerow([src, spk, Path(tg).stem, round(r, 4)]); n += 1
            f.flush()
            print(f"  per-utterance table: {src}/{spk} {n} of {len(files)} recordings", flush=True)


def load_per_utt():
    d = {}
    for r in csv.DictReader(open(PER_UTT, encoding="utf-8")):
        d.setdefault((r["source"], r["speaker"]), {})[r["utt_id"]] = float(r["reduction_ratio"])
    return d


def report(label, nat, vn_by_source, names):
    nm, nsd = np.mean(nat), np.std(nat, ddof=1)
    print(f"\n{label}")
    print(f"  native: mean {nm:.3f}, SD {nsd:.3f}  {nat}")
    for src, vals in vn_by_source:
        lo, hi = boot_ci(nat, vals)
        print(f"  {src}")
        print(f"    Vietnamese {vals}  mean {np.mean(vals):.3f}")
        print(f"    d (native - VN) = {cohens_d(nat, vals):+.3f}   exact p = {exact_perm_p(nat, vals):.3f}"
              f"   boot 95% CI [{lo:+.2f}, {hi:+.2f}]")
        print("    z vs native: " + "  ".join(f"{s} {(v - nm) / nsd:+.2f}" for s, v in zip(names, vals)))
        print(f"    every Vietnamese speaker above every native speaker: {min(vals) > max(nat)}"
              f"   (lowest VN {min(vals):.3f}, highest native {max(nat):.3f})")
    (_, old), (_, new) = vn_by_source
    print(f"  share of the native-Vietnamese gap removed by re-alignment: "
          f"{1 - (np.mean(new) - nm) / (np.mean(old) - nm):.0%}")


def main():
    print("Function-word reduction ratio of the Vietnamese speakers:")
    print("corpus TextGrids (MFA 1.0.0) vs re-aligned with the native set-up (MFA 3.3.9)")
    build_per_utt()
    d = load_per_utt()
    vn, nat_spk = L2_SPEAKERS["vn"], NATIVE_SPEAKERS
    mean3 = lambda dct, ids: round(float(np.mean([dct[u] for u in ids])), 3)

    # (A) arctic_a0001-a0300, native values of the main analysis
    full = list(csv.DictReader(open(OUTPUTS / "reduction_full.csv", encoding="utf-8")))
    nat_main = [float(r["reduction"]) for r in full if r["group"] == "Native"]
    a300 = lambda dct: [u for u in sorted(dct) if u.startswith("arctic_a") and int(u[8:12]) <= 300]
    print(f"\n(A) arctic_a0001-a0300 (main analysis recordings)")
    print(f"  {'speaker':8} {'n':>4} {'corpus':>7} {'MFA':>7}   paired diff (MFA-corpus)")
    rows, old_a, new_a = [], [], []
    for s in vn:
        oc, nw = d[("vn_corpus", s)], d[("vn_mfa", s)]
        ids = [u for u in a300(oc) if u in nw]
        diff = np.array([nw[u] - oc[u] for u in ids])
        old_a.append(mean3(oc, a300(oc))); new_a.append(mean3(nw, a300(nw)))
        print(f"  {s:8} {len(ids):4d} {old_a[-1]:7.3f} {new_a[-1]:7.3f}   mean {diff.mean():+.3f} (SD {diff.std(ddof=1):.3f})")
        rows.append([s, old_a[-1], new_a[-1]])
    report("(A) speaker level, native reference = reduction_full.csv", nat_main,
           [("corpus TextGrids (main analysis)", old_a), ("re-aligned with MFA 3.3.9", new_a)], vn)

    # (B) all prompts shared by the eight speakers, native recomputed with the same function
    common = set.intersection(*[set(d[("native_mfa", s)]) for s in nat_spk],
                              *[set(d[(src, s)]) for src in ("vn_corpus", "vn_mfa") for s in vn])
    ids = sorted(common)
    nat_b = [mean3(d[("native_mfa", s)], ids) for s in nat_spk]
    old_b = [mean3(d[("vn_corpus", s)], ids) for s in vn]
    new_b = [mean3(d[("vn_mfa", s)], ids) for s in vn]
    for r, o, n in zip(rows, old_b, new_b):
        r += [o, n]
    report(f"(B) all {len(ids)} prompts shared by the eight speakers", nat_b,
           [("corpus TextGrids", old_b), ("re-aligned with MFA 3.3.9", new_b)], vn)

    with open(OUTPUTS / "reduction_realign_check.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["speaker", "a300_corpus_tg", "a300_mfa_tg", "allprompts_corpus_tg", "allprompts_mfa_tg"])
        w.writerows(rows)
    print("\n-> saved outputs/reduction_realign_check.csv")


if __name__ == "__main__":
    main()

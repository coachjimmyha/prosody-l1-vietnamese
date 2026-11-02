"""extract_rhythm_features.py — rhythm metrics from the MFA phone tier (Stage A; TextGrids only).

  %V      100 * sum(vowel durations) / total duration of vowels and consonants  (Ramus et al., 1999)
  VarcoV  100 * SD(vowel durations) / mean(vowel durations)                     (Dellwo, 2006)
  nPVI    100 * mean over successive vowels of |d_k - d_k+1| / ((d_k + d_k+1)/2) (Grabe and Low, 2002)

Outputs (outputs/): speechocean762_rhythm_per_utt.csv, rhythm_native_per_utt.csv,
rhythm_vn_per_utt.csv

Run:  python scripts/extract_rhythm_features.py
"""

import csv
import re
from pathlib import Path
import numpy as np
import textgrid


from paths import OUTPUTS, SPEECHOCEAN_TEXTGRIDS, cmu_dir, l2_dir
OUTPUT_DIR = OUTPUTS
OUTPUT_DIR.mkdir(exist_ok=True)

# speechocean762 (training corpus) — flat TextGrid dir + utt_ids from the v2 CSV
TG_DIR_SO = SPEECHOCEAN_TEXTGRIDS
V2_CSV = OUTPUT_DIR / "speechocean762_full_per_utt_v2.csv"

# native (CMU ARCTIC) + Vietnamese (L2-ARCTIC) — per-speaker dirs (mirrors your existing script)
NATIVE_SPEAKERS = {
    "bdl": cmu_dir("bdl"),
    "slt": cmu_dir("slt"),
    "clb": cmu_dir("clb"),
    "rms": cmu_dir("rms"),
}
VN_SPEAKERS = {spk: l2_dir(spk) for spk in ("HQTV", "PNV", "THV", "TLV")}

ARPABET_VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AX", "AY", "EH", "ER", "EY",
    "IH", "IY", "OW", "OY", "UH", "UW",
}
SILENCE = {"sil", "sp", "spn", "[silence]", ""}


def strip_stress(label):
    return re.sub(r"\d+", "", label).strip()


def get_tier(tg, name_starts):
    for tier in tg.tiers:
        if tier.name.strip().lower().startswith(name_starts):
            return tier
    return None


def compute_rhythm(tg_path):
    """Return dict of npvi/percent_v/varcov for one utterance, or None if unusable."""
    try:
        tg = textgrid.TextGrid.fromFile(str(tg_path))
    except Exception:
        return None
    phones = get_tier(tg, "phone")
    if phones is None:
        return None

    vowel_durs = []
    vowel_sum = 0.0
    cons_sum = 0.0
    for iv in phones.intervals:
        label = (iv.mark or "").strip()
        dur = float(iv.maxTime - iv.minTime)
        if not label or label in SILENCE:
            continue  # pauses excluded from %V denominator and from vowel sequence
        base = strip_stress(label)
        if base in ARPABET_VOWELS:
            vowel_durs.append(dur)
            vowel_sum += dur
        else:
            cons_sum += dur

    if len(vowel_durs) < 2:
        return None  # need >= 2 vowels for nPVI / VarcoV

    v = np.array(vowel_durs, dtype=float)
    denom = vowel_sum + cons_sum
    percent_v = 100.0 * vowel_sum / denom if denom > 0 else 0.0
    varcov = 100.0 * (v.std() / v.mean()) if v.mean() > 0 else 0.0
    diffs = np.abs(v[:-1] - v[1:])
    means = (v[:-1] + v[1:]) / 2.0
    npvi = 100.0 * float(np.mean(diffs / means))

    return {
        "npvi": round(npvi, 2),
        "percent_v": round(percent_v, 2),
        "varcov": round(varcov, 2),
    }


def _write(path, rows):
    if not rows:
        print(f"    [WARN] no rows for {path.name}")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _stats(rows):
    if not rows:
        return
    for c in ("npvi", "percent_v", "varcov"):
        vals = [r[c] for r in rows]
        print(f"      {c:10} mean={np.mean(vals):.2f}  std={np.std(vals):.2f}")


def process_speechocean762():
    print("\n[speechocean762] reading utt_ids from v2 CSV...")
    if not V2_CSV.exists():
        print(f"    [WARN] {V2_CSV.name} not found — skipping speechocean762")
        return
    utt_ids = []
    with open(V2_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            utt_ids.append(row["utt_id"])
    rows, nfail = [], 0
    for utt_id in utt_ids:
        tg = TG_DIR_SO / f"{utt_id}.TextGrid"
        if not tg.exists():
            nfail += 1
            continue
        r = compute_rhythm(tg)
        if r is None:
            nfail += 1
            continue
        rows.append({"utt_id": utt_id, **r})
    out = OUTPUT_DIR / "speechocean762_rhythm_per_utt.csv"
    _write(out, rows)
    print(f"    {len(rows)} processed, {nfail} failed -> {out.name}")
    _stats(rows)


def process_corpus(speakers, label, out_name):
    print(f"\n[{label}] processing {len(speakers)} speakers...")
    rows = []
    for spk, spk_dir in speakers.items():
        tg_dir = spk_dir / "textgrid"
        if not tg_dir.exists():
            print(f"    skip {spk}: no textgrid dir ({tg_dir})")
            continue
        n = 0
        for tg in sorted(tg_dir.glob("*.TextGrid")):
            r = compute_rhythm(tg)
            if r is None:
                continue
            rows.append({"speaker": spk, "utt_id": tg.stem, **r})
            n += 1
        print(f"    {spk}: {n} processed")
    out = OUTPUT_DIR / out_name
    _write(out, rows)
    _stats(rows)
    return rows


def main():
    print("=" * 70)
    print("Rhythm features (nPVI, %V, VarcoV) from MFA TextGrids")
    print("=" * 70)
    process_speechocean762()
    nat = process_corpus(NATIVE_SPEAKERS, "Native (CMU ARCTIC)", "rhythm_native_per_utt.csv")
    vn = process_corpus(VN_SPEAKERS, "Vietnamese (L2-ARCTIC)", "rhythm_vn_per_utt.csv")

    if nat and vn:
        print("\n[sanity] Native vs Vietnamese (expect native nPVI/VarcoV higher, %V lower if VN is more syllable-timed):")
        for c in ("npvi", "percent_v", "varcov"):
            print(f"    {c:10} native={np.mean([r[c] for r in nat]):.2f}   VN={np.mean([r[c] for r in vn]):.2f}")

    print("\nDone. Send me the printed stats and I'll wire rhythm into training + retrain.")


if __name__ == "__main__":
    main()

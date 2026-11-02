"""f0_range_robust.py — robust utterance F0 range (Section 3.10).

The model feature f0_range_hz is max − min F0 in Hz with one 75–600 Hz search range for all voices,
which a few spurious pitch points can inflate. The robust measure uses Praat autocorrelation pitch
with a sex-specific range (male 75–300 Hz, female 100–500 Hz) and takes the 5th-to-95th percentile
span of the voiced frames in semitones (the 10th-to-90th span is also stored). The original measure
is recomputed on the same file for comparison.

Speakers: 4 CMU ARCTIC + 16 L2-ARCTIC (Vietnamese, Korean, Spanish, Chinese), all prompts.
Usage:  python scripts/f0_range_robust.py                 # all five groups
        python scripts/f0_range_robust.py Native Korean   # selected groups (runs can be split)
Each group is written to outputs/f0_range_robust_parts/<group>.csv; when all five parts exist
they are merged into outputs/f0_range_robust_per_utt.csv (statistics: f0_range_robust_stats.py).
"""
import csv, os, re, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import parselmouth

from paths import OUTPUTS, cmu_dir, l2_dir
OUT = OUTPUTS / "f0_range_robust_per_utt.csv"

# sex from CMU ARCTIC documentation and the L2-ARCTIC README speaker table
SPK = {"Native":     {"bdl": "M", "rms": "M", "clb": "F", "slt": "F"},
       "Vietnamese": {"HQTV": "M", "TLV": "M", "PNV": "F", "THV": "F"},
       "Korean":     {"HKK": "M", "YKWK": "M", "HJK": "F", "YDCK": "F"},
       "Spanish":    {"EBVS": "M", "ERMS": "M", "MBMPS": "F", "NJS": "F"},
       "Chinese":    {"BWC": "M", "TXHC": "M", "LXC": "F", "NCC": "F"}}
RANGE = {"M": (75, 300), "F": (100, 500)}
UTT = re.compile(r"^arctic_[ab]\d{4}\.wav$")


def wav_dir(group, spk):
    return cmu_dir(spk) / "wav" if group == "Native" else l2_dir(spk) / "wav"


def measure(job):
    group, spk, sex, wav = job
    snd = parselmouth.Sound(str(wav))
    f_old = snd.to_pitch_ac(pitch_floor=75, pitch_ceiling=600).selected_array["frequency"]
    f_old = f_old[f_old > 0]
    lo, hi = RANGE[sex]
    f = snd.to_pitch_ac(pitch_floor=lo, pitch_ceiling=hi).selected_array["frequency"]
    f = f[f > 0]
    if len(f) < 10 or len(f_old) < 3:
        return None
    st = 12 * np.log2(f)
    return [group, spk, sex, wav.stem, round(float(f_old.max() - f_old.min()), 2),
            round(float(np.percentile(st, 95) - np.percentile(st, 5)), 3),
            round(float(np.percentile(st, 90) - np.percentile(st, 10)), 3),
            round(float(np.median(f)), 1), len(f)]


HEADER = ["group", "speaker", "sex", "utt_id", "old_maxmin_hz", "f0_range_p5p95_st",
          "f0_range_p10p90_st", "median_f0_hz", "n_voiced_frames"]
PARTS = OUTPUTS / "f0_range_robust_parts"


def main():
    groups = sys.argv[1:] or list(SPK)
    PARTS.mkdir(parents=True, exist_ok=True)
    for g in groups:
        jobs = [(g, s, x, w) for s, x in SPK[g].items()
                for w in sorted(wav_dir(g, s).glob("arctic_*.wav")) if UTT.match(w.name)]
        if not jobs:
            sys.exit(f"No audio for {g}. Run python scripts/check_data.py.")
        t0 = time.time()
        with Pool(max(1, os.cpu_count() or 1)) as pool:
            rows = [r for r in pool.imap(measure, jobs, chunksize=20) if r]
        with open(PARTS / f"{g}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(HEADER); w.writerows(rows)
        print(f"{g}: {len(rows)}/{len(jobs)} utterances, {time.time() - t0:.0f}s", flush=True)
    if all((PARTS / f"{g}.csv").exists() for g in SPK):
        with open(OUT, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(HEADER)
            for g in SPK:
                w.writerows(list(csv.reader(open(PARTS / f"{g}.csv", encoding="utf-8")))[1:])
        print(f"merged -> {OUT}")


if __name__ == "__main__":
    main()

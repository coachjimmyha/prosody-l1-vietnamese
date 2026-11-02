"""extract_native_stress_contrast.py — per-word stress contrasts for the four CMU ARCTIC speakers.

Processes the first 200 recordings of each speaker (arctic_a0001–a0200) with the MFA TextGrids
(english_us_arpa). Each multisyllabic word gets a duration ratio, F0 ratio and intensity difference
between its primary-stressed vowel and its unstressed vowels, with the same definition and filters as
the training corpus (stress_contrast.py).

Input : data/cmu_arctic/cmu_us_{bdl,slt,clb,rms}_arctic/{wav,textgrid}
Output: outputs/stress_contrast_native_per_word.csv

Run:  python scripts/extract_native_stress_contrast.py
"""
import csv
import numpy as np
from paths import OUTPUTS, NATIVE_SPEAKERS, cmu_dir
from stress_contrast import word_contrasts

N_UTTERANCES_PER_SPEAKER = 200          # arctic_a0001–a0200
OUTPUT_CSV = OUTPUTS / "stress_contrast_native_per_word.csv"
FIELDS = ["speaker", "utt_id", "word", "n_vowels", "stressed_vowel", "dur_ratio", "f0_ratio",
          "intensity_diff_db"]


def speaker_rows(spk, wav_paths, tg_dir):
    """Per-word rows (output schema) for the given recordings of one speaker."""
    rows = []
    for wav_path in wav_paths:
        tg_path = tg_dir / f"{wav_path.stem}.TextGrid"
        if not tg_path.exists():
            continue
        for r in word_contrasts(tg_path, wav_path):
            rows.append({"speaker": spk, "utt_id": wav_path.stem, "word": r["word"],
                         "n_vowels": r["n_vowels"], "stressed_vowel": r["stressed_vowel"],
                         "dur_ratio": r["duration_ratio"], "f0_ratio": r["f0_ratio"],
                         "intensity_diff_db": r["intensity_diff"]})
    return rows


def recordings(spk):
    d = cmu_dir(spk)
    return sorted((d / "wav").glob("*.wav"))[:N_UTTERANCES_PER_SPEAKER], d / "textgrid"


def main():
    print(f"Within-word stress contrasts — CMU ARCTIC, first {N_UTTERANCES_PER_SPEAKER} recordings per speaker")
    all_rows = []
    for spk in NATIVE_SPEAKERS:
        wavs, tg_dir = recordings(spk)
        rows = speaker_rows(spk, wavs, tg_dir)
        all_rows.extend(rows)
        print(f"  {spk}: {len({r['utt_id'] for r in rows})} of {len(wavs)} utterances, {len(rows)} words")
    if not all_rows:
        print("[ERROR] No words measured. Run python scripts/check_data.py.")
        return
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    print(f"Saved {OUTPUT_CSV.name}: {len(all_rows)} words")
    for col in ["dur_ratio", "f0_ratio", "intensity_diff_db"]:
        v = [r[col] for r in all_rows]
        print(f"  {col:18} mean={np.mean(v):.3f}  sd={np.std(v):.3f}")


if __name__ == "__main__":
    main()

"""extract_vn_stress_contrast.py — per-word stress contrasts for the four Vietnamese L2-ARCTIC speakers.

Uses all recordings of each speaker with the TextGrids distributed with L2-ARCTIC (release v5.0).
Each multisyllabic word gets a duration ratio, F0 ratio and intensity difference between its
primary-stressed vowel and its unstressed vowels, with the same definition and filters as the
training corpus (stress_contrast.py).

Input : data/l2arctic/{HQTV,PNV,THV,TLV}/{wav,textgrid}
Output: outputs/stress_contrast_vn_per_word.csv

Run:  python scripts/extract_vn_stress_contrast.py
"""
import csv
import numpy as np
from paths import OUTPUTS, L2_SPEAKERS, l2_dir
from extract_native_stress_contrast import FIELDS, speaker_rows

OUTPUT_CSV = OUTPUTS / "stress_contrast_vn_per_word.csv"


def recordings(spk):
    d = l2_dir(spk)
    return sorted((d / "wav").glob("*.wav")), d / "textgrid"


def main():
    print("Within-word stress contrasts — L2-ARCTIC Vietnamese speakers, all recordings")
    all_rows = []
    for spk in L2_SPEAKERS["vn"]:
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

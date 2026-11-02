"""extract_extra_features.py — three utterance-level features for Speechocean762 (Stage A).

  f0_range_hz         maximum minus minimum F0 (Hz) over 100 points across the utterance
  speaking_rate       vowels per second of total utterance duration
  silence_proportion  share of the utterance labelled as silence in the phone tier

Input : outputs/speechocean762_full_per_utt.csv (eight stress features + target)
Output: outputs/speechocean762_full_per_utt_v2.csv (eleven features + target)

Run:  python scripts/extract_extra_features.py
"""

import csv
import re
from pathlib import Path
import numpy as np
import parselmouth
import textgrid


from paths import OUTPUTS, SPEECHOCEAN_ROOT, SPEECHOCEAN_TEXTGRIDS, find_wav
WAVE_DIR = SPEECHOCEAN_ROOT / "WAVE"
TG_DIR = SPEECHOCEAN_TEXTGRIDS

OUTPUT_DIR = OUTPUTS
EXISTING_CSV = OUTPUT_DIR / "speechocean762_full_per_utt.csv"
NEW_CSV = OUTPUT_DIR / "speechocean762_full_per_utt_v2.csv"

ARPABET_VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AX", "AY", "EH", "ER", "EY",
    "IH", "IY", "OW", "OY", "UH", "UW",
}

F0_MIN_HZ = 75.0
F0_MAX_HZ = 600.0


def strip_stress(label):
    return re.sub(r"\d+", "", label).strip()


def get_tier(tg, name_starts):
    for tier in tg.tiers:
        if tier.name.strip().lower().startswith(name_starts):
            return tier
    return None


def find_wav_file(utt_id):
    return find_wav(WAVE_DIR, utt_id)


def compute_extra_features(utt_id):
    """Compute f0_range_hz, speaking_rate, silence_proportion for one utterance."""
    tg_path = TG_DIR / f"{utt_id}.TextGrid"
    wav_path = find_wav_file(utt_id)
    if not tg_path.exists() or wav_path is None:
        return None

    try:
        tg = textgrid.TextGrid.fromFile(str(tg_path))
        sound = parselmouth.Sound(str(wav_path))
    except Exception:
        return None

    phones_tier = get_tier(tg, "phone")
    if phones_tier is None:
        return None

    total_duration = tg.maxTime

    # ---- Feature 1: F0 range ----
    try:
        pitch_obj = sound.to_pitch_ac(pitch_floor=F0_MIN_HZ, pitch_ceiling=F0_MAX_HZ)
        # Sample F0 values across entire utterance at regular intervals
        n_samples = 100
        times = np.linspace(0.05, total_duration - 0.05, n_samples)
        f0_values = []
        for t in times:
            f0 = pitch_obj.get_value_at_time(t)
            if f0 is not None and not np.isnan(f0) and F0_MIN_HZ <= f0 <= F0_MAX_HZ:
                f0_values.append(f0)
        if len(f0_values) < 3:
            f0_range_hz = 0.0
        else:
            f0_range_hz = float(np.max(f0_values) - np.min(f0_values))
    except Exception:
        f0_range_hz = 0.0

    # ---- Feature 2: Speaking rate (vowels per second) ----
    vowel_count = 0
    silence_duration = 0.0
    for iv in phones_tier.intervals:
        label = (iv.mark or "").strip()
        if not label or label in {"sil", "sp", "spn", "[silence]", ""}:
            silence_duration += (iv.maxTime - iv.minTime)
            continue
        base = strip_stress(label)
        if base in ARPABET_VOWELS:
            vowel_count += 1

    speaking_rate = vowel_count / total_duration if total_duration > 0 else 0.0

    # ---- Feature 3: Silence proportion ----
    silence_proportion = silence_duration / total_duration if total_duration > 0 else 0.0

    return {
        "f0_range_hz": round(float(f0_range_hz), 2),
        "speaking_rate": round(float(speaking_rate), 3),
        "silence_proportion": round(float(silence_proportion), 3),
    }


def main():
    print("=" * 76)
    print("Extract 3 Extra Features + Merge with Existing 8")
    print("=" * 76)

    # Load existing CSV
    print(f"\n[1] Loading existing 8-feature CSV: {EXISTING_CSV.name}")
    existing_rows = []
    with open(EXISTING_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_rows = list(reader)
    print(f"    Loaded {len(existing_rows)} utterances")

    # Compute extra features
    print(f"\n[2] Computing extra features from TextGrids + audio...")
    new_rows = []
    n_processed = 0
    n_failed = 0
    for i, row in enumerate(existing_rows):
        utt_id = row["utt_id"]
        extra = compute_extra_features(utt_id)
        if extra is None:
            n_failed += 1
            continue
        merged = dict(row)
        merged.update(extra)
        new_rows.append(merged)
        n_processed += 1
        if (i + 1) % 500 == 0:
            print(f"    Progress: {i+1}/{len(existing_rows)}")

    print(f"\n    Processed: {n_processed}, Failed: {n_failed}")

    # Save new CSV
    if new_rows:
        fieldnames = list(new_rows[0].keys())
        with open(NEW_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(new_rows)
        print(f"\n    Saved: {NEW_CSV.name}")

        # Show statistics of new features
        print(f"\n[3] New feature statistics:")
        for col in ["f0_range_hz", "speaking_rate", "silence_proportion"]:
            vals = [float(r[col]) for r in new_rows]
            print(f"    {col:20} mean={np.mean(vals):.3f}  std={np.std(vals):.3f}  "
                  f"min={np.min(vals):.3f}  max={np.max(vals):.3f}")
    else:
        print("[ERROR] No rows processed")



if __name__ == "__main__":
    main()

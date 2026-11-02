"""extract_speechocean762_features_full.py — within-word stress features for Speechocean762 (Stage A).

For every MFA-aligned utterance, each multisyllabic word gets a duration ratio, F0 ratio and intensity
difference between its primary-stressed vowel and its unstressed vowels (stress_contrast.py, the same
definition used for the native and Vietnamese speakers). The word values are summarised per utterance
into the eight stress features (Table 3.2).

Inputs : Speechocean762 folder (paths.SPEECHOCEAN_ROOT): WAVE/, resource/scores.json
         data/speechocean762_textgrids/*.TextGrid (paths.SPEECHOCEAN_TEXTGRIDS)
Outputs: outputs/speechocean762_full_per_word.csv, speechocean762_full_per_utt.csv

Run:  python scripts/extract_speechocean762_features_full.py
"""

import json
import csv
import numpy as np
from stress_contrast import word_contrasts


# ====================================================================
# Configuration
# ====================================================================
from paths import OUTPUTS, SPEECHOCEAN_ROOT, SPEECHOCEAN_TEXTGRIDS, find_wav
WAVE_DIR = SPEECHOCEAN_ROOT / "WAVE"
SCORES_JSON = SPEECHOCEAN_ROOT / "resource" / "scores.json"
TG_DIR = SPEECHOCEAN_TEXTGRIDS

OUTPUT_DIR = OUTPUTS
OUTPUT_DIR.mkdir(exist_ok=True)

PER_WORD_CSV = OUTPUT_DIR / "speechocean762_full_per_word.csv"
PER_UTT_CSV = OUTPUT_DIR / "speechocean762_full_per_utt.csv"


def process_utterance(utt_id, prosodic_score):
    """Per-word rows and the eight utterance-level stress features of one utterance."""
    tg_path = TG_DIR / f"{utt_id}.TextGrid"
    wav_path = find_wav(WAVE_DIR, utt_id)
    if not tg_path.exists() or wav_path is None:
        return [], None
    per_word_rows = [{"utt_id": utt_id, "prosodic_score": prosodic_score, **r}
                     for r in word_contrasts(tg_path, wav_path)]
    if not per_word_rows:
        return per_word_rows, None

    dur_ratios = [r["duration_ratio"] for r in per_word_rows]
    f0_ratios = [r["f0_ratio"] for r in per_word_rows]
    int_diffs = [r["intensity_diff"] for r in per_word_rows]

    utt_summary = {
        "utt_id": utt_id,
        "prosodic_score": prosodic_score,
        "n_multisyllabic_words": len(per_word_rows),
        "mean_duration_ratio": round(float(np.mean(dur_ratios)), 4),
        "std_duration_ratio": round(float(np.std(dur_ratios)), 4) if len(dur_ratios) > 1 else 0.0,
        "max_duration_ratio": round(float(np.max(dur_ratios)), 4),
        "mean_f0_ratio": round(float(np.mean(f0_ratios)), 4),
        "std_f0_ratio": round(float(np.std(f0_ratios)), 4) if len(f0_ratios) > 1 else 0.0,
        "mean_intensity_diff": round(float(np.mean(int_diffs)), 4),
        "std_intensity_diff": round(float(np.std(int_diffs)), 4) if len(int_diffs) > 1 else 0.0,
    }

    return per_word_rows, utt_summary


# ====================================================================
# Main
# ====================================================================
def main():
    print("=" * 76)
    print("Speechocean762 FULL — Feature Extraction (5,000 utterances)")
    print("=" * 76)

    print("\nLoading scores.json...")
    with open(SCORES_JSON, "r", encoding="utf-8") as f:
        scores = json.load(f)

    tg_files = sorted(TG_DIR.rglob("*.TextGrid"))
    print(f"  TextGrids available: {len(tg_files):,}")
    print(f"  Estimated feature extraction time: ~5-10 minutes")

    all_word_rows = []
    all_utt_rows = []
    n_no_features = 0

    print("\nProcessing...")
    for i, tg_path in enumerate(tg_files):
        utt_id = tg_path.stem
        prosodic = scores.get(utt_id, {}).get("prosodic", None)
        if prosodic is None:
            continue

        word_rows, utt_summary = process_utterance(utt_id, prosodic)
        if utt_summary is None:
            n_no_features += 1
        else:
            all_word_rows.extend(word_rows)
            all_utt_rows.append(utt_summary)

        if (i + 1) % 500 == 0:
            print(f"  Progress: {i+1}/{len(tg_files)} — "
                  f"{len(all_utt_rows)} utterances with features so far")

    # Save
    print(f"\n" + "=" * 76)
    print("Extraction summary")
    print("=" * 76)
    print(f"  Utterances processed: {len(tg_files):,}")
    print(f"  Utterances with features: {len(all_utt_rows):,}")
    print(f"  Utterances all-monosyllabic (skipped): {n_no_features:,}")
    print(f"  Total word-level rows: {len(all_word_rows):,}")

    if not all_utt_rows:
        print("[ERROR] No features extracted.")
        return

    with open(PER_WORD_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_word_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_word_rows)
    print(f"\n  Per-word CSV: {PER_WORD_CSV.name}")

    with open(PER_UTT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_utt_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_utt_rows)
    print(f"  Per-utterance CSV: {PER_UTT_CSV.name}")

    # Statistics
    print(f"\n" + "=" * 76)
    print("Feature distribution statistics")
    print("=" * 76)

    print("\nPer-word (all multisyllabic words):")
    for col in ["duration_ratio", "f0_ratio", "intensity_diff"]:
        vals = [r[col] for r in all_word_rows]
        print(f"  {col:20} mean={np.mean(vals):.3f}  std={np.std(vals):.3f}  "
              f"CI95=[{np.percentile(vals, 2.5):.3f}, {np.percentile(vals, 97.5):.3f}]")

    print("\nPer-utterance target distribution:")
    prosodic_vals = [r["prosodic_score"] for r in all_utt_rows]
    print(f"  prosodic_score  mean={np.mean(prosodic_vals):.2f}  std={np.std(prosodic_vals):.2f}  "
          f"min={np.min(prosodic_vals):.1f}  max={np.max(prosodic_vals):.1f}")


if __name__ == "__main__":
    main()

"""extract_extra_features_native_vn.py — utterance-level features for CMU ARCTIC and Vietnamese L2-ARCTIC.

Same definitions as extract_extra_features.py (F0 range, speaking rate, silence proportion), applied
to every recording of the four native and the four Vietnamese speakers.

Outputs: outputs/extra_features_native_per_utt.csv, extra_features_vn_per_utt.csv

Run:  python scripts/extract_extra_features_native_vn.py
"""

import csv
import re
from pathlib import Path
import numpy as np
import parselmouth
import textgrid


from paths import OUTPUTS, cmu_dir, l2_dir
OUTPUT_DIR = OUTPUTS
OUTPUT_DIR.mkdir(exist_ok=True)

NATIVE_SPEAKERS = {
    "bdl": cmu_dir("bdl"),
    "slt": cmu_dir("slt"),
    "clb": cmu_dir("clb"),
    "rms": cmu_dir("rms"),
}

VN_SPEAKERS = {}
for spk in ("HQTV", "PNV", "THV", "TLV"):
    VN_SPEAKERS[spk] = l2_dir(spk)

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


def compute_features(wav_path, tg_path):
    try:
        tg = textgrid.TextGrid.fromFile(str(tg_path))
        sound = parselmouth.Sound(str(wav_path))
    except Exception as e:
        return None

    phones_tier = get_tier(tg, "phone")
    if phones_tier is None:
        return None

    total_duration = tg.maxTime

    # F0 range
    try:
        pitch_obj = sound.to_pitch_ac(pitch_floor=F0_MIN_HZ, pitch_ceiling=F0_MAX_HZ)
        n_samples = 100
        if total_duration < 0.2:
            f0_range_hz = 0.0
        else:
            times = np.linspace(0.05, total_duration - 0.05, n_samples)
            f0_values = []
            for t in times:
                f0 = pitch_obj.get_value_at_time(t)
                if f0 is not None and not np.isnan(f0) and F0_MIN_HZ <= f0 <= F0_MAX_HZ:
                    f0_values.append(f0)
            f0_range_hz = float(np.max(f0_values) - np.min(f0_values)) if len(f0_values) >= 3 else 0.0
    except Exception:
        f0_range_hz = 0.0

    # Speaking rate + silence proportion
    vowel_count = 0
    silence_duration = 0.0
    for iv in phones_tier.intervals:
        label = (iv.mark or "").strip()
        dur = iv.maxTime - iv.minTime
        if not label or label in {"sil", "sp", "spn", "[silence]", ""}:
            silence_duration += dur
            continue
        base = strip_stress(label)
        if base in ARPABET_VOWELS:
            vowel_count += 1

    speaking_rate = vowel_count / total_duration if total_duration > 0 else 0.0
    silence_proportion = silence_duration / total_duration if total_duration > 0 else 0.0

    return {
        "f0_range_hz": round(float(f0_range_hz), 2),
        "speaking_rate": round(float(speaking_rate), 3),
        "silence_proportion": round(float(silence_proportion), 3),
    }


def process_corpus(speakers_dict, corpus_label, out_csv):
    print(f"\n[{corpus_label}] Processing {len(speakers_dict)} speakers...")
    rows = []
    for spk, spk_dir in speakers_dict.items():
        wav_dir = spk_dir / "wav"
        tg_dir = spk_dir / "textgrid"
        if not wav_dir.exists() or not tg_dir.exists():
            print(f"    Skipping {spk}: paths missing")
            continue

        n_proc = 0
        n_fail = 0
        for wav_path in sorted(wav_dir.glob("*.wav")):
            utt_id = wav_path.stem
            tg_path = tg_dir / f"{utt_id}.TextGrid"
            if not tg_path.exists():
                n_fail += 1
                continue
            feats = compute_features(wav_path, tg_path)
            if feats is None:
                n_fail += 1
                continue
            rows.append({
                "speaker": spk,
                "utt_id": utt_id,
                **feats,
            })
            n_proc += 1
        print(f"    {spk}: {n_proc} processed, {n_fail} failed")

    if rows:
        fieldnames = list(rows[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"    Saved: {out_csv.name} ({len(rows)} rows)")

        # Stats
        for col in ["f0_range_hz", "speaking_rate", "silence_proportion"]:
            vals = [r[col] for r in rows]
            print(f"    {col:20} mean={np.mean(vals):.3f}  std={np.std(vals):.3f}")


def main():
    print("=" * 76)
    print("Extract Extra Features: Native + Vietnamese")
    print("=" * 76)

    process_corpus(
        NATIVE_SPEAKERS,
        "Native (CMU ARCTIC)",
        OUTPUT_DIR / "extra_features_native_per_utt.csv",
    )
    process_corpus(
        VN_SPEAKERS,
        "Vietnamese (L2-ARCTIC)",
        OUTPUT_DIR / "extra_features_vn_per_utt.csv",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()

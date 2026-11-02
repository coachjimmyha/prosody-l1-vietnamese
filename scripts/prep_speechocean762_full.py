"""prep_speechocean762_full.py — prepare the 5,000 Speechocean762 utterances for forced alignment.

Copies each WAVE/*.wav file and writes a matching .lab transcript (the "text" field of
resource/scores.json) into
data/speechocean762_mfa_input/, the input folder for Montreal Forced Aligner
(english_us_arpa acoustic model and dictionary; Section 3.3).

Run:  python scripts/prep_speechocean762_full.py
"""

import json
import shutil
from pathlib import Path


from paths import SPEECHOCEAN_ROOT, SPEECHOCEAN_MFA_INPUT, SPEECHOCEAN_TEXTGRIDS, find_wav
WAVE_DIR = SPEECHOCEAN_ROOT / "WAVE"
SCORES_JSON = SPEECHOCEAN_ROOT / "resource" / "scores.json"

MFA_INPUT_DIR = SPEECHOCEAN_MFA_INPUT


def find_wav_file(utt_id, wave_root):
    return find_wav(wave_root, utt_id)


def main():
    print("=" * 76)
    print("Preparing ALL Speechocean762 utterances for MFA")
    print("=" * 76)

    with open(SCORES_JSON, "r", encoding="utf-8") as f:
        scores = json.load(f)

    all_ids = list(scores.keys())
    print(f"\nTotal utterances: {len(all_ids):,}")

    if MFA_INPUT_DIR.exists():
        print(f"Removing existing folder: {MFA_INPUT_DIR.name}...")
        shutil.rmtree(MFA_INPUT_DIR)
    MFA_INPUT_DIR.mkdir(parents=True)

    n_success = 0
    n_missing = 0

    for i, utt_id in enumerate(all_ids):
        transcript = scores[utt_id].get("text", "").strip()
        wav_src = find_wav_file(utt_id, WAVE_DIR)

        if not transcript or wav_src is None:
            n_missing += 1
            continue

        wav_dst = MFA_INPUT_DIR / f"{utt_id}.wav"
        lab_dst = MFA_INPUT_DIR / f"{utt_id}.lab"
        shutil.copy(wav_src, wav_dst)
        with open(lab_dst, "w", encoding="utf-8") as f:
            f.write(transcript)
        n_success += 1

        if (i + 1) % 500 == 0:
            print(f"  Progress: {i+1}/{len(all_ids)}")

    print(f"\n" + "=" * 76)
    print(f"Prepared: {n_success:,}, Missing: {n_missing}")
    print("=" * 76)
    print(f"""
Next step, forced alignment (Montreal Forced Aligner, about 10-15 minutes):

  mfa align "{MFA_INPUT_DIR}" english_us_arpa english_us_arpa \\
      "{SPEECHOCEAN_TEXTGRIDS}" --clean --overwrite
""")


if __name__ == "__main__":
    main()

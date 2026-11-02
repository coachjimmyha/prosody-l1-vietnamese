"""prep_cmu_arctic_for_mfa.py — write one .lab transcript per CMU ARCTIC recording.

CMU ARCTIC stores all transcripts in etc/txt.done.data, e.g.
    ( arctic_a0001 "Author of the danger trail, Philip Steels, etc." )
MFA expects one .lab file next to each .wav file. The script processes bdl and slt under
data/cmu_arctic/ (paths.CMU_ROOT); clb and rms are prepared by add_native_speakers.py.

Run:  python scripts/prep_cmu_arctic_for_mfa.py
"""

import re
from pathlib import Path

# ====================================================================
# Configuration
# ====================================================================
from paths import CMU_ROOT

# Speakers to process — folder names under CMU_ROOT
SPEAKERS = ["cmu_us_bdl_arctic", "cmu_us_slt_arctic"]


# ====================================================================
# Helpers
# ====================================================================
# Parse CMU ARCTIC format: ( utt_id "transcript text" )
TXT_PATTERN = re.compile(r'\(\s*(\S+)\s+"(.+)"\s*\)')


def parse_txt_done_data(path):
    """Read txt.done.data and return {utterance_id: transcript}."""
    transcripts = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = TXT_PATTERN.match(line.strip())
            if m:
                transcripts[m.group(1)] = m.group(2)
    return transcripts


def find_speaker_paths(speaker_dir):
    """Locate wav/ and etc/txt.done.data under a speaker folder.
    Handles potential doubled nesting (SPEAKER/SPEAKER/wav/).
    """
    candidates = [
        (speaker_dir / "wav", speaker_dir / "etc" / "txt.done.data"),
        (speaker_dir / speaker_dir.name / "wav",
         speaker_dir / speaker_dir.name / "etc" / "txt.done.data"),
    ]
    for wav_dir, txt_path in candidates:
        if wav_dir.is_dir() and txt_path.is_file():
            return wav_dir, txt_path
    return None, None


def write_lab_files(wav_dir, transcripts):
    """Write a .lab file next to each .wav with matching utterance ID."""
    written = 0
    missing = []
    for wav_path in sorted(wav_dir.glob("*.wav")):
        utt_id = wav_path.stem
        if utt_id in transcripts:
            lab_path = wav_path.with_suffix(".lab")
            with open(lab_path, "w", encoding="utf-8") as f:
                f.write(transcripts[utt_id])
            written += 1
        else:
            missing.append(utt_id)
    return written, missing


# ====================================================================
# Main
# ====================================================================
def main():
    print("=" * 76)
    print("CMU ARCTIC → MFA .lab file generator")
    print("=" * 76)
    print(f"CMU root: {CMU_ROOT}")

    if not CMU_ROOT.is_dir():
        print("[ERROR] CMU ARCTIC folder not found. Run python scripts/check_data.py.")
        return

    for speaker in SPEAKERS:
        speaker_dir = CMU_ROOT / speaker
        print(f"\n→ Processing speaker: {speaker}")

        if not speaker_dir.is_dir():
            print(f"  [WARN] Folder not found: {speaker_dir}")
            print(f"         Skipping.")
            continue

        wav_dir, txt_path = find_speaker_paths(speaker_dir)
        if wav_dir is None:
            print(f"  [ERROR] Could not locate wav/ + etc/txt.done.data under {speaker_dir}")
            continue

        print(f"  wav directory:   {wav_dir}")
        print(f"  transcript file: {txt_path}")

        transcripts = parse_txt_done_data(txt_path)
        print(f"  Parsed {len(transcripts)} transcript entries")

        written, missing = write_lab_files(wav_dir, transcripts)
        print(f"  Wrote {written} .lab files")
        if missing:
            preview = missing[:5]
            tail = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
            print(f"  [INFO] {len(missing)} .wav files have no matching transcript: {preview}{tail}")

    print("\nDone.")


if __name__ == "__main__":
    main()

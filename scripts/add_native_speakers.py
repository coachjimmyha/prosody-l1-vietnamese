"""add_native_speakers.py — download and prepare two further CMU ARCTIC voices (clb, rms).

Downloads the packed releases from festvox.org, extracts them under
data/cmu_arctic/ (paths.CMU_ROOT) and writes the .lab files needed by MFA. The MFA commands
to align the new speakers are printed at the end. Standard library only.

Run:  python scripts/add_native_speakers.py
"""

import re
import tarfile
import urllib.request
from pathlib import Path

from paths import CMU_ROOT
BASE_URL = "http://festvox.org/cmu_arctic/packed/"
SPEAKERS = ["cmu_us_clb_arctic", "cmu_us_rms_arctic"]

TXT_PATTERN = re.compile(r'\(\s*(\S+)\s+"(.+)"\s*\)')


def download(name):
    tar_name = f"{name}.tar.bz2"
    tar_path = CMU_ROOT / tar_name
    if tar_path.exists():
        print(f"  already downloaded: {tar_name}")
        return tar_path
    url = BASE_URL + tar_name
    print(f"  downloading {url}")
    urllib.request.urlretrieve(url, tar_path)
    print(f"  saved {tar_path.name} ({tar_path.stat().st_size / 1e6:.0f} MB)")
    return tar_path


def extract(tar_path, name):
    target = CMU_ROOT / name
    if target.is_dir():
        print(f"  already extracted: {name}")
        return target
    print(f"  extracting {tar_path.name} ...")
    with tarfile.open(tar_path, "r:bz2") as t:
        t.extractall(CMU_ROOT)
    print(f"  extracted -> {target.name}")
    return target


def write_labs(speaker_dir):
    for base in (speaker_dir, speaker_dir / speaker_dir.name):
        wav_dir = base / "wav"
        txt = base / "etc" / "txt.done.data"
        if wav_dir.is_dir() and txt.is_file():
            transcripts = {}
            with open(txt, encoding="utf-8") as f:
                for line in f:
                    m = TXT_PATTERN.match(line.strip())
                    if m:
                        transcripts[m.group(1)] = m.group(2)
            n = 0
            for wav in sorted(wav_dir.glob("*.wav")):
                if wav.stem in transcripts:
                    wav.with_suffix(".lab").write_text(transcripts[wav.stem], encoding="utf-8")
                    n += 1
            print(f"  wrote {n} .lab files in {wav_dir}")
            return wav_dir
    print(f"  [ERROR] could not find wav/ + etc/txt.done.data under {speaker_dir}")
    return None


def main():
    print("=" * 70)
    print("Add native speakers: clb (female), rms (male)")
    print("=" * 70)
    CMU_ROOT.mkdir(parents=True, exist_ok=True)
    for name in SPEAKERS:
        print(f"\n{name}")
        try:
            tar = download(name)
        except Exception as e:
            print(f"  [ERROR] download failed: {e}")
            print(f"  Manual fallback: download {BASE_URL}{name}-0.95-release.tar.bz2")
            print(f"  into {CMU_ROOT} and re-run this script.")
            continue
        d = extract(tar, name)
        write_labs(d)

    print("\n" + "=" * 70)
    print("Next: align the new speakers with MFA:")
    print("=" * 70)
    for name in SPEAKERS:
        wav = CMU_ROOT / name / "wav"
        tg = CMU_ROOT / name / "textgrid"
        print(f'  mfa align "{wav}" english_us_arpa english_us_arpa "{tg}" --clean')
    print("\nThen re-run the feature extraction:")
    print("  python scripts/extract_native_stress_contrast.py")
    print("  python scripts/extract_extra_features_native_vn.py")
    print("  python scripts/extract_rhythm_features.py")
    print("  python scripts/final_evaluation_v3.py")


if __name__ == "__main__":
    main()

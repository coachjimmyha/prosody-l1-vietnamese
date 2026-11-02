"""
paths.py — every folder used by the scripts, in one place.

Repository folders (relative to the repository root, whatever the current directory):
    outputs/    feature tables and results (CSV)
    figures/    thesis figures (PNG)
    logs/       console output of each script

Raw corpora (not distributed with this repository). Default layout, under ./data or under the
folder given by the environment variable PROSODY_DATA (README.md, Section 3):

    data/
      speechocean762/                  resource/scores.json, WAVE/SPEAKERxxxx/*.WAV
      speechocean762_mfa_input/        written by prep_speechocean762_full.py (MFA input)
      speechocean762_textgrids/        MFA output, one TextGrid per utterance
      cmu_arctic/cmu_us_<spk>_arctic/  bdl, slt, clb, rms: wav/, etc/txt.done.data, textgrid/
      l2arctic/<SPK>/                  16 L2-ARCTIC speakers: wav/, textgrid/
      l2arctic_vn_mfa/<SPK>/textgrid/  MFA re-alignment of the 4 Vietnamese speakers (Appendix B.15)

A speaker folder unzipped one level too deep (l2arctic/HQTV/HQTV/wav) is also accepted. If a corpus
is stored elsewhere, point to it directly with PROSODY_SPEECHOCEAN, PROSODY_SPEECHOCEAN_TEXTGRIDS,
PROSODY_CMU or PROSODY_L2ARCTIC. Run `python scripts/check_data.py` to verify the layout.
Only Stage A (extraction) and Stage E (audio-based analyses) read these files.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs"
FIGURES = REPO_ROOT / "figures"
LOGS = REPO_ROOT / "logs"

DATA_ROOT = Path(os.environ.get("PROSODY_DATA", REPO_ROOT / "data")).expanduser()


def _corpus(env_name, *candidates, marker=None):
    """Folder given by env_name, else the first candidate that exists (containing marker, if given)."""
    if os.environ.get(env_name):
        return Path(os.environ[env_name]).expanduser()
    for c in candidates:
        if c.is_dir() and (marker is None or (c / marker).exists()):
            return c
    return candidates[0]


# GitHub download unzips to speechocean762-main/ (sometimes nested twice); OpenSLR to speechocean762/
SPEECHOCEAN_ROOT = _corpus("PROSODY_SPEECHOCEAN",
                           DATA_ROOT / "speechocean762",
                           DATA_ROOT / "speechocean762-main",
                           DATA_ROOT / "speechocean762-main" / "speechocean762-main",
                           marker="resource/scores.json")
SPEECHOCEAN_MFA_INPUT = DATA_ROOT / "speechocean762_mfa_input"
SPEECHOCEAN_TEXTGRIDS = _corpus("PROSODY_SPEECHOCEAN_TEXTGRIDS", DATA_ROOT / "speechocean762_textgrids")
CMU_ROOT = _corpus("PROSODY_CMU", DATA_ROOT / "cmu_arctic")
L2ARCTIC_ROOT = _corpus("PROSODY_L2ARCTIC", DATA_ROOT / "l2arctic", DATA_ROOT / "l2arctic_release_v5.0")

NATIVE_SPEAKERS = ["bdl", "slt", "clb", "rms"]
L2_SPEAKERS = {"vn": ["HQTV", "PNV", "THV", "TLV"], "korean": ["HKK", "HJK", "YDCK", "YKWK"],
               "spanish": ["EBVS", "ERMS", "MBMPS", "NJS"], "chinese": ["BWC", "LXC", "NCC", "TXHC"]}


def cmu_dir(spk):
    """CMU ARCTIC speaker folder (holding wav/, etc/, textgrid/), e.g. cmu_dir('bdl')."""
    d = CMU_ROOT / f"cmu_us_{spk}_arctic"
    return d / d.name if (d / d.name / "wav").is_dir() else d


def l2_dir(spk):
    """L2-ARCTIC speaker folder (holding wav/, textgrid/), e.g. l2_dir('HQTV')."""
    d = L2ARCTIC_ROOT / spk
    return d / spk if (d / spk / "wav").is_dir() else d


def find_wav(folder, utt_id):
    """Return the audio file for utt_id in folder or its subfolders (.wav or .WAV), or None."""
    folder = Path(folder)
    for ext in (".wav", ".WAV"):
        direct = folder / f"{utt_id}{ext}"
        if direct.exists():
            return direct
    for ext in (".wav", ".WAV"):
        matches = list(folder.rglob(f"{utt_id}{ext}"))
        if matches:
            return matches[0]
    return None

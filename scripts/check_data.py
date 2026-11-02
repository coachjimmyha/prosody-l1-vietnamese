"""
check_data.py — check that the raw corpora are where the extraction scripts expect them.

Run after downloading the corpora (README.md, Section 3):
    python scripts/check_data.py
For each item the script prints OK, or MISSING with a hint. Items marked "after MFA" only exist once
forced alignment has been run (see run_extraction.sh). The folders are defined in paths.py.
"""
from pathlib import Path
from paths import (DATA_ROOT, SPEECHOCEAN_ROOT, SPEECHOCEAN_TEXTGRIDS, CMU_ROOT, L2ARCTIC_ROOT,
                   NATIVE_SPEAKERS, L2_SPEAKERS, cmu_dir, l2_dir)

GROUP_NAMES = {"vn": "Vietnamese", "korean": "Korean", "spanish": "Spanish", "chinese": "Chinese"}
problems = 0


def count(folder, pattern):
    folder = Path(folder)
    return len(list(folder.glob(pattern))) if folder.is_dir() else 0


def report(ok, label, detail="", hint=""):
    global problems
    print(f"  [{'OK' if ok else 'MISSING'}] {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        problems += 1
        if hint:
            print(f"            -> {hint}")


print(f"Data folder: {DATA_ROOT}")

print(f"\nSpeechocean762: {SPEECHOCEAN_ROOT}")
n_wav = count(SPEECHOCEAN_ROOT / "WAVE", "*/*.[wW][aA][vV]")
report((SPEECHOCEAN_ROOT / "resource" / "scores.json").exists(), "resource/scores.json", "",
       "unzip the corpus so that data/speechocean762/resource/scores.json exists")
report(n_wav >= 5000, "WAVE/SPEAKERxxxx/*.wav", f"{n_wav} files, expected at least 5,000")

print(f"\nSpeechocean762 TextGrids (after MFA): {SPEECHOCEAN_TEXTGRIDS}")
n_tg = count(SPEECHOCEAN_TEXTGRIDS, "*.TextGrid")
report(n_tg >= 4996, "*.TextGrid", f"{n_tg} files, expected 4,996",
       "run scripts/prep_speechocean762_full.py, then the mfa align command in run_extraction.sh")

print(f"\nCMU ARCTIC (native reference): {CMU_ROOT}")
for s in NATIVE_SPEAKERS:
    d = cmu_dir(s)
    nw, nt = count(d / "wav", "*.wav"), count(d / "textgrid", "*.TextGrid")
    report(nw >= 1132 and (d / "etc" / "txt.done.data").exists(), f"cmu_us_{s}_arctic/wav + etc/txt.done.data",
           f"{nw} wav files",
           f"extract cmu_us_{s}_arctic-0.95-release.tar.bz2 into data/cmu_arctic/"
           + (" (or run scripts/add_native_speakers.py)" if s in ("clb", "rms") else ""))
    report(nt >= 1131, f"cmu_us_{s}_arctic/textgrid (after MFA)", f"{nt} files",
           "align with MFA into cmu_us_<spk>_arctic/textgrid (command in run_extraction.sh)")

print(f"\nL2-ARCTIC v5.0: {L2ARCTIC_ROOT}")
for group, spks in L2_SPEAKERS.items():
    for s in spks:
        d = l2_dir(s)
        nw, nt = count(d / "wav", "*.wav"), count(d / "textgrid", "*.TextGrid")
        report(nw >= 1000 and nt >= 1000, f"{GROUP_NAMES[group]:10} {s}", f"{nw} wav, {nt} TextGrid",
               f"unzip {s}.zip into data/l2arctic/ so that data/l2arctic/{s}/wav and {s}/textgrid exist")

print("\n" + ("All corpora are in place." if problems == 0 else
              f"{problems} item(s) missing. The analyses from the committed CSVs (run_analyses.sh) "
              "do not need these files."))

"""prep_l2arctic_vn_for_mfa.py — write one .lab transcript per L2-ARCTIC recording of the four
Vietnamese speakers, so that they can be force-aligned with the same MFA set-up as the native
speakers (robustness check of the function-word reduction ratio, Section 5.6, Appendix B.15).

L2-ARCTIC stores one transcript per recording in transcript/<utt>.txt. MFA expects a .lab file next
to each .wav file, as for CMU ARCTIC (prep_cmu_arctic_for_mfa.py). The corpus TextGrids are not
modified; the new alignment is written to a separate folder.

Run:  python scripts/prep_l2arctic_vn_for_mfa.py
Then, in the MFA conda environment (MFA 3.3.9), for each speaker <SPK> (D = data folder):
  mfa align "$D/l2arctic/<SPK>/wav" english_us_arpa english_us_arpa \
            "$D/l2arctic_vn_mfa/<SPK>/textgrid" --clean
MFA resamples the 44.1 kHz recordings to 16 kHz internally.
"""
from paths import L2_SPEAKERS, l2_dir


def main():
    print("L2-ARCTIC (Vietnamese speakers) -> MFA .lab files")
    for spk in L2_SPEAKERS["vn"]:
        d = l2_dir(spk)
        wav_dir, tr_dir = d / "wav", d / "transcript"
        if not wav_dir.is_dir() or not tr_dir.is_dir():
            print(f"  [{spk}] wav/ or transcript/ not found under {d} — skipped")
            continue
        written, missing = 0, []
        for wav in sorted(wav_dir.glob("*.wav")):
            txt = tr_dir / f"{wav.stem}.txt"
            if not txt.is_file():
                missing.append(wav.stem)
                continue
            text = " ".join(txt.read_text(encoding="utf-8", errors="ignore").split())
            wav.with_suffix(".lab").write_text(text, encoding="utf-8")
            written += 1
        print(f"  [{spk}] {written} .lab files written" + (f"; {len(missing)} without transcript" if missing else ""))
    print("Done.")


if __name__ == "__main__":
    main()

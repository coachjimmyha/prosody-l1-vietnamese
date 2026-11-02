"""build_X14.py — 14-feature matrices for the native and Vietnamese speakers (input to RQ2).

Aggregates the per-word stress contrasts into the eight utterance-level stress features (as in
final_evaluation_v3.py) and joins the utterance-level and rhythm features:
  outputs/X14_native_fresh.csv  722 utterances (arctic_a0001–a0200, 4 speakers)
  outputs/X14_vn_fresh.csv      4,129 utterances (all prompts, 4 speakers)
rq2_matched_content.py restricts the Vietnamese set to the native prompts.

Run:  python scripts/build_X14.py
"""
import csv
from pathlib import Path
from final_evaluation_v3 import (FEATURE_COLS, build_X_from_per_word, load_by_speaker_utt,
                                 NATIVE_PER_WORD_CSV, VN_PER_WORD_CSV, NATIVE_EXTRA_CSV,
                                 VN_EXTRA_CSV, NATIVE_RHYTHM_CSV, VN_RHYTHM_CSV, OUTPUT_DIR)

EXTRA = ["f0_range_hz", "speaking_rate", "silence_proportion"]
RHY = ["npvi", "percent_v", "varcov"]

for name, pw, ex, rh in [("native", NATIVE_PER_WORD_CSV, NATIVE_EXTRA_CSV, NATIVE_RHYTHM_CSV),
                         ("vn", VN_PER_WORD_CSV, VN_EXTRA_CSV, VN_RHYTHM_CSV)]:
    X, spk, utt = build_X_from_per_word(pw, load_by_speaker_utt(ex, EXTRA), load_by_speaker_utt(rh, RHY))
    out = OUTPUT_DIR / f"X14_{name}_fresh.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["speaker", "utt_id"] + FEATURE_COLS)
        for s, u, row in zip(spk, utt, X):
            w.writerow([s, u] + [float(v) for v in row])
    print(f"saved {out.name}: {len(X)} utterances, {len(set(spk))} speakers")

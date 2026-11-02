#!/usr/bin/env bash
# run_extraction.sh — rebuild the feature tables in outputs/ from the raw corpora (Stage A) and
# rerun the audio-based supporting analyses (Stage E). Requires the corpora in data/ (README.md,
# Section 3; check with: python scripts/check_data.py) and the MFA TextGrids. Runtime: several hours.
#
# Usage:   bash run_extraction.sh
#          PROSODY_DATA=/path/to/data bash run_extraction.sh     (corpora outside the repository)
#
# Forced alignment is run beforehand with Montreal Forced Aligner 3.3.9 in its own conda environment
# (D = data folder):
#   python scripts/prep_speechocean762_full.py
#   mfa align "$D/speechocean762_mfa_input" english_us_arpa english_us_arpa \
#             "$D/speechocean762_textgrids" --clean
#   python scripts/prep_cmu_arctic_for_mfa.py     # bdl, slt
#   python scripts/add_native_speakers.py         # clb, rms (download + .lab files)
#   mfa align "$D/cmu_arctic/cmu_us_<spk>_arctic/wav" english_us_arpa english_us_arpa \
#             "$D/cmu_arctic/cmu_us_<spk>_arctic/textgrid" --clean
#   python scripts/prep_l2arctic_vn_for_mfa.py    # Vietnamese speakers, re-alignment check only (B.15)
#   mfa align "$D/l2arctic/<SPK>/wav" english_us_arpa english_us_arpa \
#             "$D/l2arctic_vn_mfa/<SPK>/textgrid" --clean
# Apart from that check, the L2-ARCTIC TextGrids are used as distributed (release v5.0).
set -eu
cd "$(dirname "$0")"
PY="${PYTHON:-$(command -v python || command -v python3)}"   # the active environment's Python
mkdir -p logs outputs
DATA="${PROSODY_DATA:-$PWD/data}"
HERE_RE=$(printf '%s' "$PWD/" | sed 's/[][\.*^$|/]/\\&/g')
DATA_RE=$(printf '%s' "$DATA/" | sed 's/[][\.*^$|/]/\\&/g')
clean () { sed -e "s/$DATA_RE/data\//g" -e "s/$HERE_RE//g"; }
run () {
    echo; echo "########## $* ##########"
    "$PY" "scripts/$1" "${@:2}" 2>&1 | clean
}
pilot () {   # pilot <log name> <script> [NUTT]
    echo; echo "########## $2 (NUTT=${3:-default}) -> logs/$1.log ##########"
    if [ -n "${3:-}" ]; then NUTT="$3" "$PY" "scripts/$2"; else "$PY" "scripts/$2"; fi \
        2>&1 | clean | tee "logs/$1.log"
}

echo "== Stage A1: Speechocean762 training table =="
run extract_speechocean762_features_full.py     # 8 stress features + prosody score
run extract_extra_features.py                   # + F0 range, speaking rate, silence proportion
run extract_rhythm_features.py                  # rhythm metrics (all three corpora)
run retrain_with_rhythm.py                      # -> speechocean762_full_per_utt_v3.csv (14 features)

echo "== Stage A2: native and Vietnamese speakers =="
run extract_native_stress_contrast.py
run extract_vn_stress_contrast.py
run extract_extra_features_native_vn.py
run build_X14.py

echo "== Stage A3: comparison groups, reduction ratio, robust F0 range =="
run extract_contrast_l1.py
run extract_reduction_feature.py                # 20 speakers -> reduction_full.csv
run extract_reduction_feature.py speechocean    # -> reduction_speechocean.csv
run f0_range_robust.py                          # -> f0_range_robust_per_utt.csv
rm -f outputs/reduction_realign_per_utt.csv     # rebuilt from the TextGrids by the next line
run reduction_realign_check.py                  # -> reduction_realign_per_utt.csv (needs data/l2arctic_vn_mfa)

echo "== Stage E: audio-based supporting analyses (Section 5.4) =="
pilot pilot_vowel_centralisation      vowel_centralization_pilot.py 24
pilot pilot_vowel_centralisation_N99  vowel_centralization_pilot.py 99
pilot pilot_accent_density            accent_density_full.py 24
pilot pilot_accent_density_N99        accent_density_full.py 99
pilot pilot_reduction_f0_intensity    reduction_f0_int_pilot.py 12
pilot pilot_typology_5groups          contrast_full_pilot.py

echo; echo "########## done — now run: bash run_analyses.sh ##########"

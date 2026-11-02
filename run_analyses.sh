#!/usr/bin/env bash
# run_analyses.sh — reproduce every model result, statistic and data figure of the thesis from the
# committed feature tables in outputs/ (Stages B–D and the figures). No audio is needed.
# The scripts are in scripts/; run this file from anywhere.
#
# Usage:   bash run_analyses.sh            full run (the three 1,000-model nulls take most of the time)
#          QUICK=1 bash run_analyses.sh    skip the three 1,000-model nulls and reuse their committed CSVs
#
# Console output of each script is saved to logs/<script>.log (quoted in Appendix B of the thesis).
# Extraction from audio (Stage A) and the audio-based analyses (Stage E) are in run_extraction.sh.
set -u
cd "$(dirname "$0")"
PY="${PYTHON:-$(command -v python || command -v python3)}"   # the active environment's Python
mkdir -p logs figures outputs
HERE_RE=$(printf '%s' "$PWD/" | sed 's/[][\.*^$|/]/\\&/g')

run () {
    echo; echo "########## $* ##########"
    "$PY" "scripts/$1" "${@:2}" 2>&1 | sed "s/$HERE_RE//g" | tee "logs/${1%.py}.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || echo "[warning] $1 exited with an error"
}

echo "== Stage A (from committed CSVs): model input matrices for the native and Vietnamese speakers =="
run build_X14.py

echo "== Stage B: model, evaluation, robustness =="
run final_evaluation_v3.py
run thesis_robustness_checks.py
run hyperparameter_sensitivity.py
run feature_correlations.py
if [ "${QUICK:-0}" = "1" ]; then
    echo "QUICK=1: pcc_permutation_null.py skipped (uses committed outputs/pcc_null_1000.csv)"
else
    run pcc_permutation_null.py
fi
run speaker_disjoint_cv.py
run shap_analysis.py

echo "== Stage C: native vs Vietnamese comparisons =="
run speaker_bootstrap.py
if [ "${QUICK:-0}" = "1" ]; then
    echo "QUICK=1: transfer_reality_check_v2.py and rq2_matched_content.py skipped (committed CSVs used)"
else
    run transfer_reality_check_v2.py
    run rq2_matched_content.py
fi
run f0_range_robust_stats.py

echo "== Stage D: function-word reduction and per-learner profiles =="
run reduction_stats.py
run reduction_realign_check.py
run reduction_ablation.py
run learner_error_profiles.py

echo "== Figures (written to figures/) =="
for f in fig_ielts.py fig_lit_search.py fig_field_map.py fig_research_gap.py \
         fig_pipeline.py fig_feature_overview.py fig_stress_example.py \
         fig_eda.py fig_ablation.py \
         fig_predicted_scatter.py fig_two_nulls.py fig_forest_speaker_d.py \
         fig_permutation_importance.py fig_importance_vs_separation.py fig_shap_beeswarm_v2.py \
         fig_reduction_v2.py fig_feedback_card.py fig_pilots.py; do
    run "$f"
done

echo; echo "########## done ##########"

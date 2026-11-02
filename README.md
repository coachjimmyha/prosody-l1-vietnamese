# Automatic Detection and Interpretable Analysis of Rhythm and Lexical Stress in English Produced by L1-Vietnamese Learners

Analysis code for the MSc Data Science and Machine Learning dissertation of **Hà Thanh Trúc**
(Liverpool John Moores University; supervisor Dr Vikas Thada), December 2026.

A Random Forest regressor is trained on 14 interpretable acoustic features of Speechocean762
(L1 Mandarin, human prosody scores) and applied to Vietnamese learners (L2-ARCTIC) and native speakers
(CMU ARCTIC). Permutation importance, SHAP and speaker-level comparisons with the native reference
localise each learner's errors to named prosodic dimensions.

---

## 1. Quick start: reproduce the results from the committed feature tables

No audio is needed. Every model result, statistic and data figure in Chapters 4 and 5 is recomputed
from the CSV files in `outputs/`, except the audio-based analyses of Section 5.4 (Stage E), which need
the corpora (Section 3).

```bash
python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash run_analyses.sh                 # about 1 hour on a laptop (three 1,000-model permutation nulls)
QUICK=1 bash run_analyses.sh         # a few minutes: skips the three nulls and reuses their committed CSVs
```

On Windows, run the `.sh` files in Git Bash or WSL, or run the scripts one by one in the order listed in
`run_analyses.sh` (`python scripts/<name>.py`).

Each script writes its console output to `logs/<script>.log`. Appendix B of the thesis quotes these
logs. All random processes use seed 42, so with the pinned package versions the numbers are identical
to those in the thesis. Random Forest results can differ slightly with other scikit-learn versions.

Expected headline numbers:

| Result | Value | Script |
|---|---|---|
| Held-out test PCC (RMSE, R²), 1,789 / 383 / 384 split | 0.656 (1.03, 0.43), 95% CI [0.58, 0.72] | `final_evaluation_v3.py`, `thesis_robustness_checks.py` |
| Speaker-disjoint split / speaker-grouped 5-fold CV | 0.695 / 0.658 ± 0.038 | `speaker_disjoint_cv.py` |
| Hyperparameter sensitivity (validation PCC, 60 settings) | 0.635–0.654; chosen setting 0.647 | `hyperparameter_sensitivity.py` |
| Shuffled-label null (1,000) | max 0.35, 0/1,000 ≥ observed, p < 0.001 | `pcc_permutation_null.py` |
| RQ2 predicted-score gap, speaker level | \|d\| = 1.67, fake-model null p ≈ 0.38 | `transfer_reality_check_v2.py` |
| RQ2, content-matched prompts | \|d\| = 0.95, p ≈ 0.55 | `rq2_matched_content.py` |
| Function-word reduction, native vs Vietnamese | d = −3.58, exact p = 0.029 | `reduction_stats.py` |
| Reduction, Vietnamese re-aligned with the native MFA set-up | d = −2.63, exact p = 0.029 (all shared prompts: −2.12) | `reduction_realign_check.py` |
| Robust F0 range, native vs Vietnamese | d = −0.48, exact p = 0.54 | `speaker_bootstrap.py`, `f0_range_robust_stats.py` |

## 2. Repository structure

```
prosody-l1-vietnamese/
├── README.md, LICENSE, requirements.txt
├── run_analyses.sh     Stages B–D and all figures, from the committed CSVs (no audio needed)
├── run_extraction.sh   Stage A (feature extraction from the corpora) and Stage E (audio-based analyses)
├── scripts/            all Python scripts, one per analysis step (Section 4)
│   ├── paths.py        every folder used by the scripts, in one place
│   ├── check_data.py   checks that the downloaded corpora are in the expected folders
│   └── fig_style.py    shared figure style
├── outputs/            extracted features and analysis results (CSV)
├── logs/               console output of every script (quoted in Appendix B)
├── figures/            thesis figures (Figure_<number>_<name>.png); supplementary/ holds the SHAP plots
└── data/               raw corpora, created by you (Section 3; not in the repository)
```

Scripts can be run from any folder (`python scripts/<name>.py`): they locate `outputs/`, `figures/`,
`logs/` and `data/` through `scripts/paths.py`.

## 3. Data

The corpora are not redistributed here. `outputs/` already contains the derived acoustic
measurements (and the Speechocean762 prosody scores used as the target), which is all that
`run_analyses.sh` needs. The raw corpora are needed only to rebuild those tables (`run_extraction.sh`).

| Corpus | Role | Download | Licence |
|---|---|---|---|
| Speechocean762 (Zhang et al., 2021) | training (prosody scores) | https://github.com/jimbozhang/speechocean762 (Code → Download ZIP) or https://www.openslr.org/101/ | free for commercial and non-commercial use |
| L2-ARCTIC v5.0 (Zhao et al., 2018) | Vietnamese learners HQTV, PNV, THV, TLV; Korean, Spanish and Chinese speakers for the supplementary comparisons | https://psi.engr.tamu.edu/l2-arctic-corpus/ | CC BY-NC 4.0 |
| CMU ARCTIC (Kominek and Black, 2004) | native reference bdl, slt, clb, rms | http://festvox.org/cmu_arctic/packed/ | free use; keep the copyright notice and authors' names |

### 3.1 Where to put the files

Create a folder `data/` inside the repository and unpack the corpora into it with these folder names:

```
data/
├── speechocean762/                   ← Speechocean762 (resource/scores.json, WAVE/SPEAKERxxxx/*.WAV)
├── cmu_arctic/
│   ├── cmu_us_bdl_arctic/            ← extract cmu_us_bdl_arctic-0.95-release.tar.bz2 here
│   │   ├── wav/arctic_a0001.wav ...
│   │   ├── etc/txt.done.data
│   │   └── textgrid/                 ← written by MFA (step 3)
│   ├── cmu_us_slt_arctic/            (same structure)
│   ├── cmu_us_clb_arctic/            (downloaded by add_native_speakers.py)
│   └── cmu_us_rms_arctic/            (downloaded by add_native_speakers.py)
├── l2arctic/
│   ├── HQTV/                         ← unzip HQTV.zip here
│   │   ├── wav/arctic_a0001.wav ...
│   │   └── textgrid/arctic_a0001.TextGrid ...   (supplied with the corpus)
│   ├── PNV/, THV/, TLV/              Vietnamese
│   └── HKK/, HJK/, YDCK/, YKWK/, EBVS/, ERMS/, MBMPS/, NJS/, BWC/, LXC/, NCC/, TXHC/
│                                     Korean, Spanish, Chinese (cross-L1 comparisons only)
├── l2arctic_vn_mfa/                  ← Vietnamese speakers re-aligned by MFA (step 3; Appendix B.15 only)
├── speechocean762_mfa_input/         ← written by prep_speechocean762_full.py (step 2)
└── speechocean762_textgrids/         ← Speechocean762 TextGrids written by MFA (step 3)
```

The scripts also accept the folder names that the archives produce by default: `speechocean762-main/`
(GitHub ZIP, also when nested twice), `l2arctic_release_v5.0/` instead of `l2arctic/`, and a speaker
folder unzipped one level too deep (`l2arctic/HQTV/HQTV/wav`). To keep the corpora outside the
repository, set `PROSODY_DATA=/path/to/folder` (same layout), or point to each corpus directly with
`PROSODY_SPEECHOCEAN`, `PROSODY_SPEECHOCEAN_TEXTGRIDS`, `PROSODY_CMU` or `PROSODY_L2ARCTIC`.

### 3.2 Step by step

1. **Download and unpack** into `data/` as shown above.
   - Speechocean762: unzip `speechocean762-main.zip` into `data/` and rename the folder to `speechocean762`
     (or extract the OpenSLR archive `speechocean762.tar.gz`, which already has this name).
   - CMU ARCTIC: extract the bdl and slt archives into `data/cmu_arctic/`. Run
     `python scripts/add_native_speakers.py` to download and extract clb and rms.
   - L2-ARCTIC: unzip each speaker archive into `data/l2arctic/`, e.g. `unzip HQTV.zip -d data/l2arctic`,
     giving `data/l2arctic/HQTV/wav`.
2. **Prepare the MFA input:** `python scripts/prep_speechocean762_full.py`,
   `python scripts/prep_cmu_arctic_for_mfa.py` and, for the re-alignment check only,
   `python scripts/prep_l2arctic_vn_for_mfa.py`.
3. **Align** with Montreal Forced Aligner 3.3.9 (`english_us_arpa` acoustic model and dictionary), using the
   `mfa align` commands at the top of `run_extraction.sh`. The L2-ARCTIC TextGrids are used as
   distributed; only for the robustness check in Appendix B.15 are the four Vietnamese speakers
   re-aligned with the same set-up as the native speakers.
4. **Check the layout:** `python scripts/check_data.py` lists every expected folder as OK or MISSING,
   with a hint.
5. **Rebuild the tables:** `bash run_extraction.sh`, then `bash run_analyses.sh`.

## 4. Pipeline and scripts

All scripts are in `scripts/`.

### Stage A — feature extraction (needs the corpora; `run_extraction.sh`)

| Script | Output in `outputs/` |
|---|---|
| `prep_speechocean762_full.py`, `prep_cmu_arctic_for_mfa.py`, `add_native_speakers.py`, `prep_l2arctic_vn_for_mfa.py` | MFA input folders and .lab transcripts |
| `extract_speechocean762_features_full.py` → `extract_extra_features.py` → `extract_rhythm_features.py` → `retrain_with_rhythm.py` | `speechocean762_full_per_utt_v3.csv` (14 features + prosody score; 2,556 utterances) |
| `extract_native_stress_contrast.py`, `extract_vn_stress_contrast.py` | per-word stress contrasts, native and Vietnamese (same definition as the training corpus: `stress_contrast.py`) |
| `extract_extra_features_native_vn.py` | `extra_features_{native,vn}_per_utt.csv` |
| `extract_contrast_l1.py` | `extra_features_*`, `rhythm_*` for the Korean, Spanish and Chinese speakers |
| `build_X14.py` | `X14_native_fresh.csv` (726 utterances), `X14_vn_fresh.csv` (4,077 utterances) |
| `extract_reduction_feature.py` | `reduction_full.csv` (20 speakers); with argument `speechocean`: `reduction_speechocean.csv` |
| `f0_range_robust.py` | `f0_range_robust_per_utt.csv` (robust F0 range, 20 speakers) |

### Stages B–D — model, comparisons and diagnostics (from the CSVs; `run_analyses.sh`)

| Script | Thesis | Output |
|---|---|---|
| `final_evaluation_v3.py` | §4.4, §5.1, B.1 | canonical model, test PCC / RMSE / R² |
| `thesis_robustness_checks.py` | §4.5, Fig. 4.2, Table 5.1, B.4–B.5 | ablation, baselines, bootstrap CI, `permutation_importance_v3.csv` |
| `pcc_permutation_null.py` | §4.5, Fig. 5.2a, B.2 | `pcc_null_1000.csv` |
| `speaker_disjoint_cv.py` | §3.7.2, Table 5.1, B.3 | speaker-disjoint split and grouped CV |
| `hyperparameter_sensitivity.py` | §4.4, B.13 | validation PCC over a 60-setting hyperparameter grid (test set unused) |
| `feature_correlations.py` | §4.3.2, B.14 | Spearman correlations among the 14 features |
| `shap_analysis.py` | §3.8, §5.3, B.10 | SHAP base value; supplementary SHAP plots |
| `speaker_bootstrap.py` | §5.2–5.3, Fig. 5.3, B.7 | `speaker_level_tests.csv` (d, bootstrap CI, exact p) |
| `transfer_reality_check_v2.py` | §5.2, Fig. 5.2b, B.6 | `transfer_fake_null_1000.csv` |
| `rq2_matched_content.py` | §5.2, §5.6, B.6 | `rq2_matched_fake_null_1000.csv` |
| `f0_range_robust_stats.py` | §3.10, §5.3, B.9 | F0-range statistics (old vs robust measure) |
| `reduction_stats.py` | §5.3, Fig. 5.7, B.8 | `reduction_stats_summary.csv` |
| `reduction_ablation.py` | §5.3 | reduction as a 15th model feature (net gain ≈ 0) |
| `learner_error_profiles.py` | §3.9, §5.6, Table 5.2, B.11 | per-learner z-profiles and flags (matched prompts), with a prompt-set sensitivity check |
| `reduction_realign_check.py` | §5.3, §5.6, B.15 | `reduction_realign_check.csv`: reduction ratio with the Vietnamese speakers re-aligned by the native MFA set-up; reuses the committed `reduction_realign_per_utt.csv` (delete it to recompute from the TextGrids) |

### Stage E — audio-based supporting analyses (needs the corpora; Section 5.4)

| Script | Thesis | Log |
|---|---|---|
| `vowel_centralization_pilot.py` | §5.4.1, Fig. 5.9, B.12 | `logs/pilot_vowel_centralisation*.log` |
| `reduction_f0_int_pilot.py` | §5.4.1, B.12 | `logs/pilot_reduction_f0_intensity.log` |
| `accent_density_full.py` | §5.4.2, Fig. 5.10, B.12 | `logs/pilot_accent_density*.log` |
| `contrast_full_pilot.py` | §5.4.3, Table 5.3, Fig. 5.11, B.12 | `logs/pilot_typology_5groups.log` |

### Figures (`figures/`, drawn by `run_analyses.sh`; shared style in `scripts/fig_style.py`)

| Figure | Script | Figure | Script |
|---|---|---|---|
| 1.1 | `fig_ielts.py` | 5.1 | `fig_predicted_scatter.py` |
| 2.1 | `fig_lit_search.py` | 5.2 | `fig_two_nulls.py` |
| 2.2 | `fig_field_map.py` | 5.3 | `fig_forest_speaker_d.py` |
| 2.4 | `fig_research_gap.py` | 5.4 | `fig_permutation_importance.py` |
| 3.1 | `fig_pipeline.py` | 5.5 | `fig_importance_vs_separation.py` |
| 3.5 | `fig_feature_overview.py` | 5.6 | `fig_shap_beeswarm_v2.py` |
| 3.6 | `fig_stress_example.py` | 5.7 | `fig_reduction_v2.py` |
| 4.1 | `fig_eda.py` | 5.8 | `fig_feedback_card.py` |
| 4.2 | `fig_ablation.py` | 5.9–5.11 | `fig_pilots.py` (reads the Stage E logs) |

Figures 2.3, 3.2, 3.3, 3.4, 3.7 and 3.8 are conceptual illustrations with no computed data and are
not generated by this repository.

## 5. Notes

- The model is trained on Speechocean762 only; the native and Vietnamese speakers are never used for
  training. With four speakers per group, the smallest attainable two-sided exact permutation p is
  2/70 = 0.029.
- The native per-word stress file covers the first 200 CMU ARCTIC prompts (arctic_a0001–a0200), so the
  native model input has 726 utterances; `rq2_matched_content.py` repeats the RQ2 test on matched prompts.
- The function-word reduction ratio is averaged over the first 300 recordings of each speaker
  (arctic_a0001–a0300). It is a diagnostic measure, not a model feature. The L2-ARCTIC TextGrids were
  made with MFA 1.0.0 and the native ones with MFA 3.3.9; `reduction_realign_check.py` repeats the
  comparison with the Vietnamese speakers re-aligned by MFA 3.3.9 (about two-fifths of the gap is due
  to the alignment source; the difference remains).
- `f0_range_hz` (max − min F0) is the model feature. Group comparisons and learner profiles use the
  robust F0 range from `f0_range_robust.py`.
- One CMU ARCTIC recording (bdl, arctic_a0507) has no MFA alignment, so the native direct measures
  cover 4,527 of 4,528 recordings.
- `final_feature_importance_v3_eval.csv` holds impurity-based importance and is kept for reference;
  the thesis reports permutation importance (`permutation_importance_v3.csv`).

## 6. Licence

Code: MIT licence (see `LICENSE`). The corpora remain under their own licences (Section 3).

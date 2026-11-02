"""final_evaluation_v3.py — canonical Random Forest model (Sections 4.4 and 5.1; Appendix B.1).

Trains the Random Forest regressor (200 trees, maximum depth 8, minimum leaf size 10, seed 42) on
the 14 features of Speechocean762 with a 70/15/15 split (test 15%, then 17.6% of the remainder for
validation: 1,789 / 383 / 384 utterances) and reports test PCC, RMSE and R². The model is then
applied to the native and Vietnamese feature matrices (utterance-level summaries are printed; the
speaker-level RQ2 test is transfer_reality_check_v2.py).

Outputs (outputs/): final_evaluation_results_v3.csv, final_per_speaker_predictions_v3.csv,
final_feature_importance_v3_eval.csv (impurity importance, kept for reference only)

Run:  python scripts/final_evaluation_v3.py
"""

import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats


from paths import OUTPUTS as OUTPUT_DIR

SPEECHOCEAN_CSV = OUTPUT_DIR / "speechocean762_full_per_utt_v3.csv"   # 14 features + target
NATIVE_PER_WORD_CSV = OUTPUT_DIR / "stress_contrast_native_per_word.csv"
VN_PER_WORD_CSV = OUTPUT_DIR / "stress_contrast_vn_per_word.csv"
NATIVE_EXTRA_CSV = OUTPUT_DIR / "extra_features_native_per_utt.csv"
VN_EXTRA_CSV = OUTPUT_DIR / "extra_features_vn_per_utt.csv"
NATIVE_RHYTHM_CSV = OUTPUT_DIR / "rhythm_native_per_utt.csv"
VN_RHYTHM_CSV = OUTPUT_DIR / "rhythm_vn_per_utt.csv"

OUT_RESULTS_CSV = OUTPUT_DIR / "final_evaluation_results_v3.csv"
OUT_SPEAKER_CSV = OUTPUT_DIR / "final_per_speaker_predictions_v3.csv"
OUT_FEATURE_IMP_CSV = OUTPUT_DIR / "final_feature_importance_v3_eval.csv"

FEATURE_COLS = [
    "mean_duration_ratio", "std_duration_ratio", "max_duration_ratio",
    "mean_f0_ratio", "std_f0_ratio", "mean_intensity_diff", "std_intensity_diff",
    "n_multisyllabic_words", "f0_range_hz", "speaking_rate", "silence_proportion",
    "npvi", "percent_v", "varcov",
]


def mean_ci(arr, conf=0.95):
    a = np.asarray(arr, dtype=float)
    n = len(a)
    if n < 2:
        return (float(a[0]) if n == 1 else 0.0), 0.0, 0.0
    m = float(a.mean())
    se = float(a.std(ddof=1)) / np.sqrt(n)
    z = stats.t.ppf((1 + conf) / 2, n - 1)
    return m, m - z * se, m + z * se


def load_by_speaker_utt(csv_path, cols):
    d = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d[(r["speaker"], r["utt_id"])] = {c: float(r[c]) for c in cols}
    return d


def build_X_from_per_word(csv_path, extra_dict, rhythm_dict):
    grouped = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            utt_id = r.get("utt_id")
            spk = r.get("speaker")
            if not utt_id or not spk:
                continue
            try:
                grouped[(spk, utt_id)].append({
                    "dur": float(r["dur_ratio"]),
                    "f0": float(r["f0_ratio"]) if r.get("f0_ratio") not in (None, "", "None") else None,
                    "int": float(r["intensity_diff_db"]) if r.get("intensity_diff_db") not in (None, "", "None") else None,
                })
            except (KeyError, ValueError):
                continue

    X, speakers, utt_ids = [], [], []
    n_skip = 0
    for (spk, utt_id), words in grouped.items():
        if (spk, utt_id) not in extra_dict or (spk, utt_id) not in rhythm_dict:
            n_skip += 1
            continue
        durs = [w["dur"] for w in words]
        f0s = [w["f0"] for w in words if w["f0"] is not None]
        ints = [w["int"] for w in words if w["int"] is not None]
        e = extra_dict[(spk, utt_id)]
        rh = rhythm_dict[(spk, utt_id)]
        X.append([
            float(np.mean(durs)),
            float(np.std(durs)) if len(durs) > 1 else 0.0,
            float(np.max(durs)),
            float(np.mean(f0s)) if f0s else 1.0,
            float(np.std(f0s)) if len(f0s) > 1 else 0.0,
            float(np.mean(ints)) if ints else 0.0,
            float(np.std(ints)) if len(ints) > 1 else 0.0,
            len(words),
            e["f0_range_hz"], e["speaking_rate"], e["silence_proportion"],
            rh["npvi"], rh["percent_v"], rh["varcov"],
        ])
        speakers.append(spk)
        utt_ids.append(utt_id)
    if n_skip:
        print(f"    (skipped {n_skip} utterances missing extra/rhythm)")
    return np.array(X), speakers, utt_ids


def main():
    print("=" * 76)
    print("Final Cross-L1 Evaluation — 14 features (with rhythm)")
    print("=" * 76)

    print("\n[1] Loading Speechocean762 v3 (14 features + target)...")
    rows = []
    with open(SPEECHOCEAN_CSV, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({c: float(r[c]) for c in FEATURE_COLS + ["prosodic_score"]})
            except (KeyError, ValueError):
                continue
    X = np.array([[r[c] for c in FEATURE_COLS] for r in rows])
    y = np.array([r["prosodic_score"] for r in rows])
    print(f"    {len(y)} utterances, {X.shape[1]} features")

    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.176, random_state=42)
    print(f"    Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    model = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=10,
                                  random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)
    test_pcc = stats.pearsonr(y_test, test_pred)[0]
    test_rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
    test_r2 = float(r2_score(y_test, test_pred))
    print("\n" + "=" * 76)
    print("SPEECHOCEAN762 RESULTS (held-out test)")
    print("=" * 76)
    print(f"  Test PCC:  {test_pcc:.4f}   Test RMSE: {test_rmse:.4f}   Test R2: {test_r2:.4f}")
    print(f"  Validation PCC: {stats.pearsonr(y_val, model.predict(X_val))[0]:.4f}   (hyperparameters set a priori; checked here, not tuned)")

    imp_pairs = sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1])
    with open(OUT_FEATURE_IMP_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["feature", "importance"])
        for feat, imp in imp_pairs:
            w.writerow([feat, round(float(imp), 4)])
    print("\n  Feature importance:")
    for feat, imp in imp_pairs:
        print(f"    {feat:25} {imp:.3f} {'#'*int(imp*60)}")

    print("\n[2] Applying model to Native + VN (14 features)...")
    native_extra = load_by_speaker_utt(NATIVE_EXTRA_CSV, ["f0_range_hz", "speaking_rate", "silence_proportion"])
    vn_extra = load_by_speaker_utt(VN_EXTRA_CSV, ["f0_range_hz", "speaking_rate", "silence_proportion"])
    native_rhythm = load_by_speaker_utt(NATIVE_RHYTHM_CSV, ["npvi", "percent_v", "varcov"])
    vn_rhythm = load_by_speaker_utt(VN_RHYTHM_CSV, ["npvi", "percent_v", "varcov"])

    X_native, spk_native, _ = build_X_from_per_word(NATIVE_PER_WORD_CSV, native_extra, native_rhythm)
    X_vn, spk_vn, _ = build_X_from_per_word(VN_PER_WORD_CSV, vn_extra, vn_rhythm)
    pred_native = model.predict(X_native)
    pred_vn = model.predict(X_vn)
    print(f"    Native: {len(pred_native)} preds / {len(set(spk_native))} speakers")
    print(f"    VN:     {len(pred_vn)} preds / {len(set(spk_vn))} speakers")

    per_speaker_rows = []
    print("\n" + "=" * 76)
    print("PER-SPEAKER PREDICTIONS (mean +/- 95% CI)")
    print("=" * 76)
    for group, preds, speakers in [("Native", pred_native, spk_native), ("Vietnamese", pred_vn, spk_vn)]:
        print(f"\n  {group}")
        for sp in sorted(set(speakers)):
            mask = np.array([s == sp for s in speakers])
            p = preds[mask]
            m, lo, hi = mean_ci(p)
            print(f"    {sp:<10} n={len(p):>5}  {m:.3f} [{lo:.3f}, {hi:.3f}]")
            per_speaker_rows.append({"speaker": sp, "group": group, "n": len(p),
                                     "mean_pred": round(m, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
    with open(OUT_SPEAKER_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_speaker_rows[0].keys()))
        w.writeheader(); w.writerows(per_speaker_rows)

    print("\n" + "=" * 76)
    print("CROSS-L1 DISCRIMINATION (14-feature model)")
    print("=" * 76)
    n_mean, n_lo, n_hi = mean_ci(pred_native)
    v_mean, v_lo, v_hi = mean_ci(pred_vn)
    gap = n_mean - v_mean
    t_stat, p_val = stats.ttest_ind(pred_native, pred_vn, equal_var=False)
    pooled = np.sqrt((pred_native.var(ddof=1) + pred_vn.var(ddof=1)) / 2)
    cohens_d = (pred_native.mean() - pred_vn.mean()) / pooled if pooled > 0 else 0
    print(f"  Native:      n={len(pred_native)}, {n_mean:.3f} [{n_lo:.3f}, {n_hi:.3f}]")
    print(f"  Vietnamese:  n={len(pred_vn)}, {v_mean:.3f} [{v_lo:.3f}, {v_hi:.3f}]")
    print(f"  Gap:         {gap:+.3f}")
    print(f"  p-value:     {p_val:.4g}")
    print(f"  Cohen's d:   {cohens_d:.3f}")

    # Direct rhythm measurements native vs VN (independent of ML)
    print("\n" + "=" * 76)
    print("DIRECT RHYTHM MEASUREMENTS (native vs VN, independent of ML)")
    print("=" * 76)
    idx = {"npvi": 11, "percent_v": 12, "varcov": 13,
           "speaking_rate": 9, "mean_duration_ratio": 0}
    direct_results = []
    for feat in ["mean_duration_ratio", "speaking_rate", "npvi", "percent_v", "varcov"]:
        n_arr = X_native[:, idx[feat]]
        v_arr = X_vn[:, idx[feat]]
        pooled_f = np.sqrt((n_arr.var(ddof=1) + v_arr.var(ddof=1)) / 2)
        d_feat = (n_arr.mean() - v_arr.mean()) / pooled_f if pooled_f > 0 else 0
        _, p_feat = stats.ttest_ind(n_arr, v_arr, equal_var=False)
        print(f"  {feat:22} native={n_arr.mean():.3f}  VN={v_arr.mean():.3f}  d={d_feat:+.3f}  (p={p_feat:.2e})")
        direct_results.append({"feature": feat, "cohens_d": round(float(d_feat), 4), "p_value": float(p_feat)})

    with open(OUT_RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["metric", "value"])
        w.writerow(["speechocean_test_pcc", round(test_pcc, 4)])
        w.writerow(["cross_l1_gap", round(gap, 4)])
        w.writerow(["cross_l1_cohens_d", round(float(cohens_d), 4)])
        w.writerow(["cross_l1_p_value", float(p_val)])
        for r in direct_results:
            w.writerow([f"direct_{r['feature']}_cohens_d", r["cohens_d"]])

    print("\n" + "=" * 76)
    print("KEY NUMBERS (14-feature model)")
    print("=" * 76)
    print(f"  Speechocean762 Test PCC: {test_pcc:.3f}")
    print(f"  Cross-L1 ML Cohen's d:   {cohens_d:.3f} (p = {p_val:.2e})")
    print(f"  Files: {OUT_RESULTS_CSV.name}, {OUT_SPEAKER_CSV.name}, {OUT_FEATURE_IMP_CSV.name}")


if __name__ == "__main__":
    main()

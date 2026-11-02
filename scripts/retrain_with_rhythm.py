"""retrain_with_rhythm.py — build the final 14-feature training table (Stage A).

Merges the three rhythm metrics into the eleven-feature Speechocean762 table by utt_id and writes
outputs/speechocean762_full_per_utt_v3.csv (14 features + prosodic_score), the training file
read by every model script. For reference it also fits the Random Forest on the 11- and 14-feature
tables and prints both test PCCs and an impurity-based importance ranking
(outputs/final_feature_importance_v3.csv).

Run:  python scripts/retrain_with_rhythm.py
"""

import csv
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from scipy import stats


from paths import OUTPUTS as OUTPUT_DIR

V2_CSV = OUTPUT_DIR / "speechocean762_full_per_utt_v2.csv"
RHYTHM_CSV = OUTPUT_DIR / "speechocean762_rhythm_per_utt.csv"
MERGED_CSV = OUTPUT_DIR / "speechocean762_full_per_utt_v3.csv"
IMP_CSV = OUTPUT_DIR / "final_feature_importance_v3.csv"

FEATURE_COLS_V2 = [
    "mean_duration_ratio", "std_duration_ratio", "max_duration_ratio",
    "mean_f0_ratio", "std_f0_ratio", "mean_intensity_diff", "std_intensity_diff",
    "n_multisyllabic_words", "f0_range_hz", "speaking_rate", "silence_proportion",
]
RHYTHM_COLS = ["npvi", "percent_v", "varcov"]
FEATURE_COLS_V3 = FEATURE_COLS_V2 + RHYTHM_COLS
TARGET_COL = "prosodic_score"


def train_and_evaluate(X, y, feature_names, label, save_imp=None):
    print(f"\n{'='*76}\nTraining: {label} ({len(feature_names)} features)\n{'='*76}")
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.176, random_state=42)
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    model = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=10,
                                  random_state=42, n_jobs=-1)
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
    print(f"  5-fold CV R2: {cv.mean():.3f} +/- {cv.std():.3f}")
    model.fit(X_train, y_train)

    val_pcc = stats.pearsonr(y_val, model.predict(X_val))[0]
    test_pred = model.predict(X_test)
    test_pcc = stats.pearsonr(y_test, test_pred)[0]
    print(f"  Validation PCC: {val_pcc:.3f}")
    print(f"  Test PCC:       {test_pcc:.3f}  *")
    print(f"  Test R2:        {r2_score(y_test, test_pred):.3f}")

    print(f"\n  Feature importance:")
    imp_pairs = sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1])
    for feat, imp in imp_pairs:
        print(f"    {feat:25} {imp:.3f} {'#'*int(imp*60)}")

    if save_imp:
        with open(save_imp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["feature", "importance"])
            for feat, imp in imp_pairs:
                w.writerow([feat, round(float(imp), 4)])
        print(f"\n  Saved importance -> {Path(save_imp).name}")

    return {"test_pcc": test_pcc, "cv_r2_mean": cv.mean()}


def main():
    print("=" * 76)
    print("Retrain: 11 features (V2) vs 14 features (V3, + rhythm)")
    print("=" * 76)

    # rhythm lookup by utt_id
    rhythm = {}
    with open(RHYTHM_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rhythm[r["utt_id"]] = {c: float(r[c]) for c in RHYTHM_COLS}
    print(f"\nLoaded rhythm for {len(rhythm)} utterances")

    rows, merged_for_save, n_missing = [], [], 0
    with open(V2_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            uid = r["utt_id"]
            if uid not in rhythm:
                n_missing += 1
                continue
            try:
                base = {c: float(r[c]) for c in FEATURE_COLS_V2 + [TARGET_COL]}
            except (KeyError, ValueError):
                continue
            base.update(rhythm[uid])
            rows.append(base)
            merged_for_save.append({
                "utt_id": uid,
                **{c: r[c] for c in FEATURE_COLS_V2},
                **rhythm[uid],
                TARGET_COL: r[TARGET_COL],
            })
    print(f"Merged {len(rows)} rows ({n_missing} without rhythm)")

    if merged_for_save:
        with open(MERGED_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(merged_for_save[0].keys()))
            w.writeheader()
            w.writerows(merged_for_save)
        print(f"Saved merged -> {MERGED_CSV.name}")

    y = np.array([r[TARGET_COL] for r in rows])
    X_v2 = np.array([[r[c] for c in FEATURE_COLS_V2] for r in rows])
    X_v3 = np.array([[r[c] for c in FEATURE_COLS_V3] for r in rows])

    res_v2 = train_and_evaluate(X_v2, y, FEATURE_COLS_V2, "V2 (11 features)")
    res_v3 = train_and_evaluate(X_v3, y, FEATURE_COLS_V3, "V3 (14 features, + rhythm)", save_imp=IMP_CSV)

    print(f"\n{'='*76}\nSUMMARY\n{'='*76}")
    d = res_v3["test_pcc"] - res_v2["test_pcc"]
    print(f"  Test PCC:  V2(11)={res_v2['test_pcc']:.3f}   V3(14)={res_v3['test_pcc']:.3f}   delta={d:+.3f}")
    print(f"  CV R2:     V2(11)={res_v2['cv_r2_mean']:.3f}   V3(14)={res_v3['cv_r2_mean']:.3f}")
    if d > 0.02:
        print("  -> Rhythm features add signal.")
    elif d > -0.02:
        print("  -> Rhythm features are roughly neutral (as the means suggested).")
    else:
        print("  -> Rhythm features slightly hurt (noise).")
    print("  Check the rank of npvi / percent_v / varcov in the V3 importance list above.")


if __name__ == "__main__":
    main()

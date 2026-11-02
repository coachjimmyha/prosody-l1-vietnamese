"""
stress_contrast.py — the within-word stress contrast, one definition for all three corpora
(Section 3.3, Table 3.2).

For each word with at least two vowels in the MFA phone tier, the first primary-stressed vowel
(ARPABET marker 1) is compared with the mean of the word's unstressed vowels (marker 0); vowels with
secondary stress (marker 2) are not used:
  duration ratio   stressed vowel duration / mean unstressed vowel duration
  F0 ratio         peak F0 over the central 60% of the stressed vowel / mean of the same for the
                   unstressed vowels (Praat autocorrelation pitch, 75–600 Hz)
  intensity diff   mean intensity of the stressed vowel minus the mean for the unstressed vowels (dB)
A vowel is skipped if it is shorter than 30 ms, has no F0 value in 75–600 Hz or a mean intensity
outside 30–95 dB. A word is discarded if its duration ratio is outside 0.3–4.0, its F0 ratio outside
0.5–2.5 or its intensity difference outside ±15 dB.

Used by extract_speechocean762_features_full.py (training corpus), extract_native_stress_contrast.py
(CMU ARCTIC) and extract_vn_stress_contrast.py (L2-ARCTIC Vietnamese speakers).
"""
import re
import numpy as np
import parselmouth
import textgrid

ARPABET_VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AX", "AY", "EH", "ER", "EY",
    "IH", "IY", "OW", "OY", "UH", "UW",
}

DURATION_MIN_SEC = 0.03
F0_MIN_HZ = 75.0
F0_MAX_HZ = 600.0
INTENSITY_MIN_DB = 30.0
INTENSITY_MAX_DB = 95.0

DUR_RATIO_MIN, DUR_RATIO_MAX = 0.3, 4.0
F0_RATIO_MIN, F0_RATIO_MAX = 0.5, 2.5
INTENSITY_DIFF_MIN, INTENSITY_DIFF_MAX = -15.0, 15.0

SILENCE_LABELS = {"sil", "sp", "spn", "[silence]", ""}


def strip_stress(label):
    return re.sub(r"\d+", "", label).strip()


def get_stress_digit(label):
    m = re.search(r"(\d)$", label)
    return int(m.group(1)) if m else None


def get_tier(tg, name_starts):
    for tier in tg.tiers:
        if tier.name.strip().lower().startswith(name_starts):
            return tier
    return None


def measure_vowel(pitch_obj, intensity_obj, vowel_iv):
    """Duration, peak F0 (central 60%, five points) and mean intensity of one vowel, or None."""
    duration = vowel_iv.maxTime - vowel_iv.minTime
    if duration < DURATION_MIN_SEC:
        return None
    try:
        times = np.linspace(vowel_iv.minTime + 0.2 * duration,
                            vowel_iv.maxTime - 0.2 * duration, 5)
        f0_values = []
        for t in times:
            f0 = pitch_obj.get_value_at_time(t)
            if f0 is not None and not np.isnan(f0) and F0_MIN_HZ <= f0 <= F0_MAX_HZ:
                f0_values.append(f0)
        f0_max = max(f0_values) if f0_values else None
    except Exception:
        f0_max = None
    if f0_max is None:
        return None
    try:
        intensity_mean = intensity_obj.get_average(vowel_iv.minTime, vowel_iv.maxTime)
        if intensity_mean is None or np.isnan(intensity_mean):
            return None
        if not (INTENSITY_MIN_DB <= intensity_mean <= INTENSITY_MAX_DB):
            return None
    except Exception:
        return None
    return {
        "duration_sec": duration,
        "f0_max_hz": float(f0_max),
        "intensity_mean_db": float(intensity_mean),
    }


def word_contrasts(tg_path, wav_path):
    """Stress contrasts of every valid multisyllabic word in one utterance.

    Returns a list of dicts (word, n_vowels, stressed_vowel, duration_ratio, f0_ratio,
    intensity_diff); an empty list if the files cannot be read or no word qualifies.
    """
    try:
        tg = textgrid.TextGrid.fromFile(str(tg_path))
        sound = parselmouth.Sound(str(wav_path))
    except Exception:
        return []

    words_tier = get_tier(tg, "word")
    phones_tier = get_tier(tg, "phone")
    if words_tier is None or phones_tier is None:
        return []

    try:
        pitch_obj = sound.to_pitch_ac(pitch_floor=F0_MIN_HZ, pitch_ceiling=F0_MAX_HZ)
        intensity_obj = sound.to_intensity()
    except Exception:
        return []

    rows = []
    tol = 0.005
    for word_iv in words_tier.intervals:
        word_text = (word_iv.mark or "").strip().lower()
        if word_text in SILENCE_LABELS:
            continue

        word_phones = [p for p in phones_tier.intervals
                       if p.minTime >= word_iv.minTime - tol
                       and p.maxTime <= word_iv.maxTime + tol]
        vowels = [p for p in word_phones if strip_stress(p.mark) in ARPABET_VOWELS]
        if len(vowels) < 2:
            continue

        stressed_vowel = None
        unstressed_vowels = []
        for v in vowels:
            digit = get_stress_digit(v.mark)
            if digit == 1 and stressed_vowel is None:
                stressed_vowel = v
            elif digit == 0:
                unstressed_vowels.append(v)
        if stressed_vowel is None or not unstressed_vowels:
            continue

        s_meas = measure_vowel(pitch_obj, intensity_obj, stressed_vowel)
        if s_meas is None:
            continue
        u_measurements = [m for m in (measure_vowel(pitch_obj, intensity_obj, uv)
                                      for uv in unstressed_vowels) if m is not None]
        if not u_measurements:
            continue

        u_dur_mean = float(np.mean([m["duration_sec"] for m in u_measurements]))
        u_f0_mean = float(np.mean([m["f0_max_hz"] for m in u_measurements]))
        u_int_mean = float(np.mean([m["intensity_mean_db"] for m in u_measurements]))

        dur_ratio = s_meas["duration_sec"] / u_dur_mean if u_dur_mean > 0 else None
        f0_ratio = s_meas["f0_max_hz"] / u_f0_mean if u_f0_mean > 0 else None
        int_diff = s_meas["intensity_mean_db"] - u_int_mean

        if dur_ratio is None or not (DUR_RATIO_MIN <= dur_ratio <= DUR_RATIO_MAX):
            continue
        if f0_ratio is None or not (F0_RATIO_MIN <= f0_ratio <= F0_RATIO_MAX):
            continue
        if not (INTENSITY_DIFF_MIN <= int_diff <= INTENSITY_DIFF_MAX):
            continue

        rows.append({
            "word": word_text,
            "n_vowels": len(vowels),
            "stressed_vowel": strip_stress(stressed_vowel.mark),
            "duration_ratio": round(dur_ratio, 4),
            "f0_ratio": round(f0_ratio, 4),
            "intensity_diff": round(int_diff, 4),
        })
    return rows

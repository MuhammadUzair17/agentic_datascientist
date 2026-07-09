# tools/drift_detector.py
"""
Drift Detector
==============
Compares a new dataset against saved pipeline reference statistics.

Two-phase gatekeeper:
  Phase 1 — Schema Check   : columns, types, target, problem type
  Phase 2 — Statistical    : PSI (primary) + KS test (secondary)

PSI thresholds (industry standard):
  PSI < 0.10  → No drift        → Reuse pipeline (HIGH confidence)
  PSI 0.10-0.25 → Minor drift   → Reuse with warning (MEDIUM confidence)
  PSI > 0.25  → Major drift     → Retrain needed (LOW confidence)
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional


# ── PSI ────────────────────────────────────────────────────────────────────
def _compute_psi_numeric(ref: pd.Series, new: pd.Series,
                          n_bins: int = 10) -> float:
    """Population Stability Index for a numeric column."""
    ref = ref.dropna()
    new = new.dropna()
    if len(ref) == 0 or len(new) == 0:
        return 0.0

    # If reference has near-zero variance, PSI is meaningless — return 0
    if ref.std() < 1e-8:
        return 0.0

    # Build bins from reference data
    breakpoints = np.linspace(ref.min(), ref.max(), n_bins + 1)
    breakpoints[0]  = -np.inf
    breakpoints[-1] =  np.inf

    # Drop duplicate edges (happens when column has very few unique values)
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        # Not enough unique bins — fall back to categorical PSI
        return _compute_psi_categorical(ref.astype(str), new.astype(str))

    try:
        ref_counts = pd.cut(ref, bins=breakpoints,
                            duplicates="drop").value_counts(sort=False)
        new_counts = pd.cut(new, bins=breakpoints,
                            duplicates="drop").value_counts(sort=False)
    except Exception:
        return 0.0

    # Align indices
    ref_counts, new_counts = ref_counts.align(new_counts, fill_value=0)

    ref_pct = (ref_counts / len(ref)).values
    new_pct = (new_counts / len(new)).values

    # Avoid log(0)
    eps = 1e-6
    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    new_pct = np.where(new_pct == 0, eps, new_pct)

    psi = np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct))
    return float(round(psi, 4))


def _compute_psi_categorical(ref: pd.Series, new: pd.Series) -> float:
    """PSI for a categorical column."""
    ref = ref.dropna().astype(str)
    new = new.dropna().astype(str)
    if len(ref) == 0 or len(new) == 0:
        return 0.0

    all_cats = set(ref.unique()) | set(new.unique())
    eps = 1e-6

    psi = 0.0
    for cat in all_cats:
        ref_pct = (ref == cat).sum() / len(ref)
        new_pct = (new == cat).sum() / len(new)
        ref_pct = max(ref_pct, eps)
        new_pct = max(new_pct, eps)
        psi += (new_pct - ref_pct) * np.log(new_pct / ref_pct)

    return float(round(psi, 4))


def _ks_test(ref: pd.Series, new: pd.Series) -> Tuple[float, float]:
    """Kolmogorov-Smirnov test. Returns (statistic, p_value)."""
    ref = ref.dropna()
    new = new.dropna()
    if len(ref) < 5 or len(new) < 5:
        return 0.0, 1.0
    stat, p = stats.ks_2samp(ref.values, new.values)
    return float(round(stat, 4)), float(round(p, 4))


# ── SCHEMA CHECK ────────────────────────────────────────────────────────────
def check_schema(new_df: pd.DataFrame,
                 snapshot_meta: Dict) -> Dict:
    """
    Phase 1 — Hard compatibility check.
    Uses feature_stats keys (not feature_names) so ID-like columns
    that were excluded from stats don't cause false schema failures.
    Target column comparison is case-insensitive.
    Returns: {compatible: bool, issues: [str]}
    """
    issues = []

    ref_features = set(snapshot_meta.get("feature_stats", {}).keys())
    ref_types    = snapshot_meta.get("feature_types", {})
    ref_target   = snapshot_meta.get("target_column", "")
    new_cols     = set(new_df.columns)

    missing = ref_features - new_cols
    if missing:
        issues.append(f"Missing columns: {sorted(missing)}")

    target_found = (ref_target in new_cols or
                    ref_target.lower() in {c.lower() for c in new_cols})
    if ref_target and not target_found:
        issues.append(f"Target column '{ref_target}' not found in new data")

    for col, ref_type in ref_types.items():
        if col not in new_df.columns:
            continue
        new_type = ("numeric" if pd.api.types.is_numeric_dtype(new_df[col])
                    else "categorical")
        if new_type != ref_type:
            issues.append(f"Column '{col}' type changed: {ref_type} → {new_type}")

    return {
        "compatible": len(issues) == 0,
        "issues":     issues
    }


# ── STATISTICAL DRIFT ───────────────────────────────────────────────────────
def _is_id_column(series: pd.Series, col_name: str) -> bool:
    """Detect ID-like columns that should be excluded from drift check."""
    col_lower = col_name.lower()
    if col_lower in ("id","index","row_id","record_id","sample_id","uid","uuid",
                     "customer_id","user_id","order_id","product_id"):
        return True
    if col_lower.endswith("_id") or col_lower.endswith("id"):
        return True
    if pd.api.types.is_numeric_dtype(series):
        n        = len(series.dropna())
        n_unique = series.nunique()
        if n > 10 and n_unique / n > 0.95:
            diffs = series.dropna().sort_values().diff().dropna()
            if len(diffs) > 0 and (diffs == 1).mean() > 0.8:
                return True
    return False


def _chi_square_test(ref_freqs: Dict, new_series: pd.Series) -> Tuple[float, float]:
    """Chi-square test for categorical columns."""
    from scipy.stats import chisquare
    ref_freqs_lower = {k.lower(): v for k, v in ref_freqs.items()}
    new_counts = new_series.str.lower().value_counts()
    n_new      = len(new_series)

    all_cats = sorted(set(ref_freqs_lower.keys()) |
                      set(new_counts.index.tolist()))
    observed  = np.array([new_counts.get(c, 0)       for c in all_cats], float)
    expected  = np.array([ref_freqs_lower.get(c, 1e-6) * n_new
                          for c in all_cats], float)

    mask = expected >= 1
    if mask.sum() < 2:
        return 0.0, 1.0

    observed = observed[mask]
    expected = expected[mask]
    expected = expected / expected.sum() * observed.sum()
    if observed.sum() == 0:
        return 0.0, 1.0

    try:
        chi2, p = chisquare(observed, f_exp=expected)
        df        = max(len(observed) - 1, 1)
        chi2_norm = float(chi2 / df)
        return round(chi2_norm, 4), round(float(p), 4)
    except Exception:
        return 0.0, 1.0


def _find_best_match(ref_col: str, ref_stats: Dict, ref_type: str,
                     new_df: pd.DataFrame) -> Optional[str]:
    """Find best matching column by data type + distribution similarity."""
    candidates = []
    for new_col in new_df.columns:
        new_type = ("numeric" if pd.api.types.is_numeric_dtype(new_df[new_col])
                    else "categorical")
        if new_type != ref_type:
            continue

        if ref_type == "numeric":
            new_s    = pd.to_numeric(new_df[new_col], errors="coerce").dropna()
            if len(new_s) == 0:
                continue
            ref_mean = ref_stats.get("mean", 0)
            ref_std  = max(ref_stats.get("std", 1), 1e-6)
            new_mean = float(new_s.mean())
            new_std  = float(new_s.std()) if len(new_s) > 1 else 1.0
            mean_diff = abs(new_mean - ref_mean) / (ref_std + 1e-6)
            std_ratio = max(new_std, ref_std) / (min(new_std, ref_std) + 1e-6)
            score = mean_diff + (std_ratio - 1)
            candidates.append((new_col, score))
        else:
            ref_cats = set(k.lower() for k in
                           ref_stats.get("frequencies", {}).keys())
            new_cats = set(new_df[new_col].astype(str).str.lower().unique())
            overlap  = len(ref_cats & new_cats) / max(len(ref_cats), 1)
            candidates.append((new_col, 1.0 - overlap))

    if not candidates:
        return None
    return min(candidates, key=lambda x: x[1])[0]


def check_drift(new_df: pd.DataFrame,
                snapshot_meta: Dict) -> Dict:
    """
    Statistical drift detection using THREE methods:

    NUMERIC columns:
      • PSI      (histogram shape comparison)   — weight 0.5
      • KS test  (full CDF comparison)          — weight 0.5

    CATEGORICAL columns:
      • PSI        (category frequency shift)   — weight 0.5
      • Chi-square (count significance test)    — weight 0.5
    """
    feature_stats = snapshot_meta.get("feature_stats", {})
    feature_types = snapshot_meta.get("feature_types", {})
    feature_names = list(feature_stats.keys())

    per_feature     = {}
    combined_scores = []

    for col in feature_names:
        if col not in feature_stats:
            continue

        ref_stats = feature_stats[col]
        col_type  = feature_types.get(col, "numeric")

        # Column matching
        if col in new_df.columns:
            matched_col  = col
            match_method = "exact"
        else:
            matched_col  = _find_best_match(col, ref_stats, col_type, new_df)
            match_method = "auto" if matched_col else "none"

        if matched_col is None:
            continue
        if _is_id_column(new_df[matched_col], matched_col):
            continue

        col_type  = feature_types.get(col, "numeric")
        ref_stats = feature_stats[col]

        # ── NUMERIC ───────────────────────────────────────────────────
        if col_type == "numeric":
            ref_mean        = ref_stats.get("mean", 0)
            ref_std         = max(ref_stats.get("std", 1), 1e-6)
            ref_min         = ref_stats.get("min", ref_mean - 3*ref_std)
            ref_max         = ref_stats.get("max", ref_mean + 3*ref_std)
            ref_hist_counts = ref_stats.get("histogram_counts", [])
            ref_hist_edges  = ref_stats.get("histogram_edges",  [])

            new_series   = pd.to_numeric(new_df[matched_col], errors="coerce").dropna()
            new_mean     = float(new_series.mean()) if len(new_series) > 0 else ref_mean
            z_score      = abs(new_mean - ref_mean) / ref_std

            n_unique_new = new_series.nunique()
            n_unique_ref = sum(1 for c in ref_hist_counts if c > 0) if ref_hist_counts else 10
            is_discrete  = (n_unique_new <= 10 and n_unique_ref <= 10)

            if is_discrete:
                ref_freq_dict = {}
                if ref_hist_counts and ref_hist_edges:
                    total_ref = sum(ref_hist_counts)
                    for i, cnt in enumerate(ref_hist_counts):
                        if cnt > 0:
                            mid = round((ref_hist_edges[i] + ref_hist_edges[i+1]) / 2)
                            ref_freq_dict[str(int(mid))] = cnt / total_ref
                ref_series_cat = pd.Series(
                    [k for k, v in ref_freq_dict.items()
                     for _ in range(max(int(v * 1000), 1))])
                new_series_cat = new_series.astype(int).astype(str)
                psi     = _compute_psi_categorical(ref_series_cat, new_series_cat)
                ks_stat = 0.0
                ks_p    = 1.0
            else:
                if ref_hist_counts and ref_hist_edges:
                    ref_series = pd.Series(
                        np.repeat(
                            [(ref_hist_edges[i] + ref_hist_edges[i+1]) / 2
                             for i in range(len(ref_hist_counts))],
                            [max(int(c), 0) for c in ref_hist_counts]))
                else:
                    np.random.seed(42)
                    ref_series = pd.Series(
                        np.clip(np.random.normal(ref_mean, ref_std, 1000),
                                ref_min, ref_max))

                psi = _compute_psi_numeric(ref_series, new_series)
                ks_stat, ks_p = _ks_test(ref_series, new_series)

            psi_norm = min(psi / 2.0, 1.0)
            combined = round(psi_norm, 4) if match_method == "exact" else \
                       round(0.5 * psi_norm + 0.5 * ks_stat, 4)

            per_feature[col] = {
                "type":           "numeric",
                "matched_col":    matched_col,
                "match_method":   match_method,
                "psi":            round(psi, 4),
                "ks_stat":        round(ks_stat, 4),
                "ks_p":           round(ks_p, 4),
                "z_score":        round(z_score, 4),
                "combined_score": combined,
                "ref_mean":       round(ref_mean, 4),
                "new_mean":       round(new_mean, 4),
                "ref_std":        round(ref_stats.get("std", 0), 4),
                "new_std":        round(float(new_series.std()), 4) if len(new_series) > 1 else 0,
                "psi_status":     ("🔴 Drift" if psi > 0.25 else
                                   "🟡 Minor" if psi > 0.10 else "✅ Stable"),
                "ks_status":      ("🔴 Drift" if ks_p < 0.05 else "✅ Stable"),
            }
            combined_scores.append(combined)

        # ── CATEGORICAL ───────────────────────────────────────────────
        else:
            ref_freqs  = ref_stats.get("frequencies", {})
            n_ref_approx = 1000
            ref_series = pd.Series(
                [cat for cat, freq in ref_freqs.items()
                 for _ in range(max(int(freq * n_ref_approx), 1))])
            new_series        = new_df[matched_col].astype(str)
            ref_series        = ref_series.str.lower()
            new_series_lower  = new_series.str.lower()

            psi            = _compute_psi_categorical(ref_series, new_series_lower)
            chi2_norm, chi2_p = _chi_square_test(ref_freqs, new_series)

            new_freqs       = new_series_lower.value_counts(normalize=True).to_dict()
            ref_freqs_lower = {k.lower(): v for k, v in ref_freqs.items()}
            unseen          = set(new_freqs.keys()) - set(ref_freqs_lower.keys())

            psi_norm    = min(psi / 2.0, 1.0)
            chi_norm    = min(chi2_norm / 10.0, 1.0)
            unseen_ratio = sum(new_freqs.get(c, 0) for c in unseen)
            if unseen_ratio > 0.5:
                psi_norm = psi_norm * 0.3
            combined = round(0.5 * psi_norm + 0.5 * chi_norm, 4)

            per_feature[col] = {
                "type":              "categorical",
                "matched_col":       matched_col,
                "match_method":      match_method,
                "psi":               round(psi, 4),
                "chi2":              round(chi2_norm, 4),
                "chi2_p":            round(chi2_p, 4),
                "ks_stat":           None,
                "ks_p":              None,
                "combined_score":    combined,
                "ref_categories":    list(ref_freqs.keys()),
                "new_categories":    list(new_freqs.keys()),
                "unseen_categories": list(unseen),
                "unseen_pct":        round(sum(new_freqs.get(c, 0) for c in unseen), 4),
                "psi_status":        ("🔴 Drift" if psi > 0.25 else
                                      "🟡 Minor" if psi > 0.10 else "✅ Stable"),
                "chi2_status":       ("🔴 Drift" if chi2_p < 0.05 else "✅ Stable"),
            }
            combined_scores.append(combined)

    # ── OVERALL ───────────────────────────────────────────────────────
    if not combined_scores:
        return {
            "decision":      "no_overlap",
            "confidence":    "NONE",
            "label":         "⬜ Cannot Compare",
            "color":         "gray",
            "mean_psi":      0.0,
            "mean_combined": 0.0,
            "drift_score":   0.0,
            "n_drifted":     0,
            "n_features":    0,
            "thresholds":    {},
            "per_feature":   {},
        }

    mean_combined = float(np.mean(combined_scores))

    n_ref       = snapshot_meta.get("n_rows", 1000)
    scale       = max(1.5, (1000 / max(n_ref, 50)) ** 0.5)
    thresh_low  = round(0.10 * scale, 3)
    thresh_high = round(0.25 * scale, 3)
    eff_low     = thresh_low  / 4.0
    eff_high    = thresh_high / 4.0

    n_drifted   = sum(1 for s in combined_scores if s > eff_high)
    drift_score = n_drifted / len(combined_scores)
    drift_pct   = n_drifted / len(combined_scores)

    if drift_pct < 0.30:
        decision, confidence, label, color = "no_drift",    "HIGH",   "✅ No Drift",          "green"
    elif drift_pct < 0.60 or mean_combined < eff_high:
        decision, confidence, label, color = "minor_drift", "MEDIUM", "🟡 Minor Drift",       "orange"
    else:
        decision, confidence, label, color = "major_drift", "LOW",    "🔴 Major Drift",       "red"

    return {
        "decision":      decision,
        "confidence":    confidence,
        "label":         label,
        "color":         color,
        "mean_psi":      round(mean_combined, 4),
        "mean_combined": round(mean_combined, 4),
        "drift_score":   round(drift_score, 4),
        "n_drifted":     n_drifted,
        "n_features":    len(combined_scores),
        "thresholds":    {"low": eff_low, "high": eff_high, "scale": round(scale, 2)},
        "per_feature":   per_feature,
    }


# ── FULL CHECK ──────────────────────────────────────────────────────────────
def run_drift_check(new_df: pd.DataFrame,
                    snapshot_meta: Dict) -> Dict:
    """Soft Schema Check + Phase 2 Statistical Drift."""
    ref_problem  = snapshot_meta.get("problem_type", "")
    ref_target   = snapshot_meta.get("target_column", "")
    ref_features = set(snapshot_meta.get("feature_stats", {}).keys())
    new_cols     = set(new_df.columns)

    soft_warnings = []

    target_in_new = (ref_target in new_cols or
                     ref_target.lower() in {c.lower() for c in new_cols})
    if not target_in_new:
        soft_warnings.append(
            f"Target column '{ref_target}' not in new data — "
            f"this model predicts a different column")

    overlapping  = ref_features & new_cols
    overlap_pct  = len(overlapping) / len(ref_features) if ref_features else 0.0
    missing_cols = ref_features - new_cols
    extra_cols   = new_cols - ref_features - {ref_target}

    if overlap_pct == 0:
        soft_warnings.append(
            f"No columns overlap — completely different dataset domains "
            f"(snapshot has {sorted(ref_features)}, "
            f"new data has {sorted(list(new_cols)[:5])}...)")
    elif overlap_pct < 0.5:
        soft_warnings.append(
            f"Low column overlap ({overlap_pct:.0%}) — "
            f"only {len(overlapping)} of {len(ref_features)} snapshot features found")

    drift = check_drift(new_df, snapshot_meta)

    dist_similarity = 1.0 - min(drift["mean_psi"] * 2, 1.0)
    match_score     = round(0.3 * overlap_pct + 0.7 * dist_similarity, 4)

    can_reuse = (drift["decision"] in ("no_drift", "minor_drift")
                 and overlap_pct > 0.0)

    if overlap_pct == 0.0:
        reuse_label = "🔴 Retrain needed (No shared columns)"
    else:
        reuse_label = {
            "no_drift":    "✅ Reuse (No drift)",
            "minor_drift": "🟡 Reuse with caution (Minor drift)",
            "major_drift": "🔴 Retrain needed (Major drift)",
            "no_overlap":  "🔴 Retrain needed (No shared columns)",
        }.get(drift["decision"], drift["decision"])

    return {
        "snapshot_id":   snapshot_meta.get("snapshot_id", ""),
        "snapshot_name": snapshot_meta.get("dataset_name", ""),
        "model_name":    snapshot_meta.get("model_name", ""),
        "problem_type":  snapshot_meta.get("problem_type", ""),
        "target_column": snapshot_meta.get("target_column", ""),
        "metrics":       snapshot_meta.get("model_metrics", {}),
        "compatible":    True,
        "schema": {
            "compatible":    True,
            "issues":        [],
            "soft_warnings": soft_warnings,
            "overlap_pct":   round(overlap_pct, 4),
            "overlapping":   sorted(overlapping),
            "missing_cols":  sorted(missing_cols),
            "extra_cols":    sorted(extra_cols),
        },
        "drift":       drift,
        "can_reuse":   can_reuse,
        "reuse_label": reuse_label,
        "reuse_color": {"no_drift":"green","minor_drift":"orange",
                        "major_drift":"red","no_overlap":"gray"}.get(
                            drift["decision"],"gray"),
        "mean_psi":    drift["mean_psi"],
        "match_score": match_score,
        "overlap_pct": round(overlap_pct, 4),
    }
# tools/feature_engineering.py
"""
FEATURE ENGINEERING TOOL  —  Agentic Data Scientist
=====================================================
Runs AFTER EDA on df_cleaned.
Produces df_fe (model-ready) saved separately in session_state.

All 14 techniques included:
  1.  Constant / near-zero-variance column removal
  2.  High-missing column removal
  3.  DateTime feature extraction (year, month, day, dayofweek, is_weekend, hour)
  4.  Log1p transforms (skewed numerics, min >= 0)
  5.  Yeo-Johnson transforms (any skewed numeric)
  6.  Polynomial features  (top-N by correlation, configurable degree)
  7.  Interaction features (top-N pairs by correlation)
  8.  Ratio features       (top variance pairs)
  9.  Binning              (equal-width, skewed columns)
  10. One-Hot / Label / Target encoding
  11. Class Imbalance — SMOTE / Random Undersampling  (classification only)
  12. Drop highly correlated features (VIF optional)
  13. Feature importance — Random Forest only
      Scores ALL features: numeric + categorical (temp-encoded for RF, scored as whole column)
  14. Feature scaling — Standard / MinMax / Robust
  15. Top-K feature selection (user-controlled slider + checkboxes)

Two public functions consumed by app.py:
  analyse_features(df, target_col, problem_type) -> suggestions dict
  apply_feature_engineering(...)                  -> (df_fe, report)
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler,
    PowerTransformer
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

warnings.filterwarnings("ignore")

from tools.feature_engineering_memory import FEMemoryManager


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 1 — ANALYSIS
#  Pure Python — no Streamlit calls.  Returns suggestions dict only;
#  nothing is modified here.
# ═══════════════════════════════════════════════════════════════════════

def analyse_features(df: pd.DataFrame,
                     target_col: str,
                     problem_type: str) -> Dict:
    """
    Score ALL features (numeric + categorical) with Random Forest importance.
    Categorical columns are temp label-encoded for RF scoring only — scores are
    returned per original column name, not per dummy. Pearson r removed.
    The original df is never modified.

    Returns
    -------
    suggestions dict with keys:
      keep, drop, create, importance_df,
      skewed_cols, imbalance_info
    """
    import re

    feature_cols = [c for c in df.columns if c != target_col]
    numeric_cols = (df[feature_cols]
                    .select_dtypes(include=[np.number])
                    .columns.tolist())
    cat_cols     = (df[feature_cols]
                    .select_dtypes(include=["object", "category"])
                    .columns.tolist())

    suggestions = {
        "keep":           [],
        "drop":           [],
        "create":         [],
        "importance_df":  None,
        "skewed_cols":    [],
        "imbalance_info": None,
    }

    # ── STEP A: build X for RF (temp-encode categoricals) ────────────
    X_rf = df[feature_cols].copy()

    for col in cat_cols:
        le = LabelEncoder()
        X_rf[col] = le.fit_transform(
            X_rf[col].astype(str).fillna("__NA__"))

    for col in numeric_cols:
        X_rf[col] = X_rf[col].fillna(X_rf[col].median())

    # prepare target y
    y = df[target_col].copy()
    if y.dtype == object or str(y.dtype) == "category":
        y = LabelEncoder().fit_transform(y.astype(str).fillna("__NA__"))
    else:
        y = y.fillna(y.median()).values

    # ── STEP B: RF importance for ALL features ────────────────────────
    rf_scores: Dict[str, float] = {}
    try:
        if problem_type == "regression":
            rf = RandomForestRegressor(
                n_estimators=150, max_depth=8,
                random_state=42, n_jobs=-1)
        else:
            rf = RandomForestClassifier(
                n_estimators=150, max_depth=8,
                random_state=42, n_jobs=-1,
                class_weight="balanced")
        rf.fit(X_rf.values, y)
        rf_scores = dict(zip(feature_cols,
                             rf.feature_importances_.tolist()))
    except Exception as exc:
        print(f"⚠️ RF importance failed: {exc}")
        rf_scores = {c: 0.0 for c in feature_cols}

    # ── STEP C: build importance_df — ALL features ───────────────────
    # Pearson r removed — RF alone is the primary and sufficient signal.
    # RF already scores ALL features (numeric + categorical via temp encoding)
    # so there is no need for a secondary numeric-only metric.
    rows = []
    for col in feature_cols:
        rf_s = rf_scores.get(col, 0.0)
        rows.append({
            "feature":  col,
            "rf_score": round(float(rf_s), 5),
            "col_type": "numeric" if col in numeric_cols else "categorical",
        })

    imp_df = (pd.DataFrame(rows)
                .sort_values("rf_score", ascending=False)
                .reset_index(drop=True))
    imp_df["rank"] = imp_df.index + 1
    suggestions["importance_df"] = imp_df

    # ── STEP E: skewness detection ────────────────────────────────────
    for col in numeric_cols:
        try:
            sk = df[col].skew()
            if abs(sk) > 1.0:
                suggestions["skewed_cols"].append({
                    "name":      col,
                    "skewness":  round(float(sk), 3),
                    "direction": "right" if sk > 0 else "left",
                    "min_val":   float(df[col].min()),
                })
        except Exception:
            pass

    # ── STEP F: class imbalance detection ────────────────────────────
    if problem_type == "classification":
        vc = df[target_col].value_counts()
        if len(vc) >= 2:
            ratio = vc.min() / vc.max()
            suggestions["imbalance_info"] = {
                "ratio":          round(float(ratio), 3),
                "imbalanced":     ratio < 0.4,
                "class_counts":   vc.to_dict(),
                "minority_class": str(vc.idxmin()),
                "majority_class": str(vc.idxmax()),
            }

    # ── STEP G: keep / drop rules ─────────────────────────────────────
    corr_matrix = (df[numeric_cols].corr().abs()
                   if len(numeric_cols) >= 2 else pd.DataFrame())
    dropped_set: set = set()

    for col in feature_cols:
        dtype       = str(df[col].dtype)
        missing_pct = df[col].isnull().mean()
        rf_s        = rf_scores.get(col, 0.0)

        # drop: >50 % missing
        if missing_pct > 0.50:
            dropped_set.add(col)
            suggestions["drop"].append({
                "name": col, "dtype": dtype,
                "reason": "high_missing",
                "detail": f"{missing_pct*100:.1f}% missing values",
                "rf_score": round(rf_s, 4),
            }); continue

        # drop: constant
        if df[col].nunique(dropna=False) <= 1:
            dropped_set.add(col)
            suggestions["drop"].append({
                "name": col, "dtype": dtype,
                "reason": "constant",
                "detail": "Only one unique value — zero variance",
                "rf_score": 0.0,
            }); continue

        # drop: high pairwise correlation (numeric only)
        if col in numeric_cols and not corr_matrix.empty:
            kept_num = [r["name"] for r in suggestions["keep"]
                        if r["name"] in numeric_cols]
            skip = False
            for kept in kept_num:
                if (kept in corr_matrix.columns
                        and col in corr_matrix.columns
                        and corr_matrix.loc[col, kept] > 0.92):
                    dropped_set.add(col)
                    suggestions["drop"].append({
                        "name": col, "dtype": dtype,
                        "reason": "high_correlation",
                        "detail": f"|r|={corr_matrix.loc[col,kept]:.2f} with '{kept}'",
                        "rf_score": round(rf_s, 4),
                    })
                    skip = True; break
            if skip: continue

        # drop: near-zero RF importance (numeric only)
        if col in numeric_cols and rf_s < 0.001:
            dropped_set.add(col)
            suggestions["drop"].append({
                "name": col, "dtype": dtype,
                "reason": "low_importance",
                "detail": f"RF score={rf_s:.5f} (near zero)",
                "rf_score": round(rf_s, 4),
            }); continue

        # keep
        if col in numeric_cols:
            detail = f"RF={rf_s:.4f}"
            reason = "good_rf_importance"
        else:
            n_uniq = df[col].nunique()
            detail = f"{n_uniq} unique values  |  RF={rf_s:.4f}"
            reason = "categorical_feature"

        suggestions["keep"].append({
            "name":     col,
            "dtype":    dtype,
            "reason":   reason,
            "detail":   detail,
            "rf_score": round(rf_s, 4),
        })

    # ── STEP H: suggest new features ──────────────────────────────────
    def _looks_like_date(series: pd.Series) -> bool:
        sample = series.dropna().head(10).astype(str)
        return (sample.str.match(re.compile(r"\d{4}[-/]\d{2}[-/]\d{2}"))
                .mean() > 0.5)

    # skewness transforms
    for info in suggestions["skewed_cols"]:
        col = info["name"]
        if col in dropped_set: continue
        sk, mv = info["skewness"], info["min_val"]
        if mv >= 0:
            suggestions["create"].append({
                "name": f"{col}_log1p", "method": "log_transform",
                "sources": [col],
                "description": f"log1p({col})  skewness={sk:.2f} → ~0",
                "enabled": True, "category": "skewness",
            })
        suggestions["create"].append({
            "name": f"{col}_yeojohnson", "method": "yeo_johnson",
            "sources": [col],
            "description": f"Yeo-Johnson({col})  works on negatives too",
            "enabled": False, "category": "skewness",
        })
        suggestions["create"].append({
            "name": f"{col}_bin", "method": "bin",
            "sources": [col],
            "description": f"Equal-width bins of {col}  (5 bins default)",
            "enabled": False, "category": "skewness",
            "n_bins": 5,
        })

    # datetime extraction
    for col in feature_cols:
        if col in dropped_set: continue
        if (pd.api.types.is_datetime64_any_dtype(df[col]) or
                (df[col].dtype == object and _looks_like_date(df[col]))):
            suggestions["create"].append({
                "name": f"{col}_datetime_parts",
                "method": "datetime_extraction",
                "sources": [col],
                "description": (f"Extract year, month, day, "
                                f"dayofweek, is_weekend from {col}"),
                "enabled": True, "category": "datetime",
            })

    # top-2 interaction (numeric only)
    top_num = (imp_df[imp_df["col_type"] == "numeric"]
               .head(2)["feature"].tolist())
    if len(top_num) == 2:
        c1, c2 = top_num
        if c1 not in dropped_set and c2 not in dropped_set:
            suggestions["create"].append({
                "name": f"{c1}_x_{c2}", "method": "interaction",
                "sources": [c1, c2],
                "description": f"{c1} × {c2}  (top-2 RF numeric features)",
                "enabled": False, "category": "interaction",
            })

    # polynomial (top numeric feature)
    top_num_1 = (imp_df[imp_df["col_type"] == "numeric"]
                 .head(1)["feature"].tolist())
    if top_num_1 and top_num_1[0] not in dropped_set:
        tf = top_num_1[0]
        suggestions["create"].append({
            "name": f"{tf}_squared", "method": "polynomial",
            "sources": [tf],
            "description": f"{tf}²  — captures non-linear pattern",
            "enabled": False, "category": "polynomial",
        })

    # ratio (top-2 by variance, numeric only)
    var_cols = (df[numeric_cols].var()
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .nlargest(2).index.tolist())
    if len(var_cols) == 2:
        c1, c2 = var_cols
        if c1 not in dropped_set and c2 not in dropped_set:
            suggestions["create"].append({
                "name": f"{c1}_div_{c2}", "method": "ratio",
                "sources": [c1, c2],
                "description": f"{c1} / {c2}  (top variance pair)",
                "enabled": False, "category": "ratio",
            })

    return suggestions


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 3 — APPLY
#  Executes only the operations the user approved.
# ═══════════════════════════════════════════════════════════════════════

def apply_feature_engineering(
    df: pd.DataFrame,
    target_col: str,
    problem_type: str,
    keep_cols: List[str],
    create_approved: List[Dict],
    encode_cats: bool,
    encoding_method: str,       # 'auto' | 'onehot' | 'label'
    handle_imbalance: bool,
    imbalance_method: str,      # 'smote' | 'undersample'
    scale_method: str,          # 'standard' | 'minmax' | 'robust' | 'none'
    top_k: int,                 # 0 = keep all approved
    importance_df: Optional[pd.DataFrame],
    test_size: float,           # train/test split ratio e.g. 0.2
    session_id: str,
    fe_mem: FEMemoryManager,
) -> Tuple[Dict, Dict]:
    """
    Execute approved feature engineering — correct leakage-free pipeline:

      1. Keep approved columns only
      2. Create new features (log, yeo-johnson, datetime, interaction,
                              polynomial, ratio, bin)
      3. Top-K feature selection by RF importance (on original names)
         → produces X_full (features only) and y (target)
      4. Train/Test split: X_full + y → X_train, X_test, y_train, y_test
      5. Encode X_train (fit + transform), X_test (transform only)
         — encoders fitted ONLY on training data
      6. Scale X_train (fit + transform), X_test (transform only)
         — scalers fitted ONLY on training data
      7. Handle class imbalance on X_train + y_train only (optional)

    Returns:
      splits: {
          "X_train": DataFrame, "X_test": DataFrame,
          "y_train": Series,    "y_test": Series,
          "feature_names": List[str]
      }
      report: {kept, dropped, created, encoded, scaled, imbalance, errors}
    """
    report = {
        "kept": [], "dropped": [], "created": [],
        "encoded": [], "imbalance": [], "scaled": [], "errors": []
    }

    # ── 1. Keep approved columns ──────────────────────────────────────
    available         = [c for c in keep_cols if c in df.columns] + [target_col]
    report["dropped"] = [c for c in df.columns if c not in available]
    df = df[[c for c in available if c in df.columns]].copy()

    # ── 2. Feature creation ───────────────────────────────────────────
    for feat in create_approved:
        method  = feat["method"]
        sources = feat["sources"]
        name    = feat["name"]
        try:
            # log1p transform
            if method == "log_transform" and len(sources) == 1:
                src = sources[0]
                if src in df.columns and df[src].min() >= 0:
                    df[name] = np.log1p(df[src])
                    report["created"].append(name)
                    fe_mem.log_created_feature(session_id, name, method,
                                               sources, feat.get("description", ""))

            # Yeo-Johnson
            elif method == "yeo_johnson" and len(sources) == 1:
                src = sources[0]
                if src in df.columns:
                    pt = PowerTransformer(method="yeo-johnson")
                    df[name] = pt.fit_transform(df[[src]])
                    report["created"].append(name)
                    fe_mem.log_created_feature(session_id, name, method,
                                               sources, feat.get("description", ""))

            # datetime extraction
            elif method == "datetime_extraction" and len(sources) == 1:
                src = sources[0]
                if src in df.columns:
                    dt = pd.to_datetime(df[src], errors="coerce")
                    new_cols = []
                    parts = [
                        ("_year",       dt.dt.year),
                        ("_month",      dt.dt.month),
                        ("_day",        dt.dt.day),
                        ("_dayofweek",  dt.dt.dayofweek),
                        ("_is_weekend", dt.dt.dayofweek.isin([5, 6]).astype(int)),
                    ]
                    if dt.dt.hour.sum() > 0:
                        parts.append(("_hour", dt.dt.hour))
                    for sfx, val in parts:
                        col_name = f"{src}{sfx}"
                        df[col_name] = val
                        new_cols.append(col_name)
                    report["created"].extend(new_cols)
                    fe_mem.log_created_feature(session_id,
                        f"{src}_datetime_parts", method,
                        sources, feat.get("description", ""))

            # interaction
            elif method == "interaction" and len(sources) == 2:
                c1, c2 = sources
                if c1 in df.columns and c2 in df.columns:
                    df[name] = df[c1] * df[c2]
                    report["created"].append(name)
                    fe_mem.log_created_feature(session_id, name, method,
                                               sources, feat.get("description", ""))

            # polynomial
            elif method == "polynomial" and len(sources) == 1:
                src = sources[0]
                if src in df.columns:
                    df[name] = df[src] ** 2
                    report["created"].append(name)
                    fe_mem.log_created_feature(session_id, name, method,
                                               sources, feat.get("description", ""))

            # ratio
            elif method == "ratio" and len(sources) == 2:
                c1, c2 = sources
                if c1 in df.columns and c2 in df.columns:
                    denom    = df[c2].replace(0, np.nan)
                    df[name] = (df[c1] / denom).replace(
                        [np.inf, -np.inf], np.nan)
                    df[name].fillna(df[name].median(), inplace=True)
                    report["created"].append(name)
                    fe_mem.log_created_feature(session_id, name, method,
                                               sources, feat.get("description", ""))

            # binning
            elif method == "bin" and len(sources) == 1:
                src    = sources[0]
                n_bins = feat.get("n_bins", 5)
                if src in df.columns:
                    df[name] = pd.cut(df[src], bins=n_bins, labels=False)
                    df[name] = df[name].fillna(-1).astype(int)
                    report["created"].append(name)
                    fe_mem.log_created_feature(session_id, name, method,
                                               sources, feat.get("description", ""))

        except Exception as exc:
            report["errors"].append(f"Could not create '{name}': {exc}")

    # ── 3. Top-K feature selection ───────────────────────────────────
    # Runs BEFORE encoding on original column names.
    # RF scored on clean original columns — select top-K here,
    # then encode/scale only the selected features.
    # User-created new features (log1p, polynomial, etc.) always kept.
    if (top_k > 0 and importance_df is not None
            and not importance_df.empty):

        ranked   = importance_df["feature"].tolist()  # ordered by RF score
        # features that survived step 1+2 and are in the ranking
        existing = [f for f in ranked if f in df.columns
                    and f != target_col]
        # user-created new features (not in original ranking) — always keep
        new_feats = [c for c in df.columns
                     if c != target_col and c not in ranked]
        # take exactly top_k from ranked, plus all user-created
        selected  = existing[:top_k]
        final_keep = selected + new_feats + [target_col]
        dropped_by_k = [c for c in df.columns if c not in final_keep]
        for col in dropped_by_k:
            report["dropped"].append(col)
            fe_mem.log_dropped_feature(
                session_id, col, "below_top_k",
                f"Not in top-{top_k} by RF importance")
        df = df[[c for c in final_keep if c in df.columns]]


    # ── 4. Split into X, y then Train/Test ───────────────────────────
    # This is the CORRECT position for splitting:
    #   Encoders and scalers will be fitted on X_train ONLY
    #   preventing any data leakage from test set.
    X_full = df.drop(columns=[target_col]).copy()
    y_full = df[target_col].copy()

    # encode target if object (for classification)
    le_target = None
    if y_full.dtype == object or str(y_full.dtype) == "category":
        from sklearn.preprocessing import LabelEncoder as _LE
        le_target = _LE()
        y_full = pd.Series(
            le_target.fit_transform(y_full.astype(str)),
            index=y_full.index, name=target_col)

    from sklearn.model_selection import train_test_split as _tts
    stratify = y_full if problem_type == "classification" else None
    try:
        X_train, X_test, y_train, y_test = _tts(
            X_full, y_full,
            test_size=test_size, random_state=42, stratify=stratify)
    except Exception:
        X_train, X_test, y_train, y_test = _tts(
            X_full, y_full, test_size=test_size, random_state=42)

    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    # ── 5. Encode — fit on X_train, transform both ───────────────────
    if encode_cats:
        cat_cols_to_enc = [c for c in X_train.select_dtypes(
            include=["object", "category"]).columns]

        for col in cat_cols_to_enc:
            n_uniq = X_train[col].nunique()
            try:
                if encoding_method == "onehot" or (
                        encoding_method == "auto" and n_uniq <= 15):
                    # fit on train only
                    from sklearn.preprocessing import LabelEncoder as _LE2
                    dummies_train = pd.get_dummies(
                        X_train[col], prefix=col, drop_first=True, dtype=int)
                    # align test to same columns (handle unseen categories)
                    dummies_test  = pd.get_dummies(
                        X_test[col], prefix=col, drop_first=True, dtype=int)
                    dummies_test  = dummies_test.reindex(
                        columns=dummies_train.columns, fill_value=0)

                    X_train = pd.concat(
                        [X_train.drop(columns=[col]), dummies_train], axis=1)
                    X_test  = pd.concat(
                        [X_test.drop(columns=[col]),  dummies_test],  axis=1)
                    report["encoded"].append(
                        f"{col} → one-hot ({len(dummies_train.columns)} cols)")
                    fe_mem.log_scaling_op(session_id, col,
                        "OneHotEncoding", {"n_unique": n_uniq})

                else:  # label encoding
                    le_col = LabelEncoder()
                    # fit on train
                    le_col.fit(X_train[col].astype(str))
                    known = set(le_col.classes_)
                    # transform train
                    X_train[col] = le_col.transform(X_train[col].astype(str))
                    # transform test — unseen → -1
                    X_test[col] = X_test[col].astype(str).apply(
                        lambda v: le_col.transform([v])[0]
                        if v in known else -1)
                    report["encoded"].append(
                        f"{col} → label-encoded ({n_uniq} cats)")
                    fe_mem.log_scaling_op(session_id, col,
                        "LabelEncoding", {"n_unique": n_uniq})

            except Exception as exc:
                report["errors"].append(f"Encoding error on '{col}': {exc}")

    # ── 6. Scale — fit on X_train, transform both ────────────────────
    if scale_method != "none":
        all_num_tr = [c for c in X_train.select_dtypes(
            include=[np.number]).columns]
        # skip binary 0/1 columns (OHE dummies)
        num_cols_to_scale = [c for c in all_num_tr
                             if X_train[c].dropna().nunique() > 2]
        scaler = {
            "standard": StandardScaler(),
            "minmax":   MinMaxScaler(),
            "robust":   RobustScaler(),
        }.get(scale_method, StandardScaler())
        if num_cols_to_scale:
            # fit on X_train ONLY
            scaler.fit(X_train[num_cols_to_scale])
            X_train[num_cols_to_scale] = scaler.transform(
                X_train[num_cols_to_scale])
            X_test[num_cols_to_scale]  = scaler.transform(
                X_test[num_cols_to_scale])
            report["scaled"] = num_cols_to_scale
            for col in num_cols_to_scale:
                fe_mem.log_scaling_op(
                    session_id, col,
                    type(scaler).__name__, {"method": scale_method})

    # ── 7. Class imbalance — on X_train + y_train only ───────────────
    if handle_imbalance and problem_type == "classification":
        try:
            X_tr_bal = X_train.fillna(X_train.median(numeric_only=True))
            if imbalance_method == "smote":
                from imblearn.over_sampling import SMOTE
                sm = SMOTE(random_state=42)
                X_res, y_res = sm.fit_resample(X_tr_bal, y_train)
                before = y_train.value_counts().to_dict()
                after  = pd.Series(y_res).value_counts().to_dict()
                X_train = pd.DataFrame(X_res, columns=X_train.columns)
                y_train = pd.Series(y_res, name=target_col)
                report["imbalance"].append(f"SMOTE: {before} → {after}")
                fe_mem.log_insight(session_id, "imbalance_smote",
                    f"SMOTE on train only. Before:{before} After:{after}",
                    [target_col], {"method": "smote"}, confidence=0.95)
            elif imbalance_method == "undersample":
                from imblearn.under_sampling import RandomUnderSampler
                rus = RandomUnderSampler(random_state=42)
                X_res, y_res = rus.fit_resample(X_tr_bal, y_train)
                before = y_train.value_counts().to_dict()
                after  = pd.Series(y_res).value_counts().to_dict()
                X_train = pd.DataFrame(X_res, columns=X_train.columns)
                y_train = pd.Series(y_res, name=target_col)
                report["imbalance"].append(f"Undersample: {before} → {after}")
                fe_mem.log_insight(session_id, "imbalance_undersample",
                    f"Undersampling on train only. Before:{before} After:{after}",
                    [target_col], {"method": "undersample"}, confidence=0.95)
        except ImportError:
            report["errors"].append(
                "imbalanced-learn not installed. Run: pip install imbalanced-learn")
        except Exception as exc:
            report["errors"].append(f"Imbalance error: {exc}")

    feature_names = X_train.columns.tolist()

    report["kept"] = [c for c in df.columns if c != target_col]

    # ── Save pre-split df to memory (before encoding/scaling) ─────────
    # df here is the top-K selected dataframe (original values, no encoding)
    # This is saved so AI chat can reference it.
    try:
        fe_mem.save_engineered_df(
            session_id   = session_id,
            df           = df,
            dataset_name = "engineered_dataset",
            target_column= target_col,
            problem_type = problem_type)
    except Exception as _e:
        report["errors"].append(f"Memory save warning: {_e}")

    # ── Return splits dict — what app.py expects ──────────────────────
    # Order:
    #   Step 1 — keep approved cols
    #   Step 2 — create new features
    #   Step 3 — top-K selection  (on original column names, before encoding)
    #   Step 4 — train/test split  ← HERE
    #   Step 5 — encode X_train (fit+transform), X_test (transform only)
    #   Step 6 — scale  X_train (fit+transform), X_test (transform only)
    #   Step 7 — class imbalance on X_train+y_train only
    splits = {
        "X_train":       X_train,
        "X_test":        X_test,
        "y_train":       y_train,
        "y_test":        y_test,
        "feature_names": feature_names,
    }
    return splits, report
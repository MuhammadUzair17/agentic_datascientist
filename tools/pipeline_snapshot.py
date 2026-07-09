# tools/pipeline_snapshot.py
"""
Pipeline Snapshot Manager
=========================
Saves and loads complete end-to-end pipeline snapshots.

Each snapshot contains:
  - Schema info (columns, types, target, problem type)
  - Reference statistics per feature (for drift detection)
  - Fitted preprocessing objects (encoders + scaler)
  - Trained model + metrics
  - Feature engineering config

Storage:
  pipeline_snapshots/
      index.db                  ← SQLite index of all snapshots
      SNAP-YYYYMMDD-HHMMSS/
          metadata.json         ← lightweight (loaded for drift check)
          pipeline.pkl          ← heavy (loaded only on reuse)
"""

import os
import json
import pickle
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any


SNAPSHOTS_DIR = "pipeline_snapshots"
INDEX_DB      = os.path.join(SNAPSHOTS_DIR, "index.db")


# ── INDEX DATABASE ──────────────────────────────────────────────────────────
def _init_index_db():
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    conn = sqlite3.connect(INDEX_DB)
    cur  = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS snapshots (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        snapshot_id     TEXT    UNIQUE NOT NULL,
        created_at      TEXT    NOT NULL,
        dataset_name    TEXT,
        target_column   TEXT,
        problem_type    TEXT,
        model_name      TEXT,
        n_features      INTEGER,
        n_rows          INTEGER,
        feature_names   TEXT,
        metrics_json    TEXT,
        snapshot_dir    TEXT    NOT NULL
    )""")
    conn.commit()
    conn.close()


# ── SAVE SNAPSHOT ───────────────────────────────────────────────────────────
def save_snapshot(
        df_original:    pd.DataFrame,
        target_col:     str,
        problem_type:   str,
        feature_names:  List[str],
        encoders:       Dict,
        scaler:         Any,
        model:          Any,
        model_name:     str,
        model_metrics:  Dict,
        fe_config:      Dict,
        dataset_name:   str = "dataset") -> str:
    """
    Save a complete pipeline snapshot.
    df_original = cleaned df BEFORE encoding/scaling (for reference stats).
    Returns snapshot_id.
    """
    _init_index_db()

    snapshot_id  = f"SNAP-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    snap_dir     = os.path.join(SNAPSHOTS_DIR, snapshot_id)
    os.makedirs(snap_dir, exist_ok=True)

    # ── Feature types ────────────────────────────────────────────────
    feature_types = {}
    for col in feature_names:
        if col in df_original.columns:
            feature_types[col] = (
                "numeric" if pd.api.types.is_numeric_dtype(df_original[col])
                else "categorical")

    # ── Reference statistics per feature ─────────────────────────────
    # Skip ID-like columns — sequential integers have no distribution meaning
    # and comparing them across datasets always shows false drift.
    def _is_id_col(series: pd.Series, col_name: str) -> bool:
        col_lower = col_name.lower()
        if col_lower in ("id","index","row_id","record_id","sample_id",
                         "uid","uuid","customer_id","user_id","order_id"):
            return True
        if col_lower.endswith("_id") or col_lower == "id":
            return True
        if pd.api.types.is_numeric_dtype(series):
            n, n_unique = len(series.dropna()), series.nunique()
            if n > 10 and n_unique / n > 0.95:
                diffs = series.dropna().sort_values().diff().dropna()
                if len(diffs) > 0 and (diffs == 1).mean() > 0.8:
                    return True
        return False

    feature_stats = {}
    for col in feature_names:
        if col not in df_original.columns:
            continue
        series = df_original[col].dropna()
        # Skip ID-like columns — no drift meaning
        if _is_id_col(series, col):
            feature_types.pop(col, None)
            continue
        if feature_types.get(col) == "numeric":
            series_num = pd.to_numeric(series, errors="coerce").dropna()
            # Store histogram for accurate PSI later
            if len(series_num) > 0:
                counts, edges = np.histogram(series_num, bins=10)
                feature_stats[col] = {
                    "mean":   float(series_num.mean()),
                    "std":    float(series_num.std()),
                    "min":    float(series_num.min()),
                    "max":    float(series_num.max()),
                    "q25":    float(series_num.quantile(0.25)),
                    "median": float(series_num.median()),
                    "q75":    float(series_num.quantile(0.75)),
                    "histogram_counts": counts.tolist(),
                    "histogram_edges":  edges.tolist(),
                }
        else:
            freqs = series.astype(str).value_counts(normalize=True).to_dict()
            feature_stats[col] = {"frequencies": freqs}

    # ── Target statistics ─────────────────────────────────────────────
    if target_col in df_original.columns:
        tgt = df_original[target_col].dropna()
        if problem_type == "classification":
            target_stats = {
                "frequencies": tgt.astype(str).value_counts(normalize=True).to_dict()
            }
        else:
            tgt_num = pd.to_numeric(tgt, errors="coerce").dropna()
            target_stats = {
                "mean": float(tgt_num.mean()),
                "std":  float(tgt_num.std()),
                "min":  float(tgt_num.min()),
                "max":  float(tgt_num.max()),
            }
    else:
        target_stats = {}

    # ── Class labels (classification only) ───────────────────────────
    class_labels = []
    if problem_type == "classification" and target_col in df_original.columns:
        class_labels = sorted(df_original[target_col].dropna().astype(str).unique().tolist())

    # ── metadata.json (lightweight — used for drift check) ────────────
    metadata = {
        "snapshot_id":   snapshot_id,
        "created_at":    datetime.now().isoformat(),
        "dataset_name":  dataset_name,
        "target_column": target_col,
        "problem_type":  problem_type,
        "class_labels":  class_labels,
        "feature_names": feature_names,
        "feature_types": feature_types,
        "feature_stats": feature_stats,
        "target_stats":  target_stats,
        "n_rows":        len(df_original),
        "n_features":    len(feature_names),
        "model_name":    model_name,
        "model_metrics": model_metrics,
        "fe_config":     fe_config,
    }

    meta_path = os.path.join(snap_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # ── pipeline.pkl (heavy — loaded only on reuse) ───────────────────
    pipeline_obj = {
        "snapshot_id":  snapshot_id,
        "encoders":     encoders,
        "scaler":       scaler,
        "model":        model,
        "model_name":   model_name,
        "feature_names":feature_names,
        "target_col":   target_col,
        "problem_type": problem_type,
        "class_labels": class_labels,
        "fe_config":    fe_config,
    }
    pkl_path = os.path.join(snap_dir, "pipeline.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(pipeline_obj, f)

    # ── Update index DB ───────────────────────────────────────────────
    conn = sqlite3.connect(INDEX_DB)
    cur  = conn.cursor()
    cur.execute("""
    INSERT OR REPLACE INTO snapshots
        (snapshot_id, created_at, dataset_name, target_column,
         problem_type, model_name, n_features, n_rows,
         feature_names, metrics_json, snapshot_dir)
    VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
    (snapshot_id, metadata["created_at"], dataset_name, target_col,
     problem_type, model_name, len(feature_names), len(df_original),
     json.dumps(feature_names), json.dumps(model_metrics), snap_dir))
    conn.commit()
    conn.close()

    return snapshot_id


# ── LOAD SNAPSHOT ────────────────────────────────────────────────────────────
def load_snapshot_metadata(snapshot_id: str) -> Optional[Dict]:
    """Load lightweight metadata (for drift check). Fast."""
    snap_dir  = os.path.join(SNAPSHOTS_DIR, snapshot_id)
    meta_path = os.path.join(snap_dir, "metadata.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f)


def load_snapshot_pipeline(snapshot_id: str) -> Optional[Dict]:
    """Load full pipeline object (encoders + scaler + model). Slow."""
    snap_dir = os.path.join(SNAPSHOTS_DIR, snapshot_id)
    pkl_path = os.path.join(snap_dir, "pipeline.pkl")
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# ── LIST ALL SNAPSHOTS ───────────────────────────────────────────────────────
def list_snapshots() -> List[Dict]:
    """Return all saved snapshots from index DB, newest first."""
    if not os.path.exists(INDEX_DB):
        return []
    try:
        conn = sqlite3.connect(INDEX_DB)
        cur  = conn.cursor()
        cur.execute("""
        SELECT snapshot_id, created_at, dataset_name, target_column,
               problem_type, model_name, n_features, n_rows, metrics_json
        FROM snapshots ORDER BY created_at DESC""")
        rows = cur.fetchall()
        conn.close()
        result = []
        for row in rows:
            snap_dir  = os.path.join(SNAPSHOTS_DIR, row[0])
            meta_path = os.path.join(snap_dir, "metadata.json")
            if os.path.exists(meta_path):
                metrics = json.loads(row[8]) if row[8] else {}
                result.append({
                    "snapshot_id":   row[0],
                    "created_at":    row[1],
                    "dataset_name":  row[2],
                    "target_column": row[3],
                    "problem_type":  row[4],
                    "model_name":    row[5],
                    "n_features":    row[6],
                    "n_rows":        row[7],
                    "metrics":       metrics,
                })
        return result
    except Exception:
        return []


# ── DELETE SNAPSHOT ───────────────────────────────────────────────────────────
def delete_snapshot(snapshot_id: str) -> bool:
    """Remove snapshot from disk and index."""
    import shutil
    snap_dir = os.path.join(SNAPSHOTS_DIR, snapshot_id)
    if os.path.exists(snap_dir):
        shutil.rmtree(snap_dir)
    if os.path.exists(INDEX_DB):
        conn = sqlite3.connect(INDEX_DB)
        conn.execute("DELETE FROM snapshots WHERE snapshot_id=?", (snapshot_id,))
        conn.commit()
        conn.close()
    return True
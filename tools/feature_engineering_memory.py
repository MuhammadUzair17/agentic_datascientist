# tools/feature_engineering_memory.py
"""
Feature Engineering Memory Management System
Tracks all FE operations for AI chat assistant integration.

Database: memory/feature_engineering/fe_metadata.db

Tables (new additions marked ★):
- fe_sessions        : Track FE sessions
- selected_features  : Final selected features per session
- dropped_features   : Features dropped and why
- created_features   : New features created
- scaling_ops        : Scaling / encoding operations
- importance_scores  : RF importance + Pearson r scores  ★ updated
- skewness_ops       : Which columns were skewness-transformed ★ new
- imbalance_ops      : Class balancing operations ★ new
- fe_insights        : Plain-language insights for chat
"""

import sqlite3
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import os


class FEMemoryManager:
    """
    Feature Engineering Memory Manager.

    AI chat can answer:
      "Which features were selected and their importance?"
      "Why was column X dropped?"
      "What new features were created?"
      "Was SMOTE applied?"
      "Which columns were skewness-transformed?"
      "What was the Pearson r of feature Y?"
    """

    def __init__(self, db_path: str = "memory/feature_engineering/fe_metadata.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn    = None
        self._init_database()

    # ──────────────────────────────────────────────────────────────────
    def _init_database(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur       = self.conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS fe_sessions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT    UNIQUE NOT NULL,
            dataset_name    TEXT,
            timestamp       TEXT    NOT NULL,
            problem_type    TEXT,
            target_column   TEXT,
            input_features  INTEGER,
            output_features INTEGER,
            rows_before     INTEGER,
            rows_after      INTEGER,
            notes           TEXT
        )""")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS selected_features (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id       TEXT    NOT NULL,
            feature_name     TEXT    NOT NULL,
            data_type        TEXT,
            rf_importance    REAL,
            pearson_r        REAL,
            selection_method TEXT,
            timestamp        TEXT    NOT NULL,
            FOREIGN KEY (session_id) REFERENCES fe_sessions(session_id)
        )""")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS dropped_features (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   TEXT    NOT NULL,
            feature_name TEXT    NOT NULL,
            reason       TEXT,
            detail       TEXT,
            rf_importance REAL,
            pearson_r    REAL,
            timestamp    TEXT    NOT NULL,
            FOREIGN KEY (session_id) REFERENCES fe_sessions(session_id)
        )""")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS created_features (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT    NOT NULL,
            feature_name    TEXT    NOT NULL,
            creation_method TEXT,
            source_columns  TEXT,
            description     TEXT,
            category        TEXT,
            timestamp       TEXT    NOT NULL,
            FOREIGN KEY (session_id) REFERENCES fe_sessions(session_id)
        )""")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS scaling_ops (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   TEXT    NOT NULL,
            feature_name TEXT    NOT NULL,
            operation    TEXT,
            parameters   TEXT,
            timestamp    TEXT    NOT NULL,
            FOREIGN KEY (session_id) REFERENCES fe_sessions(session_id)
        )""")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS importance_scores (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   TEXT    NOT NULL,
            feature_name TEXT    NOT NULL,
            method       TEXT,
            score        REAL,
            pearson_r    REAL,
            rank         INTEGER,
            timestamp    TEXT    NOT NULL,
            FOREIGN KEY (session_id) REFERENCES fe_sessions(session_id)
        )""")

        # ★ NEW: skewness operations
        cur.execute("""
        CREATE TABLE IF NOT EXISTS skewness_ops (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT    NOT NULL,
            feature_name    TEXT    NOT NULL,
            original_skewness REAL,
            transform_method TEXT,
            new_feature_name TEXT,
            timestamp       TEXT    NOT NULL,
            FOREIGN KEY (session_id) REFERENCES fe_sessions(session_id)
        )""")

        # ★ NEW: imbalance operations
        cur.execute("""
        CREATE TABLE IF NOT EXISTS imbalance_ops (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT    NOT NULL,
            method          TEXT    NOT NULL,
            target_column   TEXT,
            class_counts_before TEXT,
            class_counts_after  TEXT,
            rows_before     INTEGER,
            rows_after      INTEGER,
            timestamp       TEXT    NOT NULL,
            FOREIGN KEY (session_id) REFERENCES fe_sessions(session_id)
        )""")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS fe_insights (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id       TEXT    NOT NULL,
            insight_type     TEXT    NOT NULL,
            description      TEXT    NOT NULL,
            columns_involved TEXT,
            statistics       TEXT,
            confidence       REAL,
            timestamp        TEXT    NOT NULL,
            FOREIGN KEY (session_id) REFERENCES fe_sessions(session_id)
        )""")

        # ★ NEW: store the final engineered dataframe as CSV
        cur.execute("""
        CREATE TABLE IF NOT EXISTS engineered_datasets (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT    NOT NULL,
            dataset_name    TEXT,
            target_column   TEXT,
            problem_type    TEXT,
            rows            INTEGER,
            columns         INTEGER,
            feature_names   TEXT,
            csv_data        TEXT    NOT NULL,
            timestamp       TEXT    NOT NULL,
            FOREIGN KEY (session_id) REFERENCES fe_sessions(session_id)
        )""")

        # ── MIGRATION: add missing columns to existing tables ────────────
        # Must run AFTER all CREATE TABLE statements so tables exist first.
        # On a fresh DB this is a no-op. On old DBs it adds missing columns.
        def _migrate(table, cols):
            existing = {row[1] for row in cur.execute(f"PRAGMA table_info({table})")}
            for col, typedef in cols:
                if col not in existing:
                    cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typedef}")

        _migrate("fe_sessions", [
            ("rows_before",     "INTEGER"),
            ("rows_after",      "INTEGER"),
            ("output_features", "INTEGER"),
            ("notes",           "TEXT"),
        ])
        _migrate("selected_features", [
            ("data_type",        "TEXT"),
            ("rf_importance",    "REAL"),
            ("pearson_r",        "REAL"),
            ("selection_method", "TEXT"),
        ])
        _migrate("dropped_features", [
            ("detail",        "TEXT"),
            ("rf_importance", "REAL"),
            ("pearson_r",     "REAL"),
        ])
        _migrate("importance_scores", [
            ("pearson_r", "REAL"),
            ("rank",      "INTEGER"),
        ])
        _migrate("created_features", [
            ("category", "TEXT"),
        ])

        self.conn.commit()

    # ── SESSION ───────────────────────────────────────────────────────
    def create_fe_session(self, dataset_name: str, problem_type: str,
                          target_column: str, input_features: int,
                          rows: int) -> str:
        session_id = f"FE-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO fe_sessions
            (session_id, dataset_name, timestamp, problem_type,
             target_column, input_features, rows_before)
        VALUES (?,?,?,?,?,?,?)""",
        (session_id, dataset_name, datetime.now().isoformat(),
         problem_type, target_column, input_features, rows))
        self.conn.commit()
        return session_id

    def update_session_output(self, session_id: str, output_features: int,
                               rows_after: int = None, notes: str = ""):
        cur = self.conn.cursor()
        cur.execute("""
        UPDATE fe_sessions
        SET output_features=?, rows_after=?, notes=?
        WHERE session_id=?""",
        (output_features, rows_after, notes, session_id))
        self.conn.commit()

    # ── SELECTED FEATURES ─────────────────────────────────────────────
    def log_selected_features(self, session_id: str, features: List[Dict]):
        """
        features: [{"name":str, "dtype":str, "rf_score":float,
                    "pearson_r":float, "method":str}]
        """
        cur = self.conn.cursor()
        ts  = datetime.now().isoformat()
        for f in features:
            cur.execute("""
            INSERT INTO selected_features
                (session_id, feature_name, data_type, rf_importance,
                 pearson_r, selection_method, timestamp)
            VALUES (?,?,?,?,?,?,?)""",
            (session_id, f["name"], f.get("dtype",""),
             f.get("rf_score"), f.get("pearson_r"),
             f.get("method","user_approved"), ts))
        self.conn.commit()

    # ── DROPPED FEATURES ──────────────────────────────────────────────
    def log_dropped_feature(self, session_id: str, feature_name: str,
                             reason: str, detail: str = "",
                             rf_score: float = None, pearson_r: float = None):
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO dropped_features
            (session_id, feature_name, reason, detail,
             rf_importance, pearson_r, timestamp)
        VALUES (?,?,?,?,?,?,?)""",
        (session_id, feature_name, reason, detail,
         rf_score, pearson_r, datetime.now().isoformat()))
        self.conn.commit()

    # ── CREATED FEATURES ──────────────────────────────────────────────
    def log_created_feature(self, session_id: str, feature_name: str,
                             creation_method: str, source_columns: List[str],
                             description: str = "", category: str = ""):
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO created_features
            (session_id, feature_name, creation_method,
             source_columns, description, category, timestamp)
        VALUES (?,?,?,?,?,?,?)""",
        (session_id, feature_name, creation_method,
         json.dumps(source_columns), description, category,
         datetime.now().isoformat()))
        self.conn.commit()

    # ── SCALING / ENCODING OPS ────────────────────────────────────────
    def log_scaling_op(self, session_id: str, feature_name: str,
                        operation: str, parameters: Dict = None):
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO scaling_ops
            (session_id, feature_name, operation, parameters, timestamp)
        VALUES (?,?,?,?,?)""",
        (session_id, feature_name, operation,
         json.dumps(parameters or {}), datetime.now().isoformat()))
        self.conn.commit()

    # ── IMPORTANCE SCORES (RF + Pearson) ─────────────────────────────
    def log_importance_scores(self, session_id: str, method: str,
                               scores: List[Tuple],
                               pearson_map: Dict[str, float] = None):
        """
        scores: [(feature_name, score), ...]
        pearson_map: {feature_name: pearson_r}  (optional)
        """
        cur = self.conn.cursor()
        ts  = datetime.now().isoformat()
        for rank, (feat, score) in enumerate(scores, 1):
            pr = pearson_map.get(feat, None) if pearson_map else None
            cur.execute("""
            INSERT INTO importance_scores
                (session_id, feature_name, method, score, pearson_r, rank, timestamp)
            VALUES (?,?,?,?,?,?,?)""",
            (session_id, feat, method, float(score), pr, rank, ts))
        self.conn.commit()

    # ── SKEWNESS OPS ★ ────────────────────────────────────────────────
    def log_skewness_op(self, session_id: str, feature_name: str,
                         original_skewness: float, transform_method: str,
                         new_feature_name: str):
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO skewness_ops
            (session_id, feature_name, original_skewness,
             transform_method, new_feature_name, timestamp)
        VALUES (?,?,?,?,?,?)""",
        (session_id, feature_name, original_skewness,
         transform_method, new_feature_name, datetime.now().isoformat()))
        self.conn.commit()

    # ── IMBALANCE OPS ★ ───────────────────────────────────────────────
    def log_imbalance_op(self, session_id: str, method: str,
                          target_column: str,
                          class_counts_before: Dict,
                          class_counts_after: Dict,
                          rows_before: int, rows_after: int):
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO imbalance_ops
            (session_id, method, target_column,
             class_counts_before, class_counts_after,
             rows_before, rows_after, timestamp)
        VALUES (?,?,?,?,?,?,?,?)""",
        (session_id, method, target_column,
         json.dumps({str(k): int(v) for k, v in class_counts_before.items()}),
         json.dumps({str(k): int(v) for k, v in class_counts_after.items()}),
         rows_before, rows_after, datetime.now().isoformat()))
        self.conn.commit()

    # ── INSIGHTS ──────────────────────────────────────────────────────
    def log_insight(self, session_id: str, insight_type: str,
                     description: str, columns: List[str],
                     statistics: Dict = None, confidence: float = None):
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO fe_insights
            (session_id, insight_type, description, columns_involved,
             statistics, confidence, timestamp)
        VALUES (?,?,?,?,?,?,?)""",
        (session_id, insight_type, description,
         json.dumps(columns), json.dumps(statistics or {}),
         confidence, datetime.now().isoformat()))
        self.conn.commit()

    # ── RETRIEVAL ─────────────────────────────────────────────────────
    def get_latest_session(self) -> Optional[Dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM fe_sessions ORDER BY timestamp DESC LIMIT 1")
        row = cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

    def get_session_summary(self, session_id: str) -> Dict:
        cur = self.conn.cursor()

        def fetch(table):
            cur.execute(f"SELECT * FROM {table} WHERE session_id=?", (session_id,))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

        return {
            "session":           fetch("fe_sessions"),
            "selected_features": fetch("selected_features"),
            "dropped_features":  fetch("dropped_features"),
            "created_features":  fetch("created_features"),
            "scaling_ops":       fetch("scaling_ops"),
            "importance_scores": fetch("importance_scores"),
            "skewness_ops":      fetch("skewness_ops"),
            "imbalance_ops":     fetch("imbalance_ops"),
            "insights":          fetch("fe_insights"),
        }

    def get_selected_features_list(self, session_id: str) -> List[str]:
        cur = self.conn.cursor()
        cur.execute("SELECT feature_name FROM selected_features WHERE session_id=?",
                    (session_id,))
        return [r[0] for r in cur.fetchall()]

    # ── ENGINEERED DATASET STORAGE ★ ─────────────────────────────────
    def save_engineered_df(self, session_id: str, df, dataset_name: str,
                            target_column: str, problem_type: str):
        """
        Save the model-ready engineered DataFrame as CSV in SQLite.
        This allows retrieval for download or AI chat queries.
        """
        import io
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        csv_str = buf.getvalue()

        cur = self.conn.cursor()
        # delete any previous save for this session
        cur.execute("DELETE FROM engineered_datasets WHERE session_id=?",
                    (session_id,))
        cur.execute("""
        INSERT INTO engineered_datasets
            (session_id, dataset_name, target_column, problem_type,
             rows, columns, feature_names, csv_data, timestamp)
        VALUES (?,?,?,?,?,?,?,?,?)""",
        (session_id, dataset_name, target_column, problem_type,
         len(df), len(df.columns),
         json.dumps(df.columns.tolist()),
         csv_str, datetime.now().isoformat()))
        self.conn.commit()

    def get_engineered_df(self, session_id: str):
        """Retrieve the stored engineered DataFrame for a session."""
        import io
        import pandas as _pd
        cur = self.conn.cursor()
        cur.execute(
            "SELECT csv_data FROM engineered_datasets WHERE session_id=?",
            (session_id,))
        row = cur.fetchone()
        if row:
            return _pd.read_csv(io.StringIO(row[0]))
        return None

    def get_engineered_csv_bytes(self, session_id: str) -> bytes:
        """Return raw CSV bytes for download button."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT csv_data FROM engineered_datasets WHERE session_id=?",
            (session_id,))
        row = cur.fetchone()
        return row[0].encode("utf-8") if row else b""

    def close(self):
        if self.conn:
            self.conn.close()
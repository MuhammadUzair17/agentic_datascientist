# tools/model_memory.py
"""
Model Training Memory Manager
==============================
Stores all modelling, evaluation and explainability results in SQLite.
Enables AI chat to answer questions like:
  "Which model performed best?"
  "What was the F1 score of XGBoost?"
  "Which feature was most important?"
  "Was the model overfitting?"

Tables
------
training_sessions   : one row per training run
trained_models      : one row per model trained
model_metrics       : detailed metrics per model
feature_importance  : SHAP / model importance values
model_insights      : plain-text summaries for AI chat
"""

import sqlite3
import json
import os
import io
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd


class ModelMemoryManager:

    def __init__(self,
                 db_path: str = "memory/model_training/model_metadata.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn    = sqlite3.connect(db_path, check_same_thread=False)
        self._init_database()

    # ──────────────────────────────────────────────────────────────────
    def _init_database(self):
        cur = self.conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS training_sessions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT    UNIQUE NOT NULL,
            timestamp       TEXT    NOT NULL,
            dataset_name    TEXT,
            target_column   TEXT,
            problem_type    TEXT,
            n_features      INTEGER,
            n_rows          INTEGER,
            test_size       REAL,
            cv_folds        INTEGER,
            tuning_mode     TEXT,
            best_model_name TEXT,
            notes           TEXT
        )""")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS trained_models (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT    NOT NULL,
            model_name      TEXT    NOT NULL,
            model_params    TEXT,
            train_score     REAL,
            test_score      REAL,
            cv_score        REAL,
            cv_std          REAL,
            fit_status      TEXT,
            retrained       INTEGER DEFAULT 0,
            retrain_params  TEXT,
            retrain_score   REAL,
            timestamp       TEXT    NOT NULL,
            FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
        )""")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT    NOT NULL,
            model_name  TEXT    NOT NULL,
            metric_name TEXT    NOT NULL,
            metric_value REAL,
            timestamp   TEXT    NOT NULL,
            FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
        )""")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS feature_importance (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT    NOT NULL,
            model_name      TEXT    NOT NULL,
            feature_name    TEXT    NOT NULL,
            importance_type TEXT,
            importance_value REAL,
            rank            INTEGER,
            timestamp       TEXT    NOT NULL,
            FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
        )""")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS model_insights (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT    NOT NULL,
            insight_type    TEXT    NOT NULL,
            description     TEXT    NOT NULL,
            model_name      TEXT,
            statistics      TEXT,
            timestamp       TEXT    NOT NULL,
            FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
        )""")

        # ── MIGRATION: add any missing columns ──────────────────────
        def _migrate(table, cols):
            existing = {r[1] for r in cur.execute(
                f"PRAGMA table_info({table})")}
            for col, typedef in cols:
                if col not in existing:
                    cur.execute(
                        f"ALTER TABLE {table} ADD COLUMN {col} {typedef}")

        _migrate("training_sessions", [
            ("cv_folds",        "INTEGER"),
            ("tuning_mode",     "TEXT"),
            ("best_model_name", "TEXT"),
        ])
        _migrate("trained_models", [
            ("cv_score",       "REAL"),
            ("cv_std",         "REAL"),
            ("fit_status",     "TEXT"),
            ("retrained",      "INTEGER"),
            ("retrain_params", "TEXT"),
            ("retrain_score",  "REAL"),
        ])

        self.conn.commit()

    # ── SESSION ───────────────────────────────────────────────────────
    def create_session(self, dataset_name: str, target_column: str,
                       problem_type: str, n_features: int, n_rows: int,
                       test_size: float, cv_folds: int,
                       tuning_mode: str) -> str:
        session_id = f"MT-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO training_sessions
            (session_id, timestamp, dataset_name, target_column,
             problem_type, n_features, n_rows, test_size,
             cv_folds, tuning_mode)
        VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (session_id, datetime.now().isoformat(), dataset_name,
         target_column, problem_type, n_features, n_rows,
         test_size, cv_folds, tuning_mode))
        self.conn.commit()
        return session_id

    def update_best_model(self, session_id: str, best_model_name: str,
                          notes: str = ""):
        cur = self.conn.cursor()
        cur.execute("""
        UPDATE training_sessions
        SET best_model_name=?, notes=?
        WHERE session_id=?""",
        (best_model_name, notes, session_id))
        self.conn.commit()

    # ── MODEL ─────────────────────────────────────────────────────────
    def log_model(self, session_id: str, model_name: str,
                  model_params: Dict, train_score: float,
                  test_score: float, cv_score: float = None,
                  cv_std: float = None, fit_status: str = "good"):
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO trained_models
            (session_id, model_name, model_params, train_score,
             test_score, cv_score, cv_std, fit_status, timestamp)
        VALUES (?,?,?,?,?,?,?,?,?)""",
        (session_id, model_name, json.dumps(model_params),
         train_score, test_score, cv_score, cv_std,
         fit_status, datetime.now().isoformat()))
        self.conn.commit()

    def log_retrain(self, session_id: str, model_name: str,
                    retrain_params: Dict, retrain_score: float):
        cur = self.conn.cursor()
        cur.execute("""
        UPDATE trained_models
        SET retrained=1, retrain_params=?, retrain_score=?
        WHERE session_id=? AND model_name=?""",
        (json.dumps(retrain_params), retrain_score,
         session_id, model_name))
        self.conn.commit()

    # ── METRICS ───────────────────────────────────────────────────────
    def log_metrics(self, session_id: str, model_name: str,
                    metrics: Dict[str, float]):
        cur = self.conn.cursor()
        ts  = datetime.now().isoformat()
        for name, value in metrics.items():
            cur.execute("""
            INSERT INTO model_metrics
                (session_id, model_name, metric_name, metric_value, timestamp)
            VALUES (?,?,?,?,?)""",
            (session_id, model_name, name,
             float(value) if value is not None else None, ts))
        self.conn.commit()

    # ── FEATURE IMPORTANCE ────────────────────────────────────────────
    def log_feature_importance(self, session_id: str, model_name: str,
                                importance_type: str,
                                importances: List[Dict]):
        """
        importances: [{"feature": str, "value": float, "rank": int}]
        """
        cur = self.conn.cursor()
        ts  = datetime.now().isoformat()
        for item in importances:
            cur.execute("""
            INSERT INTO feature_importance
                (session_id, model_name, feature_name,
                 importance_type, importance_value, rank, timestamp)
            VALUES (?,?,?,?,?,?,?)""",
            (session_id, model_name, item["feature"],
             importance_type, float(item["value"]),
             item.get("rank", 0), ts))
        self.conn.commit()

    # ── INSIGHTS ──────────────────────────────────────────────────────
    def log_insight(self, session_id: str, insight_type: str,
                    description: str, model_name: str = None,
                    statistics: Dict = None):
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO model_insights
            (session_id, insight_type, description,
             model_name, statistics, timestamp)
        VALUES (?,?,?,?,?,?)""",
        (session_id, insight_type, description, model_name,
         json.dumps(statistics or {}), datetime.now().isoformat()))
        self.conn.commit()

    # ── RETRIEVAL ─────────────────────────────────────────────────────
    def get_latest_session(self) -> Optional[Dict]:
        cur = self.conn.cursor()
        cur.execute("""
        SELECT * FROM training_sessions
        ORDER BY timestamp DESC LIMIT 1""")
        row = cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

    def get_session_models(self, session_id: str) -> List[Dict]:
        cur = self.conn.cursor()
        cur.execute("""
        SELECT * FROM trained_models
        WHERE session_id=?
        ORDER BY test_score DESC""", (session_id,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def get_session_metrics(self, session_id: str,
                             model_name: str = None) -> List[Dict]:
        cur = self.conn.cursor()
        if model_name:
            cur.execute("""
            SELECT * FROM model_metrics
            WHERE session_id=? AND model_name=?""",
            (session_id, model_name))
        else:
            cur.execute("""
            SELECT * FROM model_metrics
            WHERE session_id=?""", (session_id,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def get_top_features(self, session_id: str, model_name: str,
                          top_n: int = 10) -> List[Dict]:
        cur = self.conn.cursor()
        cur.execute("""
        SELECT * FROM feature_importance
        WHERE session_id=? AND model_name=?
        ORDER BY importance_value DESC
        LIMIT ?""", (session_id, model_name, top_n))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def get_session_summary(self, session_id: str) -> Dict:
        cur = self.conn.cursor()
        def fetch(table, extra=""):
            cur.execute(
                f"SELECT * FROM {table} WHERE session_id=? {extra}",
                (session_id,))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
        return {
            "session":            fetch("training_sessions"),
            "models":             fetch("trained_models",
                                        "ORDER BY test_score DESC"),
            "metrics":            fetch("model_metrics"),
            "feature_importance": fetch("feature_importance",
                                        "ORDER BY importance_value DESC"),
            "insights":           fetch("model_insights"),
        }

    def close(self):
        if self.conn:
            self.conn.close()
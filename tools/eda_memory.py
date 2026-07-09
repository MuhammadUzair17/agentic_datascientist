# tools/eda_memory.py
"""
EDA Memory Management System
Tracks all EDA operations for AI chat assistant integration

CHANGE FROM ORIGINAL:
  db_path default changed from "memory/eda_metadata.db"
                            to "memory/eda/eda_metadata.db"
  (matches the standard memory/<tool>/<file>.db pattern)
"""

import sqlite3
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
import os


class EDAMemoryManager:
    """
    EDA Memory Manager - Tracks all EDA operations

    Database: memory/eda/eda_metadata.db        ← UPDATED PATH

    Tables:
    - eda_sessions   : Track EDA sessions
    - distributions  : Distribution analysis results
    - correlations   : Correlation analysis results
    - outliers       : Outlier detection results
    - visualizations : All plots created
    - insights       : Analytical insights
    """

    def __init__(self, db_path: str = "memory/eda/eda_metadata.db"):  # ← ONLY CHANGE
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn = None
        self._init_database()

    def _init_database(self):
        """Create database and tables if they don't exist"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.conn.cursor()

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS eda_sessions (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id        TEXT UNIQUE NOT NULL,
            dataset_name      TEXT,
            timestamp         TEXT NOT NULL,
            row_count         INTEGER,
            column_count      INTEGER,
            numeric_columns   TEXT,
            categorical_columns TEXT,
            memory_usage_mb   REAL
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS distributions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            column_name TEXT NOT NULL,
            plot_type   TEXT,
            mean        REAL,
            median      REAL,
            std         REAL,
            min_val     REAL,
            max_val     REAL,
            skewness    REAL,
            kurtosis    REAL,
            timestamp   TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES eda_sessions (session_id)
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS correlations (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id         TEXT NOT NULL,
            method             TEXT NOT NULL,
            correlation_matrix TEXT,
            top_positive       TEXT,
            top_negative       TEXT,
            timestamp          TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES eda_sessions (session_id)
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS outliers (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id         TEXT NOT NULL,
            column_name        TEXT NOT NULL,
            detection_method   TEXT NOT NULL,
            outlier_count      INTEGER,
            outlier_percentage REAL,
            lower_bound        REAL,
            upper_bound        REAL,
            outlier_indices    TEXT,
            timestamp          TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES eda_sessions (session_id)
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS visualizations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            viz_type    TEXT NOT NULL,
            x_column    TEXT,
            y_column    TEXT,
            z_column    TEXT,
            hue_column  TEXT,
            plot_method TEXT,
            parameters  TEXT,
            timestamp   TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES eda_sessions (session_id)
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS insights (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id       TEXT NOT NULL,
            insight_type     TEXT NOT NULL,
            description      TEXT NOT NULL,
            columns_involved TEXT,
            statistics       TEXT,
            confidence       REAL,
            timestamp        TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES eda_sessions (session_id)
        )
        ''')

        self.conn.commit()

    def create_eda_session(self, df: pd.DataFrame,
                           dataset_name: str = "current_dataset") -> str:
        """Create new EDA session and return session_id"""
        session_id = f"EDA-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        numeric_cols     = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(
            include=['object', 'category']).columns.tolist()

        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO eda_sessions (
            session_id, dataset_name, timestamp, row_count, column_count,
            numeric_columns, categorical_columns, memory_usage_mb
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            dataset_name,
            datetime.now().isoformat(),
            len(df),
            len(df.columns),
            json.dumps(numeric_cols),
            json.dumps(categorical_cols),
            df.memory_usage(deep=True).sum() / 1024 ** 2
        ))
        self.conn.commit()
        return session_id

    def log_distribution_analysis(self, session_id, column, plot_type, stats):
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO distributions (
            session_id, column_name, plot_type, mean, median, std,
            min_val, max_val, skewness, kurtosis, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id, column, plot_type,
            stats.get('mean'), stats.get('median'), stats.get('std'),
            stats.get('min'),  stats.get('max'),
            stats.get('skewness'), stats.get('kurtosis'),
            datetime.now().isoformat()
        ))
        self.conn.commit()

    def log_correlation_analysis(self, session_id, method, corr_matrix,
                                  top_positive, top_negative):
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO correlations (
            session_id, method, correlation_matrix,
            top_positive, top_negative, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            session_id, method, corr_matrix.to_json(),
            json.dumps(top_positive), json.dumps(top_negative),
            datetime.now().isoformat()
        ))
        self.conn.commit()

    def log_outlier_detection(self, session_id, column, method,
                               outlier_count, outlier_pct,
                               lower_bound, upper_bound, outlier_indices):
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO outliers (
            session_id, column_name, detection_method, outlier_count,
            outlier_percentage, lower_bound, upper_bound,
            outlier_indices, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id, column, method, outlier_count, outlier_pct,
            lower_bound, upper_bound,
            json.dumps(outlier_indices[:100]) if outlier_indices else json.dumps([]),
            datetime.now().isoformat()
        ))
        self.conn.commit()

    def log_visualization(self, session_id, viz_type,
                           x_col=None, y_col=None, z_col=None,
                           hue_col=None, plot_method=None, parameters=None):
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO visualizations (
            session_id, viz_type, x_column, y_column, z_column,
            hue_column, plot_method, parameters, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id, viz_type, x_col, y_col, z_col,
            hue_col, plot_method,
            json.dumps(parameters) if parameters else None,
            datetime.now().isoformat()
        ))
        self.conn.commit()

    def log_insight(self, session_id, insight_type, description,
                    columns, statistics=None, confidence=None):
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO insights (
            session_id, insight_type, description, columns_involved,
            statistics, confidence, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id, insight_type, description,
            json.dumps(columns), json.dumps(statistics or {}),
            confidence, datetime.now().isoformat()
        ))
        self.conn.commit()

    def get_latest_session(self) -> Optional[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT * FROM eda_sessions ORDER BY timestamp DESC LIMIT 1')
        row = cursor.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cursor.description]
        return dict(zip(cols, row))

    def get_session_summary(self, session_id) -> Dict[str, Any]:
        cursor = self.conn.cursor()

        def fetch(table):
            cursor.execute(
                f'SELECT * FROM {table} WHERE session_id = ?', (session_id,))
            cols = [d[0] for d in cursor.description]
            return [dict(zip(cols, r)) for r in cursor.fetchall()]

        return {
            'session_info':    fetch('eda_sessions'),
            'distributions':   fetch('distributions'),
            'correlations':    fetch('correlations'),
            'outliers':        fetch('outliers'),
            'visualizations':  fetch('visualizations'),
            'insights':        fetch('insights'),
        }

    def close(self):
        if self.conn:
            self.conn.close()
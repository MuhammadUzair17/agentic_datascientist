# tools/data_cleaning.py
"""
DATA CLEANING TOOL - Agentic Data Scientist
=============================================
PURPOSE: Fix data quality issues ONLY.
         Encoding and scaling are handled in Feature Engineering.

Operations included:
  1. Remove duplicates
  2. Handle missing values (mean/median/mode/ffill/bfill/interpolate/knn/constant/drop)
  3. Handle outliers (IQR / Z-score / modified Z-score / percentile)
  4. Clean text (trim / lowercase / remove URLs, emails, HTML, etc.)
  5. Convert data types (numeric / datetime / categorical / string)

Operations REMOVED from cleaning (moved to Feature Engineering):
  ✗ encode_categorical
  ✗ scale_features
"""

import pandas as pd
import numpy as np
import os
import json
import sqlite3
from datetime import datetime
from typing import Union, Dict, Any, List, Optional, Tuple
import io
import pickle
import hashlib
import re
from scipy import stats
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════
# MEMORY SYSTEM  (unchanged — SQLite + FAISS)
# ═══════════════════════════════════════════════════════════════════════

class DataCleaningMemory:
    def __init__(self, memory_dir: str = "./memory/cleaning"):
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)
        self.sqlite_path        = os.path.join(memory_dir, "cleaning_metadata.db")
        self.faiss_index_path   = os.path.join(memory_dir, "cleaning_vectors.index")
        self.faiss_metadata_path= os.path.join(memory_dir, "cleaning_vectors_metadata.pkl")
        self._initialize_sqlite()
        self._initialize_faiss()
        if OPENAI_AVAILABLE:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.openai_client  = OpenAI(api_key=api_key)
                self.embedding_model = "text-embedding-3-small"
            else:
                self.openai_client   = None
                self.embedding_model = None
        else:
            self.openai_client   = None
            self.embedding_model = None

    def _initialize_sqlite(self):
        conn   = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cleaning_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, session_id TEXT,
                dataset_name TEXT, validation_id INTEGER,
                rows_before INTEGER, rows_after INTEGER,
                columns_before INTEGER, columns_after INTEGER,
                total_operations INTEGER, cleaning_summary TEXT,
                memory_usage_mb REAL
            )""")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cleaning_operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cleaning_id INTEGER, operation_type TEXT,
                operation_details TEXT, affected_columns TEXT,
                items_affected INTEGER, success BOOLEAN,
                FOREIGN KEY (cleaning_id) REFERENCES cleaning_history(id)
            )""")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cleaning_id INTEGER, snapshot_type TEXT,
                data_hash TEXT, data_csv TEXT, created_at TEXT,
                FOREIGN KEY (cleaning_id) REFERENCES cleaning_history(id)
            )""")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, query_text TEXT NOT NULL,
                query_type TEXT, results_count INTEGER,
                execution_time_ms REAL
            )""")
        conn.commit()
        conn.close()

    def _initialize_faiss(self):
        if not FAISS_AVAILABLE or not OPENAI_AVAILABLE:
            self.faiss_index    = None
            self.faiss_metadata = []
            return
        self.dimension = 1536
        if os.path.exists(self.faiss_index_path) and os.path.exists(self.faiss_metadata_path):
            self.faiss_index = faiss.read_index(self.faiss_index_path)
            with open(self.faiss_metadata_path, 'rb') as f:
                self.faiss_metadata = pickle.load(f)
        else:
            self.faiss_index    = faiss.IndexFlatL2(self.dimension)
            self.faiss_metadata = []

    def _save_faiss(self):
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, self.faiss_index_path)
            with open(self.faiss_metadata_path, 'wb') as f:
                pickle.dump(self.faiss_metadata, f)

    def _generate_embedding(self, text: str):
        if self.openai_client is None:
            return None
        try:
            response  = self.openai_client.embeddings.create(
                model=self.embedding_model, input=text, encoding_format="float")
            return np.array(response.data[0].embedding, dtype='float32')
        except Exception:
            return None

    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        return hashlib.md5(df.to_csv(index=False).encode()).hexdigest()

    def store_cleaning(self, cleaning_report: Dict, df_before: pd.DataFrame,
                       df_after: pd.DataFrame) -> int:
        conn   = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO cleaning_history
                (timestamp, session_id, dataset_name, validation_id,
                 rows_before, rows_after, columns_before, columns_after,
                 total_operations, cleaning_summary, memory_usage_mb)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (cleaning_report.get('timestamp'),
                 cleaning_report.get('session_id'),
                 cleaning_report.get('dataset_name'),
                 cleaning_report.get('validation_id'),
                 cleaning_report.get('rows_before'),
                 cleaning_report.get('rows_after'),
                 cleaning_report.get('columns_before'),
                 cleaning_report.get('columns_after'),
                 len(cleaning_report.get('operations', [])),
                 cleaning_report.get('summary', ''),
                 df_after.memory_usage(deep=True).sum() / 1024**2))
            cleaning_id = cursor.lastrowid
            for op in cleaning_report.get('operations', []):
                cursor.execute("""
                    INSERT INTO cleaning_operations
                    (cleaning_id, operation_type, operation_details,
                     affected_columns, items_affected, success)
                    VALUES (?,?,?,?,?,?)""",
                    (cleaning_id, op.get('type',''), op.get('details',''),
                     ','.join(op.get('columns',[])), op.get('count',0), True))
            for stype, df in [('before', df_before), ('after', df_after)]:
                csv_buf = io.StringIO()
                df.to_csv(csv_buf, index=False)
                cursor.execute("""
                    INSERT INTO data_snapshots
                    (cleaning_id, snapshot_type, data_hash, data_csv, created_at)
                    VALUES (?,?,?,?,?)""",
                    (cleaning_id, stype,
                     self._compute_data_hash(df), csv_buf.getvalue(),
                     datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            if self.faiss_index is not None:
                text = (f"Dataset:{cleaning_report.get('dataset_name')} "
                        f"Rows:{cleaning_report.get('rows_before')}→"
                        f"{cleaning_report.get('rows_after')} "
                        f"Ops:{len(cleaning_report.get('operations',[]))}")
                emb = self._generate_embedding(text)
                if emb is not None:
                    self.faiss_index.add(np.array([emb]).astype('float32'))
                    self.faiss_metadata.append({
                        'cleaning_id':  cleaning_id,
                        'timestamp':    cleaning_report.get('timestamp'),
                        'dataset_name': cleaning_report.get('dataset_name'),
                        'operations_count': len(cleaning_report.get('operations',[])),
                        'text': text})
                    self._save_faiss()
            conn.commit()
            return cleaning_id
        except Exception as e:
            conn.rollback()
            print(f"❌ Error storing cleaning: {e}")
            return None
        finally:
            conn.close()

    def get_all_cleanings(self) -> pd.DataFrame:
        conn = sqlite3.connect(self.sqlite_path)
        df   = pd.read_sql_query(
            "SELECT * FROM cleaning_history ORDER BY timestamp DESC", conn)
        conn.close()
        return df

    def get_cleaning_by_id(self, cleaning_id: int) -> Optional[pd.DataFrame]:
        conn   = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT data_csv FROM data_snapshots WHERE cleaning_id=? AND snapshot_type='after'",
            (cleaning_id,))
        result = cursor.fetchone()
        conn.close()
        return pd.read_csv(io.StringIO(result[0])) if result else None

    def get_memory_stats(self) -> Dict:
        conn   = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM cleaning_history")
        total = cursor.fetchone()[0]
        conn.close()
        return {
            'total_cleanings': total,
            'vector_count':    self.faiss_index.ntotal if self.faiss_index else 0}

    def reset_memory(self, confirm: bool = False):
        if not confirm:
            return
        if os.path.exists(self.sqlite_path):
            os.remove(self.sqlite_path)
            self._initialize_sqlite()
        for p in [self.faiss_index_path, self.faiss_metadata_path]:
            if os.path.exists(p):
                os.remove(p)
        self._initialize_faiss()


# ═══════════════════════════════════════════════════════════════════════
# MAIN CLEANING TOOL
# ═══════════════════════════════════════════════════════════════════════

class DataCleaningTool:
    """
    Data Quality Cleaning Tool.

    Handles: duplicates, missing values, outliers, text, type conversion.
    Does NOT handle: encoding, scaling (those belong in Feature Engineering).
    """

    def __init__(self, memory_dir: str = "./memory/cleaning", session_id: str = None):
        self.memory              = DataCleaningMemory(memory_dir)
        self.session_id          = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_df          = None
        self.current_df_cleaned  = None
        self.current_cleaning_report = None

    # ── 1. DUPLICATES ────────────────────────────────────────────────────────

    def _remove_duplicates(self, df: pd.DataFrame,
                            subset: List[str] = None,
                            keep: str = 'first') -> Tuple[pd.DataFrame, Dict]:
        rows_before = len(df)
        df_clean    = df.drop_duplicates(subset=subset, keep=keep)
        removed     = rows_before - len(df_clean)
        return df_clean, {
            'type':    'remove_duplicates',
            'details': f'Removed {removed} duplicate rows (keep={keep})',
            'columns': subset if subset else list(df.columns),
            'count':   removed}

    # ── 2. MISSING VALUES ────────────────────────────────────────────────────

    def _handle_missing_values(self, df: pd.DataFrame,
                                strategy: str = 'auto',
                                threshold: float = 0.5,
                                knn_neighbors: int = 5) -> Tuple[pd.DataFrame, List[Dict]]:
        df_clean   = df.copy()
        operations = []
        missing_cols = df_clean.columns[df_clean.isnull().any()].tolist()
        if not missing_cols:
            return df_clean, []

        for col in missing_cols:
            missing_count = df_clean[col].isnull().sum()
            missing_pct   = (missing_count / len(df_clean)) * 100
            is_numeric    = df_clean[col].dtype in ['float64', 'int64']

            if strategy == 'auto':
                if missing_pct > threshold * 100:
                    df_clean = df_clean.drop(columns=[col])
                    operations.append({'type': 'drop_column',
                        'details': f'Dropped {col} ({missing_pct:.1f}% missing)',
                        'columns': [col], 'count': 1})
                elif is_numeric:
                    val = df_clean[col].median()
                    df_clean[col].fillna(val, inplace=True)
                    operations.append({'type': 'fill_median',
                        'details': f'Filled {col} with median={val:.2f}',
                        'columns': [col], 'count': missing_count})
                else:
                    val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                    df_clean[col].fillna(val, inplace=True)
                    operations.append({'type': 'fill_mode',
                        'details': f'Filled {col} with mode={val}',
                        'columns': [col], 'count': missing_count})

            elif strategy == 'drop_rows':
                df_clean = df_clean.dropna(subset=[col])
                operations.append({'type': 'drop_rows',
                    'details': f'Dropped {missing_count} rows with missing {col}',
                    'columns': [col], 'count': missing_count})

            elif strategy == 'drop_cols':
                if missing_pct > threshold * 100:
                    df_clean = df_clean.drop(columns=[col])
                    operations.append({'type': 'drop_column',
                        'details': f'Dropped {col} ({missing_pct:.1f}% missing)',
                        'columns': [col], 'count': 1})

            elif strategy == 'mean' and is_numeric:
                val = df_clean[col].mean()
                df_clean[col].fillna(val, inplace=True)
                operations.append({'type': 'fill_mean',
                    'details': f'Filled {col} with mean={val:.2f}',
                    'columns': [col], 'count': missing_count})

            elif strategy == 'median' and is_numeric:
                val = df_clean[col].median()
                df_clean[col].fillna(val, inplace=True)
                operations.append({'type': 'fill_median',
                    'details': f'Filled {col} with median={val:.2f}',
                    'columns': [col], 'count': missing_count})

            elif strategy == 'mode':
                val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col].fillna(val, inplace=True)
                operations.append({'type': 'fill_mode',
                    'details': f'Filled {col} with mode={val}',
                    'columns': [col], 'count': missing_count})

            elif strategy == 'forward_fill':
                df_clean[col].fillna(method='ffill', inplace=True)
                df_clean[col].fillna(method='bfill', inplace=True)
                operations.append({'type': 'forward_fill',
                    'details': f'Forward filled {col}',
                    'columns': [col], 'count': missing_count})

            elif strategy == 'backward_fill':
                df_clean[col].fillna(method='bfill', inplace=True)
                df_clean[col].fillna(method='ffill', inplace=True)
                operations.append({'type': 'backward_fill',
                    'details': f'Backward filled {col}',
                    'columns': [col], 'count': missing_count})

            elif strategy == 'interpolate' and is_numeric:
                df_clean[col].interpolate(method='linear', inplace=True)
                operations.append({'type': 'interpolate',
                    'details': f'Interpolated {col}',
                    'columns': [col], 'count': missing_count})

            elif strategy == 'knn' and is_numeric:
                imputer = KNNImputer(n_neighbors=knn_neighbors)
                df_clean[col] = imputer.fit_transform(df_clean[[col]])
                operations.append({'type': 'knn_impute',
                    'details': f'KNN imputed {col} (k={knn_neighbors})',
                    'columns': [col], 'count': missing_count})

            elif strategy == 'constant':
                fill_val = 0 if is_numeric else 'Unknown'
                df_clean[col].fillna(fill_val, inplace=True)
                operations.append({'type': 'fill_constant',
                    'details': f'Filled {col} with constant={fill_val}',
                    'columns': [col], 'count': missing_count})

        return df_clean, operations

    # ── 3. OUTLIERS ──────────────────────────────────────────────────────────

    def _handle_outliers(self, df: pd.DataFrame,
                          method: str = 'iqr', action: str = 'remove',
                          iqr_multiplier: float = 1.5,
                          zscore_threshold: float = 3.0,
                          percentile_lower: float = 1.0,
                          percentile_upper: float = 99.0) -> Tuple[pd.DataFrame, List[Dict]]:
        df_clean     = df.copy()
        operations   = []
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if df_clean[col].notna().sum() < 4:
                continue

            if method in ('iqr', 'iqr_custom'):
                Q1, Q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
                IQR    = Q3 - Q1
                if IQR == 0: continue
                m  = iqr_multiplier if method == 'iqr_custom' else 1.5
                lb = Q1 - m * IQR
                ub = Q3 + m * IQR
                mask = (df_clean[col] < lb) | (df_clean[col] > ub)
                method_name = f'IQR (×{m})'

            elif method == 'zscore':
                mean, std = df_clean[col].mean(), df_clean[col].std()
                if std == 0: continue
                z    = np.abs((df_clean[col] - mean) / std)
                mask = z > zscore_threshold
                lb   = mean - zscore_threshold * std
                ub   = mean + zscore_threshold * std
                method_name = f'Z-score (>{zscore_threshold})'

            elif method == 'modified_zscore':
                median = df_clean[col].median()
                mad    = np.median(np.abs(df_clean[col] - median))
                if mad == 0: continue
                mz   = 0.6745 * (df_clean[col] - median) / mad
                mask = np.abs(mz) > 3.5
                lb, ub = df_clean[col].min(), df_clean[col].max()
                method_name = 'Modified Z-score'

            elif method == 'percentile':
                lb   = df_clean[col].quantile(percentile_lower / 100)
                ub   = df_clean[col].quantile(percentile_upper / 100)
                mask = (df_clean[col] < lb) | (df_clean[col] > ub)
                method_name = f'Percentile ({percentile_lower}-{percentile_upper})'
            else:
                continue

            outlier_count = mask.sum()
            if outlier_count == 0:
                continue

            if action == 'remove':
                df_clean = df_clean[~mask]
                operations.append({'type': 'remove_outliers',
                    'details': f'Removed {outlier_count} outliers in {col} ({method_name})',
                    'columns': [col], 'count': outlier_count})

            elif action == 'cap':
                df_clean.loc[df_clean[col] < lb, col] = lb
                df_clean.loc[df_clean[col] > ub, col] = ub
                operations.append({'type': 'cap_outliers',
                    'details': f'Capped {outlier_count} outliers in {col} to [{lb:.2f},{ub:.2f}]',
                    'columns': [col], 'count': outlier_count})

            elif action == 'flag':
                flag_col = f'{col}_outlier_flag'
                df_clean[flag_col] = mask
                operations.append({'type': 'flag_outliers',
                    'details': f'Flagged {outlier_count} outliers in {col}',
                    'columns': [col, flag_col], 'count': outlier_count})

        return df_clean, operations

    # ── 4. TEXT CLEANING ─────────────────────────────────────────────────────

    def _clean_text(self, df: pd.DataFrame,
                    operations: List[str] = None) -> Tuple[pd.DataFrame, List[Dict]]:
        if operations is None:
            operations = ['lowercase', 'trim', 'remove_extra_spaces']
        df_clean     = df.copy()
        cleaning_ops = []
        text_cols    = df_clean.select_dtypes(include=['object']).columns

        for col in text_cols:
            if df_clean[col].astype(str).str.match(r'^\d+$').sum() > len(df_clean) * 0.8:
                continue
            modified = False
            applied  = []

            if 'lowercase'         in operations:
                df_clean[col] = df_clean[col].astype(str).str.lower()
                applied.append('lowercase'); modified = True
            if 'uppercase'         in operations:
                df_clean[col] = df_clean[col].astype(str).str.upper()
                applied.append('uppercase'); modified = True
            if 'titlecase'         in operations:
                df_clean[col] = df_clean[col].astype(str).str.title()
                applied.append('titlecase'); modified = True
            if 'trim'              in operations:
                df_clean[col] = df_clean[col].astype(str).str.strip()
                applied.append('trim'); modified = True
            if 'remove_extra_spaces' in operations:
                df_clean[col] = df_clean[col].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
                applied.append('remove_extra_spaces'); modified = True
            if 'standardize_whitespace' in operations:
                df_clean[col] = df_clean[col].astype(str).str.replace(r'[\t\n\r]+', ' ', regex=True)
                applied.append('standardize_whitespace'); modified = True
            if 'remove_special'    in operations:
                df_clean[col] = df_clean[col].astype(str).str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
                applied.append('remove_special'); modified = True
            if 'remove_numbers'    in operations:
                df_clean[col] = df_clean[col].astype(str).str.replace(r'\d+', '', regex=True)
                applied.append('remove_numbers'); modified = True
            if 'remove_punctuation' in operations:
                df_clean[col] = df_clean[col].astype(str).str.replace(r'[.,!?;:\'"()-]', '', regex=True)
                applied.append('remove_punctuation'); modified = True
            if 'remove_urls'       in operations:
                df_clean[col] = df_clean[col].astype(str).str.replace(
                    r'http[s]?://\S+|www\.\S+', '', regex=True)
                applied.append('remove_urls'); modified = True
            if 'remove_emails'     in operations:
                df_clean[col] = df_clean[col].astype(str).str.replace(
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '', regex=True)
                applied.append('remove_emails'); modified = True
            if 'remove_phone'      in operations:
                df_clean[col] = df_clean[col].astype(str).str.replace(
                    r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{3,}', '', regex=True)
                applied.append('remove_phone'); modified = True
            if 'remove_html'       in operations:
                df_clean[col] = df_clean[col].astype(str).str.replace(r'<[^>]+>', '', regex=True)
                applied.append('remove_html'); modified = True
            if 'fix_encoding'      in operations:
                fixes = {'â€™':"'",'â€œ':'"','â€':'"','â€"':'-','Ã©':'é','Ã¨':'è'}
                for bad, good in fixes.items():
                    df_clean[col] = df_clean[col].astype(str).str.replace(bad, good)
                applied.append('fix_encoding'); modified = True

            if modified:
                cleaning_ops.append({'type': 'clean_text',
                    'details': f'Cleaned {col}: {", ".join(applied)}',
                    'columns': [col], 'count': len(df_clean)})

        return df_clean, cleaning_ops

    # ── 5. TYPE CONVERSION ───────────────────────────────────────────────────

    def _convert_data_types(self, df: pd.DataFrame,
                             conversions: Dict[str, str] = None) -> Tuple[pd.DataFrame, List[Dict]]:
        df_clean   = df.copy()
        operations = []
        if not conversions:
            return df_clean, []

        for col, target_type in conversions.items():
            if col not in df_clean.columns:
                continue
            try:
                if target_type == 'numeric':
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                elif target_type == 'datetime':
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                elif target_type == 'categorical':
                    df_clean[col] = df_clean[col].astype('category')
                elif target_type == 'string':
                    df_clean[col] = df_clean[col].astype(str)
                operations.append({'type': f'convert_{target_type}',
                    'details': f'Converted {col} to {target_type}',
                    'columns': [col], 'count': 1})
            except Exception as e:
                print(f"⚠️ Could not convert {col} to {target_type}: {e}")

        return df_clean, operations

    # ── MAIN METHOD ──────────────────────────────────────────────────────────

    def clean_data(self, df: pd.DataFrame = None,
                   operations: Dict[str, Any] = None,
                   dataset_name: str = None,
                   validation_report: Dict = None) -> pd.DataFrame:
        """
        Run data quality cleaning pipeline.

        Supported operations keys:
          remove_duplicates : True | {'keep': 'first'/'last'/'none', 'subset': [...]}
          handle_missing    : 'auto'/'mean'/'median'/'mode'/'forward_fill'/
                              'backward_fill'/'interpolate'/'knn'/'constant'/
                              'drop_rows'/'drop_cols'
          handle_outliers   : {'method': 'iqr'/'iqr_custom'/'zscore'/
                               'modified_zscore'/'percentile',
                               'action': 'remove'/'cap'/'flag', ...}
          clean_text        : ['lowercase','trim','remove_extra_spaces', ...]
          convert_types     : {'col': 'numeric'/'datetime'/'categorical'/'string'}

        NOT supported (moved to Feature Engineering):
          encode_categorical, scale_features
        """
        if df is None:
            print("❌ No DataFrame provided")
            return None

        if operations is None:
            operations = {
                'remove_duplicates': True,
                'handle_missing':    'auto',
                'handle_outliers':   {'method': 'iqr', 'action': 'cap'},
                'clean_text':        ['lowercase', 'trim', 'remove_extra_spaces']
            }

        df_original = df.copy()
        df_clean    = df.copy()
        rows_before = len(df_clean)
        cols_before = len(df_clean.columns)

        print("\n" + "="*60)
        print("🧹 DATA CLEANING  (quality fix — no encoding/scaling)")
        print("="*60)
        print(f"   Dataset : {dataset_name or 'Unknown'}")
        print(f"   Shape   : {rows_before} rows × {cols_before} cols")

        all_ops = []

        # 1. Duplicates
        if operations.get('remove_duplicates'):
            cfg = operations['remove_duplicates']
            if isinstance(cfg, dict):
                df_clean, op = self._remove_duplicates(
                    df_clean, subset=cfg.get('subset'), keep=cfg.get('keep','first'))
            else:
                df_clean, op = self._remove_duplicates(df_clean)
            if op['count'] > 0:
                all_ops.append(op)
                print(f"   ✅ {op['details']}")
            else:
                print("   ℹ️ No duplicates found")

        # 2. Type conversion (before missing/outliers so types are correct)
        if operations.get('convert_types'):
            df_clean, ops = self._convert_data_types(df_clean, operations['convert_types'])
            all_ops.extend(ops)
            if ops:
                print(f"   ✅ Converted {len(ops)} columns")

        # 3. Missing values
        if operations.get('handle_missing'):
            strategy = operations['handle_missing']
            knn_k    = operations.get('missing_knn_neighbors', 5)
            thresh   = operations.get('missing_threshold', 0.5)
            df_clean, ops = self._handle_missing_values(
                df_clean, strategy=strategy,
                threshold=thresh, knn_neighbors=knn_k)
            all_ops.extend(ops)
            print(f"   ✅ Missing values handled ({len(ops)} ops)" if ops else "   ℹ️ No missing values")

        # 4. Outliers
        if operations.get('handle_outliers'):
            cfg = operations['handle_outliers']
            if isinstance(cfg, dict):
                df_clean, ops = self._handle_outliers(
                    df_clean,
                    method           = cfg.get('method','iqr'),
                    action           = cfg.get('action','cap'),
                    iqr_multiplier   = cfg.get('iqr_multiplier', 1.5),
                    zscore_threshold = cfg.get('zscore_threshold', 3.0),
                    percentile_lower = cfg.get('percentile_lower', 1.0),
                    percentile_upper = cfg.get('percentile_upper', 99.0))
                all_ops.extend(ops)
                print(f"   ✅ Outliers handled ({len(ops)} cols)" if ops else "   ℹ️ No outliers detected")

        # 5. Text cleaning
        if operations.get('clean_text'):
            df_clean, ops = self._clean_text(df_clean, operations['clean_text'])
            all_ops.extend(ops)
            print(f"   ✅ Cleaned {len(ops)} text cols" if ops else "   ℹ️ No text cols to clean")

        # Guard: warn if caller accidentally passes encode/scale
        if 'encode_categorical' in operations:
            print("   ⚠️  encode_categorical ignored — use Feature Engineering page")
        if 'scale_features' in operations:
            print("   ⚠️  scale_features ignored — use Feature Engineering page")

        rows_after = len(df_clean)
        cols_after = len(df_clean.columns)

        cleaning_report = {
            'timestamp':      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'session_id':     self.session_id,
            'dataset_name':   dataset_name or 'unknown',
            'validation_id':  validation_report.get('validation_id') if validation_report else None,
            'rows_before':    rows_before,
            'rows_after':     rows_after,
            'columns_before': cols_before,
            'columns_after':  cols_after,
            'operations':     all_ops,
            'summary':        (f"Cleaned {rows_before}→{rows_after} rows, "
                               f"{cols_before}→{cols_after} cols. "
                               f"{len(all_ops)} operations.")
        }

        cleaning_id = self.memory.store_cleaning(cleaning_report, df_original, df_clean)
        cleaning_report['cleaning_id'] = cleaning_id

        self.current_df             = df_original
        self.current_df_cleaned     = df_clean
        self.current_cleaning_report = cleaning_report

        print(f"\n✅ CLEANING COMPLETE")
        print(f"   Rows   : {rows_before:,} → {rows_after:,}")
        print(f"   Columns: {cols_before} → {cols_after}")
        print(f"   Ready for EDA (no encoding done — categorical columns intact)")
        print("="*60)

        return df_clean

    # ── HELPERS ──────────────────────────────────────────────────────────────

    def get_cleaned_csv_buffer(self) -> Optional[io.StringIO]:
        if self.current_df_cleaned is None:
            return None
        buf = io.StringIO()
        self.current_df_cleaned.to_csv(buf, index=False)
        buf.seek(0)
        return buf

    def get_output_for_eda(self) -> Optional[Dict]:
        if self.current_df_cleaned is None:
            return None
        return {
            'dataframe':        self.current_df_cleaned,
            'cleaning_report':  self.current_cleaning_report,
            'metadata': {
                'cleaning_id':  self.current_cleaning_report.get('cleaning_id'),
                'dataset_name': self.current_cleaning_report.get('dataset_name'),
                'rows_after':   self.current_cleaning_report.get('rows_after'),
                'columns_after':self.current_cleaning_report.get('columns_after'),
            }
        }

    def get_memory_stats(self) -> Dict:
        return self.memory.get_memory_stats()

    def reset_memory(self, confirm: bool = False):
        self.memory.reset_memory(confirm=confirm)


def create_cleaning_tool(memory_dir: str = "./memory/cleaning") -> DataCleaningTool:
    return DataCleaningTool(memory_dir=memory_dir)
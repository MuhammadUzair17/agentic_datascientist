# tools/data_validation1.py - WITH STRICT FORMAT VALIDATION
"""
DATA VALIDATION TOOL - Healthcare AutoML Data Science Agent
=============================================================
STRICT FORMAT VALIDATION:
- CSV can only be validated as CSV
- JSON can only be validated as JSON  
- API can only be validated as API
- Mismatched formats are REJECTED
"""

import pandas as pd
import numpy as np
import os
import json
import sqlite3
from datetime import datetime
from typing import Union, Dict, Any, List, Optional
import io
import pickle
import hashlib
import re
from scipy import stats

# Environment variables
from dotenv import load_dotenv
load_dotenv()

# Vector DB imports
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS not installed. Install with: pip install faiss-cpu")

# OpenAI imports for embeddings
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI not installed. Install with: pip install openai")


class DataValidationMemory:
    """
    Dual Memory System for Data Validation Tool
    - SQLite3: Structured validation metadata storage
    - FAISS: Vector embeddings for semantic search
    """
    
    def __init__(self, memory_dir: str = "./memory/validation"):
        """Initialize dual memory system"""
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)
        
        # SQLite database path
        self.sqlite_path = os.path.join(memory_dir, "validation_metadata.db")
        
        # FAISS index path
        self.faiss_index_path = os.path.join(memory_dir, "validation_vectors.index")
        self.faiss_metadata_path = os.path.join(memory_dir, "validation_vectors_metadata.pkl")
        
        # Initialize SQLite
        self._initialize_sqlite()
        
        # Initialize FAISS Vector DB
        self._initialize_faiss()
        
        # Initialize OpenAI client for embeddings
        if OPENAI_AVAILABLE:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                print("⚠️ OPENAI_API_KEY not found in environment variables")
                print("💡 Please add OPENAI_API_KEY to your .env file")
                self.openai_client = None
                self.embedding_model = None
            else:
                self.openai_client = OpenAI(api_key=api_key)
                self.embedding_model = "text-embedding-3-small"
                print(f"✅ OpenAI embeddings initialized (Model: {self.embedding_model})")
        else:
            self.openai_client = None
            self.embedding_model = None
    
    def _initialize_sqlite(self):
        """Create SQLite database schema"""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        # Main validation history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS validation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                session_id TEXT,
                dataset_name TEXT,
                ingestion_id INTEGER,
                total_rows INTEGER,
                total_columns INTEGER,
                validation_status TEXT,
                total_issues INTEGER,
                memory_usage_mb REAL,
                ingested_format TEXT,
                expected_format TEXT,
                format_match BOOLEAN
            )
        """)
        
        # Validation issues table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS validation_issues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                validation_id INTEGER,
                issue_category TEXT,
                issue_type TEXT,
                severity TEXT,
                affected_columns TEXT,
                issue_count INTEGER,
                issue_details TEXT,
                FOREIGN KEY (validation_id) REFERENCES validation_history(id)
            )
        """)
        
        # Data statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS validation_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                validation_id INTEGER,
                column_name TEXT,
                data_type TEXT,
                count INTEGER,
                unique_count INTEGER,
                missing_count INTEGER,
                mean_value REAL,
                std_dev REAL,
                min_value REAL,
                max_value REAL,
                quartile_25 REAL,
                quartile_50 REAL,
                quartile_75 REAL,
                FOREIGN KEY (validation_id) REFERENCES validation_history(id)
            )
        """)
        
        # Outlier detection results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS outlier_detection (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                validation_id INTEGER,
                column_name TEXT,
                method TEXT,
                outlier_count INTEGER,
                outlier_percentage REAL,
                lower_bound REAL,
                upper_bound REAL,
                outlier_details TEXT,
                FOREIGN KEY (validation_id) REFERENCES validation_history(id)
            )
        """)
        
        # Schema validation table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_validation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                validation_id INTEGER,
                column_name TEXT,
                expected_type TEXT,
                actual_type TEXT,
                is_valid BOOLEAN,
                conversion_possible BOOLEAN,
                FOREIGN KEY (validation_id) REFERENCES validation_history(id)
            )
        """)
        
        # NLP Query Log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                query_text TEXT NOT NULL,
                query_type TEXT,
                results_count INTEGER,
                execution_time_ms REAL
            )
        """)
        
        conn.commit()
        conn.close()
        print(f"✅ SQLite Memory initialized at: {self.sqlite_path}")
    
    def _initialize_faiss(self):
        """Initialize FAISS vector database"""
        if not FAISS_AVAILABLE or not OPENAI_AVAILABLE:
            print("⚠️ FAISS or OpenAI not available. Vector search disabled.")
            self.faiss_index = None
            self.faiss_metadata = []
            return
        
        # Dimension for text-embedding-3-small model (1536 dimensions)
        self.dimension = 1536
        
        # Load existing index or create new
        if os.path.exists(self.faiss_index_path) and os.path.exists(self.faiss_metadata_path):
            self.faiss_index = faiss.read_index(self.faiss_index_path)
            with open(self.faiss_metadata_path, 'rb') as f:
                self.faiss_metadata = pickle.load(f)
            print(f"✅ FAISS Vector DB loaded: {self.faiss_index.ntotal} vectors")
        else:
            # Create new index (L2 distance)
            self.faiss_index = faiss.IndexFlatL2(self.dimension)
            self.faiss_metadata = []
            print("✅ New FAISS Vector DB created")
    
    def _save_faiss(self):
        """Save FAISS index to disk"""
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, self.faiss_index_path)
            with open(self.faiss_metadata_path, 'wb') as f:
                pickle.dump(self.faiss_metadata, f)
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding vector for text using OpenAI API"""
        if self.openai_client is None or self.embedding_model is None:
            return None
        
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text,
                encoding_format="float"
            )
            embedding = np.array(response.data[0].embedding, dtype='float32')
            return embedding
        except Exception as e:
            print(f"⚠️ Error generating embedding: {str(e)}")
            return None
    
    def store_validation(self, validation_report: Dict[str, Any], df: pd.DataFrame) -> int:
        """Store validation results in dual memory system"""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        try:
            # Insert main validation record with format info
            cursor.execute("""
                INSERT INTO validation_history 
                (timestamp, session_id, dataset_name, ingestion_id, total_rows, total_columns, 
                 validation_status, total_issues, memory_usage_mb, ingested_format, expected_format, format_match)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                validation_report.get('timestamp'),
                validation_report.get('session_id'),
                validation_report.get('dataset_name'),
                validation_report.get('ingestion_id'),
                validation_report.get('total_rows'),
                validation_report.get('total_columns'),
                validation_report.get('validation_status'),
                validation_report.get('total_issues'),
                df.memory_usage(deep=True).sum() / 1024**2,
                validation_report.get('ingested_format'),
                validation_report.get('expected_format'),
                validation_report.get('format_match', False)
            ))
            
            validation_id = cursor.lastrowid
            
            # Store validation issues
            for category, issues in validation_report.items():
                if category in ['format_issues', 'structure_issues', 'missing_values', 
                               'duplicates', 'text_issues']:
                    if issues:
                        cursor.execute("""
                            INSERT INTO validation_issues 
                            (validation_id, issue_category, issue_type, severity, affected_columns, 
                             issue_count, issue_details)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            validation_id,
                            category,
                            str(type(issues).__name__),
                            self._determine_severity(category),
                            json.dumps(list(issues.keys()) if isinstance(issues, dict) else []),
                            len(issues) if isinstance(issues, (dict, list)) else 1,
                            json.dumps(issues, default=str)[:1000]
                        ))
            
            # Store statistics
            if 'statistics' in validation_report:
                for col, stats in validation_report['statistics'].items():
                    cursor.execute("""
                        INSERT INTO validation_statistics 
                        (validation_id, column_name, data_type, count, unique_count, missing_count,
                         mean_value, std_dev, min_value, max_value, quartile_25, quartile_50, quartile_75)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        validation_id,
                        col,
                        str(stats.get('dtype')),
                        stats.get('count'),
                        stats.get('unique'),
                        stats.get('missing'),
                        stats.get('mean'),
                        stats.get('std'),
                        stats.get('min'),
                        stats.get('max'),
                        stats.get('25%'),
                        stats.get('50%'),
                        stats.get('75%')
                    ))
            
            # Store outlier detection results
            if 'outliers' in validation_report:
                for method, method_results in validation_report['outliers'].items():
                    for col, outlier_info in method_results.items():
                        cursor.execute("""
                            INSERT INTO outlier_detection 
                            (validation_id, column_name, method, outlier_count, outlier_percentage,
                             lower_bound, upper_bound, outlier_details)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            validation_id,
                            col,
                            method,
                            outlier_info.get('count'),
                            outlier_info.get('percentage'),
                            outlier_info.get('lower_bound'),
                            outlier_info.get('upper_bound'),
                            json.dumps(outlier_info, default=str)[:500]
                        ))
            
            # Add to FAISS vector DB
            if self.faiss_index is not None:
                search_text = f"""
                Validation Report: {validation_report.get('dataset_name')}
                Status: {validation_report.get('validation_status')}
                Rows: {validation_report.get('total_rows')}
                Columns: {validation_report.get('total_columns')}
                Total Issues: {validation_report.get('total_issues')}
                Format Match: {validation_report.get('format_match')}
                """
                
                embedding = self._generate_embedding(search_text)
                if embedding is not None:
                    self.faiss_index.add(np.array([embedding]).astype('float32'))
                    self.faiss_metadata.append({
                        'validation_id': validation_id,
                        'timestamp': validation_report.get('timestamp'),
                        'dataset_name': validation_report.get('dataset_name'),
                        'validation_status': validation_report.get('validation_status'),
                        'total_issues': validation_report.get('total_issues'),
                        'text': search_text
                    })
                    self._save_faiss()
            
            conn.commit()
            print(f"✅ Validation stored in memory (ID: {validation_id})")
            return validation_id
            
        except Exception as e:
            conn.rollback()
            print(f"❌ Error storing validation: {str(e)}")
            return None
        finally:
            conn.close()
    
    def _determine_severity(self, category: str) -> str:
        """Determine severity level of validation issue"""
        critical = ['format_issues']
        high = ['structure_issues', 'missing_values', 'duplicates']
        medium = ['text_issues', 'outliers']
        
        if category in critical:
            return "CRITICAL"
        elif category in high:
            return "HIGH"
        else:
            return "MEDIUM"
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search validation history using semantic similarity"""
        if self.faiss_index is None or self.embedding_model is None:
            print("⚠️ Vector search not available")
            return []
        
        if self.faiss_index.ntotal == 0:
            print("ℹ️ No data in vector database yet")
            return []
        
        query_embedding = self._generate_embedding(query)
        if query_embedding is None:
            return []
        
        start_time = datetime.now()
        distances, indices = self.faiss_index.search(
            np.array([query_embedding]).astype('float32'), 
            min(top_k, self.faiss_index.ntotal)
        )
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.faiss_metadata):
                result = self.faiss_metadata[idx].copy()
                result['similarity_score'] = float(1 / (1 + distances[0][i]))
                results.append(result)
        
        self._log_query(query, 'semantic_search', len(results), execution_time)
        return results
    
    def _log_query(self, query_text: str, query_type: str, results_count: int, execution_time: float):
        """Log NLP queries for tracking"""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO query_log (timestamp, query_text, query_type, results_count, execution_time_ms)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            query_text,
            query_type,
            results_count,
            execution_time
        ))
        
        conn.commit()
        conn.close()
    
    def get_all_validations(self) -> pd.DataFrame:
        """Get all validation history"""
        conn = sqlite3.connect(self.sqlite_path)
        df = pd.read_sql_query("SELECT * FROM validation_history ORDER BY timestamp DESC", conn)
        conn.close()
        return df
    
    def get_validation_by_id(self, validation_id: int) -> Dict:
        """Retrieve detailed validation results by ID"""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM validation_history WHERE id = ?", (validation_id,))
        result = cursor.fetchone()
        
        if result:
            validation_data = {
                'validation_id': result[0],
                'timestamp': result[1],
                'dataset_name': result[3],
                'total_rows': result[5],
                'total_columns': result[6],
                'validation_status': result[7],
                'total_issues': result[8]
            }
            
            # Get issues
            issues_df = pd.read_sql_query(
                "SELECT * FROM validation_issues WHERE validation_id = ?",
                conn,
                params=(validation_id,)
            )
            validation_data['issues'] = issues_df.to_dict('records')
            
            conn.close()
            return validation_data
        
        conn.close()
        return {}
    
    def reset_memory(self, confirm: bool = False):
        """Reset all memory (SQLite + FAISS)"""
        if not confirm:
            print("⚠️ Memory reset requires confirmation. Set confirm=True")
            return
        
        if os.path.exists(self.sqlite_path):
            os.remove(self.sqlite_path)
            self._initialize_sqlite()
        
        if os.path.exists(self.faiss_index_path):
            os.remove(self.faiss_index_path)
        if os.path.exists(self.faiss_metadata_path):
            os.remove(self.faiss_metadata_path)
        
        self._initialize_faiss()
        print("✅ Memory reset complete!")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        stats = {}
        cursor.execute("SELECT COUNT(*) FROM validation_history")
        stats['total_validations'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM query_log")
        stats['total_queries'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(memory_usage_mb) FROM validation_history")
        stats['total_data_mb'] = cursor.fetchone()[0] or 0
        
        stats['vector_count'] = self.faiss_index.ntotal if self.faiss_index else 0
        stats['sqlite_size_mb'] = os.path.getsize(self.sqlite_path) / 1024**2 if os.path.exists(self.sqlite_path) else 0
        stats['faiss_size_mb'] = os.path.getsize(self.faiss_index_path) / 1024**2 if os.path.exists(self.faiss_index_path) else 0
        
        conn.close()
        return stats


class DataValidationTool:
    """
    Professional Data Validation Tool with STRICT FORMAT VALIDATION
    
    STRICT RULE: Data format from ingestion MUST match validation format
    - CSV → Only validate as CSV
    - JSON → Only validate as JSON
    - API → Only validate as API
    """
    
    def __init__(self, memory_dir: str = "./memory/validation", session_id: str = None):
        """Initialize Data Validation Tool"""
        self.memory = DataValidationMemory(memory_dir)
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_df = None
        self.current_validation_report = None
        
        print("="*70)
        print("DATA VALIDATION TOOL - Healthcare AutoML System")
        print("="*70)
        print(f"Session ID: {self.session_id}")
        print(f"Memory Location: {memory_dir}")
        print("="*70)
    
    def _validate_format_match(self, ingested_format: str, expected_format: str) -> Dict[str, Any]:
        """
        STRICT FORMAT VALIDATION
        Returns error if formats don't match
        """
        # Normalize formats
        ingested_format = ingested_format.upper().strip()
        expected_format = expected_format.upper().strip()
        
        # Check for match
        if ingested_format != expected_format:
            error_report = {
                'format_match': False,
                'error': 'FORMAT_MISMATCH',
                'ingested_format': ingested_format,
                'expected_format': expected_format,
                'message': f"""
❌ FORMAT MISMATCH DETECTED!

╔══════════════════════════════════════════════════════════════╗
║  VALIDATION REJECTED - Format Mismatch                       ║
╚══════════════════════════════════════════════════════════════╝

📥 Data Ingested As:     {ingested_format}
🔍 Validation Expected:  {expected_format}

⚠️  STRICT RULE VIOLATION:
    Data ingested as {ingested_format} can ONLY be validated as {ingested_format}.
    
🚫 VALIDATION STOPPED - Pipeline Halted

💡 SOLUTION:
    → Go to Validation page
    → Select "{ingested_format}" as expected format
    → Run validation again
    
📋 FORMAT RULES:
    • CSV data  → Must validate as CSV
    • JSON data → Must validate as JSON
    • API data  → Must validate as API
    
⛔ Cross-format validation is NOT allowed for data integrity.
                """,
                'severity': 'CRITICAL',
                'validation_status': 'FORMAT_REJECTED',
                'total_issues': 1,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            return error_report
        
        # Format match successful
        return {
            'format_match': True,
            'ingested_format': ingested_format,
            'expected_format': expected_format,
            'message': f'✅ Format validation passed: {ingested_format}'
        }
    
    def _validate_input_format(self, input_data: Any) -> Dict:
        """Validate that input is from Data Ingestion Tool"""
        format_issues = {}
        
        if not isinstance(input_data, dict):
            format_issues['invalid_input'] = f"Expected dict from Data Ingestion Tool, got {type(input_data)}"
            return format_issues
        
        required_keys = ['dataframe', 'metadata', 'summary']
        missing_keys = [key for key in required_keys if key not in input_data]
        if missing_keys:
            format_issues['missing_keys'] = missing_keys
        
        if 'dataframe' in input_data:
            df = input_data['dataframe']
            if not isinstance(df, pd.DataFrame):
                format_issues['invalid_dataframe'] = f"Expected DataFrame, got {type(df)}"
            elif df.empty:
                format_issues['empty_dataframe'] = "DataFrame is empty"
        
        if 'metadata' in input_data:
            metadata = input_data['metadata']
            if not isinstance(metadata, dict):
                format_issues['invalid_metadata'] = f"Expected dict, got {type(metadata)}"
            else:
                expected_meta = ['data_format', 'file_name']
                missing_meta = [key for key in expected_meta if key not in metadata]
                if missing_meta:
                    format_issues['missing_metadata'] = missing_meta
                
                if 'data_format' in metadata:
                    valid_formats = ['CSV', 'JSON', 'API']
                    if metadata['data_format'] not in valid_formats:
                        format_issues['invalid_format'] = f"Format {metadata['data_format']} not in {valid_formats}"
        
        return format_issues
    
    def _validate_structure(self, df: pd.DataFrame) -> Dict:
        """Validate DataFrame structure"""
        structure_issues = {}
        
        empty_cols = [col for col in df.columns if df[col].isna().all()]
        if empty_cols:
            structure_issues['empty_columns'] = empty_cols
        
        unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
        if unnamed_cols:
            structure_issues['unnamed_columns'] = unnamed_cols
        
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            structure_issues['duplicate_column_names'] = duplicate_cols
        
        if len(df) == 1:
            structure_issues['single_row'] = "Dataset contains only 1 row"
        
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        if constant_cols:
            structure_issues['constant_columns'] = constant_cols
        
        if len(df.columns) > len(df):
            structure_issues['more_columns_than_rows'] = f"{len(df.columns)} columns but only {len(df)} rows"
        
        return structure_issues
    
    def _check_missing_values(self, df: pd.DataFrame) -> Dict:
        """Comprehensive missing value analysis"""
        missing_info = {}
        
        total_missing = df.isnull().sum().sum()
        if total_missing > 0:
            missing_per_column = df.isnull().sum()
            missing_per_column = missing_per_column[missing_per_column > 0]
            
            missing_info['total_missing'] = int(total_missing)
            missing_info['missing_percentage'] = round((total_missing / (df.shape[0] * df.shape[1])) * 100, 2)
            missing_info['columns_with_missing'] = {k: int(v) for k, v in missing_per_column.to_dict().items()}
            
            high_missing = missing_per_column[missing_per_column > len(df) * 0.5]
            if not high_missing.empty:
                missing_info['high_missing_columns'] = {
                    k: {
                        'count': int(v),
                        'percentage': round((v / len(df)) * 100, 2)
                    } for k, v in high_missing.to_dict().items()
                }
            
            missing_rows = df.isnull().any(axis=1).sum()
            missing_info['rows_with_missing'] = int(missing_rows)
            missing_info['complete_rows'] = len(df) - missing_rows
        
        return missing_info
    
    def _check_duplicates(self, df: pd.DataFrame) -> Dict:
        """Check for duplicate rows"""
        duplicate_info = {}
        
        total_duplicates = df.duplicated().sum()
        if total_duplicates > 0:
            duplicate_info['total_duplicates'] = int(total_duplicates)
            duplicate_info['duplicate_percentage'] = round((total_duplicates / len(df)) * 100, 2)
            duplicate_info['unique_rows'] = len(df) - total_duplicates
            
            duplicate_indices = df[df.duplicated(keep=False)].index.tolist()
            duplicate_info['duplicate_row_count'] = len(duplicate_indices)
            
            for col in df.columns:
                col_duplicates = df[col].duplicated().sum()
                if col_duplicates > 0 and col_duplicates < len(df):
                    if 'column_duplicates' not in duplicate_info:
                        duplicate_info['column_duplicates'] = {}
                    duplicate_info['column_duplicates'][col] = {
                        'count': int(col_duplicates),
                        'percentage': round((col_duplicates / len(df)) * 100, 2)
                    }
        
        return duplicate_info
    
    def _detect_outliers_iqr(self, df: pd.DataFrame) -> Dict:
        """Detect outliers using IQR method"""
        outliers = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].notna().sum() < 4:
                continue
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:
                continue
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                outliers[col] = {
                    'count': int(outlier_count),
                    'percentage': round((outlier_count / len(df)) * 100, 2),
                    'lower_bound': round(lower_bound, 4),
                    'upper_bound': round(upper_bound, 4),
                    'min_outlier': round(df[outlier_mask][col].min(), 4),
                    'max_outlier': round(df[outlier_mask][col].max(), 4)
                }
        
        return outliers
    
    def _detect_outliers_zscore(self, df: pd.DataFrame, threshold: float = 3.0) -> Dict:
        """Detect outliers using Z-score method"""
        outliers = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].notna().sum() < 4:
                continue
            
            mean = df[col].mean()
            std = df[col].std()
            
            if std == 0:
                continue
            
            z_scores = np.abs((df[col] - mean) / std)
            outlier_mask = z_scores > threshold
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                outliers[col] = {
                    'count': int(outlier_count),
                    'percentage': round((outlier_count / len(df)) * 100, 2),
                    'threshold': threshold,
                    'mean': round(mean, 4),
                    'std_dev': round(std, 4),
                    'max_zscore': round(z_scores.max(), 4),
                    'min_outlier': round(df[outlier_mask][col].min(), 4),
                    'max_outlier': round(df[outlier_mask][col].max(), 4)
                }
        
        return outliers
    
    def _detect_outliers_stddev(self, df: pd.DataFrame, num_std: float = 3.0) -> Dict:
        """Detect outliers using Standard Deviation method"""
        outliers = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].notna().sum() < 4:
                continue
            
            mean = df[col].mean()
            std = df[col].std()
            
            if std == 0:
                continue
            
            lower_bound = mean - num_std * std
            upper_bound = mean + num_std * std
            
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                outliers[col] = {
                    'count': int(outlier_count),
                    'percentage': round((outlier_count / len(df)) * 100, 2),
                    'num_std': num_std,
                    'mean': round(mean, 4),
                    'std_dev': round(std, 4),
                    'lower_bound': round(lower_bound, 4),
                    'upper_bound': round(upper_bound, 4),
                    'min_outlier': round(df[outlier_mask][col].min(), 4),
                    'max_outlier': round(df[outlier_mask][col].max(), 4)
                }
        
        return outliers
    
    def _check_text_issues(self, df: pd.DataFrame) -> Dict:
        """Check for text-related issues"""
        text_issues = {}
        text_cols = df.select_dtypes(include=['object']).columns
        
        for col in text_cols:
            if df[col].notna().sum() < 1:
                continue
            
            col_issues = {}
            
            numeric_count = df[col].dropna().astype(str).str.match(r'^\d+\.?\d*$').sum()
            if numeric_count > len(df[col].dropna()) * 0.8:
                continue
            
            special_char_pattern = r'[^a-zA-Z0-9\s,.\-_@]'
            special_chars = df[col].astype(str).str.contains(special_char_pattern, regex=True, na=False)
            if special_chars.any():
                examples = df[special_chars][col].head(3).tolist()
                col_issues['special_characters'] = {
                    'count': int(special_chars.sum()),
                    'percentage': round((special_chars.sum() / len(df)) * 100, 2),
                    'examples': examples
                }
            
            excessive_punct = df[col].astype(str).str.count(r'[.,!?;:]') > 5
            if excessive_punct.any():
                col_issues['excessive_punctuation'] = {
                    'count': int(excessive_punct.sum()),
                    'percentage': round((excessive_punct.sum() / len(df)) * 100, 2)
                }
            
            non_null = df[col].notna()
            whitespace = (df[non_null][col].astype(str).str.strip() != df[non_null][col].astype(str))
            if whitespace.any():
                col_issues['whitespace_issues'] = {
                    'count': int(whitespace.sum()),
                    'percentage': round((whitespace.sum() / len(df)) * 100, 2)
                }
            
            non_ascii = df[col].astype(str).str.contains(r'[^\x00-\x7F]', regex=True, na=False)
            if non_ascii.any():
                examples = df[non_ascii][col].head(3).tolist()
                col_issues['non_ascii_characters'] = {
                    'count': int(non_ascii.sum()),
                    'percentage': round((non_ascii.sum() / len(df)) * 100, 2),
                    'examples': examples
                }
            
            multi_spaces = df[col].astype(str).str.contains(r'\s{2,}', regex=True, na=False)
            if multi_spaces.any():
                col_issues['multiple_spaces'] = {
                    'count': int(multi_spaces.sum()),
                    'percentage': round((multi_spaces.sum() / len(df)) * 100, 2)
                }
            
            has_text = df[col].astype(str).str.len() > 2
            if has_text.any():
                values = df[has_text][col].astype(str)
                all_upper = values.str.isupper().sum()
                all_lower = values.str.islower().sum()
                title_case = values.str.istitle().sum()
                total = len(values)
                
                if all_upper < total * 0.8 and all_lower < total * 0.8 and title_case < total * 0.8:
                    col_issues['inconsistent_casing'] = {
                        'all_upper': int(all_upper),
                        'all_lower': int(all_lower),
                        'title_case': int(title_case),
                        'mixed': int(total - all_upper - all_lower - title_case)
                    }
            
            if col_issues:
                text_issues[col] = col_issues
        
        return text_issues
    
    def _validate_schema(self, df: pd.DataFrame, expected_schema: Dict = None) -> Dict:
        """Validate data types and schema"""
        schema_info = {
            'current_schema': df.dtypes.astype(str).to_dict(),
            'column_count': len(df.columns),
            'columns': df.columns.tolist()
        }
        
        type_issues = {}
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_numeric(df[col].dropna(), errors='raise')
                    type_issues[col] = {
                        'current_type': 'object',
                        'suggested_type': 'numeric',
                        'reason': 'Column contains only numeric values'
                    }
                except:
                    pass
                
                try:
                    pd.to_datetime(df[col].dropna(), errors='raise')
                    type_issues[col] = {
                        'current_type': 'object',
                        'suggested_type': 'datetime',
                        'reason': 'Column contains datetime values'
                    }
                except:
                    pass
        
        if type_issues:
            schema_info['type_conversion_suggestions'] = type_issues
        
        if expected_schema:
            schema_validation = {}
            for col, expected_type in expected_schema.items():
                if col in df.columns:
                    actual_type = str(df[col].dtype)
                    schema_validation[col] = {
                        'expected': expected_type,
                        'actual': actual_type,
                        'valid': expected_type == actual_type
                    }
                else:
                    schema_validation[col] = {
                        'expected': expected_type,
                        'actual': None,
                        'valid': False,
                        'error': 'Column not found'
                    }
            schema_info['schema_validation'] = schema_validation
        
        return schema_info
    
    def _generate_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive statistics"""
        statistics = {}
        
        for col in df.columns:
            col_stats = {
                'dtype': str(df[col].dtype),
                'count': int(df[col].count()),
                'missing': int(df[col].isnull().sum()),
                'unique': int(df[col].nunique())
            }
            
            if df[col].dtype in [np.int64, np.float64]:
                col_stats.update({
                    'mean': round(df[col].mean(), 4) if pd.notna(df[col].mean()) else None,
                    'std': round(df[col].std(), 4) if pd.notna(df[col].std()) else None,
                    'min': round(df[col].min(), 4) if pd.notna(df[col].min()) else None,
                    'max': round(df[col].max(), 4) if pd.notna(df[col].max()) else None,
                    '25%': round(df[col].quantile(0.25), 4) if pd.notna(df[col].quantile(0.25)) else None,
                    '50%': round(df[col].quantile(0.50), 4) if pd.notna(df[col].quantile(0.50)) else None,
                    '75%': round(df[col].quantile(0.75), 4) if pd.notna(df[col].quantile(0.75)) else None,
                })
            
            if df[col].dtype == 'object':
                value_counts = df[col].value_counts().head(5)
                col_stats['top_values'] = value_counts.to_dict()
            
            statistics[col] = col_stats
        
        return statistics
    
    def validate(self, input_data: Union[Dict, pd.DataFrame], 
                 dataset_name: str = None,
                 expected_format: str = None,
                 expected_schema: Dict = None) -> Dict[str, Any]:
        """
        Main validation method with STRICT FORMAT VALIDATION
        
        Args:
            input_data: Output from Data Ingestion Tool OR DataFrame
            dataset_name: Name of dataset for tracking
            expected_format: Expected format (csv/json/api) - MUST MATCH ingested format
            expected_schema: Expected schema (optional)
        
        Returns:
            Comprehensive validation report OR format mismatch error
        """
        
        print("\n" + "="*70)
        print("🔍 STARTING DATA VALIDATION")
        print("="*70)
        
        # Handle input
# Handle input
        if isinstance(input_data, pd.DataFrame):
            # ⭐ FIX: Read format from session state instead of hardcoding
            import streamlit as st
            ingested_format = st.session_state.get('data_format', 'CSV')  # ✅ Read from session
            
            df = input_data
            metadata = {
                'data_format': ingested_format,  # ✅ Use actual format
                'file_name': dataset_name or 'unknown'
            }
            ingestion_id = None
            
            print(f"✅ DataFrame input detected")
            print(f"   Format from session state: {ingested_format}")
        elif isinstance(input_data, dict):
            print("\n📥 Receiving data from Data Ingestion Tool...")
            
            format_issues = self._validate_input_format(input_data)
            if format_issues:
                print("\n❌ Invalid input format from Data Ingestion Tool")
                for issue, details in format_issues.items():
                    print(f"   • {issue}: {details}")
                return {'error': 'Invalid input format', 'details': format_issues}
            
            df = input_data['dataframe']
            metadata = input_data.get('metadata', {})
            ingestion_id = metadata.get('ingestion_id')
            dataset_name = dataset_name or metadata.get('file_name', 'unknown')
            ingested_format = metadata.get('data_format', 'CSV')
            
            print(f"✅ Input validated from Data Ingestion Tool")
            print(f"   Source: {metadata.get('data_format')}")
            print(f"   File: {metadata.get('file_name')}")
        else:
            return {'error': f'Invalid input type: {type(input_data)}'}
        
        # ============================================
        # STRICT FORMAT VALIDATION - CRITICAL CHECK
        # ============================================
        if expected_format:
            print(f"\n🔒 STRICT FORMAT CHECK:")
            print(f"   Ingested as: {ingested_format}")
            print(f"   Expected as: {expected_format.upper()}")
            
            format_check = self._validate_format_match(ingested_format, expected_format)
            
            if not format_check.get('format_match'):
                # FORMAT MISMATCH - REJECT VALIDATION
                print("\n" + "="*70)
                print("❌ FORMAT MISMATCH - VALIDATION REJECTED!")
                print("="*70)
                print(format_check['message'])
                print("="*70)
                
                # Store rejection in memory
                rejection_report = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'session_id': self.session_id,
                    'dataset_name': dataset_name,
                    'ingestion_id': ingestion_id,
                    'total_rows': len(df),
                    'total_columns': len(df.columns),
                    'validation_status': 'FORMAT_REJECTED',
                    'total_issues': 1,
                    'ingested_format': ingested_format,
                    'expected_format': expected_format.upper(),
                    'format_match': False,
                    'format_issues': format_check
                }
                
                self.memory.store_validation(rejection_report, df)
                
                return format_check
            
            print(f"   ✅ Format match confirmed: {ingested_format}")
        
        # Continue with normal validation if format matches
        validation_report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'session_id': self.session_id,
            'dataset_name': dataset_name,
            'ingestion_id': ingestion_id,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'data_format': metadata.get('data_format'),
            'source_file': metadata.get('file_name'),
            'ingested_format': ingested_format,
            'expected_format': expected_format.upper() if expected_format else ingested_format,
            'format_match': True
        }
        
        print(f"\n📊 Dataset: {dataset_name}")
        print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"   Format: {metadata.get('data_format')}")
        
        # Run validations
        issues_found = 0
        
        print("\n🏗️ Checking structure...")
        structure_issues = self._validate_structure(df)
        if structure_issues:
            validation_report['structure_issues'] = structure_issues
            issues_found += len(structure_issues)
            print(f"   ⚠️ Found {len(structure_issues)} structure issue(s)")
        else:
            print("   ✅ Structure validation passed")
        
        print("\n🔍 Checking missing values...")
        missing_values = self._check_missing_values(df)
        if missing_values:
            validation_report['missing_values'] = missing_values
            issues_found += 1
            print(f"   ⚠️ Found {missing_values.get('total_missing', 0)} missing values")
        else:
            print("   ✅ No missing values")
        
        print("\n🔄 Checking duplicates...")
        duplicates = self._check_duplicates(df)
        if duplicates:
            validation_report['duplicates'] = duplicates
            issues_found += 1
            print(f"   ⚠️ Found {duplicates.get('total_duplicates', 0)} duplicate rows")
        else:
            print("   ✅ No duplicates")
        
        print("\n📊 Detecting outliers...")
        outliers = {}
        
        print("   • IQR method...")
        outliers_iqr = self._detect_outliers_iqr(df)
        if outliers_iqr:
            outliers['iqr'] = outliers_iqr
            print(f"     ⚠️ IQR: Found outliers in {len(outliers_iqr)} column(s)")
        
        print("   • Z-score method...")
        outliers_zscore = self._detect_outliers_zscore(df)
        if outliers_zscore:
            outliers['zscore'] = outliers_zscore
            print(f"     ⚠️ Z-score: Found outliers in {len(outliers_zscore)} column(s)")
        
        print("   • Standard Deviation method...")
        outliers_stddev = self._detect_outliers_stddev(df)
        if outliers_stddev:
            outliers['stddev'] = outliers_stddev
            print(f"     ⚠️ Std Dev: Found outliers in {len(outliers_stddev)} column(s)")
        
        if outliers:
            validation_report['outliers'] = outliers
            issues_found += 1
        else:
            print("   ✅ No outliers detected")
        
        print("\n📝 Checking text issues...")
        text_issues = self._check_text_issues(df)
        if text_issues:
            validation_report['text_issues'] = text_issues
            issues_found += 1
            print(f"   ⚠️ Found text issues in {len(text_issues)} column(s)")
        else:
            print("   ✅ No text issues")
        
        print("\n📋 Validating schema...")
        schema_info = self._validate_schema(df, expected_schema)
        validation_report['schema'] = schema_info
        if 'type_conversion_suggestions' in schema_info:
            print(f"   ℹ️ Type conversion suggestions for {len(schema_info['type_conversion_suggestions'])} column(s)")
        else:
            print("   ✅ Schema looks good")
        
        print("\n📈 Generating statistics...")
        statistics = self._generate_statistics(df)
        validation_report['statistics'] = statistics
        print(f"   ✅ Statistics generated for {len(statistics)} columns")
        
        validation_report['shape_info'] = {
            'rows': len(df),
            'columns': len(df.columns),
            'total_cells': len(df) * len(df.columns),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 4),
            'size_category': 'small' if len(df) < 1000 else 'medium' if len(df) < 100000 else 'large'
        }
        
        # Finalize report
        validation_report['total_issues'] = issues_found
        validation_report['validation_status'] = 'PASS' if issues_found == 0 else 'ISSUES_FOUND'
        
        # Store in memory
        print("\n💾 Storing validation results in memory...")
        validation_id = self.memory.store_validation(validation_report, df)
        validation_report['validation_id'] = validation_id
        
        # Store current state
        self.current_df = df
        self.current_validation_report = validation_report
        
        # Final summary
        print("\n" + "="*70)
        if issues_found == 0:
            print("✅ VALIDATION COMPLETE - NO ISSUES FOUND!")
        else:
            print(f"⚠️ VALIDATION COMPLETE - {issues_found} ISSUE CATEGORIES FOUND")
        print("="*70)
        print(f"\nValidation ID: {validation_id}")
        print(f"Status: {validation_report['validation_status']}")
        print(f"Total Issues: {issues_found}")
        print(f"\n🎯 Output ready for Data Cleaning Tool")
        print("="*70)
        
        return validation_report
    
    def get_output_for_cleaning(self) -> Optional[Dict[str, Any]]:
        """Get structured output for Data Cleaning Tool"""
        if self.current_df is None:
            print("⚠️ No validated data available. Please run validation first.")
            return None
        
        issues_summary = {
            'missing_values': self.current_validation_report.get('missing_values', {}),
            'duplicates': self.current_validation_report.get('duplicates', {}),
            'outliers': self.current_validation_report.get('outliers', {}),
            'text_issues': self.current_validation_report.get('text_issues', {}),
            'structure_issues': self.current_validation_report.get('structure_issues', {})
        }
        
        return {
            'dataframe': self.current_df,
            'validation_report': self.current_validation_report,
            'issues_summary': issues_summary,
            'metadata': {
                'validation_id': self.current_validation_report.get('validation_id'),
                'dataset_name': self.current_validation_report.get('dataset_name'),
                'validation_status': self.current_validation_report.get('validation_status'),
                'total_issues': self.current_validation_report.get('total_issues')
            }
        }
    
    def query(self, query_text: str, method: str = 'semantic') -> Any:
        """NLP Query Interface"""
        print(f"\n🔍 Processing query: '{query_text}'")
        
        if method == 'semantic':
            results = self.memory.semantic_search(query_text, top_k=5)
            if results:
                print(f"✅ Found {len(results)} results")
                for i, result in enumerate(results, 1):
                    print(f"\n  Result {i}:")
                    print(f"    Dataset: {result['dataset_name']}")
                    print(f"    Status: {result['validation_status']}")
                    print(f"    Issues: {result['total_issues']}")
                    print(f"    Similarity: {result['similarity_score']:.3f}")
                return results
            else:
                print("ℹ️ No results found")
                return []
        else:
            print(f"⚠️ Unknown query method: {method}")
            return None
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        return self.memory.get_memory_stats()
    
    def reset_memory(self, confirm: bool = False):
        """Reset all memory"""
        self.memory.reset_memory(confirm=confirm)


# ==================== HELPER FUNCTIONS ====================

def create_validation_tool(memory_dir: str = "./memory/validation") -> DataValidationTool:
    """Factory function to create Data Validation Tool"""
    return DataValidationTool(memory_dir=memory_dir)


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    """Example usage of Data Validation Tool with STRICT FORMAT VALIDATION"""
    
    tool = create_validation_tool()
    
    # Example: CSV data
    sample_csv_data = pd.DataFrame({
        'patient_id': ['P001', 'P002', 'P003', 'P002', 'P004'],
        'age': [45, 62, 38, 62, 200],
        'diagnosis': ['diabetes', 'hypertension', 'asthma', 'hypertension', 'healthy'],
        'blood_pressure': [140, 165, 118, 165, 120],
        'notes': ['Normal  ', 'OK!@#', 'Fine', 'OK!@#', None]
    })
    
    ingestion_output = {
        'dataframe': sample_csv_data,
        'metadata': {
            'data_format': 'CSV',
            'file_name': 'sample_patients.csv',
            'ingestion_id': 1
        },
        'summary': {
            'shape': sample_csv_data.shape,
            'columns': sample_csv_data.columns.tolist()
        }
    }
    
    # Test 1: Correct format match (CSV → CSV)
    print("\n" + "="*70)
    print("TEST 1: Correct Format Match (CSV → CSV)")
    print("="*70)
    validation_report = tool.validate(ingestion_output, dataset_name='sample_patients', expected_format='csv')
    
    # Test 2: Format mismatch (CSV → JSON) - SHOULD BE REJECTED
    print("\n\n" + "="*70)
    print("TEST 2: Format Mismatch (CSV → JSON) - SHOULD REJECT")
    print("="*70)
    validation_report_rejected = tool.validate(ingestion_output, dataset_name='sample_patients', expected_format='json')
    
    # Check rejection
    if validation_report_rejected.get('format_match') == False:
        print("\n✅ FORMAT VALIDATION WORKING CORRECTLY!")
        print("   Mismatched format was properly rejected.")
    else:
        print("\n❌ ERROR: Format validation not working!")
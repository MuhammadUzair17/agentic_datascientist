# tools/data_ingesion1.py
"""
DATA INGESTION TOOL - Healthcare AutoML Data Science Agent
============================================================
Part of Automated Data Science (ADS) System
Accepts: CSV, JSON, API only
Output: Standardized CSV DataFrame for Data Validation Tool

Features:
- Strict format validation (CSV, JSON, API only)
- Automatic conversion to DataFrame
- Dual Memory System (SQLite3 + FAISS Vector DB)
- NLP Query Support
- Context for Agent
- Memory Reset Capability
"""

import pandas as pd
import numpy as np
import os
import json
import requests
import sqlite3
from datetime import datetime
from typing import Union, Dict, Any, List, Optional
import io
import pickle
import hashlib

# Environment variables
from dotenv import load_dotenv
load_dotenv()  # Load .env file

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


class DataIngestionMemory:
    """
    Dual Memory System for Data Ingestion Tool
    - SQLite3: Structured metadata storage
    - FAISS: Vector embeddings for semantic search
    """
    
    def __init__(self, memory_dir: str = "./memory/ingestion"):
        """Initialize dual memory system"""
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)
        
        # SQLite database path
        self.sqlite_path = os.path.join(memory_dir, "ingestion_metadata.db")
        
        # FAISS index path
        self.faiss_index_path = os.path.join(memory_dir, "ingestion_vectors.index")
        self.faiss_metadata_path = os.path.join(memory_dir, "ingestion_vectors_metadata.pkl")
        
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
                self.embedding_model = "text-embedding-3-small"  # OpenAI's efficient embedding model
                print(f"✅ OpenAI embeddings initialized (Model: {self.embedding_model})")
        else:
            self.openai_client = None
            self.embedding_model = None
    
    def _initialize_sqlite(self):
        """Create SQLite database schema"""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        # Main ingestion history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                session_id TEXT,
                data_source TEXT NOT NULL,
                data_format TEXT NOT NULL,
                source_type TEXT,
                file_name TEXT,
                row_count INTEGER,
                column_count INTEGER,
                memory_usage_mb REAL,
                status TEXT DEFAULT 'success'
            )
        """)
        
        # Column metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS column_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ingestion_id INTEGER,
                column_name TEXT NOT NULL,
                data_type TEXT,
                null_count INTEGER,
                unique_count INTEGER,
                sample_values TEXT,
                FOREIGN KEY (ingestion_id) REFERENCES ingestion_history(id)
            )
        """)
        
        # Data statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ingestion_id INTEGER,
                column_name TEXT,
                mean_value REAL,
                median_value REAL,
                std_dev REAL,
                min_value REAL,
                max_value REAL,
                quartile_25 REAL,
                quartile_75 REAL,
                FOREIGN KEY (ingestion_id) REFERENCES ingestion_history(id)
            )
        """)
        
        # Data snapshot storage (serialized CSV)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ingestion_id INTEGER,
                data_hash TEXT UNIQUE,
                data_csv TEXT,
                created_at TEXT,
                FOREIGN KEY (ingestion_id) REFERENCES ingestion_history(id)
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
            # Call OpenAI Embeddings API
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text,
                encoding_format="float"
            )
            
            # Extract embedding vector
            embedding = np.array(response.data[0].embedding, dtype='float32')
            return embedding
            
        except Exception as e:
            print(f"⚠️ Error generating embedding: {str(e)}")
            return None
    
    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of DataFrame for deduplication"""
        csv_string = df.to_csv(index=False)
        return hashlib.md5(csv_string.encode()).hexdigest()
    
    def store_ingestion(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> int:
        """
        Store ingestion data in dual memory system
        Returns: ingestion_id
        """
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        try:
            # Insert main ingestion record
            cursor.execute("""
                INSERT INTO ingestion_history 
                (timestamp, session_id, data_source, data_format, source_type, 
                 file_name, row_count, column_count, memory_usage_mb, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                metadata.get('session_id', 'default'),
                metadata.get('data_source', 'unknown'),
                metadata.get('data_format', 'unknown'),
                metadata.get('source_type', 'unknown'),
                metadata.get('file_name', None),
                len(df),
                len(df.columns),
                df.memory_usage(deep=True).sum() / 1024**2,  # MB
                'success'
            ))
            
            ingestion_id = cursor.lastrowid
            
            # Store column metadata
            for col in df.columns:
                sample_values = df[col].dropna().head(5).tolist()
                cursor.execute("""
                    INSERT INTO column_metadata 
                    (ingestion_id, column_name, data_type, null_count, unique_count, sample_values)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    ingestion_id,
                    col,
                    str(df[col].dtype),
                    int(df[col].isnull().sum()),
                    int(df[col].nunique()),
                    json.dumps(sample_values, default=str)
                ))
            
            # Store statistics for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                stats = df[col].describe()
                cursor.execute("""
                    INSERT INTO data_statistics 
                    (ingestion_id, column_name, mean_value, median_value, std_dev, 
                     min_value, max_value, quartile_25, quartile_75)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ingestion_id,
                    col,
                    float(stats['mean']) if 'mean' in stats else None,
                    float(df[col].median()),
                    float(stats['std']) if 'std' in stats else None,
                    float(stats['min']) if 'min' in stats else None,
                    float(stats['max']) if 'max' in stats else None,
                    float(stats['25%']) if '25%' in stats else None,
                    float(stats['75%']) if '75%' in stats else None,
                ))
            
            # Store data snapshot
            data_hash = self._compute_data_hash(df)
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            try:
                cursor.execute("""
                    INSERT INTO data_snapshots (ingestion_id, data_hash, data_csv, created_at)
                    VALUES (?, ?, ?, ?)
                """, (
                    ingestion_id,
                    data_hash,
                    csv_data,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ))
            except sqlite3.IntegrityError:
                print(f"ℹ️ Duplicate data detected (hash: {data_hash[:8]}...)")
            
            # Add to FAISS vector DB
            if self.faiss_index is not None:
                # Create searchable text representation
                search_text = f"""
                Dataset: {metadata.get('file_name', 'unknown')}
                Format: {metadata.get('data_format', 'unknown')}
                Columns: {', '.join(df.columns.tolist())}
                Rows: {len(df)}
                Numeric columns: {', '.join(numeric_cols.tolist())}
                Sample data: {df.head(2).to_string()}
                """
                
                embedding = self._generate_embedding(search_text)
                if embedding is not None:
                    self.faiss_index.add(np.array([embedding]).astype('float32'))
                    self.faiss_metadata.append({
                        'ingestion_id': ingestion_id,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'file_name': metadata.get('file_name', 'unknown'),
                        'columns': df.columns.tolist(),
                        'row_count': len(df),
                        'text': search_text
                    })
                    self._save_faiss()
            
            conn.commit()
            print(f"✅ Ingestion stored in memory (ID: {ingestion_id})")
            return ingestion_id
            
        except Exception as e:
            conn.rollback()
            print(f"❌ Error storing ingestion: {str(e)}")
            return None
        finally:
            conn.close()
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search ingestion history using semantic similarity"""
        if self.faiss_index is None or self.embedding_model is None:
            print("⚠️ Vector search not available")
            return []
        
        if self.faiss_index.ntotal == 0:
            print("ℹ️ No data in vector database yet")
            return []
        
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        if query_embedding is None:
            return []
        
        # Search
        start_time = datetime.now()
        distances, indices = self.faiss_index.search(
            np.array([query_embedding]).astype('float32'), 
            min(top_k, self.faiss_index.ntotal)
        )
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Get results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.faiss_metadata):
                result = self.faiss_metadata[idx].copy()
                result['similarity_score'] = float(1 / (1 + distances[0][i]))  # Convert distance to similarity
                results.append(result)
        
        # Log query
        self._log_query(query, 'semantic_search', len(results), execution_time)
        
        return results
    
    def sql_search(self, query_params: Dict[str, Any]) -> pd.DataFrame:
        """Search ingestion history using SQL conditions"""
        conn = sqlite3.connect(self.sqlite_path)
        
        # Build SQL query
        conditions = []
        params = []
        
        if 'data_format' in query_params:
            conditions.append("data_format = ?")
            params.append(query_params['data_format'])
        
        if 'min_rows' in query_params:
            conditions.append("row_count >= ?")
            params.append(query_params['min_rows'])
        
        if 'max_rows' in query_params:
            conditions.append("row_count <= ?")
            params.append(query_params['max_rows'])
        
        if 'file_name' in query_params:
            conditions.append("file_name LIKE ?")
            params.append(f"%{query_params['file_name']}%")
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
            SELECT * FROM ingestion_history 
            WHERE {where_clause}
            ORDER BY timestamp DESC
        """
        
        start_time = datetime.now()
        df = pd.read_sql_query(query, conn, params=params)
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        conn.close()
        
        # Log query
        self._log_query(str(query_params), 'sql_search', len(df), execution_time)
        
        return df
    
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
    
    def get_ingestion_by_id(self, ingestion_id: int) -> Optional[pd.DataFrame]:
        """Retrieve DataFrame from memory by ingestion ID"""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT data_csv FROM data_snapshots 
            WHERE ingestion_id = ?
        """, (ingestion_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return pd.read_csv(io.StringIO(result[0]))
        return None
    
    def get_all_ingestions(self) -> pd.DataFrame:
        """Get all ingestion history"""
        conn = sqlite3.connect(self.sqlite_path)
        df = pd.read_sql_query("SELECT * FROM ingestion_history ORDER BY timestamp DESC", conn)
        conn.close()
        return df
    
    def get_column_metadata(self, ingestion_id: int) -> pd.DataFrame:
        """Get column metadata for specific ingestion"""
        conn = sqlite3.connect(self.sqlite_path)
        df = pd.read_sql_query(
            "SELECT * FROM column_metadata WHERE ingestion_id = ?", 
            conn, 
            params=(ingestion_id,)
        )
        conn.close()
        return df
    
    def reset_memory(self, confirm: bool = False):
        """Reset all memory (SQLite + FAISS)"""
        if not confirm:
            print("⚠️ Memory reset requires confirmation. Set confirm=True")
            return
        
        # Reset SQLite
        if os.path.exists(self.sqlite_path):
            os.remove(self.sqlite_path)
            self._initialize_sqlite()
        
        # Reset FAISS
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
        
        # SQLite stats
        cursor.execute("SELECT COUNT(*) FROM ingestion_history")
        stats['total_ingestions'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM query_log")
        stats['total_queries'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(memory_usage_mb) FROM ingestion_history")
        stats['total_data_mb'] = cursor.fetchone()[0] or 0
        
        # FAISS stats
        stats['vector_count'] = self.faiss_index.ntotal if self.faiss_index else 0
        
        # File sizes
        stats['sqlite_size_mb'] = os.path.getsize(self.sqlite_path) / 1024**2 if os.path.exists(self.sqlite_path) else 0
        stats['faiss_size_mb'] = os.path.getsize(self.faiss_index_path) / 1024**2 if os.path.exists(self.faiss_index_path) else 0
        
        conn.close()
        return stats


class DataIngestionTool:
    """
    Professional Data Ingestion Tool for Healthcare AutoML
    
    STRICT REQUIREMENTS:
    - Only accepts: CSV, JSON, API
    - All data converted to DataFrame
    - Dual memory system (SQLite + FAISS)
    - NLP query support
    - Output ready for Data Validation Tool
    """
    
    ALLOWED_FORMATS = ['csv', 'json', 'api']
    
    def __init__(self, memory_dir: str = "./memory/ingestion", session_id: str = None):
        """Initialize Data Ingestion Tool"""
        self.memory = DataIngestionMemory(memory_dir)
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_df = None
        self.current_metadata = None
        
        print("="*70)
        print("DATA INGESTION TOOL - Healthcare AutoML System")
        print("="*70)
        print(f"Session ID: {self.session_id}")
        print(f"Memory Location: {memory_dir}")
        print(f"Allowed Formats: {', '.join(self.ALLOWED_FORMATS).upper()}")
        print("="*70)
    
    def _validate_format(self, data_format: str) -> bool:
        """Strict format validation"""
        data_format = data_format.lower().strip()
        
        if data_format not in self.ALLOWED_FORMATS:
            print("\n" + "="*70)
            print("❌ INVALID DATA FORMAT")
            print("="*70)
            print(f"You entered: '{data_format}'")
            print(f"\n⚠️ This tool ONLY accepts: {', '.join(self.ALLOWED_FORMATS).upper()}")
            print("\nSupported formats:")
            print("  • CSV  - Comma-separated values (.csv files)")
            print("  • JSON - JavaScript Object Notation (.json files or JSON data)")
            print("  • API  - RESTful API endpoints (returns JSON)")
            print("\n💡 Please provide data in one of these formats only.")
            print("="*70)
            return False
        
        return True
    
    def _ingest_csv(self, source: Union[str, Any]) -> Optional[pd.DataFrame]:
        """
        Ingest CSV data
        Source can be: file path, uploaded file object, or file-like object
        """
        try:
            print("\n📥 Ingesting CSV data...")
            
            # Handle different source types
            if hasattr(source, 'read'):  # File-like object (e.g., Streamlit upload)
                df = pd.read_csv(source)
                file_name = getattr(source, 'name', 'uploaded_file.csv')
                source_type = 'upload'
            elif isinstance(source, str) and os.path.exists(source):  # File path
                df = pd.read_csv(source)
                file_name = os.path.basename(source)
                source_type = 'file_path'
            else:
                raise ValueError("Invalid CSV source. Provide file path or file object.")
            
            print(f"✅ CSV loaded successfully!")
            print(f"   Rows: {len(df)}")
            print(f"   Columns: {len(df.columns)}")
            print(f"   File: {file_name}")
            
            # Store metadata
            self.current_metadata = {
                'session_id': self.session_id,
                'data_source': str(source)[:200],
                'data_format': 'CSV',
                'source_type': source_type,
                'file_name': file_name
            }
            
            return df
            
        except Exception as e:
            print(f"❌ CSV Ingestion Error: {str(e)}")
            print("💡 Ensure the file is a valid CSV with proper encoding (UTF-8)")
            return None
    
    def _ingest_json(self, source: Union[str, Dict, Any]) -> Optional[pd.DataFrame]:
        """
        Ingest JSON data and convert to DataFrame
        Source can be: file path, dict, JSON string, or uploaded file
        """
        try:
            print("\n📥 Ingesting JSON data...")
            
            # Handle different source types
            if hasattr(source, 'read'):  # File-like object
                json_data = json.load(source)
                file_name = getattr(source, 'name', 'uploaded_file.json')
                source_type = 'upload'
            elif isinstance(source, str) and os.path.exists(source):  # File path
                with open(source, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                file_name = os.path.basename(source)
                source_type = 'file_path'
            elif isinstance(source, dict):  # Dictionary
                json_data = source
                file_name = 'json_dict.json'
                source_type = 'dict'
            elif isinstance(source, str):  # JSON string
                json_data = json.loads(source)
                file_name = 'json_string.json'
                source_type = 'json_string'
            else:
                raise ValueError("Invalid JSON source")
            
            # Convert to DataFrame
            print("🔄 Converting JSON to DataFrame...")
            
            if isinstance(json_data, list):
                # List of objects -> each object is a row
                df = pd.json_normalize(json_data)
            elif isinstance(json_data, dict):
                # Check if it's a nested structure
                if any(isinstance(v, list) for v in json_data.values()):
                    # Find the list and use it
                    list_key = next(k for k, v in json_data.items() if isinstance(v, list))
                    df = pd.json_normalize(json_data[list_key])
                    print(f"ℹ️ Extracted data from key: '{list_key}'")
                else:
                    # Single object -> one row
                    df = pd.DataFrame([json_data])
            else:
                raise ValueError("Unexpected JSON structure")
            
            print(f"✅ JSON converted to DataFrame!")
            print(f"   Rows: {len(df)}")
            print(f"   Columns: {len(df.columns)}")
            print(f"   Features extracted: {', '.join(df.columns.tolist()[:5])}{'...' if len(df.columns) > 5 else ''}")
            
            # Store metadata
            self.current_metadata = {
                'session_id': self.session_id,
                'data_source': str(source)[:200],
                'data_format': 'JSON',
                'source_type': source_type,
                'file_name': file_name
            }
            
            return df
            
        except Exception as e:
            print(f"❌ JSON Ingestion Error: {str(e)}")
            print("💡 Ensure the JSON is valid and contains tabular-like data")
            return None
    
    def _ingest_api(self, api_url: str, params: Dict = None, headers: Dict = None, 
                    method: str = 'GET') -> Optional[pd.DataFrame]:
        """
        Ingest data from API endpoint and convert to DataFrame
        """
        try:
            print(f"\n📥 Fetching data from API: {api_url}")
            
            # Make API request
            if method.upper() == 'GET':
                response = requests.get(api_url, params=params, headers=headers, timeout=30)
            elif method.upper() == 'POST':
                response = requests.post(api_url, json=params, headers=headers, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            
            print(f"✅ API response received (Status: {response.status_code})")
            
            # Parse JSON
            api_data = response.json()
            
            # Convert to DataFrame
            print("🔄 Converting API response to DataFrame...")
            
            if isinstance(api_data, list):
                df = pd.json_normalize(api_data)
            elif isinstance(api_data, dict):
                # Try to find the data array
                if 'data' in api_data and isinstance(api_data['data'], list):
                    df = pd.json_normalize(api_data['data'])
                    print("ℹ️ Extracted data from 'data' key")
                elif 'results' in api_data and isinstance(api_data['results'], list):
                    df = pd.json_normalize(api_data['results'])
                    print("ℹ️ Extracted data from 'results' key")
                elif any(isinstance(v, list) for v in api_data.values()):
                    list_key = next(k for k, v in api_data.items() if isinstance(v, list))
                    df = pd.json_normalize(api_data[list_key])
                    print(f"ℹ️ Extracted data from '{list_key}' key")
                else:
                    df = pd.DataFrame([api_data])
            else:
                raise ValueError("Unexpected API response format")
            
            print(f"✅ API data converted to DataFrame!")
            print(f"   Rows: {len(df)}")
            print(f"   Columns: {len(df.columns)}")
            print(f"   Features extracted: {', '.join(df.columns.tolist()[:5])}{'...' if len(df.columns) > 5 else ''}")
            
            # Store metadata
            self.current_metadata = {
                'session_id': self.session_id,
                'data_source': api_url,
                'data_format': 'API',
                'source_type': f'api_{method.lower()}',
                'file_name': f"api_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            }
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"❌ API Request Error: {str(e)}")
            print("💡 Check: URL correctness, internet connection, API availability")
            return None
        except Exception as e:
            print(f"❌ API Ingestion Error: {str(e)}")
            print("💡 Ensure API returns valid JSON data in expected format")
            return None
    
    def ingest(self, source: Any, data_format: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        Main ingestion method - STRICT FORMAT CONTROL
        
        Args:
            source: Data source (file path, file object, dict, or API URL)
            data_format: MUST be one of: 'csv', 'json', 'api'
            **kwargs: Additional parameters (api_params, api_headers, api_method)
        
        Returns:
            pd.DataFrame if successful, None otherwise
        """
        
        # STRICT FORMAT VALIDATION
        if not self._validate_format(data_format):
            return None
        
        data_format = data_format.lower().strip()
        
        # Route to appropriate method
        try:
            if data_format == 'csv':
                df = self._ingest_csv(source)
            
            elif data_format == 'json':
                df = self._ingest_json(source)
            
            elif data_format == 'api':
                api_params = kwargs.get('api_params', None)
                api_headers = kwargs.get('api_headers', None)
                api_method = kwargs.get('api_method', 'GET')
                df = self._ingest_api(source, api_params, api_headers, api_method)
            
            # Validate result
            if df is None or df.empty:
                print("\n⚠️ No data was ingested")
                return None
            
            # Store in memory
            print("\n💾 Storing in memory system...")
            ingestion_id = self.memory.store_ingestion(df, self.current_metadata)
            
            if ingestion_id:
                self.current_df = df
                print(f"\n{'='*70}")
                print("✅ DATA INGESTION COMPLETE")
                print(f"{'='*70}")
                print(f"Ingestion ID: {ingestion_id}")
                print(f"Format: {data_format.upper()}")
                print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                print(f"\n📊 Preview:")
                print(df.head(3).to_string())
                print(f"{'='*70}")
                print("\n🎯 Output ready for Data Validation Tool")
                
                return df
            else:
                print("⚠️ Data ingested but memory storage failed")
                return df
                
        except Exception as e:
            print(f"\n❌ Unexpected Error: {str(e)}")
            print("💡 Please check your input and try again")
            return None
    
    def get_output_for_validation(self) -> Optional[Dict[str, Any]]:
        """
        Get structured output for Data Validation Tool
        
        Returns:
            Dict containing:
                - dataframe: The actual DataFrame
                - metadata: All metadata about ingestion
                - summary: Quick summary statistics
        """
        if self.current_df is None:
            print("⚠️ No data available. Please ingest data first.")
            return None
        
        return {
            'dataframe': self.current_df,
            'metadata': self.current_metadata,
            'summary': {
                'shape': self.current_df.shape,
                'columns': self.current_df.columns.tolist(),
                'dtypes': self.current_df.dtypes.to_dict(),
                'missing_values': self.current_df.isnull().sum().to_dict(),
                'memory_usage_mb': self.current_df.memory_usage(deep=True).sum() / 1024**2
            }
        }
    
    def query(self, query_text: str, method: str = 'semantic') -> Any:
        """
        NLP Query Interface
        
        Args:
            query_text: Natural language query
            method: 'semantic' or 'sql'
        
        Returns:
            Query results
        """
        print(f"\n🔍 Processing query: '{query_text}'")
        
        if method == 'semantic':
            results = self.memory.semantic_search(query_text, top_k=5)
            if results:
                print(f"✅ Found {len(results)} results")
                for i, result in enumerate(results, 1):
                    print(f"\n  Result {i}:")
                    print(f"    File: {result['file_name']}")
                    print(f"    Rows: {result['row_count']}")
                    print(f"    Similarity: {result['similarity_score']:.3f}")
                return results
            else:
                print("ℹ️ No results found")
                return []
        
        elif method == 'sql':
            # Parse query to SQL conditions (simplified)
            # In production, use more sophisticated NLP parsing
            print("ℹ️ SQL query method - provide query_params dict instead")
            return None
        
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

def create_ingestion_tool(memory_dir: str = "./memory/ingestion") -> DataIngestionTool:
    """Factory function to create Data Ingestion Tool"""
    return DataIngestionTool(memory_dir=memory_dir)


def inspect_memory(memory_dir: str = "./memory/ingestion"):
    """
    Inspect memory contents (for debugging and verification)
    """
    print("\n" + "="*70)
    print("MEMORY INSPECTION")
    print("="*70)
    
    sqlite_path = os.path.join(memory_dir, "ingestion_metadata.db")
    
    if not os.path.exists(sqlite_path):
        print("⚠️ No memory database found")
        return
    
    conn = sqlite3.connect(sqlite_path)
    
    # Ingestion history
    print("\n📋 INGESTION HISTORY:")
    df = pd.read_sql_query("SELECT * FROM ingestion_history ORDER BY timestamp DESC LIMIT 10", conn)
    print(df.to_string())
    
    # Column metadata
    print("\n📊 COLUMN METADATA (Latest):")
    df = pd.read_sql_query("""
        SELECT cm.* FROM column_metadata cm
        JOIN ingestion_history ih ON cm.ingestion_id = ih.id
        ORDER BY ih.timestamp DESC LIMIT 20
    """, conn)
    print(df.to_string())
    
    # Query log
    print("\n🔍 QUERY LOG:")
    df = pd.read_sql_query("SELECT * FROM query_log ORDER BY timestamp DESC LIMIT 10", conn)
    print(df.to_string())
    
    conn.close()
    
    print("\n" + "="*70)


# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    # Create tool
    tool = create_ingestion_tool()
    
    # Example 2: JSON ingestion (wrap list in a dict)
    sample_json = {
        "data": [
            {"patient_id": 1, "age": 45, "diagnosis": "diabetes", "risk_score": 0.7},
            {"patient_id": 2, "age": 62, "diagnosis": "hypertension", "risk_score": 0.5},
            {"patient_id": 3, "age": 38, "diagnosis": "asthma", "risk_score": 0.3}
        ]
    }
    df = tool.ingest(sample_json, data_format="json")
    
    # Example 3: API ingestion (public API)
    # df = tool.ingest(
    #     "https://api.example.com/patients",
    #     data_format="api",
    #     api_params={"limit": 100}
    # )
    
    # Get output for next tool
    if df is not None:
        output = tool.get_output_for_validation()
        print("\n📤 Output structure for Data Validation Tool:")
        print(f"Keys: {output.keys()}")
    
    # Query memory
    # tool.query("Show me diabetes patient data", method='semantic')
    
    # Get stats
    stats = tool.get_memory_stats()
    print(f"\n📊 Memory Stats: {stats}")
    
    # Inspect memory
    inspect_memory()
    
    # Reset memory (if needed)
    # tool.reset_memory(confirm=True)
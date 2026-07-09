# tools/memory_utils.py
# Enhanced Unified Memory Management for All Tools
# Tracks context from: Ingestion, Validation, Cleaning, and EDA

import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional
import streamlit as st

class UnifiedMemoryManager:
    """
    Unified Memory Manager that consolidates context from all tools:
    - Data Ingestion
    - Data Validation
    - Data Cleaning
    - EDA Analysis
    
    Provides a single source of truth for the AI agent
    """
    
    def __init__(self, db_path: str = "unified_memory.db"):
        """Initialize unified memory manager"""
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """Create unified memory database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Unified context table - stores all tool contexts
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS unified_context (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    ingestion_context TEXT,
                    validation_context TEXT,
                    cleaning_context TEXT,
                    eda_context TEXT,
                    current_state TEXT,
                    metadata TEXT
                )
            """)
            
            # Operation log - tracks all operations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS operation_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    operation_type TEXT NOT NULL,
                    operation_details TEXT,
                    success BOOLEAN,
                    error_message TEXT
                )
            """)
            
            conn.commit()
            conn.close()
        except Exception as e:
            if 'st' in dir():
                st.error(f"⚠️ Unified memory initialization error: {str(e)}")
    
    def get_current_session_id(self) -> str:
        """Get or create current session ID"""
        if 'session_id' not in st.session_state:
            st.session_state['session_id'] = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return st.session_state['session_id']
    
    def update_ingestion_context(self, context: Dict):
        """Update ingestion context in unified memory"""
        session_id = self.get_current_session_id()
        
        # Get existing context
        full_context = self.get_full_context()
        full_context['ingestion'] = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'data_source': context.get('data_source', 'Unknown'),
            'data_format': context.get('data_format', 'Unknown'),
            'rows': context.get('row_count', 0),
            'columns': context.get('column_count', 0),
            'column_names': context.get('columns', []) if isinstance(context.get('columns'), list) else [],
            'file_name': context.get('file_name', 'Unknown')
        }
        
        # Update database
        self._save_context(session_id, full_context)
        
        # Log operation
        self.log_operation('ingestion', f"Loaded data: {context.get('row_count', 0)} rows", True)
        
        # Update session state
        st.session_state['unified_context'] = full_context
    
    def update_validation_context(self, context: Dict):
        """Update validation context in unified memory"""
        session_id = self.get_current_session_id()
        
        # Get existing context
        full_context = self.get_full_context()
        full_context['validation'] = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_issues': context.get('total_issues', 0),
            'validation_status': context.get('validation_status', 'Unknown'),
            'missing_values': context.get('missing_values', {}),
            'duplicates': context.get('duplicates', {}),
            'outliers': context.get('outliers', {}),
            'text_issues': context.get('text_issues', {}),
            'format_issues': context.get('format_issues', {}),
            'structure_issues': context.get('structure_issues', {})
        }
        
        # Update database
        self._save_context(session_id, full_context)
        
        # Log operation
        self.log_operation('validation', f"Validation completed: {context.get('total_issues', 0)} issues", True)
        
        # Update session state
        st.session_state['unified_context'] = full_context
    
    def update_cleaning_context(self, context: Dict):
        """Update cleaning context in unified memory"""
        session_id = self.get_current_session_id()
        
        # Get existing context
        full_context = self.get_full_context()
        full_context['cleaning'] = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'rows_before': context.get('rows_before', 0),
            'rows_after': context.get('rows_after', 0),
            'columns_before': context.get('columns_before', 0),
            'columns_after': context.get('columns_after', 0),
            'operations': context.get('operations', []),
            'summary': context.get('summary', ''),
            'rows_removed': context.get('rows_before', 0) - context.get('rows_after', 0)
        }
        
        # Update database
        self._save_context(session_id, full_context)
        
        # Log operation
        self.log_operation('cleaning', f"Cleaned data: {len(context.get('operations', []))} operations", True)
        
        # Update session state
        st.session_state['unified_context'] = full_context
    
    def update_eda_context(self, context: Dict):
        """Update EDA context in unified memory"""
        session_id = self.get_current_session_id()
        
        # Get existing context
        full_context = self.get_full_context()
        full_context['eda'] = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'dataset_shape': context.get('dataset_shape', (0, 0)),
            'numeric_columns': context.get('numeric_columns', []),
            'categorical_columns': context.get('categorical_columns', []),
            'distributions': context.get('distributions', {}),
            'correlations': context.get('correlations', {}),
            'outliers': context.get('outliers', {}),
            'missing_values': context.get('missing_values', {}),
            'categorical_analysis': context.get('categorical_analysis', {})
        }
        
        # Update database
        self._save_context(session_id, full_context)
        
        # Log operation
        self.log_operation('eda', f"EDA completed: {len(context.get('numeric_columns', []))} numeric cols analyzed", True)
        
        # Update session state
        st.session_state['unified_context'] = full_context
    
    def _save_context(self, session_id: str, full_context: Dict):
        """Save full context to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Prepare JSON strings
            ingestion_json = json.dumps(full_context.get('ingestion', {}))
            validation_json = json.dumps(full_context.get('validation', {}))
            cleaning_json = json.dumps(full_context.get('cleaning', {}))
            eda_json = json.dumps(full_context.get('eda', {}))
            current_state = self._determine_current_state(full_context)
            metadata_json = json.dumps({'session_start': timestamp})
            
            # Check if session exists
            cursor.execute("SELECT id FROM unified_context WHERE session_id = ?", (session_id,))
            result = cursor.fetchone()
            
            if result:
                # Update existing
                cursor.execute("""
                    UPDATE unified_context 
                    SET timestamp = ?, ingestion_context = ?, validation_context = ?,
                        cleaning_context = ?, eda_context = ?, current_state = ?
                    WHERE session_id = ?
                """, (timestamp, ingestion_json, validation_json, cleaning_json, 
                      eda_json, current_state, session_id))
            else:
                # Insert new
                cursor.execute("""
                    INSERT INTO unified_context 
                    (session_id, timestamp, ingestion_context, validation_context,
                     cleaning_context, eda_context, current_state, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (session_id, timestamp, ingestion_json, validation_json,
                      cleaning_json, eda_json, current_state, metadata_json))
            
            conn.commit()
            conn.close()
        except Exception as e:
            if 'st' in dir():
                st.error(f"⚠️ Error saving context: {str(e)}")
    
    def get_full_context(self) -> Dict:
        """Get complete unified context for current session"""
        # Try session state first
        if 'unified_context' in st.session_state:
            return st.session_state['unified_context']
        
        # Otherwise get from database
        session_id = self.get_current_session_id()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT ingestion_context, validation_context, cleaning_context, eda_context
                FROM unified_context WHERE session_id = ?
            """, (session_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'ingestion': json.loads(result[0]) if result[0] else {},
                    'validation': json.loads(result[1]) if result[1] else {},
                    'cleaning': json.loads(result[2]) if result[2] else {},
                    'eda': json.loads(result[3]) if result[3] else {}
                }
            else:
                return {'ingestion': {}, 'validation': {}, 'cleaning': {}, 'eda': {}}
        except Exception as e:
            return {'ingestion': {}, 'validation': {}, 'cleaning': {}, 'eda': {}}
    
    def _determine_current_state(self, context: Dict) -> str:
        """Determine current pipeline state"""
        if context.get('eda'):
            return 'analyzed'
        elif context.get('cleaning'):
            return 'cleaned'
        elif context.get('validation'):
            return 'validated'
        elif context.get('ingestion'):
            return 'loaded'
        else:
            return 'initial'
    
    def log_operation(self, operation_type: str, details: str, success: bool, error: str = None):
        """Log an operation to the operation log"""
        session_id = self.get_current_session_id()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            cursor.execute("""
                INSERT INTO operation_log 
                (session_id, timestamp, operation_type, operation_details, success, error_message)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, timestamp, operation_type, details, success, error))
            
            conn.commit()
            conn.close()
        except Exception as e:
            pass  # Silently fail for logging
    
    def get_operation_log(self) -> List[Dict]:
        """Get operation log for current session"""
        session_id = self.get_current_session_id()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT timestamp, operation_type, operation_details, success
                FROM operation_log WHERE session_id = ?
                ORDER BY timestamp DESC
            """, (session_id,))
            
            results = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'timestamp': r[0],
                    'type': r[1],
                    'details': r[2],
                    'success': bool(r[3])
                }
                for r in results
            ]
        except Exception as e:
            return []
    
    def clear_session_context(self):
        """Clear current session context"""
        if 'unified_context' in st.session_state:
            del st.session_state['unified_context']
        if 'session_id' in st.session_state:
            del st.session_state['session_id']
    
    def create_agent_context_summary(self) -> str:
        """
        Create comprehensive context summary for AI agent
        Includes all tool contexts in formatted text
        """
        context = self.get_full_context()
        
        summary_parts = []
        summary_parts.append("=" * 70)
        summary_parts.append("UNIFIED DATA PIPELINE CONTEXT")
        summary_parts.append("=" * 70)
        
        # Current state
        state = self._determine_current_state(context)
        summary_parts.append(f"\n🔄 Pipeline State: {state.upper()}")
        
        # Ingestion Context
        if context.get('ingestion'):
            ing = context['ingestion']
            summary_parts.append(f"\n📊 DATA INGESTION ({ing.get('timestamp', 'N/A')})")
            summary_parts.append(f"  • Source: {ing.get('file_name', 'Unknown')}")
            summary_parts.append(f"  • Format: {ing.get('data_format', 'Unknown')}")
            summary_parts.append(f"  • Rows: {ing.get('rows', 0):,}")
            summary_parts.append(f"  • Columns: {ing.get('columns', 0)}")
            if ing.get('column_names'):
                cols = ing['column_names'][:10]
                summary_parts.append(f"  • Column Names: {', '.join(map(str, cols))}")
        
        # Validation Context
        if context.get('validation'):
            val = context['validation']
            summary_parts.append(f"\n🔍 VALIDATION RESULTS ({val.get('timestamp', 'N/A')})")
            summary_parts.append(f"  • Status: {val.get('validation_status', 'Unknown')}")
            summary_parts.append(f"  • Total Issues: {val.get('total_issues', 0)}")
            
            if val.get('missing_values'):
                mv = val['missing_values']
                summary_parts.append(f"  • Missing Values: {mv.get('total_missing', 0)} ({mv.get('missing_percentage', 0):.1f}%)")
                if mv.get('columns_with_missing'):
                    cols = list(mv['columns_with_missing'].keys())[:5]
                    summary_parts.append(f"    - Affected Columns: {', '.join(cols)}")
            
            if val.get('duplicates'):
                dup = val['duplicates']
                summary_parts.append(f"  • Duplicates: {dup.get('total_duplicates', 0)} rows ({dup.get('duplicate_percentage', 0):.1f}%)")
            
            if val.get('outliers'):
                out = val['outliers']
                summary_parts.append(f"  • Outliers: Detected in {len(out)} columns")
                for col, info in list(out.items())[:3]:
                    summary_parts.append(f"    - {col}: {info.get('count', 0)} outliers")
        
        # Cleaning Context
        if context.get('cleaning'):
            cln = context['cleaning']
            summary_parts.append(f"\n🧹 CLEANING OPERATIONS ({cln.get('timestamp', 'N/A')})")
            summary_parts.append(f"  • Rows: {cln.get('rows_before', 0):,} → {cln.get('rows_after', 0):,} (removed: {cln.get('rows_removed', 0)})")
            summary_parts.append(f"  • Columns: {cln.get('columns_before', 0)} → {cln.get('columns_after', 0)}")
            summary_parts.append(f"  • Operations Performed: {len(cln.get('operations', []))}")
            
            if cln.get('operations'):
                summary_parts.append("  • Details:")
                for op in cln['operations'][:5]:
                    summary_parts.append(f"    - {op.get('type', 'Unknown')}: {op.get('details', 'No details')}")
        
        # EDA Context
        if context.get('eda'):
            eda = context['eda']
            summary_parts.append(f"\n📈 EDA ANALYSIS ({eda.get('timestamp', 'N/A')})")
            
            shape = eda.get('dataset_shape', (0, 0))
            summary_parts.append(f"  • Dataset Shape: {shape[0]:,} rows × {shape[1]} columns")
            
            if eda.get('numeric_columns'):
                num_cols = eda['numeric_columns']
                summary_parts.append(f"  • Numeric Columns ({len(num_cols)}): {', '.join(num_cols[:5])}")
            
            if eda.get('categorical_columns'):
                cat_cols = eda['categorical_columns']
                summary_parts.append(f"  • Categorical Columns ({len(cat_cols)}): {', '.join(cat_cols[:5])}")
            
            # Distributions
            if eda.get('distributions'):
                summary_parts.append(f"  • Distribution Analysis:")
                for col, stats in list(eda['distributions'].items())[:3]:
                    skew = stats.get('skewness', 0)
                    skew_desc = "right-skewed" if skew > 0.5 else "left-skewed" if skew < -0.5 else "symmetric"
                    summary_parts.append(f"    - {col}: μ={stats.get('mean', 0):.2f}, σ={stats.get('std', 0):.2f} ({skew_desc})")
            
            # Correlations
            if eda.get('correlations'):
                corr = eda['correlations']
                strong_corrs = []
                for col1, correlations in corr.items():
                    for col2, corr_val in correlations.items():
                        if abs(corr_val) > 0.5:
                            strong_corrs.append((col1, col2, corr_val))
                
                if strong_corrs:
                    strong_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
                    summary_parts.append(f"  • Top Correlations:")
                    for col1, col2, val in strong_corrs[:5]:
                        summary_parts.append(f"    - {col1} ↔ {col2}: {val:.3f}")
            
            # Outliers from EDA
            if eda.get('outliers'):
                summary_parts.append(f"  • Outliers (EDA):")
                for col, info in list(eda['outliers'].items())[:3]:
                    summary_parts.append(f"    - {col}: {info.get('count', 0)} ({info.get('percentage', 0):.1f}%)")
        
        summary_parts.append("\n" + "=" * 70)
        
        return "\n".join(summary_parts)


class AnalysisContextManager:
    """Legacy context manager - kept for backward compatibility"""
    
    @staticmethod
    def create_context_summary(analysis_context: Dict) -> str:
        """Create summary from analysis context (legacy method)"""
        if not analysis_context:
            return "No analysis context available."
        
        summary_parts = []
        
        if "dataset_shape" in analysis_context:
            shape = analysis_context["dataset_shape"]
            summary_parts.append(f"Dataset: {shape[0]} rows × {shape[1]} columns")
        
        if "numeric_columns" in analysis_context:
            num_cols = analysis_context["numeric_columns"]
            summary_parts.append(f"Numeric columns ({len(num_cols)}): {', '.join(num_cols[:5])}")
        
        if "categorical_columns" in analysis_context:
            cat_cols = analysis_context["categorical_columns"]
            summary_parts.append(f"Categorical columns ({len(cat_cols)}): {', '.join(cat_cols[:5])}")
        
        if "correlations" in analysis_context:
            corr = analysis_context["correlations"]
            summary_parts.append(f"\nTop Correlations:")
            
            strong_corrs = []
            for col1, correlations in corr.items():
                for col2, corr_val in correlations.items():
                    if abs(corr_val) > 0.5:
                        strong_corrs.append((col1, col2, corr_val))
            
            strong_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            for col1, col2, val in strong_corrs[:5]:
                summary_parts.append(f"  • {col1} ↔ {col2}: {val:.3f}")
        
        return "\n".join(summary_parts)
    
    @staticmethod
    def get_column_insights(analysis_context: Dict, column_name: str) -> str:
        """Get insights for specific column (legacy method)"""
        if not analysis_context:
            return f"No analysis data available for '{column_name}'"
        
        insights = [f"Insights for '{column_name}':"]
        
        if "distributions" in analysis_context and column_name in analysis_context["distributions"]:
            stats = analysis_context["distributions"][column_name]
            insights.append(f"\n📊 Distribution:")
            insights.append(f"  • Mean: {stats.get('mean', 0):.2f}")
            insights.append(f"  • Median: {stats.get('median', 0):.2f}")
            insights.append(f"  • Std Dev: {stats.get('std', 0):.2f}")
        
        return "\n".join(insights)


# Global instance for easy access
unified_memory = UnifiedMemoryManager()
# agent.py - Complete version with proper tool connections
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
import os
import streamlit as st
from typing import Dict, Any, Optional
import pandas as pd

# Import tool classes
from tools.data_ingestion1 import DataIngestionTool
from tools.data_validation1 import DataValidationTool
from tools.data_cleaning import DataCleaningTool  # ⭐ ENHANCED VERSION
from tools.eda import EDAAnalyzer
from tools.memory_utils import AnalysisContextManager, UnifiedMemoryManager

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=openai_api_key,
    temperature=0.1,
    max_tokens=2048
)

# Initialize tool instances (singleton pattern - one instance for entire session)
data_ingestion_tool = DataIngestionTool()
data_validation_tool = DataValidationTool()
data_cleaning_tool = DataCleaningTool()
eda_analyzer = EDAAnalyzer()
context_manager = AnalysisContextManager()
unified_memory = UnifiedMemoryManager()


# ==================== TOOL WRAPPER FUNCTIONS ====================
# These connect the agent to the tools via Streamlit session state

def smart_data_ingestion(query: str) -> str:
    """
    Check if data is loaded in session state from Streamlit upload
    
    Data flow: 
    app.py uploads file → stores in st.session_state['df'] → this function reads it
    """
    try:
        # Check session state for uploaded data
        if 'df' in st.session_state and st.session_state['df'] is not None:
            df = st.session_state['df']
            
            # Update unified memory
            unified_memory.update_ingestion_context({
                'data_source': 'streamlit_upload',
                'data_format': st.session_state.get('data_format', 'CSV'),
                'row_count': len(df),
                'column_count': len(df.columns),
                'columns': df.columns.tolist(),
                'file_name': st.session_state.get('file_name', 'uploaded_data')
            })
            
            # Create detailed summary
            summary = f"✅ Data loaded successfully!\n\n"
            summary += f"📊 Dataset: {st.session_state.get('file_name', 'Unknown')}\n"
            summary += f"📏 Size: {len(df):,} rows × {len(df.columns)} columns\n"
            summary += f"💾 Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n"
            summary += f"📋 Columns ({len(df.columns)}):\n"
            summary += f"{', '.join(df.columns.tolist()[:10])}"
            if len(df.columns) > 10:
                summary += f"... and {len(df.columns) - 10} more"
            
            return summary
        else:
            return "⚠️ No data loaded. Please go to the 'Data Upload' page and upload a CSV or JSON file first."
            
    except Exception as e:
        unified_memory.log_operation('ingestion', f"Error: {str(e)}", False, str(e))
        return f"❌ Error checking data: {str(e)}\n\nPlease upload data in the Data Upload page."


def smart_data_validation(query: str) -> str:
    """
    Validate data from session state
    
    Data flow:
    st.session_state['df'] → data_validation_tool.validate() → results
    """
    try:
        # Get DataFrame from session state
        df = st.session_state.get('df')
        
        if df is None:
            return "⚠️ No data found. Please upload data first using the Data Upload page."
        
        # Validate using the tool
        print(f"\n🔍 Validating dataset: {st.session_state.get('file_name', 'unknown')}")
        
        validation_report = data_validation_tool.validate(
            input_data=df,
            dataset_name=st.session_state.get('file_name', 'uploaded_dataset')
        )
        
        # Store validation report in session state
        st.session_state['validation_report'] = validation_report
        
        # Update unified memory
        unified_memory.update_validation_context(validation_report)
        
        # Parse query to determine what to show
        query_lower = query.lower()
        
        # Specific query handling
        if 'missing' in query_lower:
            missing_info = validation_report.get('missing_values', {})
            if missing_info:
                result = f"📊 Missing Values Analysis:\n\n"
                result += f"Total missing: {missing_info.get('total_missing', 0):,} cells "
                result += f"({missing_info.get('missing_percentage', 0):.2f}%)\n\n"
                
                cols_with_missing = missing_info.get('columns_with_missing', {})
                if cols_with_missing:
                    result += f"Affected columns ({len(cols_with_missing)}):\n"
                    for col, count in list(cols_with_missing.items())[:10]:
                        pct = (count / len(df)) * 100
                        result += f"• {col}: {count:,} ({pct:.2f}%)\n"
                    if len(cols_with_missing) > 10:
                        result += f"... and {len(cols_with_missing) - 10} more columns\n"
                
                return result
            else:
                return "✅ No missing values found!"
        
        elif 'duplicate' in query_lower:
            dup_info = validation_report.get('duplicates', {})
            if dup_info:
                result = f"📊 Duplicate Analysis:\n\n"
                result += f"Duplicate rows: {dup_info.get('total_duplicates', 0):,} "
                result += f"({dup_info.get('duplicate_percentage', 0):.2f}%)\n"
                result += f"Unique rows: {dup_info.get('unique_rows', 0):,}\n"
                return result
            else:
                return "✅ No duplicate rows found!"
        
        elif 'outlier' in query_lower:
            outliers = validation_report.get('outliers', {})
            if outliers:
                iqr_outliers = outliers.get('iqr', {})
                if iqr_outliers:
                    result = f"📊 Outlier Analysis (IQR Method):\n\n"
                    result += f"Columns with outliers: {len(iqr_outliers)}\n\n"
                    
                    for col, info in list(iqr_outliers.items())[:10]:
                        result += f"• {col}: {info['count']} outliers ({info['percentage']:.2f}%)\n"
                        result += f"  Range: {info['outlier_range']}\n"
                    
                    if len(iqr_outliers) > 10:
                        result += f"\n... and {len(iqr_outliers) - 10} more columns"
                    
                    return result
                else:
                    return "✅ No outliers detected!"
            else:
                return "✅ No outliers detected!"
        
        else:
            # General validation summary
            total_issues = validation_report.get('total_issues', 0)
            status = validation_report.get('validation_status', 'UNKNOWN')
            
            if total_issues == 0:
                return "✅ Validation PASSED!\n\nNo data quality issues found. Your dataset is clean and ready for analysis."
            else:
                summary = f"📊 Validation Report\n\n"
                summary += f"Status: {status}\n"
                summary += f"Issues found: {total_issues} categories\n\n"
                summary += "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                
                # Missing values
                if validation_report.get('missing_values'):
                    mv = validation_report['missing_values']
                    summary += f"❓ Missing Values:\n"
                    summary += f"   {mv.get('total_missing', 0):,} cells ({mv.get('missing_percentage', 0):.2f}%)\n"
                    summary += f"   Affected: {len(mv.get('columns_with_missing', {}))} columns\n\n"
                
                # Duplicates
                if validation_report.get('duplicates'):
                    dup = validation_report['duplicates']
                    summary += f"🔄 Duplicates:\n"
                    summary += f"   {dup.get('total_duplicates', 0):,} rows ({dup.get('duplicate_percentage', 0):.2f}%)\n\n"
                
                # Outliers
                if validation_report.get('outliers'):
                    outliers_dict = validation_report['outliers']
                    iqr_count = len(outliers_dict.get('iqr', {}))
                    if iqr_count > 0:
                        summary += f"📊 Outliers:\n"
                        summary += f"   Detected in {iqr_count} columns\n\n"
                
                # Text issues
                if validation_report.get('text_issues'):
                    summary += f"📝 Text Issues:\n"
                    summary += f"   Found in {len(validation_report['text_issues'])} columns\n\n"
                
                summary += "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                summary += "\n💡 Tip: Use 'Clean the data' to fix these issues automatically."
                
                return summary
                
    except Exception as e:
        unified_memory.log_operation('validation', f"Error: {str(e)}", False, str(e))
        return f"❌ Validation error: {str(e)}\n\nPlease ensure data is loaded correctly."


def smart_data_cleaning(query: str) -> str:
    """
    Clean data from session state based on natural language query
    
    Data flow:
    st.session_state['df'] → data_cleaning_tool.clean_data() → cleaned df → st.session_state['df']
    """
    try:
        df = st.session_state.get('df')
        
        if df is None:
            return "⚠️ No data found. Please upload data first."
        
        query_lower = query.lower()
        
        # Parse query to determine operations
        operations = {}
        
        # Duplicates
        if any(word in query_lower for word in ['duplicate', 'duplicates', 'remove duplicate']):
            operations['remove_duplicates'] = True
        
        # Missing values
        if any(word in query_lower for word in ['missing', 'null', 'nan', 'fill', 'impute']):
            if 'mean' in query_lower:
                operations['handle_missing'] = 'mean'
            elif 'median' in query_lower:
                operations['handle_missing'] = 'median'
            elif 'mode' in query_lower:
                operations['handle_missing'] = 'mode'
            elif 'drop' in query_lower:
                operations['handle_missing'] = 'drop'
            else:
                operations['handle_missing'] = 'auto'
        
        # Outliers
        if any(word in query_lower for word in ['outlier', 'outliers', 'anomal']):
            if 'remove' in query_lower:
                operations['handle_outliers'] = {'method': 'iqr', 'action': 'remove'}
            elif 'cap' in query_lower or 'clip' in query_lower:
                operations['handle_outliers'] = {'method': 'iqr', 'action': 'cap'}
            else:
                operations['handle_outliers'] = {'method': 'iqr', 'action': 'cap'}
        
        # Text cleaning
        text_ops = []
        if 'lowercase' in query_lower or 'lower case' in query_lower:
            text_ops.append('lowercase')
        if 'trim' in query_lower or 'whitespace' in query_lower:
            text_ops.extend(['trim', 'remove_extra_spaces'])
        if 'special' in query_lower:
            text_ops.append('remove_special')
        
        if text_ops or 'text' in query_lower or 'clean text' in query_lower:
            operations['clean_text'] = text_ops if text_ops else ['lowercase', 'trim', 'remove_extra_spaces']
        
        # Encoding
        if any(word in query_lower for word in ['encode', 'encoding', 'categorical']):
            if 'onehot' in query_lower or 'one hot' in query_lower:
                operations['encode_categorical'] = 'onehot'
            elif 'label' in query_lower:
                operations['encode_categorical'] = 'label'
            else:
                operations['encode_categorical'] = 'auto'
        
        # Standardization
        if any(word in query_lower for word in ['standard', 'normalize', 'scale']):
            operations['standardize'] = True
        
        # If no operations specified, do comprehensive cleaning
        if not operations:
            operations = {
                'remove_duplicates': True,
                'handle_missing': 'auto',
                'handle_outliers': {'method': 'iqr', 'action': 'cap'},
                'clean_text': ['lowercase', 'trim', 'remove_extra_spaces']
            }
        
        # Store original metrics
        rows_before = len(df)
        cols_before = len(df.columns)
        
        print(f"\n🧹 Cleaning data with operations: {list(operations.keys())}")
        
        # Perform cleaning
        df_clean = data_cleaning_tool.clean_data(
            df=df,
            operations=operations,
            dataset_name=st.session_state.get('file_name', 'dataset')
        )
        
        # Update session state with cleaned data
        st.session_state['df'] = df_clean
        
        # Store metrics
        rows_after = len(df_clean)
        cols_after = len(df_clean.columns)
        
        # Update unified memory
        cleaning_context = {
            'rows_before': rows_before,
            'rows_after': rows_after,
            'columns_before': cols_before,
            'columns_after': cols_after,
            'operations': [
                {
                    'type': k,
                    'details': str(v),
                    'columns': [],
                    'count': 0
                }
                for k, v in operations.items()
            ],
            'summary': f"Cleaned {rows_before} → {rows_after} rows"
        }
        unified_memory.update_cleaning_context(cleaning_context)
        
        # Generate detailed summary
        summary = f"✅ Data Cleaning Complete!\n\n"
        summary += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        summary += f"📊 Results:\n"
        summary += f"   Rows: {rows_before:,} → {rows_after:,}"
        
        if rows_before != rows_after:
            removed = rows_before - rows_after
            summary += f" ({removed:,} removed)\n"
        else:
            summary += "\n"
        
        summary += f"   Columns: {cols_before} → {cols_after}\n\n"
        
        summary += f"🔧 Operations Performed ({len(operations)}):\n\n"
        
        if operations.get('remove_duplicates'):
            summary += "   ✓ Removed duplicate rows\n"
        
        if operations.get('handle_missing'):
            strategy = operations['handle_missing']
            summary += f"   ✓ Handled missing values (strategy: {strategy})\n"
        
        if operations.get('handle_outliers'):
            outlier_config = operations['handle_outliers']
            method = outlier_config.get('method', 'iqr')
            action = outlier_config.get('action', 'cap')
            summary += f"   ✓ Handled outliers (method: {method}, action: {action})\n"
        
        if operations.get('clean_text'):
            summary += f"   ✓ Cleaned text columns\n"
        
        if operations.get('encode_categorical'):
            encoding = operations['encode_categorical']
            summary += f"   ✓ Encoded categorical variables ({encoding})\n"
        
        if operations.get('standardize'):
            summary += "   ✓ Standardized numeric features\n"
        
        summary += f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        summary += f"\n💾 Cleaned data saved to session. Ready for EDA!"
        
        return summary
        
    except Exception as e:
        unified_memory.log_operation('cleaning', f"Error: {str(e)}", False, str(e))
        return f"❌ Cleaning error: {str(e)}"


def smart_eda(query: str) -> str:
    """
    Perform EDA on data from session state
    
    Data flow:
    st.session_state['df'] → eda_analyzer.perform_comprehensive_eda() → insights
    """
    try:
        df = st.session_state.get('df')
        
        if df is None:
            return "⚠️ No data found. Please upload data first."
        
        query_lower = query.lower()
        
        # Perform comprehensive EDA
        print(f"\n📊 Performing EDA on: {st.session_state.get('file_name', 'dataset')}")
        
        context = eda_analyzer.perform_comprehensive_eda(df)
        
        # Store in session state
        st.session_state['analysis_context'] = context
        
        # Update unified memory
        unified_memory.update_eda_context(context)
        
        # Answer based on query
        if 'correlation' in query_lower:
            corr_info = context.get('correlations', {})
            if corr_info:
                # Find strong correlations
                top_corrs = []
                for col1, correlations in corr_info.items():
                    for col2, corr_val in correlations.items():
                        if abs(corr_val) > 0.5:
                            top_corrs.append((col1, col2, corr_val))
                
                top_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
                
                if top_corrs:
                    result = f"📊 Top Correlations (|r| > 0.5):\n\n"
                    for i, (col1, col2, corr) in enumerate(top_corrs[:10], 1):
                        strength = "Strong" if abs(corr) > 0.7 else "Moderate"
                        direction = "positive" if corr > 0 else "negative"
                        result += f"{i}. {col1} ↔ {col2}\n"
                        result += f"   r = {corr:.3f} ({strength} {direction})\n\n"
                    
                    if len(top_corrs) > 10:
                        result += f"... and {len(top_corrs) - 10} more correlations\n"
                    
                    return result
                else:
                    return "No strong correlations found (threshold: |r| > 0.5)"
            else:
                return "⚠️ No correlation data available. Need at least 2 numeric columns."
        
        elif 'outlier' in query_lower:
            outlier_info = context.get('outliers', {})
            if outlier_info:
                result = f"📊 Outlier Analysis:\n\n"
                result += f"Columns with outliers: {len(outlier_info)}\n\n"
                
                for col, info in list(outlier_info.items())[:15]:
                    result += f"• {col}:\n"
                    result += f"  {info['count']} outliers ({info['percentage']:.2f}%)\n"
                
                if len(outlier_info) > 15:
                    result += f"\n... and {len(outlier_info) - 15} more columns\n"
                
                return result
            else:
                return "✅ No outliers detected in any column."
        
        elif 'distribution' in query_lower or 'skew' in query_lower:
            dist_info = context.get('distributions', {})
            if dist_info:
                result = f"📊 Distribution Analysis:\n\n"
                
                for col, stats in list(dist_info.items())[:10]:
                    skew = stats['skewness']
                    if abs(skew) < 0.5:
                        skew_desc = "Symmetric"
                    elif skew > 0.5:
                        skew_desc = "Right-skewed (positively skewed)"
                    else:
                        skew_desc = "Left-skewed (negatively skewed)"
                    
                    result += f"• {col}:\n"
                    result += f"  Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}\n"
                    result += f"  Skewness: {skew:.3f} ({skew_desc})\n\n"
                
                if len(dist_info) > 10:
                    result += f"... and {len(dist_info) - 10} more columns\n"
                
                return result
            else:
                return "⚠️ No distribution data available (need numeric columns)."
        
        elif 'missing' in query_lower:
            missing_info = context.get('missing_values', {})
            if missing_info:
                total_missing = sum(missing_info.values())
                result = f"📊 Missing Values:\n\n"
                result += f"Total missing: {total_missing:,} cells\n"
                result += f"Affected columns: {len(missing_info)}\n\n"
                
                for col, count in list(missing_info.items())[:15]:
                    pct = (count / context['dataset_shape'][0]) * 100
                    result += f"• {col}: {count:,} ({pct:.2f}%)\n"
                
                if len(missing_info) > 15:
                    result += f"\n... and {len(missing_info) - 15} more columns\n"
                
                return result
            else:
                return "✅ No missing values in the dataset."
        
        elif 'summary' in query_lower or 'overview' in query_lower:
            # Comprehensive summary
            return eda_analyzer.generate_text_summary(df)
        
        else:
            # General EDA overview
            shape = context['dataset_shape']
            n_numeric = len(context.get('numeric_columns', []))
            n_categorical = len(context.get('categorical_columns', []))
            
            summary = f"📊 Dataset Overview\n\n"
            summary += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            summary += f"📏 Shape: {shape[0]:,} rows × {shape[1]} columns\n\n"
            summary += f"📊 Column Types:\n"
            summary += f"   • Numeric: {n_numeric}\n"
            summary += f"   • Categorical: {n_categorical}\n\n"
            
            # Missing values
            missing_info = context.get('missing_values', {})
            if missing_info:
                total_missing = sum(missing_info.values())
                summary += f"❓ Missing Values:\n"
                summary += f"   {total_missing:,} cells in {len(missing_info)} columns\n\n"
            else:
                summary += f"✅ No missing values\n\n"
            
            # Outliers
            outlier_info = context.get('outliers', {})
            if outlier_info:
                summary += f"📊 Outliers:\n"
                summary += f"   Detected in {len(outlier_info)} columns\n\n"
            else:
                summary += f"✅ No outliers detected\n\n"
            
            summary += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            summary += f"💡 Ask me:\n"
            summary += f"   • 'Show correlations'\n"
            summary += f"   • 'Analyze distributions'\n"
            summary += f"   • 'Show outlier details'\n"
            
            return summary
            
    except Exception as e:
        return f"❌ EDA error: {str(e)}"



def smart_feature_engineering(query: str) -> str:
    """
    Answer questions about feature engineering results from session state.

    Reads:
      st.session_state['fe_report']      — FE summary (features kept/dropped/created)
      st.session_state['fe_done']        — whether FE has been run
      st.session_state['fe_suggestions'] — importance scores from analyse_features
      st.session_state['df_fe']          — the engineered dataframe

    Data flow:
      app.py runs analyse_features() + apply_feature_engineering()
      → saves results to session_state → this function reads them
    """
    try:
        fe_done      = st.session_state.get("fe_done", False)
        fe_report    = st.session_state.get("fe_report", {})
        fe_suggestions = st.session_state.get("fe_suggestions", {})
        df_fe        = st.session_state.get("df_fe")
        query_lower  = query.lower()

        # Fallback: try to load from memory if session state is missing
        if (not fe_done or not fe_report):
            try:
                from tools.feature_engineering_memory import FEMemoryManager
                _fe_m    = FEMemoryManager()
                _session = _fe_m.get_latest_session()
                if _session:
                    _summary = _fe_m.get_session_summary(_session["session_id"])
                    sel  = [f["feature_name"] for f in _summary.get("selected_features", [])]
                    drop = [f["feature_name"] for f in _summary.get("dropped_features",  [])]
                    cre  = [f["feature_name"] for f in _summary.get("created_features",  [])]
                    enc  = [s["operation"]    for s in _summary.get("scaling_ops",       [])]
                    return (
                        f"⚙️ Feature Engineering (from memory)\n\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"🎯 Target  : {_session.get('target_column','N/A')}\n"
                        f"📋 Problem : {_session.get('problem_type','N/A')}\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"✅ Selected : {len(sel)} features\n"
                        f"🗑️ Dropped  : {len(drop)} features\n"
                        f"✨ Created  : {len(cre)} features\n"
                        f"🔢 Encoded  : {len(enc)} operations\n"
                        f"\nNote: Re-run FE page to see full interactive report."
                    )
            except Exception:
                pass
            return (
                "⚠️ Feature Engineering has not been run yet.\n\n"
                "Please go to the ⚙️ Feature Engineering page, select a target column, "
                "click 'Analyse & Get Suggestions', review the suggestions, "
                "then click 'Apply & Finalise Feature Engineering'."
            )

        # ── importance / which feature ────────────────────────────────
        if any(w in query_lower for w in ["importance", "important", "top feature",
                                           "best feature", "highest score", "rf score"]):
            imp_df = fe_report.get("importance_df")
            if imp_df is not None and not imp_df.empty:
                result = "📊 Feature Importance (Random Forest):\n\n"
                for _, row in imp_df.head(10).iterrows():
                    prs = row.get("pearson_r")
                    prs_str = f"{prs:+.4f}" if prs is not None else "N/A"
                    result += (f"  {int(row['rank']):>2}. {row['feature']:<30} "
                               f"RF={row['rf_score']:.4f}  Pearson r={prs_str}\n")
                return result
            return "No importance scores available. Run analysis first."

        # ── which features were selected / kept ───────────────────────
        if any(w in query_lower for w in ["selected", "kept", "which feature",
                                           "final feature", "chosen"]):
            sel = fe_report.get("selected_features", [])
            if sel:
                result = f"✅ {len(sel)} features selected for model training:\n\n"
                for f in sel:
                    result += f"  • {f}\n"
                return result
            return "No feature selection data found."

        # ── dropped features ─────────────────────────────────────────
        if any(w in query_lower for w in ["drop", "removed", "excluded", "why"]):
            dropped = fe_report.get("dropped_features", [])
            drop_info = fe_suggestions.get("drop", []) if fe_suggestions else []
            if dropped:
                result = f"🗑️ {len(dropped)} features were dropped:\n\n"
                for col in dropped:
                    reason_entry = next(
                        (d for d in drop_info if d.get("name") == col), None)
                    reason = reason_entry.get("reason", "user decision") if reason_entry else "user decision"
                    detail = reason_entry.get("detail", "") if reason_entry else ""
                    result += f"  • {col}  ({reason}"
                    if detail:
                        result += f": {detail}"
                    result += ")\n"
                return result
            return "No features were dropped."

        # ── created features ─────────────────────────────────────────
        if any(w in query_lower for w in ["create", "new feature", "engineered",
                                           "generated", "log", "polynomial"]):
            created = fe_report.get("created_features", [])
            if created:
                result = f"✨ {len(created)} new features created:\n\n"
                for f in created:
                    result += f"  • {f}\n"
                return result
            return "No new features were created in this run."

        # ── encoding ─────────────────────────────────────────────────
        if any(w in query_lower for w in ["encod", "categorical", "one-hot", "label"]):
            encoded = fe_report.get("encoded", [])
            if encoded:
                result = f"🔢 Encoding applied to {len(encoded)} columns:\n\n"
                for e in encoded:
                    result += f"  • {e}\n"
                return result
            return "No encoding was applied."

        # ── scaling ──────────────────────────────────────────────────
        if any(w in query_lower for w in ["scal", "normaliz", "standard"]):
            scaled = fe_report.get("scaled", [])
            problem = fe_report.get("problem_type", "")
            if scaled:
                return (f"📏 Scaling applied to {len(scaled)} numeric features\n"
                        f"  Problem type: {problem}\n"
                        f"  Columns: {', '.join(scaled[:10])}"
                        f"{'...' if len(scaled) > 10 else ''}")
            return "No scaling was applied."

        # ── class imbalance ──────────────────────────────────────────
        if any(w in query_lower for w in ["imbalanc", "smote", "oversample",
                                           "undersample", "class"]):
            imbalance = fe_report.get("imbalance", [])
            if imbalance:
                result = f"⚖️ Class Imbalance Handling:\n\n"
                for item in imbalance:
                    result += f"  • {item}\n"
                return result
            return "No class imbalance handling was applied."

        # ── errors ───────────────────────────────────────────────────
        if "error" in query_lower:
            errors = fe_report.get("errors", [])
            if errors:
                result = f"⚠️ {len(errors)} error(s) during feature engineering:\n\n"
                for e in errors:
                    result += f"  • {e}\n"
                return result
            return "✅ No errors during feature engineering."

        # ── shape / size / download ──────────────────────────────────
        if any(w in query_lower for w in ["shape", "size", "rows", "columns",
                                           "ready", "download", "csv"]):
            if df_fe is not None:
                result = (
                    f"📊 Engineered dataset shape: "
                    f"{df_fe.shape[0]:,} rows × {df_fe.shape[1]} columns\n"
                    f"  Target column : {fe_report.get('target_column', 'N/A')}\n"
                    f"  Problem type  : {fe_report.get('problem_type', 'N/A')}\n"
                    f"  Features      : {len(fe_report.get('selected_features', []))}\n"
                    f"\n✅ Data is ready for model training.\n"
                    f"   Go to ⚙️ Feature Engineering → Preview tab → "
                    f"'Download Engineered CSV' button."
                )
                # Also check memory
                try:
                    from tools.feature_engineering_memory import FEMemoryManager
                    _fe_m = FEMemoryManager()
                    _sid  = fe_report.get("session_id", "")
                    _df   = _fe_m.get_engineered_df(_sid) if _sid else None
                    if _df is not None:
                        result += f"\n   Memory-verified shape: {_df.shape[0]:,}×{_df.shape[1]}"
                except Exception:
                    pass
                return result
            return "Engineered dataframe not found. Run Feature Engineering first."

        # ── general summary ───────────────────────────────────────────
        target = fe_report.get("target_column", "N/A")
        problem = fe_report.get("problem_type", "N/A")
        selected = fe_report.get("selected_features", [])
        dropped = fe_report.get("dropped_features", [])
        created = fe_report.get("created_features", [])
        encoded = fe_report.get("encoded", [])
        scaled  = fe_report.get("scaled", [])

        summary  = "⚙️ Feature Engineering Summary\n\n"
        summary += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        summary += f"🎯 Target column : {target}\n"
        summary += f"📋 Problem type  : {problem}\n"
        if df_fe is not None:
            summary += f"📐 Output shape  : {df_fe.shape[0]:,} rows × {df_fe.shape[1]} cols\n"
        summary += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        summary += f"✅ Features selected : {len(selected)}\n"
        summary += f"🗑️ Features dropped  : {len(dropped)}\n"
        summary += f"✨ Features created  : {len(created)}\n"
        summary += f"🔢 Columns encoded  : {len(encoded)}\n"
        summary += f"📏 Columns scaled   : {len(scaled)}\n"
        summary += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        summary += "💡 Ask me: 'which features were selected', "
        summary += "'why was X dropped', 'what new features were created', "
        summary += "'show importance scores'"
        return summary

    except Exception as e:
        return f"❌ Feature engineering query error: {str(e)}"




def smart_model_training(query: str) -> str:
    """
    Answer questions about model training, evaluation, and explainability
    results from session state and SQLite memory.
    """
    try:
        mt_done      = st.session_state.get("mt_done", False)
        mt_results   = st.session_state.get("mt_results", {})
        best_model   = st.session_state.get("mt_best_model", "")
        problem_type = st.session_state.get("mt_problem_type", "")
        target_col   = st.session_state.get("mt_target_col", "")
        session_id   = st.session_state.get("mt_session_id", "")
        query_lower  = query.lower()

        # ── fallback: load from memory if session state gone ─────────
        if not mt_done or not mt_results:
            try:
                from tools.model_memory import ModelMemoryManager
                _mem     = ModelMemoryManager()
                _session = _mem.get_latest_session()
                if _session:
                    _sid     = _session["session_id"]
                    _models  = _mem.get_session_models(_sid)
                    _metrics = _mem.get_session_metrics(_sid)
                    result   = (
                        f"🤖 Model Training (from memory)\n\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"🎯 Target   : {_session.get('target_column','N/A')}\n"
                        f"📋 Problem  : {_session.get('problem_type','N/A')}\n"
                        f"🏆 Best     : {_session.get('best_model_name','N/A')}\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"Models trained: {len(_models)}\n"
                    )
                    for m in _models[:5]:
                        result += (f"  • {m['model_name']:<30} "
                                   f"train={m.get('train_score',0):.4f}  "
                                   f"test={m.get('test_score',0):.4f}  "
                                   f"status={m.get('fit_status','?')}\n")
                    return result
            except Exception:
                pass
            return (
                "⚠️ Model Training has not been run yet.\n\n"
                "Please go to the 🤖 Model Training page, get recommendations, "
                "select models, configure settings, and click Train."
            )

        # ── best model ────────────────────────────────────────────────
        if any(w in query_lower for w in ["best", "winner", "top model",
                                           "which model", "highest"]):
            if best_model and best_model in mt_results:
                res = mt_results[best_model]
                m   = res.get("metrics", {})
                result = (
                    f"🏆 Best Model: <b>{best_model}</b>\n\n"
                    f"  Train score : {res.get('train_score',0):.4f}\n"
                    f"  Test score  : {res.get('test_score',0):.4f}\n"
                    f"  Fit status  : {res.get('fit_status','good')}\n"
                )
                if problem_type == "classification":
                    result += (
                        f"  Accuracy   : {m.get('accuracy',0):.4f}\n"
                        f"  F1 Score   : {m.get('f1_score',0):.4f}\n"
                        f"  ROC-AUC    : {m.get('roc_auc','N/A')}\n"
                    )
                else:
                    result += (
                        f"  MAE  : {m.get('mae',0):.4f}\n"
                        f"  RMSE : {m.get('rmse',0):.4f}\n"
                        f"  R²   : {m.get('r2',0):.4f}\n"
                    )
                return result
            return f"Best model: {best_model or 'not determined yet.'}"

        # ── metrics for specific model ────────────────────────────────
        if any(w in query_lower for w in ["f1", "accuracy", "rmse", "r2",
                                           "mae", "roc", "precision",
                                           "recall", "metric", "score",
                                           "performance"]):
            result = f"📊 Model Performance ({problem_type}):\n\n"
            for model_name, res in mt_results.items():
                if "error" in res:
                    result += f"  ❌ {model_name}: {res['error']}\n"
                    continue
                m = res.get("metrics", {})
                result += f"  📌 {model_name}\n"
                result += (f"     Train: {res.get('train_score',0):.4f}  "
                           f"Test: {res.get('test_score',0):.4f}  "
                           f"Status: {res.get('fit_status','?')}\n")
                if problem_type == "classification":
                    result += (f"     Acc={m.get('accuracy',0):.4f}  "
                               f"F1={m.get('f1_score',0):.4f}  "
                               f"Precision={m.get('precision',0):.4f}  "
                               f"Recall={m.get('recall',0):.4f}\n")
                    if m.get("roc_auc"):
                        result += f"     ROC-AUC={m['roc_auc']:.4f}\n"
                else:
                    result += (f"     MAE={m.get('mae',0):.4f}  "
                               f"RMSE={m.get('rmse',0):.4f}  "
                               f"R²={m.get('r2',0):.4f}\n")
                result += "\n"
            return result

        # ── overfitting / underfitting ────────────────────────────────
        if any(w in query_lower for w in ["overfit", "underfit", "fit status",
                                           "gap", "generaliz"]):
            result = "🔍 Fit Analysis:\n\n"
            for model_name, res in mt_results.items():
                if "error" in res: continue
                fit = res.get("fit_status", "good")
                badge = {"overfit":"🔴 Overfitting","underfit":"🟡 Underfitting",
                         "good":"🟢 Good Fit"}.get(fit, fit)
                result += (f"  {badge} — {model_name}\n"
                           f"    Train: {res.get('train_score',0):.4f}  "
                           f"Test: {res.get('test_score',0):.4f}  "
                           f"Gap: {res.get('train_score',0)-res.get('test_score',0):+.4f}\n\n")
            return result

        # ── SHAP / feature importance ─────────────────────────────────
        if any(w in query_lower for w in ["shap", "important", "feature contrib",
                                           "explainab", "interpret", "lime"]):
            try:
                from tools.model_memory import ModelMemoryManager
                _mem   = ModelMemoryManager()
                top_ft = _mem.get_top_features(session_id, best_model, 10)
                if top_ft:
                    result = (f"🌊 SHAP Feature Importance "
                              f"({best_model}):\n\n")
                    for item in top_ft:
                        result += (f"  {item['rank']:>2}. "
                                   f"{item['feature_name']:<30} "
                                   f"{item['importance_value']:.4f}\n")
                    return result
            except Exception:
                pass
            return ("SHAP values not yet computed. Run Step 8 (Explainability) "
                    "on the Model Training page to generate SHAP plots.")

        # ── models trained ────────────────────────────────────────────
        if any(w in query_lower for w in ["model", "train", "which model",
                                           "how many model", "all model"]):
            ok  = [n for n, r in mt_results.items() if "error" not in r]
            err = [n for n, r in mt_results.items() if "error" in r]
            result = (f"🤖 Models Trained: {len(ok)}\n\n")
            for n in ok:
                result += f"  ✅ {n}\n"
            if err:
                result += f"\n  ❌ Failed: {', '.join(err)}\n"
            result += f"\n🏆 Best model: {best_model}"
            return result

        # ── general summary ───────────────────────────────────────────
        ok_models = [n for n, r in mt_results.items() if "error" not in r]
        summary   = (
            f"🤖 Model Training Summary\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 Target    : {target_col}\n"
            f"📋 Problem   : {problem_type}\n"
            f"🏆 Best model: {best_model}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Models trained : {len(ok_models)}\n"
        )
        if ok_models:
            summary += f"Models         : {', '.join(ok_models)}\n"
        summary += (
            "\n💡 Ask me:\n"
            "  'Which model performed best?'\n"
            "  'What was the F1 score of each model?'\n"
            "  'Was any model overfitting?'\n"
            "  'Which feature had highest SHAP importance?'"
        )
        return summary

    except Exception as exc:
        return f"❌ Model training query error: {str(exc)}"

# ==================== DEFINE LANGCHAIN TOOLS ====================

tools = [
    Tool(
        name="Model_Training",
        func=smart_model_training,
        description="""Answer questions about model training, evaluation, and explainability.
        Use when user asks about: model performance, best model, F1/accuracy/RMSE/R2 scores,
        overfitting/underfitting, confusion matrix, SHAP feature importance, LIME explanations,
        which models were trained, or model comparison.
        Examples: 'which model performed best', 'what was the F1 score of XGBoost',
        'was the model overfitting', 'which feature had highest SHAP importance',
        'show model comparison', 'was there any underfitting'"""
    ),
    Tool(
        name="Feature_Engineering",
        func=smart_feature_engineering,
        description="""Answer questions about feature engineering results.
        Use when user asks about: selected/dropped/created features, 
        RF importance scores, Pearson r, encoding, scaling, SMOTE,
        class imbalance, engineered dataset shape, or readiness for training.
        Examples: 'which features were selected', 'why was column X dropped',
        'show feature importance', 'what new features were created',
        'is data ready for model training'"""
    ),
    Tool(
        name="Check_Data_Loaded",
        func=smart_data_ingestion,
        description="""Check if data is loaded from Streamlit file upload.
        Use this FIRST to verify data is available.
        Returns dataset summary if data exists."""
    ),
    Tool(
        name="Validate_Data",
        func=smart_data_validation,
        description="""Validate data quality and identify issues.
        Checks: missing values, duplicates, outliers, text issues.
        Use AFTER confirming data is loaded.
        Examples: 'validate data', 'check for missing values', 'find duplicates'"""
    ),
    Tool(
        name="Clean_Data",
        func=smart_data_cleaning,
        description="""Clean and prepare data automatically.
        Can: remove duplicates, fill missing values, handle outliers, clean text.
        Understands natural language like: 'clean the data', 'remove duplicates and fill missing with mean'.
        Use AFTER validation."""
    ),
    Tool(
        name="Analyze_Data",
        func=smart_eda,
        description="""Perform exploratory data analysis (EDA).
        Provides: distributions, correlations, outliers, statistics.
        Examples: 'show correlations', 'analyze distributions', 'EDA summary'.
        Use AFTER cleaning data."""
    )
]

# ==================== LANGCHAIN MEMORY ====================

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="output",
    k=10  # Remember last 10 exchanges
)

# ==================== LANGCHAIN PROMPT ====================

prompt_template = """You are an expert data science assistant helping users analyze their datasets.

AVAILABLE TOOLS:
{tools}

TOOL NAMES: {tool_names}

CONVERSATION HISTORY:
{chat_history}

CURRENT CONTEXT:
{analysis_context}

USER QUERY: {input}

WORKFLOW:
1. First, check if data is loaded (Check_Data_Loaded)
2. Then validate data quality (Validate_Data)
3. Clean data if issues found (Clean_Data)
4. Analyze data (Analyze_Data)
5. For feature engineering results use (Feature_Engineering)

IMPORTANT:
- Be helpful and conversational
- Use tools to get real data, never make up information
- Explain findings clearly in plain English
- Guide users through the data science workflow

RESPONSE FORMAT:
Thought: [Your reasoning about what to do]
Action: [Tool name]
Action Input: [Input for the tool]
Observation: [Tool result]
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Final Answer: [Your clear, helpful response to the user]

{agent_scratchpad}"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["input", "chat_history", "analysis_context", "agent_scratchpad", "tools", "tool_names"]
)

# ==================== CREATE AGENT ====================

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
    early_stopping_method="generate"
)

# ==================== PUBLIC INTERFACE ====================

def run_agent_with_context(query: str, analysis_context: dict = None) -> str:
    """
    Run agent with query and context
    
    Args:
        query: User's question
        analysis_context: Optional analysis context
        
    Returns:
        Agent's response
    """
    # Get unified context
    full_context = unified_memory.get_full_context()
    context_str = unified_memory.create_agent_context_summary()
    
    # Fallback to analysis context if unified context is empty
    if not full_context or not any(full_context.values()):
        if analysis_context is None:
            analysis_context = st.session_state.get('analysis_context', {})
        
        if analysis_context:
            context_str = context_manager.create_context_summary(analysis_context)
        else:
            context_str = "No data loaded. User should upload CSV/JSON file first."
    
    try:
        result = agent_executor.invoke({
            "input": query,
            "analysis_context": context_str
        })
        return result.get("output", "I apologize, but I couldn't generate a response. Please try rephrasing your question.")
    
    except Exception as e:
        error_msg = str(e)
        
        if "Could not parse LLM output" in error_msg:
            return "I encountered an error processing your request. Could you please rephrase your question?"
        else:
            return f"Error: {error_msg}\n\nPlease try again or rephrase your question."


def run_agent(query: str) -> str:
    """Simple wrapper to run agent"""
    return run_agent_with_context(query)


def clear_agent_memory():
    """Clear all memory and context"""
    memory.clear()
    unified_memory.clear_session_context()
    
    if 'analysis_context' in st.session_state:
        del st.session_state['analysis_context']
    
    if 'validation_report' in st.session_state:
        del st.session_state['validation_report']
    
    return "✅ Memory and context cleared successfully!"


def get_conversation_history() -> list:
    """Get conversation history"""
    try:
        return memory.chat_memory.messages
    except:
        return []


# ==================== INITIALIZATION ====================

if __name__ == "__main__":
    print("="*70)
    print("AGENTIC DATA SCIENTIST - Agent Module")
    print("="*70)
    print("\n✅ Agent initialized successfully!")
    print(f"✅ Tools connected: {len(tools)}")
    print(f"✅ LLM: {llm.model_name}")
    print("\nℹ️  This module connects to:")
    print("   • data_ingestion1.py")
    print("   • data_validation1.py")
    print("   • data_cleaning.py")
    print("   • eda.py")
    print("\n📝 To use: Run 'streamlit run app.py'")
    print("="*70)
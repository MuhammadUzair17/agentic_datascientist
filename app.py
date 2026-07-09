# app.py - Agentic Data Scientist
# NOTE: If you see "ImportError: cannot import name 'dict' from 'pydantic'",
#       run: pip install --upgrade pydantic langchain langchain-openai
#       This is a pydantic v1/v2 compatibility issue in older langchain versions.
import streamlit as st
import pandas as pd
import numpy as np

# ── PyArrow / Streamlit compatibility fix ─────────────────────────────
# Streamlit 1.29 does not recognise PyArrow LargeUtf8 type (arrow 15+).
# This patch converts LargeUtf8 → Utf8 before Streamlit renders dataframes.
try:
    import pyarrow as _pa
    _orig_infer = _pa.Schema.from_pandas if hasattr(_pa.Schema, "from_pandas") else None

    def _safe_df(df, *args, **kwargs):
        """Cast all large_string (LargeUtf8) columns to string (Utf8)."""
        import pandas as _pd
        result = df.copy()
        for col in result.select_dtypes(include="object").columns:
            try:
                result[col] = result[col].astype(str).where(
                    result[col].notna(), other=None)
            except Exception:
                pass
        return result

    # Monkey-patch st.dataframe and st.data_editor to sanitise first
    _orig_dataframe    = st.dataframe
    _orig_data_editor  = st.data_editor

    def _safe_st_dataframe(data=None, *args, **kwargs):
        if isinstance(data, __import__("pandas").DataFrame):
            data = _safe_df(data)
        return _orig_dataframe(data, *args, **kwargs)

    def _safe_st_data_editor(data=None, *args, **kwargs):
        if isinstance(data, __import__("pandas").DataFrame):
            data = _safe_df(data)
        return _orig_data_editor(data, *args, **kwargs)

    st.dataframe   = _safe_st_dataframe
    st.data_editor = _safe_st_data_editor
except Exception:
    pass  # if patch fails, continue normally
# ─────────────────────────────────────────────────────────────────────
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime
import os
from typing import Optional
import base64

# Import tools and agent
from tools.data_ingestion1 import DataIngestionTool
from tools.data_validation1 import DataValidationTool
from tools.data_cleaning import DataCleaningTool
from tools.eda import EDAAnalyzer
from tools.memory_utils import UnifiedMemoryManager
from agent import run_agent_with_context, clear_agent_memory, get_conversation_history
from tools.chat_support import ChatSupportAgent
from tools.pipeline_snapshot import (save_snapshot, load_snapshot_metadata, load_snapshot_pipeline, list_snapshots, delete_snapshot)
from tools.drift_detector import run_drift_check

# Page configuration
st.set_page_config(
    page_title="Agentic Data Scientist",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a1929 0%, #1a2332 50%, #0a1929 100%);
        background-attachment: fixed;
    }
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background-image:
            radial-gradient(circle at 20% 50%, rgba(6, 182, 212, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(139, 92, 246, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 40% 20%, rgba(16, 185, 129, 0.08) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }
    .main .block-container {
        max-width: 1400px !important;
        padding: 2rem 3rem !important;
        background: rgba(15, 23, 42, 0.6);
        border-radius: 24px;
        margin: 2rem auto !important;
        box-shadow: 0 8px 32px 0 rgba(6, 182, 212, 0.15),
                    inset 0 0 0 1px rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(6, 182, 212, 0.2);
    }
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1929 0%, #1a2332 100%);
        border-right: 2px solid rgba(6, 182, 212, 0.3);
        min-width: 280px !important;
        max-width: 280px !important;
    }
    .main-header {
        font-size: 3rem !important;
        font-weight: 900;
        text-align: center;
        padding: 1.5rem 1rem;
        background: linear-gradient(90deg, #06b6d4 0%, #a855f7 25%, #10b981 50%, #06b6d4 75%, #a855f7 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s linear infinite;
        margin-bottom: 0.5rem;
    }
    @keyframes shimmer {
        0%   { background-position: 0% center; }
        100% { background-position: 200% center; }
    }
    .feature-card {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(15, 23, 42, 0.8) 100%);
        padding: 1.8rem;
        border-radius: 16px;
        margin: 1rem 0;
        border: 1px solid rgba(6, 182, 212, 0.25);
        box-shadow: 0 8px 24px rgba(6, 182, 212, 0.15);
        backdrop-filter: blur(10px);
        min-height: 320px;
        max-height: 320px;
        overflow-y: auto;
    }
    .feature-card h3 { color: #06b6d4 !important; margin-bottom: 1.2rem; font-size: 1.3rem !important; font-weight: 700 !important; }
    .metric-card {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
        padding: 1.5rem;
        border-radius: 14px;
        color: white;
        box-shadow: 0 4px 20px rgba(6, 182, 212, 0.25);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(6, 182, 212, 0.3);
        backdrop-filter: blur(10px);
        min-height: 110px;
    }
    .metric-card:hover { transform: translateY(-6px) scale(1.02); box-shadow: 0 10px 35px rgba(6, 182, 212, 0.4); }
    .success-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(6, 182, 212, 0.2) 100%);
        color: #d1fae5; padding: 1rem 1.3rem; border-radius: 12px; margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.25);
        border-left: 4px solid #10b981; border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .warning-box {
        background: linear-gradient(135deg, rgba(251, 146, 60, 0.2) 0%, rgba(245, 158, 11, 0.2) 100%);
        color: #fde68a; padding: 1rem 1.3rem; border-radius: 12px; margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(251, 146, 60, 0.25);
        border-left: 4px solid #f59e0b; border: 1px solid rgba(251, 146, 60, 0.3);
    }
    .info-box {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.2) 0%, rgba(59, 130, 246, 0.2) 100%);
        color: #bfdbfe; padding: 1rem 1.3rem; border-radius: 12px; margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(6, 182, 212, 0.25);
        border-left: 4px solid #06b6d4; border: 1px solid rgba(6, 182, 212, 0.3);
    }
    .error-box {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.2) 100%);
        color: #fecaca; padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(239, 68, 68, 0.3);
        border-left: 4px solid #ef4444; border: 1px solid rgba(239, 68, 68, 0.3);
        font-weight: 500;
    }
    .pipeline-flowchart {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.9) 100%);
        padding: 2rem; border-radius: 16px;
        border: 1px solid rgba(6, 182, 212, 0.3);
        box-shadow: 0 8px 32px rgba(6, 182, 212, 0.2);
        margin: 1rem 0; min-height: 200px;
    }
    .pipeline-step {
        display: inline-block;
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.2) 0%, rgba(8, 145, 178, 0.2) 100%);
        padding: 1rem 1.5rem; border-radius: 12px;
        border: 2px solid rgba(6, 182, 212, 0.4);
        color: #67e8f9; font-weight: 600; font-size: 0.9rem;
        margin: 0.5rem;
        box-shadow: 0 4px 16px rgba(6, 182, 212, 0.3);
        transition: all 0.3s ease; min-width: 140px; text-align: center;
    }
    .pipeline-step:hover { transform: scale(1.05); box-shadow: 0 6px 24px rgba(6, 182, 212, 0.5); }
    .pipeline-step.active {
        background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
        color: white; border-color: #06b6d4;
        box-shadow: 0 6px 24px rgba(6, 182, 212, 0.6);
    }
    .pipeline-step.blocked {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.2) 100%);
        color: #fca5a5; border-color: #ef4444; opacity: 0.6;
    }
    .pipeline-arrow { display: inline-block; color: #06b6d4; font-size: 1.5rem; margin: 0 0.5rem; vertical-align: middle; }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
        color: white; border: none; border-radius: 12px;
        padding: 0.75rem 1.5rem; font-weight: 600; font-size: 0.95rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 16px rgba(6, 182, 212, 0.35);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #0891b2 0%, #06b6d4 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(6, 182, 212, 0.5);
    }
    .chat-message {
        padding: 1.2rem 1.5rem; border-radius: 16px; margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        animation: messageSlide 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px); max-width: 85%;
    }
    @keyframes messageSlide { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    .user-message {
        background: linear-gradient(135deg, rgba(6,182,212,0.25) 0%, rgba(14,165,233,0.25) 100%);
        color: #e0f2fe; border-left: 4px solid #06b6d4; margin-left: auto;
    }
    .assistant-message {
        background: linear-gradient(135deg, rgba(139,92,246,0.25) 0%, rgba(168,85,247,0.25) 100%);
        color: #e9d5ff; border-left: 4px solid #a855f7; margin-right: auto;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px; background: rgba(15,23,42,0.7); border-radius: 14px; padding: 0.6rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(6,182,212,0.1); border-radius: 10px; padding: 0.7rem 1.3rem;
        font-weight: 600; color: #67e8f9; border: 1px solid rgba(6,182,212,0.3);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
        color: white; box-shadow: 0 4px 16px rgba(6,182,212,0.4);
    }
    .stFileUploader { background: rgba(6,182,212,0.05); border-radius: 14px; padding: 1.5rem; border: 2px dashed rgba(6,182,212,0.4); }
    .dataframe { border-radius: 12px; overflow: hidden; box-shadow: 0 4px 16px rgba(0,0,0,0.4); border: 1px solid rgba(6,182,212,0.2); }
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important; font-weight: 800;
        background: linear-gradient(135deg, #06b6d4 0%, #10b981 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    h1, h2, h3, h4, h5, h6 { color: #e2e8f0 !important; }
    p, span, div, li { color: #cbd5e1; }
    .stTextInput input, .stSelectbox select {
        background: rgba(15,23,42,0.8) !important; color: #e2e8f0 !important;
        border: 1px solid rgba(6,182,212,0.3) !important; border-radius: 10px !important;
    }
    .format-badge {
        display: inline-block; padding: 0.4rem 0.8rem; border-radius: 20px;
        font-weight: 700; font-size: 0.8rem; margin: 0.3rem;
        box-shadow: 0 0 12px currentColor; border: 2px solid currentColor;
    }
    .format-csv  { background: rgba(16,185,129,0.2); color: #10b981; }
    .format-json { background: rgba(59,130,246,0.2);  color: #3b82f6; }
    .format-api  { background: rgba(168,85,247,0.2);  color: #a855f7; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════
if 'df'                       not in st.session_state: st.session_state['df']                       = None
if 'data_format'              not in st.session_state: st.session_state['data_format']              = None
if 'file_name'                not in st.session_state: st.session_state['file_name']                = None
if 'chat_history'             not in st.session_state: st.session_state['chat_history']             = []
if 'analysis_context'         not in st.session_state: st.session_state['analysis_context']         = {}
if 'validation_report'        not in st.session_state: st.session_state['validation_report']        = None
if 'validation_expected_format' not in st.session_state: st.session_state['validation_expected_format'] = None
if 'validation_rejected'      not in st.session_state: st.session_state['validation_rejected']      = False
if 'cleaning_done'            not in st.session_state: st.session_state['cleaning_done']            = False
# ── Feature Engineering session state ──────────────────────────────────────
if 'fe_done'        not in st.session_state: st.session_state['fe_done']        = False
if 'fe_phase'       not in st.session_state: st.session_state['fe_phase']       = 1
if 'df_fe'          not in st.session_state: st.session_state['df_fe']          = None
if 'fe_report'      not in st.session_state: st.session_state['fe_report']      = {}
if 'fe_session_id'  not in st.session_state: st.session_state['fe_session_id']  = None
if 'fe_target_col'  not in st.session_state: st.session_state['fe_target_col']  = None
if 'fe_problem_type' not in st.session_state: st.session_state['fe_problem_type'] = 'auto'
if 'fe_suggestions' not in st.session_state: st.session_state['fe_suggestions'] = None
if 'fe_encode_cats' not in st.session_state: st.session_state['fe_encode_cats'] = True
if 'fe_scale_method' not in st.session_state: st.session_state['fe_scale_method'] = 'standard'
# ── Model Training session state ────────────────────────────────────────────
if 'mt_done'            not in st.session_state: st.session_state['mt_done']            = False
if 'mt_results'         not in st.session_state: st.session_state['mt_results']         = {}
if 'mt_session_id'      not in st.session_state: st.session_state['mt_session_id']      = None
if 'mt_best_model'      not in st.session_state: st.session_state['mt_best_model']      = None
if 'mt_problem_type'    not in st.session_state: st.session_state['mt_problem_type']    = None
if 'mt_target_col'      not in st.session_state: st.session_state['mt_target_col']      = None
if 'mt_recommendations' not in st.session_state: st.session_state['mt_recommendations'] = []
if 'mt_test_size'       not in st.session_state: st.session_state['mt_test_size']       = 0.2
if 'mt_mem'             not in st.session_state: st.session_state['mt_mem']             = None

# ══════════════════════════════════════════════════════════════════════════════
# TOOLS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_tools():
    return {
        'ingestion':      DataIngestionTool(),
        'validation':     DataValidationTool(),
        'cleaning':       DataCleaningTool(),
        'eda':            EDAAnalyzer(),
        'unified_memory': UnifiedMemoryManager()
    }

tools          = get_tools()
unified_memory = tools['unified_memory']


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING HELPERS  (agent analysis + apply)
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_pipeline_state():
    full_context = unified_memory.get_full_context()
    return unified_memory._determine_current_state(full_context)


def create_visual_pipeline():
    state               = get_pipeline_state()
    validation_rejected = st.session_state.get('validation_rejected', False)
    mt_done = st.session_state.get('mt_done', False)
    steps = [
        ('📥 Data Ingestion', 'loaded',    False),
        ('🔍 Validation',    'validated',  validation_rejected),
        ('🧹 Cleaning',      'cleaned',    validation_rejected),
        ('📊 EDA',           'analyzed',   validation_rejected),
        ('⚙️ Feature Eng',  'fe',         validation_rejected),
        ('🤖 Model Training','mt',         validation_rejected),
        ('💬 AI Chat',       'mt',         validation_rejected)
    ]
    state_map   = {'initial': -1, 'loaded': 0, 'validated': 1,
                   'cleaned': 2, 'analyzed': 3,
                   'fe': 4, 'mt': 5}
    # derive step index from actual session state flags
    if mt_done:
        current_step = 5
    elif st.session_state.get('fe_done'):
        current_step = 4
    else:
        base = state_map.get(state, -1)
        current_step = base
    html = '<div class="pipeline-flowchart" style="text-align: center;">'
    for i, (step_name, step_state, is_blocked) in enumerate(steps):
        is_active    = i <= current_step and not is_blocked
        active_class = 'blocked' if is_blocked else ('active' if is_active else '')
        html += f'<div class="pipeline-step {active_class}">{step_name}</div>'
        if i < len(steps) - 1:
            html += '<span class="pipeline-arrow">→</span>'
    html += '</div>'
    if validation_rejected:
        html += ('<div class="error-box" style="text-align: center; margin-top: 1rem;">'
                 '⚠️ <b>Pipeline Blocked:</b> Format mismatch. Fix to proceed.'
                 '</div>')
    return html


def get_format_badge(format_type: str) -> str:
    format_type = format_type.upper()
    cls = {'CSV': 'format-csv', 'JSON': 'format-json', 'API': 'format-api'}.get(format_type, 'format-csv')
    return f'<span class="format-badge {cls}">{format_type}</span>'


def display_dataframe_info(df: pd.DataFrame):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📊 Rows", f"{len(df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📋 Columns", f"{len(df.columns)}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("❓ Missing", f"{missing_pct:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("💾 Memory", f"{memory_mb:.1f} MB")
        st.markdown('</div>', unsafe_allow_html=True)


def display_validation_results(validation_report: dict):
    st.subheader("📋 Validation Results")
    if validation_report.get('format_match') == False:
        st.markdown(f'<div class="error-box"><pre>{validation_report.get("message", "Format mismatch detected")}</pre></div>', unsafe_allow_html=True)
        return
    # Recount actual issues (non-zero only) — validation tool counts
    # all categories regardless of whether they have actual issues
    actual_issues = []
    mv  = validation_report.get('missing_values', {})
    dup = validation_report.get('duplicates',     {})
    out = validation_report.get('outliers',       {})

    has_missing    = mv.get('total_missing',    0) > 0
    has_duplicates = dup.get('total_duplicates', 0) > 0
    has_outliers   = bool(out.get('iqr', {}))

    if has_missing:    actual_issues.append("Missing Values")
    if has_duplicates: actual_issues.append("Duplicate Rows")
    if has_outliers:   actual_issues.append("Outliers")

    total_issues = len(actual_issues)

    if total_issues == 0:
        st.markdown(
            '<div class="success-box">✅ <b>Validation Passed!</b> No missing values, no duplicates, no outliers.</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="warning-box">⚠️ <b>Found {total_issues} issue {"category" if total_issues == 1 else "categories"}: {", ".join(actual_issues)}</b></div>',
            unsafe_allow_html=True)

        # Missing Values — only if actually present
        if has_missing:
            with st.expander("❓ Missing Values", expanded=True):
                st.write(f"**Total Missing:** {mv.get('total_missing', 0):,} "
                         f"({mv.get('missing_percentage', 0):.2f}%)")
                if 'columns_with_missing' in mv:
                    st.dataframe(pd.DataFrame([
                        {'Column': col, 'Missing Count': count,
                         '%': f"{count/len(validation_report.get('all_columns',[col]))*100:.1f}%" 
                               if validation_report.get('total_rows',0) > 0 else "—"}
                        for col, count in mv['columns_with_missing'].items()
                    ]), use_container_width=True)

        # Duplicates — only if actually present
        if has_duplicates:
            with st.expander("🔄 Duplicate Rows", expanded=True):
                st.write(f"**Total Duplicates:** {dup.get('total_duplicates', 0):,} "
                         f"({dup.get('duplicate_percentage', 0):.2f}%)")
                st.info("Go to 🧹 Cleaning → remove duplicates to fix this.")

        # Outliers — only if actually present
        if has_outliers:
            with st.expander("🎯 Outliers", expanded=True):
                outlier_list = [
                    {'Column': col,
                     'Outlier Count': info['count'],
                     'Percentage': f"{info['percentage']:.2f}%",
                     'Range': str(info.get('outlier_range',''))}
                    for col, info in out['iqr'].items()]
                if outlier_list:
                    st.dataframe(pd.DataFrame(outlier_list),
                                 use_container_width=True)
                st.info("Go to 🧹 Cleaning → handle outliers to fix this.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.markdown('<h1 class="main-header">🤖 Agentic Data Scientist</h1>', unsafe_allow_html=True)

    if st.session_state.get('data_format'):
        st.markdown(f"Current Format: {get_format_badge(st.session_state['data_format'])}", unsafe_allow_html=True)

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🎯 Navigation")

        page = st.radio(
            "Select Page",
            ["🏠 Home", "📊 Data Upload", "🔍 Validation", "🧹 Cleaning",
             "📈 EDA & Visualization", "⚙️ Feature Engineering",
             "🤖 Model Training",
             "💬 AI Chat Assistant", "📜 Pipeline History",
             "🔄 Pipeline Snapshots"],
            label_visibility="collapsed"
        )

        st.divider()

        if st.session_state['df'] is not None:
            st.markdown("### 📊 Current Dataset")
            df = st.session_state['df']
            if st.session_state.get('data_format'):
                st.markdown(f"Format: {get_format_badge(st.session_state['data_format'])}", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='background: rgba(6,182,212,0.15); padding: 1rem; border-radius: 12px; margin: 0.5rem 0; border: 1px solid rgba(6,182,212,0.3);'>
                <div style='font-size: 1.3rem; font-weight: bold; color: #06b6d4;'>{len(df):,}</div>
                <div style='font-size: 0.85rem; color: #94a3b8;'>Rows</div>
            </div>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='background: rgba(139,92,246,0.15); padding: 1rem; border-radius: 12px; margin: 0.5rem 0; border: 1px solid rgba(139,92,246,0.3);'>
                <div style='font-size: 1.3rem; font-weight: bold; color: #a855f7;'>{len(df.columns)}</div>
                <div style='font-size: 0.85rem; color: #94a3b8;'>Columns</div>
            </div>""", unsafe_allow_html=True)

            if st.button("🗑️ Clear Data", key="clear_data_btn"):
                for k in ['df','data_format','file_name','analysis_context',
                          'validation_report','validation_rejected','cleaning_done',
                          'fe_done','fe_phase','df_fe','fe_report','fe_session_id',
                          'fe_suggestions','fe_target_col',
                          'mt_done','mt_results','mt_session_id','mt_best_model',
                          'mt_problem_type','mt_target_col','mt_recommendations',
                          'mt_test_size','mt_mem']:
                    st.session_state[k] = None if k in ['df','data_format','file_name',
                                                          'df_fe','fe_report','fe_session_id',
                                                          'fe_suggestions','fe_target_col',
                                                          'mt_session_id','mt_best_model',
                                                          'mt_problem_type','mt_target_col',
                                                          'mt_mem'] else \
                                          {} if k in ['analysis_context','mt_results'] else \
                                          False if k in ['validation_rejected','cleaning_done',
                                                         'fe_done','mt_done'] else \
                                          1 if k == 'fe_phase' else \
                                          [] if k == 'mt_recommendations' else \
                                          0.2 if k == 'mt_test_size' else []
                unified_memory.clear_session_context()
                st.rerun()

        st.divider()

        # Pipeline Status
        if st.session_state.get('validation_rejected'):
            st.markdown("### ⚠️ Pipeline Status")
            st.markdown('<div class="error-box">⛔ Blocked: Format Mismatch</div>', unsafe_allow_html=True)
        else:
            full_context = unified_memory.get_full_context()
            if any(full_context.values()):
                st.markdown("### 🔄 Pipeline Status")
                if full_context.get('ingestion'):  st.markdown("✅ Data Loaded")
                if full_context.get('validation'): st.markdown("✅ Validated")
                if full_context.get('cleaning'):   st.markdown("✅ Cleaned")
                if full_context.get('eda'):        st.markdown("✅ Analyzed")
                if st.session_state.get('fe_done'):  st.markdown("✅ Feature Engineering")
                if st.session_state.get('mt_done'): st.markdown("✅ Model Training")

        st.divider()
        st.markdown("""
        <div style='color: #94a3b8; font-size: 0.8rem;'>
            <b style='color: #06b6d4;'>🤖 Powered by:</b><br>
            • LangChain + OpenAI<br>• GPT-4o-mini<br>
            • FAISS Vector DB<br>• Format Validation
        </div>""", unsafe_allow_html=True)

        if st.button("🔄 Clear Memory", key="clear_memory_btn"):
            clear_agent_memory()
            st.session_state['chat_history']        = []
            st.session_state['validation_rejected'] = False
            st.success("✅ Memory cleared!")

    # ── ROUTING ──────────────────────────────────────────────────────────────
    if   page == "🏠 Home":                show_home_page()
    elif page == "📊 Data Upload":         show_upload_page()
    elif page == "🔍 Validation":          show_validation_page()
    elif page == "🧹 Cleaning":            show_cleaning_page()
    elif page == "📈 EDA & Visualization": show_eda_page()
    elif page == "⚙️ Feature Engineering": show_feature_engineering_page()
    elif page == "🤖 Model Training":      show_model_training_page()
    elif page == "💬 AI Chat Assistant":   show_chat_page()
    elif page == "📜 Pipeline History":    show_history_page()
    elif page == "🔄 Pipeline Snapshots": show_snapshots_page()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def show_home_page():
    st.markdown('<h2 style="text-align: center; color: #e2e8f0; font-weight: 700; margin-bottom: 2rem;">Welcome to Agentic Data Scientist! 🎉</h2>', unsafe_allow_html=True)
    st.markdown("### 📊 Workflow Pipeline")
    st.markdown(create_visual_pipeline(), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='feature-card'>
        <h3>🎯 What can this tool do?</h3>
        <p>✨ <b>Load Data</b> from CSV, JSON, or API</p>
        <p>🔍 <b>Validate</b> with STRICT format checking</p>
        <p>🧹 <b>Clean</b> data intelligently</p>
        <p>📈 <b>Analyze</b> with comprehensive EDA</p>
        <p>⚙️ <b>Engineer</b> features — leakage-free split + encode + scale</p>
        <p>🤖 <b>Train Models</b> — auto-recommend, tune, evaluate, explain</p>
        <p>💬 <b>Chat</b> with AI for full pipeline insights</p>
        <p>📊 <b>Visualize</b> with interactive plots</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='feature-card' style='background: linear-gradient(135deg, rgba(239,68,68,0.1) 0%, rgba(15,23,42,0.8) 100%);'>
        <h3>🔒 Format Validation Rules</h3>
        <p style='color: #fca5a5; font-weight: 600;'>STRICT ENFORCEMENT:</p>
        <ul style='color: #fecaca; line-height: 1.8;'>
        <li>CSV data → Must validate as CSV</li>
        <li>JSON data → Must validate as JSON</li>
        <li>API data → Must validate as API</li>
        <li>❌ Cross-format validation BLOCKED</li>
        <li>⛔ Pipeline stops on mismatch</li>
        </ul>
        </div>""", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
        <div class='feature-card' style='background: linear-gradient(135deg, rgba(16,185,129,0.1) 0%, rgba(15,23,42,0.8) 100%);'>
        <h3>🚀 Getting Started</h3>
        <ol style='color: #cbd5e1; line-height: 1.8;'>
        <li>Upload data (CSV, JSON, or API)</li>
        <li><b style='color: #fb923c;'>Select MATCHING format</b> in validation</li>
        <li>Clean data intelligently</li>
        <li>Run EDA &amp; save insights</li>
        <li><b style='color: #06b6d4;'>Engineer features</b> — RF score → top-K → split → encode → scale</li>
        <li><b style='color: #10b981;'>Train models</b> — select, tune hyperparams, evaluate</li>
        <li>Chat with AI for full pipeline insights</li>
        </ol>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class='feature-card' style='background: linear-gradient(135deg, rgba(251,146,60,0.1) 0%, rgba(15,23,42,0.8) 100%);'>
        <h3>📚 Pipeline Workflow</h3>
        <p style='color: #cbd5e1;'><b style='color: #10b981;'>1. Data Ingestion</b><br>Upload & store format</p>
        <p style='color: #cbd5e1;'><b style='color: #fb923c;'>2. Validation</b><br>CHECK format match ✅/❌</p>
        <p style='color: #cbd5e1;'><b style='color: #10b981;'>3. Cleaning</b><br>60+ techniques</p>
        <p style='color: #cbd5e1;'><b style='color: #10b981;'>4. EDA</b><br>Explore & visualize</p>
        <p style='color: #cbd5e1;'><b style='color: #06b6d4;'>5. Feature Engineering</b><br>Agent suggests, you decide</p>
        <p style='color: #cbd5e1;'><b style='color: #10b981;'>6. Model Training</b><br>Train, evaluate, explain</p>
        <p style='color: #cbd5e1;'><b style='color: #a855f7;'>7. AI Chat</b><br>Full pipeline context</p>
        </div>""", unsafe_allow_html=True)


def show_upload_page():
    st.header("📊 Data Upload")
    tab1, tab2, tab3 = st.tabs(["📁 Upload File", "🔗 API", "💾 Load from Memory"])

    with tab1:
        st.subheader("Upload CSV or JSON File")
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'json'])
        if uploaded_file is not None:
            try:
                file_type = uploaded_file.name.split('.')[-1].lower()
                with st.spinner(f"Loading {file_type.upper()} file..."):
                    df = tools['ingestion'].ingest(uploaded_file, data_format=file_type)
                if df is not None:
                    st.session_state['df']                 = df
                    st.session_state['df_raw']             = df.copy()  # raw before cleaning
                    st.session_state['data_format']        = file_type.upper()
                    st.session_state['file_name']          = uploaded_file.name
                    st.session_state['validation_rejected'] = False
                    st.markdown(f'<div class="success-box">✅ File loaded! <b>{len(df):,}</b> rows × <b>{len(df.columns)}</b> cols<br>Format: {get_format_badge(file_type.upper())}</div>', unsafe_allow_html=True)
                    unified_memory.update_ingestion_context({
                        'data_source': uploaded_file.name, 'data_format': file_type.upper(),
                        'row_count': len(df), 'column_count': len(df.columns),
                        'columns': df.columns.tolist(), 'file_name': uploaded_file.name
                    })
                    display_dataframe_info(df)
                    st.dataframe(df.head(10), use_container_width=True)
                    with st.expander("📋 Column Information"):
                        st.dataframe(pd.DataFrame({
                            'Column': df.columns, 'Type': df.dtypes.astype(str),
                            'Non-Null': df.count(), 'Null': df.isnull().sum(),
                            'Unique': [df[col].nunique() for col in df.columns]
                        }), use_container_width=True)
            except Exception as e:
                st.markdown(f'<div class="warning-box">❌ Error: {str(e)}</div>', unsafe_allow_html=True)

    with tab2:
        st.subheader("Load from API")
        api_url = st.text_input("API URL", placeholder="https://api.example.com/data")
        col1, col2 = st.columns(2)
        with col1: headers = st.text_area("Headers (JSON)", placeholder='{"Authorization": "Bearer token"}', height=100)
        with col2: params  = st.text_area("Parameters (JSON)", placeholder='{"limit": 100}', height=100)
        if st.button("🔗 Fetch Data", use_container_width=True):
            if api_url:
                try:
                    import json as _json
                    headers_dict = _json.loads(headers) if headers else None
                    params_dict  = _json.loads(params)  if params  else None
                    with st.spinner("Fetching..."):
                        df = tools['ingestion'].ingest(api_url, data_format='api',
                                                        api_headers=headers_dict, api_params=params_dict)
                    if df is not None:
                        st.session_state['df']          = df
                        st.session_state['data_format'] = 'API'
                        st.session_state['file_name']   = f"api_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        st.session_state['validation_rejected'] = False
                        unified_memory.update_ingestion_context({
                            'data_source': api_url, 'data_format': 'API',
                            'row_count': len(df), 'column_count': len(df.columns),
                            'columns': df.columns.tolist(), 'file_name': st.session_state['file_name']
                        })
                        st.markdown(f'<div class="success-box">✅ Data fetched!<br>Format: {get_format_badge("API")}</div>', unsafe_allow_html=True)
                        display_dataframe_info(df)
                        st.dataframe(df.head(), use_container_width=True)
                except Exception as e:
                    st.markdown(f'<div class="warning-box">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
            else:
                st.warning("Please enter an API URL")

    with tab3:
        st.subheader("📚 Load from History")
        try:
            history_df = tools['ingestion'].memory.get_all_ingestions()
            if not history_df.empty:
                st.dataframe(history_df[['id','timestamp','file_name','data_format','row_count','column_count']], use_container_width=True)
                history_id = st.number_input("Enter ID to load", min_value=1, step=1)
                if st.button("📥 Load Dataset", use_container_width=True):
                    df = tools['ingestion'].memory.get_ingestion_by_id(history_id)
                    if df is not None and not df.empty:
                        selected_row = history_df[history_df['id'] == history_id].iloc[0]
                        st.session_state['df']          = df
                        st.session_state['data_format'] = selected_row['data_format']
                        st.session_state['file_name']   = selected_row['file_name']
                        st.session_state['validation_rejected'] = False
                        st.markdown(f'<div class="success-box">✅ Loaded {len(df):,} rows</div>', unsafe_allow_html=True)
                        st.dataframe(df.head(), use_container_width=True)
            else:
                st.markdown('<div class="info-box">No history found. Upload data first!</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="warning-box">Error: {str(e)}</div>', unsafe_allow_html=True)


def show_validation_page():
    st.header("🔍 Data Validation")
    if st.session_state['df'] is None:
        st.markdown('<div class="warning-box">⚠️ No data loaded.</div>', unsafe_allow_html=True); return

    df = st.session_state['df']
    ingested_format = st.session_state.get('data_format', 'CSV')
    st.markdown(f'<div class="info-box">📥 <b>Data Ingested As:</b> {get_format_badge(ingested_format)}<br>⚠️ You MUST select <b>{ingested_format}</b> below.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        expected_format = st.selectbox("Expected Format", ["CSV", "JSON", "API"],
                                        index=["CSV","JSON","API"].index(ingested_format))
        st.session_state['validation_expected_format'] = expected_format
        if expected_format.upper() != ingested_format.upper():
            st.markdown(f'<div class="error-box">❌ <b>FORMAT MISMATCH!</b> Validation will be REJECTED!</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="success-box">✅ <b>Format Match</b></div>', unsafe_allow_html=True)
    with col2:
        if st.button("🔍 Run Validation", use_container_width=True, type="primary"):
            with st.spinner("Validating..."):
                validation_report = tools['validation'].validate(
                    df, dataset_name=st.session_state.get('file_name', 'dataset'),
                    expected_format=expected_format)
                st.session_state['validation_report'] = validation_report
                if validation_report.get('format_match') == False:
                    st.session_state['validation_rejected'] = True
                else:
                    st.session_state['validation_rejected'] = False
                    unified_memory.update_validation_context(validation_report)

    if st.session_state['validation_report']:
        display_validation_results(st.session_state['validation_report'])

        # ── DRIFT DETECTION SECTION ───────────────────────────────────
        st.markdown("---")
        st.subheader("🔄 Pipeline Snapshot Check")
        st.markdown(
            '<div class="info-box">'
            '🔄 <b>Drift Detection</b> — Checks if your new data matches any '
            'previously saved pipeline. If it matches, you can skip retraining '
            'and reuse the old model directly.'
            '</div>', unsafe_allow_html=True)

        snapshots = list_snapshots()
        df_new    = st.session_state.get("df")

        if not snapshots:
            st.markdown(
                '<div class="info-box">ℹ️ No saved pipeline snapshots found. '
                'Complete the full pipeline (up to Model Training) and save a '
                'snapshot to enable drift detection.</div>',
                unsafe_allow_html=True)
        elif df_new is None:
            st.warning("No data loaded.")
        else:
            if st.button("🔍 Run Drift Check Against All Snapshots",
                         type="primary", key="run_drift_btn"):
                drift_results = []
                # Clear old cached results first
                st.session_state.pop("drift_results", None)
                with st.spinner(f"Checking {len(snapshots)} snapshot(s)…"):
                    for snap in snapshots:
                        meta = load_snapshot_metadata(snap["snapshot_id"])
                        if meta is None:
                            continue
                        result = run_drift_check(df_new, meta)
                        drift_results.append(result)
                st.session_state["drift_results"] = drift_results

            drift_results = st.session_state.get("drift_results", [])
            if drift_results:
                st.markdown("#### 📊 Drift Check Results")

                # Summary table
                table_rows = []
                for r in drift_results:
                    _psi_val  = r.get("mean_psi")
                    psi_str   = f"{_psi_val:.4f}" if isinstance(_psi_val, (int, float)) else "—"
                    _match    = r.get("match_score")
                    match_str = f"{_match:.2%}" if isinstance(_match, (int, float)) else "—"
                    _overlap  = r.get("overlap_pct")
                    ovlp_str  = f"{_overlap:.0%}" if isinstance(_overlap, (int, float)) else "—"
                    sw        = r.get("schema", {}).get("soft_warnings", [])
                    table_rows.append({
                        "Snapshot":      r.get("snapshot_id", ""),
                        "Dataset":       r.get("snapshot_name", ""),
                        "Model":         r.get("model_name", ""),
                        "Problem":       r.get("problem_type", ""),
                        "Target":        r.get("target_column", ""),
                        "Col Overlap":   ovlp_str,
                        "Mean PSI":      psi_str,
                        "Match Score":   match_str,
                        "Decision":      r.get("reuse_label", ""),
                        "Warnings":      len(sw),
                    })
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

                # Per-snapshot detail + reuse button
                for r in drift_results:
                    drift_dec = r.get("drift", {})
                    drift_dec = drift_dec.get("decision", "") if drift_dec else ""
                    if drift_dec == "no_overlap":
                        exp_icon = "⬜"
                    elif r.get("can_reuse"):
                        exp_icon = "✅"
                    else:
                        exp_icon = "❌"
                    with st.expander(
                            f"{exp_icon} {r['snapshot_id']} — {r.get('reuse_label','')}",
                            expanded=r.get("can_reuse", False)):

                        # ── Soft schema warnings (never blocks) ───────────
                        schema = r.get("schema", {})
                        soft_w = schema.get("soft_warnings", [])
                        overlap_pct  = schema.get("overlap_pct", 0)
                        overlapping  = schema.get("overlapping", [])
                        missing_cols = schema.get("missing_cols", [])

                        # Show soft warnings
                        for w in soft_w:
                            st.warning(f"⚠️ {w}")

                        # Show overlap info
                        ov1, ov2, ov3 = st.columns(3)
                        ov1.metric("Column Overlap",
                                   f"{overlap_pct:.0%}",
                                   help="How many snapshot columns exist in new data")
                        ov2.metric("Shared Columns",
                                   f"{len(overlapping)}/{len(overlapping)+len(missing_cols)}",
                                   help="Columns compared for drift")
                        match_score = r.get("match_score", 0)
                        overlap_val = r.get("overlap_pct", 0)
                        ms_display  = ("N/A (no shared columns)"
                                       if overlap_val == 0
                                       else f"{match_score:.2%}")
                        ov3.metric("Match Score", ms_display,
                                   help="0.3×column_overlap + 0.7×distribution_similarity")

                        if overlapping:
                            st.caption(f"Columns compared: {', '.join(overlapping)}")
                        if missing_cols:
                            st.caption(f"Snapshot columns not in new data: "
                                       f"{', '.join(missing_cols)}")

                        st.markdown("---")

                        # ── Phase 2: Statistical drift ─────────────────────
                        drift = r.get("drift", {})
                        if drift is None:
                            st.info("ℹ️ Click Run Drift Check again to refresh.")
                            continue

                        if drift.get("n_features", 0) == 0:
                            st.info(
                                "ℹ️ No columns could be matched — even auto-matching "
                                "by data type found no comparable columns. "
                                "Distributions cannot be compared.")
                        else:
                            dc1, dc2, dc3 = st.columns(3)
                            dc1.metric("Mean PSI",
                                       f"{drift.get('mean_psi',0):.4f}")
                            dc2.metric("Drifted Features",
                                       f"{drift.get('n_drifted',0)}/{drift.get('n_features',0)}")
                            dc3.metric("Confidence", drift.get("confidence","—"))

                        # Per-feature PSI table
                        pf = drift.get("per_feature", {})
                        if pf:
                            pf_rows = []
                            for col, info in pf.items():
                                combined     = info.get("combined_score", 0)
                                matched_col  = info.get("matched_col", col)
                                match_method = info.get("match_method", "exact")
                                # Show matched column name with indicator
                                feat_label = (
                                    f"{col} → {matched_col} (auto)"
                                    if match_method == "auto" and matched_col != col
                                    else col)
                                row = {
                                    "Snapshot Feature": col,
                                    "Matched To":       matched_col,
                                    "Match":            ("🔗 Auto" if match_method == "auto"
                                                         else "✅ Exact"),
                                    "Type":             info["type"],
                                    "PSI":              f"{info['psi']:.4f}",
                                    "PSI Status":       info.get("psi_status", ""),
                                    "Combined Score":   f"{combined:.4f}",
                                }
                                if info["type"] == "numeric":
                                    row["KS Stat"]   = f"{info.get('ks_stat',0):.4f}"
                                    row["KS p"]      = f"{info.get('ks_p',1):.4f}"
                                    row["KS Status"] = info.get("ks_status", "")
                                    row["Z-Score"]   = f"{info.get('z_score',0):.4f}"
                                    row["Ref Mean"]  = f"{info.get('ref_mean',0):.4f}"
                                    row["New Mean"]  = f"{info.get('new_mean',0):.4f}"
                                else:
                                    row["Chi2"]        = f"{info.get('chi2',0):.4f}"
                                    row["Chi2 p"]      = f"{info.get('chi2_p',1):.4f}"
                                    row["Chi2 Status"] = info.get("chi2_status","")
                                    row["Unseen %"]    = f"{info.get('unseen_pct',0):.2%}"
                                pf_rows.append(row)
                            st.dataframe(pd.DataFrame(pf_rows),
                                         use_container_width=True)
                            # Legend
                            st.markdown(
                                '<div class="info-box" style="font-size:0.85em;">' +
                                '📊 <b>Methods used:</b> ' +
                                '<b>Numeric</b>: PSI (histogram shape) + KS Test (CDF) + Z-Score (mean shift) ' +
                                '→ Combined = 0.5×PSI + 0.5×KS_stat | ' +
                                '<b>Categorical</b>: PSI (frequency shift) + Chi-Square (count significance) ' +
                                '→ Combined = 0.5×PSI + 0.5×Chi2' +
                                '</div>', unsafe_allow_html=True)

                        # Reuse button — logic depends on drift decision
                        _drift_dec = drift.get("decision", "") if drift else ""

                        if _drift_dec == "no_overlap":
                            # No columns overlap — user can still force reuse
                            # but we warn them clearly
                            st.markdown(
                                '<div class="warning-box">'
                                '⬜ <b>Cannot measure drift</b> — no shared columns between '
                                'new data and this snapshot.<br>'
                                'PSI, KS Test, and Chi-Square were <b>not run</b> '
                                '(nothing to compare).<br>'
                                'This snapshot is from a completely different dataset domain. '
                                'Reuse is not recommended but you can force it below.'
                                '</div>', unsafe_allow_html=True)
                            if st.button(
                                    f"⚠️ Force Reuse Pipeline from {r['snapshot_id']}",
                                    key=f"reuse_{r['snapshot_id']}",
                                    type="secondary"):
                                with st.spinner("Loading saved pipeline…"):
                                    pipeline = load_snapshot_pipeline(r["snapshot_id"])
                                if pipeline:
                                    st.session_state["reused_pipeline"]    = pipeline
                                    st.session_state["reused_snapshot_id"] = r["snapshot_id"]
                                    st.session_state["reused_model"]       = pipeline["model"]
                                    st.session_state["reused_model_name"]  = pipeline["model_name"]
                                    st.session_state["mt_done"]            = True
                                    st.session_state["mt_best_model"]      = pipeline["model_name"]
                                    st.session_state["mt_problem_type"]    = pipeline["problem_type"]
                                    st.session_state["mt_target_col"]      = pipeline["target_col"]
                                    st.session_state["reuse_active"]       = True
                                    st.rerun()
                                else:
                                    st.error("❌ Could not load pipeline file.")

                        elif r.get("can_reuse"):
                            # Good match — primary reuse button
                            if st.button(
                                    f"♻️ Reuse Pipeline from {r['snapshot_id']}",
                                    key=f"reuse_{r['snapshot_id']}",
                                    type="primary"):
                                with st.spinner("Loading saved pipeline…"):
                                    pipeline = load_snapshot_pipeline(r["snapshot_id"])
                                if pipeline:
                                    st.session_state["reused_pipeline"]    = pipeline
                                    st.session_state["reused_snapshot_id"] = r["snapshot_id"]
                                    st.session_state["reused_model"]       = pipeline["model"]
                                    st.session_state["reused_model_name"]  = pipeline["model_name"]
                                    st.session_state["mt_done"]            = True
                                    st.session_state["mt_best_model"]      = pipeline["model_name"]
                                    st.session_state["mt_problem_type"]    = pipeline["problem_type"]
                                    st.session_state["mt_target_col"]      = pipeline["target_col"]
                                    st.session_state["reuse_active"]       = True
                                    st.success(
                                        f"✅ Pipeline reused from {r['snapshot_id']}! "
                                        f"Model: {pipeline['model_name']} | "
                                        f"Target: {pipeline['target_col']}")
                                    st.rerun()
                                else:
                                    st.error("❌ Could not load pipeline file.")

                        else:
                            # Major drift — warn but allow force reuse
                            st.markdown(
                                '<div class="warning-box">'
                                '🔴 <b>Major drift detected</b> — distributions differ significantly.<br>'
                                'PSI, KS Test ran on <b>{} shared columns</b>.<br>'
                                'Recommended: run full pipeline to train a new model.'
                                '</div>'.format(drift.get("n_features", 0)),
                                unsafe_allow_html=True)

        with st.expander("📜 Validation History"):
            try:
                history_df = tools['validation'].memory.get_all_validations()
                if not history_df.empty:
                    st.dataframe(history_df[['id','timestamp','total_rows','total_issues',
                                             'validation_status','ingested_format',
                                             'expected_format','format_match']], use_container_width=True)
            except Exception as e:
                st.write(f"Error: {str(e)}")


def show_cleaning_page():
    """
    Data Cleaning page.
    Handles: duplicates, missing values, outliers, text, type conversion.
    Does NOT handle: encoding or scaling (those are in Feature Engineering).
    """
    st.header("🧹 Data Cleaning")

    if st.session_state.get('validation_rejected', False):
        st.markdown(
            '<div class="error-box">⛔ <b>BLOCKED</b> — Fix format mismatch in Validation first.</div>',
            unsafe_allow_html=True); return
    if st.session_state['df'] is None:
        st.markdown('<div class="warning-box">⚠️ No data loaded.</div>',
                    unsafe_allow_html=True); return
    if not st.session_state.get('validation_report'):
        st.markdown('<div class="warning-box">⚠️ Run Validation first.</div>',
                    unsafe_allow_html=True); return

    df = st.session_state['df']
    validation_report = st.session_state['validation_report']

    st.markdown(
        '<div class="info-box">'
        '🧹 <b>Data Quality Cleaning</b> — Fix data issues only.<br>'
        '⚙️ Encoding and scaling are handled in the <b>Feature Engineering</b> page.'
        '</div>', unsafe_allow_html=True)

    with st.expander("📋 Validation Summary", expanded=False):
        st.write(f"**Status:** {validation_report.get('validation_status','UNKNOWN')}")
        st.write(f"**Total Issues:** {validation_report.get('total_issues', 0)}")

    st.markdown("---")

    # ── TABS (encoding and scaling REMOVED) ──────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔄 Duplicates & Missing",
        "📊 Outliers",
        "📝 Text Cleaning",
        "🔄 Type Conversion"
    ])

    with tab1:
        st.markdown("### 🔄 Duplicate Removal")
        col1, col2 = st.columns(2)
        with col1:
            remove_duplicates = st.checkbox("Remove Duplicate Rows", value=True)
        with col2:
            duplicate_keep = st.selectbox(
                "Keep Which Occurrence?",
                ["first", "last", "none"],
                help="first = keep first | last = keep last | none = remove all duplicates"
            ) if remove_duplicates else "first"

        st.markdown("---")
        st.markdown("### 💧 Missing Value Handling")
        col1, col2 = st.columns(2)
        with col1:
            missing_strategy = st.selectbox(
                "Strategy",
                ["auto", "mean", "median", "mode", "forward_fill",
                 "backward_fill", "interpolate", "knn", "constant",
                 "drop_rows", "drop_cols", "None"],
                help="auto = smart decision based on column type and % missing"
            )
        with col2:
            knn_neighbors = st.number_input(
                "KNN Neighbors", 1, 20, 5) if missing_strategy == "knn" else 5
            missing_threshold = st.slider(
                "Drop Threshold (%)", 10, 100, 50,
                help="Drop column if this % of values are missing"
            ) if missing_strategy == "drop_cols" else 50

    with tab2:
        st.markdown("### 📊 Outlier Detection & Handling")
        col1, col2 = st.columns(2)
        with col1:
            outlier_method = st.selectbox(
                "Detection Method",
                ["iqr", "iqr_custom", "zscore", "modified_zscore", "percentile", "None"])
            outlier_action = st.selectbox(
                "Action",
                ["cap", "remove", "flag"],
                help="cap = clip to bounds | remove = delete rows | flag = add boolean column")
        with col2:
            iqr_multiplier   = st.slider("IQR Multiplier", 1.0, 3.0, 1.5, 0.1)                                if outlier_method == "iqr_custom" else 1.5
            zscore_threshold = st.slider("Z-Score Threshold", 2.0, 4.0, 3.0, 0.1)                                if outlier_method == "zscore" else 3.0
            if outlier_method == "percentile":
                ca, cb = st.columns(2)
                with ca: percentile_lower = st.number_input("Lower %", 0.1, 10.0, 1.0, 0.1)
                with cb: percentile_upper = st.number_input("Upper %", 90.0, 99.9, 99.0, 0.1)
            else:
                percentile_lower, percentile_upper = 1.0, 99.0

    with tab3:
        st.markdown("### 📝 Text Cleaning")
        st.markdown(
            '<p style="color:#94a3b8;font-size:0.85rem;">'
            'Standardise raw text. Encoding happens in Feature Engineering.</p>',
            unsafe_allow_html=True)
        text_operations = st.multiselect(
            "Text Cleaning Operations",
            ["lowercase", "uppercase", "titlecase", "trim",
             "remove_extra_spaces", "standardize_whitespace",
             "remove_urls", "remove_emails", "remove_phone",
             "remove_html", "remove_punctuation", "remove_numbers",
             "remove_special", "fix_encoding"],
            default=["lowercase", "trim", "remove_extra_spaces"])

    with tab4:
        st.markdown("### 🔄 Data Type Conversion")
        st.markdown(
            '<p style="color:#94a3b8;font-size:0.85rem;">'
            'Convert columns to correct types before downstream steps.</p>',
            unsafe_allow_html=True)
        enable_type_conversion = st.checkbox("Enable Type Conversion", value=False)
        if enable_type_conversion:
            col1, col2 = st.columns(2)
            with col1:
                columns_to_convert = st.multiselect(
                    "Select Columns", df.columns.tolist())
            with col2:
                conversion_type = st.selectbox(
                    "Convert To",
                    ["numeric", "datetime", "categorical", "string"]
                ) if columns_to_convert else "numeric"
        else:
            columns_to_convert, conversion_type = [], None

    st.markdown("---")
    if st.button("🧹 Clean Data", use_container_width=True,
                 type="primary", key="clean_btn"):
        with st.spinner("Cleaning data…"):
            operations = {}
            if remove_duplicates:
                operations['remove_duplicates'] = {'keep': duplicate_keep}
            if missing_strategy != "None":
                operations['handle_missing'] = missing_strategy
                if missing_strategy == "knn":
                    operations['missing_knn_neighbors'] = knn_neighbors
                elif missing_strategy == "drop_cols":
                    operations['missing_threshold'] = missing_threshold / 100.0
            if outlier_method != "None":
                outlier_config = {'method': outlier_method, 'action': outlier_action}
                if outlier_method == "iqr_custom": outlier_config['iqr_multiplier'] = iqr_multiplier
                elif outlier_method == "zscore":   outlier_config['zscore_threshold'] = zscore_threshold
                elif outlier_method == "percentile":
                    outlier_config['percentile_lower'] = percentile_lower
                    outlier_config['percentile_upper'] = percentile_upper
                operations['handle_outliers'] = outlier_config
            if text_operations:
                operations['clean_text'] = text_operations
            if enable_type_conversion and columns_to_convert:
                operations['convert_types'] = {col: conversion_type
                                               for col in columns_to_convert}

            rows_before, cols_before = len(df), len(df.columns)
            try:
                from tools.data_cleaning import DataCleaningTool
                cleaner  = DataCleaningTool()
                df_clean = cleaner.clean_data(
                    df=df, operations=operations,
                    dataset_name=st.session_state.get('file_name', 'dataset'),
                    validation_report=validation_report)

                if df_clean is not None:
                    st.session_state['df']           = df_clean
                    st.session_state['df_cleaned']   = df_clean
                    st.session_state['cleaning_done'] = True

                    rows_after, cols_after = len(df_clean), len(df_clean.columns)

                    unified_memory.update_cleaning_context({
                        'rows_before':    rows_before,
                        'rows_after':     rows_after,
                        'columns_before': cols_before,
                        'columns_after':  cols_after,
                        'operations':     [{'type': k, 'details': str(v),
                                            'columns': [], 'count': 0}
                                           for k, v in operations.items()],
                        'summary': f"Cleaned {rows_before}→{rows_after} rows"
                    })

                    st.markdown(
                        '<div class="success-box">✅ Data cleaned! '
                        'Categorical columns are intact — EDA will show readable values.</div>',
                        unsafe_allow_html=True)

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Rows Before",   f"{rows_before:,}")
                    c2.metric("Rows After",    f"{rows_after:,}",
                              delta=f"{rows_after-rows_before:,}")
                    c3.metric("Cols Before",   f"{cols_before}")
                    c4.metric("Cols After",    f"{cols_after}",
                              delta=f"{cols_after-cols_before}")

                    st.dataframe(df_clean.head(20), use_container_width=True)

                    # Show remaining categorical columns info
                    cat_cols = df_clean.select_dtypes(
                        include=['object','category']).columns.tolist()
                    if cat_cols:
                        st.markdown(
                            f'<div class="info-box">ℹ️ <b>{len(cat_cols)} categorical columns</b> '
                            f'({", ".join(cat_cols[:6])}{"..." if len(cat_cols)>6 else ""}) '
                            'will be encoded in Feature Engineering.</div>',
                            unsafe_allow_html=True)

                    csv_buf = cleaner.get_cleaned_csv_buffer()
                    if csv_buf:
                        st.download_button(
                            "📥 Download Cleaned CSV",
                            data=csv_buf.getvalue(),
                            file_name=f"cleaned_{st.session_state.get('file_name','data')}"
                                      f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv", type="secondary")

            except Exception as e:
                st.markdown(
                    f'<div class="error-box">❌ Cleaning error: {str(e)}</div>',
                    unsafe_allow_html=True)
                import traceback; st.code(traceback.format_exc())

    st.markdown("""
    <div class="warning-box">
    <b>⚠️ Note</b><br>
    Encoding and scaling have been moved to <b>Feature Engineering</b>.<br>
    This ensures EDA runs on human-readable data.
    </div>""", unsafe_allow_html=True)


def show_eda_page():
    """EDA page — unchanged from original, kept in full"""
    st.header("📈 Exploratory Data Analysis")

    if st.session_state['df'] is None:
        st.markdown('<div class="warning-box">⚠️ No data loaded.</div>', unsafe_allow_html=True); return
    if st.session_state.get('validation_rejected', False):
        st.markdown('<div class="error-box">⛔ <b>EDA BLOCKED</b> — Fix format mismatch first.</div>', unsafe_allow_html=True); return

    df = st.session_state['df']

    if 'eda_memory' not in st.session_state:
        from tools.eda_memory import EDAMemoryManager
        st.session_state['eda_memory'] = EDAMemoryManager()
    eda_memory = st.session_state['eda_memory']

    try:
        from plot_interpretations import (
            generate_histogram_interpretation_ai,
            generate_boxplot_interpretation_ai,
            generate_categorical_plot_interpretation_ai,
            generate_3d_plot_interpretation_ai,
            generate_correlation_heatmap_interpretation_ai
        )
        plot_interpretations_available = True
    except ImportError:
        plot_interpretations_available = False

    plot_explainer_available = False
    try:
        from tools.plot_explainer import create_plot_explanation
        plot_explainer_available = True
    except ImportError:
        pass

    def get_categorical_columns_for_eda(df):
        original_cols   = [col for col in df.columns if col.endswith('_original')]
        display_columns = list(original_cols)
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if not col.endswith('_original'):
                display_columns.append(col)
        return {'display_cols': display_columns, 'has_backups': len(original_cols) > 0}

    def gen_scatter(df, x, y, ssize=None):
        if not plot_explainer_available: return f"Scatter: {x} vs {y}"
        try:    return create_plot_explanation(df=df, x_col=x, y_col=y, plot_type='scatter', use_ai=True,  openai_api_key=os.getenv('OPENAI_API_KEY'))
        except:
            try: return create_plot_explanation(df=df, x_col=x, y_col=y, plot_type='scatter', use_ai=False)
            except: return f"Scatter: {x} vs {y}"

    def gen_density(df, x, y):
        if not plot_explainer_available: return f"Density heatmap: {x} vs {y}"
        try:    return f"**Density Heatmap**\n\n{create_plot_explanation(df=df,x_col=x,y_col=y,plot_type='scatter',use_ai=True,openai_api_key=os.getenv('OPENAI_API_KEY'))}\n\n*Darker = more data points.*"
        except:
            try: return f"**Density Heatmap**\n\n{create_plot_explanation(df=df,x_col=x,y_col=y,plot_type='scatter',use_ai=False)}"
            except: return f"Density heatmap: {x} vs {y}"

    def gen_line(df, x, y):
        if not plot_explainer_available: return f"Line plot: {y} over {x}"
        try:    return create_plot_explanation(df=df, x_col=x, y_col=y, plot_type='line', use_ai=True,  openai_api_key=os.getenv('OPENAI_API_KEY'))
        except:
            try: return create_plot_explanation(df=df, x_col=x, y_col=y, plot_type='line', use_ai=False)
            except: return f"Line: {y} over {x}"

    st.markdown("---")
    st.info("💾 **Memory Control:** Each tab has its own 'Save to Memory' checkbox")
    st.markdown("---")

    numeric_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object','category']).columns.tolist()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview","📈 Distributions","🔗 Relationships (2D)",
        "🎯 3D Analysis","📉 Correlations","📋 Advanced Stats"
    ])

    with tab1:
        st.subheader("Dataset Overview")
        display_dataframe_info(df)
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if st.button("🔍 Generate Comprehensive Analysis", use_container_width=True, type="primary", key="gen_comp"):
                with st.spinner("Analysing..."):
                    try:
                        if 'eda_session_id' not in st.session_state:
                            st.session_state['eda_session_id'] = eda_memory.create_eda_session(df, st.session_state.get('file_name','dataset'))
                        sid = st.session_state['eda_session_id']
                        for col in numeric_cols:
                            eda_memory.log_distribution_analysis(sid, col, 'comprehensive', {
                                'mean': float(df[col].mean()), 'median': float(df[col].median()),
                                'std':  float(df[col].std()),  'min':    float(df[col].min()),
                                'max':  float(df[col].max()),  'skewness': float(df[col].skew()),
                                'kurtosis': float(df[col].kurtosis())
                            })
                        if len(numeric_cols) >= 2:
                            corr_matrix = df[numeric_cols].corr(method='pearson')
                            corr_pairs  = []
                            for i in range(len(corr_matrix.columns)):
                                for j in range(i+1, len(corr_matrix.columns)):
                                    v = corr_matrix.iloc[i,j]
                                    if not np.isnan(v):
                                        corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], float(v)))
                            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                            eda_memory.log_correlation_analysis(sid, 'pearson', corr_matrix,
                                [c for c in corr_pairs if c[2]>0][:5],
                                [c for c in corr_pairs if c[2]<0][:5])
                        for col in numeric_cols:
                            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lb, ub = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                            outs = df[(df[col]<lb)|(df[col]>ub)]
                            eda_memory.log_outlier_detection(sid, col, 'IQR', len(outs),
                                len(outs)/len(df)*100, float(lb), float(ub), outs.index.tolist())
                        eda_memory.log_insight(sid, 'comprehensive_analysis',
                            f"Comprehensive analysis: {len(df)} rows, {len(df.columns)} cols.",
                            df.columns.tolist(), {'row_count': len(df),'col_count': len(df.columns)}, 1.0)
                        st.success(f"✅ Analysis complete! Session: {sid}")
                    except Exception as e:
                        st.error(f"❌ {str(e)}")
        st.markdown("---")
        st.subheader("📋 Column Information")
        st.dataframe(pd.DataFrame({
            'Column': df.columns, 'Type': df.dtypes.astype(str),
            'Non-Null': df.count(), 'Null': df.isnull().sum(),
            'Null %': (df.isnull().sum()/len(df)*100).round(2),
            'Unique': [df[col].nunique() for col in df.columns]
        }), use_container_width=True)
        with st.expander("🔍 Data Sample"):
            st.dataframe(df.head(20), use_container_width=True)

    with tab2:
        st.subheader("📈 Distribution Analysis")
        col_mem, _ = st.columns([1,3])
        with col_mem:
            enable_dist_memory = st.checkbox("💾 Save Distributions", value=False, key="save_dist_memory")
        if enable_dist_memory:
            if 'eda_session_id' not in st.session_state:
                st.session_state['eda_session_id'] = eda_memory.create_eda_session(df, st.session_state.get('file_name','dataset'))
            st.success(f"✅ Session: {st.session_state['eda_session_id']}")
        st.markdown("---")
        if not numeric_cols:
            st.markdown('<div class="info-box">No numeric columns found</div>', unsafe_allow_html=True)
        else:
            col1, col2 = st.columns([1,3])
            with col1:
                selected_col = st.selectbox("Select Column", numeric_cols, key="dist_col")
                plot_type    = st.radio("Plot Type", ["Histogram","Both"], index=0)
            with col2:
                ptmap = {"Histogram":"hist","Both":"both"}
                fig   = tools['eda'].plot_distribution(df, selected_col, plot_type=ptmap[plot_type])
                st.pyplot(fig)
                tools['eda'].create_download_button(fig, f"dist_{selected_col}.png", "📥 Download")
                if plot_interpretations_available:
                    with st.expander("🤖 AI Interpretation", expanded=True):
                        try: st.markdown(generate_histogram_interpretation_ai(df, selected_col))
                        except: st.info(f"Distribution of {selected_col}")
                if enable_dist_memory and 'eda_session_id' in st.session_state:
                    try:
                        stats = {'mean': float(df[selected_col].mean()), 'median': float(df[selected_col].median()),
                                 'std':  float(df[selected_col].std()),  'min':    float(df[selected_col].min()),
                                 'max':  float(df[selected_col].max()),  'skewness': float(df[selected_col].skew()),
                                 'kurtosis': float(df[selected_col].kurtosis())}
                        eda_memory.log_distribution_analysis(st.session_state['eda_session_id'], selected_col, ptmap[plot_type], stats)
                        eda_memory.log_visualization(st.session_state['eda_session_id'], 'histogram', x_col=selected_col)
                    except: pass
                st.markdown("#### Boxplot")
                fig_box = tools['eda'].plot_boxplot(df, selected_col)
                st.pyplot(fig_box)
                tools['eda'].create_download_button(fig_box, f"box_{selected_col}.png", "📥 Download")
                if plot_interpretations_available:
                    with st.expander("🤖 AI Interpretation", expanded=True):
                        try: st.markdown(generate_boxplot_interpretation_ai(df, selected_col))
                        except: st.info(f"Boxplot of {selected_col}")
                if enable_dist_memory and 'eda_session_id' in st.session_state:
                    try:
                        Q1, Q3 = df[selected_col].quantile(0.25), df[selected_col].quantile(0.75)
                        IQR    = Q3 - Q1
                        lb, ub = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                        outs   = df[(df[selected_col]<lb)|(df[selected_col]>ub)]
                        eda_memory.log_outlier_detection(st.session_state['eda_session_id'],
                            selected_col, 'IQR', len(outs), len(outs)/len(df)*100,
                            float(lb), float(ub), outs.index.tolist())
                        eda_memory.log_visualization(st.session_state['eda_session_id'], 'boxplot', x_col=selected_col)
                    except: pass

        st.markdown("---")
        st.subheader("📊 Categorical Analysis")
        cat_info = get_categorical_columns_for_eda(df)
        cat_cols_viz = cat_info['display_cols']
        if not cat_cols_viz:
            st.markdown('<div class="info-box">No categorical columns found</div>', unsafe_allow_html=True)
        else:
            if cat_info['has_backups']: st.info("ℹ️ Using original categorical values for visualization")
            col1, col2 = st.columns([1,3])
            with col1:
                cat_col_options = {}
                for col in cat_cols_viz:
                    label = col.replace('_original','') + " (original)" if col.endswith('_original') else col
                    cat_col_options[label] = col
                selected_display = st.selectbox("Categorical Column", list(cat_col_options.keys()), key="cat_col")
                cat_col       = cat_col_options[selected_display]
                cat_plot_type = st.radio("Plot Type", ["Count Plot","Bar Plot","Pie Chart"], index=0)
                top_n         = st.slider("Top N", 5, 20, 10, key="cat_top_n")
            with col2:
                clean_name = cat_col.replace('_original','')
                if cat_plot_type == "Count Plot":
                    fig_c = tools['eda'].plot_countplot(df, cat_col, top_n=top_n)
                    st.pyplot(fig_c)
                    tools['eda'].create_download_button(fig_c, f"count_{clean_name}.png", "📥 Download")
                    if plot_interpretations_available:
                        with st.expander("🤖 AI Interpretation", expanded=True):
                            try: st.markdown(generate_categorical_plot_interpretation_ai(df, cat_col, "count"))
                            except: pass
                elif cat_plot_type == "Bar Plot" and numeric_cols:
                    num_for_bar = st.selectbox("Numeric Column", numeric_cols, key="bar_numeric")
                    fig_b = tools['eda'].plot_barplot(df, cat_col, num_for_bar, top_n=top_n)
                    st.pyplot(fig_b)
                    tools['eda'].create_download_button(fig_b, f"bar_{clean_name}.png", "📥 Download")
                    if plot_interpretations_available:
                        with st.expander("🤖 AI Interpretation", expanded=True):
                            try: st.markdown(generate_categorical_plot_interpretation_ai(df, cat_col, "bar"))
                            except: pass
                elif cat_plot_type == "Pie Chart":
                    fig_p = tools['eda'].plot_pie_chart(df, cat_col, top_n=top_n)
                    st.pyplot(fig_p)
                    tools['eda'].create_download_button(fig_p, f"pie_{clean_name}.png", "📥 Download")
                with st.expander("📊 Value Counts", expanded=False):
                    vc = df[cat_col].value_counts().head(top_n)
                    st.dataframe(pd.DataFrame({'Category': vc.index, 'Count': vc.values,
                                               'Percentage': (vc.values/len(df)*100).round(2)}),
                                 use_container_width=True)

    with tab3:
        st.subheader("🔗 Relationship Analysis (2D)")
        col_mem, _ = st.columns([1,3])
        with col_mem:
            enable_rel_memory = st.checkbox("💾 Save 2D Plots", value=False, key="save_rel_memory")
        if enable_rel_memory:
            if 'eda_session_id' not in st.session_state:
                st.session_state['eda_session_id'] = eda_memory.create_eda_session(df, st.session_state.get('file_name','dataset'))
            st.success(f"✅ Session: {st.session_state['eda_session_id']}")
        st.markdown("---")
        if len(numeric_cols) < 2:
            st.markdown('<div class="info-box">Need at least 2 numeric columns</div>', unsafe_allow_html=True)
        else:
            col1, col2 = st.columns([1,3])
            with col1:
                x_col = st.selectbox("X-Axis", numeric_cols, key="rel_x")
                y_col = st.selectbox("Y-Axis", [c for c in numeric_cols if c != x_col], key="rel_y")
                use_color = st.checkbox("Add Color Dimension", value=False)
                hue_col   = st.selectbox("Color By", categorical_cols+numeric_cols, key="rel_hue") if use_color and (categorical_cols+numeric_cols) else None
                st.markdown("---")
                total_points = len(df)
                if total_points > 500:
                    st.warning(f"⚠️ Large dataset ({total_points:,} points)")
                    plot_method = st.radio("Visualization Method", ["Sample Points","Density Heatmap"])
                    sample_size = st.slider("Sample Size", 10, min(10000, total_points), min(2000, total_points), 10) if plot_method == "Sample Points" else None
                else:
                    plot_method, sample_size = "All Points", None
                plot_style = st.selectbox("Plot Style", ["Scatter","Line","Scatter + Line"])
            with col2:
                st.markdown(f"### {y_col} vs {x_col}")
                df_plot = (df[[x_col,y_col]].dropna().sample(n=min(sample_size,len(df)), random_state=42)
                           if plot_method=="Sample Points" and sample_size
                           else df[[x_col,y_col]].dropna())
                scatter_shown = line_shown = False
                if plot_style in ["Scatter","Scatter + Line"]:
                    if plot_method == "Density Heatmap":
                        fig_s, ax = plt.subplots(figsize=(10,6))
                        hb = ax.hexbin(df_plot[x_col], df_plot[y_col], gridsize=30, cmap='YlOrRd', mincnt=1)
                        plt.colorbar(hb, ax=ax, label='Count')
                        ax.set_xlabel(x_col,fontsize=12); ax.set_ylabel(y_col,fontsize=12)
                        ax.set_title(f'{y_col} vs {x_col} (Density)',fontsize=14,fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig_s)
                    else:
                        fig_s = tools['eda'].plot_scatter(df_plot, x_col, y_col, hue=hue_col)
                        st.pyplot(fig_s)
                    tools['eda'].create_download_button(fig_s, f"scatter_{x_col}_{y_col}.png", "📥 Download Scatter")
                    scatter_shown = True
                    if enable_rel_memory and 'eda_session_id' in st.session_state:
                        try:
                            corr = df[[x_col,y_col]].corr().iloc[0,1]
                            eda_memory.log_visualization(st.session_state['eda_session_id'],
                                'scatter' if plot_method!="Density Heatmap" else 'density_heatmap',
                                x_col=x_col, y_col=y_col, hue_col=hue_col if use_color else None,
                                plot_method=plot_method,
                                parameters={'sample_size':sample_size} if sample_size else None)
                            insight = gen_density(df,x_col,y_col) if plot_method=="Density Heatmap" else gen_scatter(df,x_col,y_col,sample_size)
                            eda_memory.log_insight(st.session_state['eda_session_id'], '2d_relationship',
                                insight, [x_col,y_col], {'correlation':float(corr),'method':plot_method}, 0.85)
                        except: pass
                if plot_style in ["Line","Scatter + Line"] and plot_method != "Density Heatmap":
                    fig_l = tools['eda'].plot_line(df_plot, x_col, y_col)
                    st.pyplot(fig_l)
                    tools['eda'].create_download_button(fig_l, f"line_{x_col}_{y_col}.png", "📥 Download Line")
                    line_shown = True
                with st.expander("📊 What's Happening in This Plot?", expanded=True):
                    if plot_method == "Density Heatmap": st.markdown(gen_density(df,x_col,y_col))
                    elif scatter_shown and not line_shown: st.markdown(gen_scatter(df,x_col,y_col,sample_size))
                    elif line_shown and not scatter_shown: st.markdown(gen_line(df,x_col,y_col))
                    else:
                        st.markdown("#### 📊 Scatter:"); st.markdown(gen_scatter(df,x_col,y_col,sample_size))
                        st.markdown("---")
                        st.markdown("#### 📈 Line:"); st.markdown(gen_line(df,x_col,y_col))

    with tab4:
        st.subheader("🎯 3D Analysis")
        col_mem, _ = st.columns([1,3])
        with col_mem:
            enable_3d_memory = st.checkbox("💾 Save 3D Plots", value=False, key="save_3d_memory")
        if enable_3d_memory:
            if 'eda_session_id' not in st.session_state:
                st.session_state['eda_session_id'] = eda_memory.create_eda_session(df, st.session_state.get('file_name','dataset'))
            st.success(f"✅ Session: {st.session_state['eda_session_id']}")
        st.markdown("---")
        if len(numeric_cols) < 3:
            st.markdown('<div class="info-box">Need at least 3 numeric columns</div>', unsafe_allow_html=True)
        else:
            col1, col2 = st.columns([1,3])
            with col1:
                x_3d = st.selectbox("X-Axis", numeric_cols, key="3d_x")
                y_3d = st.selectbox("Y-Axis", [c for c in numeric_cols if c!=x_3d], key="3d_y")
                z_3d = st.selectbox("Z-Axis", [c for c in numeric_cols if c not in [x_3d,y_3d]], key="3d_z")
            with col2:
                fig_3d = tools['eda'].create_3d_scatter(df, x_3d, y_3d, z_3d)
                st.plotly_chart(fig_3d, use_container_width=True)
                st.download_button("📥 Download 3D (HTML)", data=fig_3d.to_html(),
                                   file_name=f"3d_{x_3d}_{y_3d}_{z_3d}.html", mime="text/html", use_container_width=True)
                if plot_interpretations_available:
                    with st.expander("🤖 AI Interpretation", expanded=True):
                        try: st.markdown(generate_3d_plot_interpretation_ai(df, x_3d, y_3d, z_3d))
                        except: st.info(f"3D: {x_3d}, {y_3d}, {z_3d}")
                if enable_3d_memory and 'eda_session_id' in st.session_state:
                    try:
                        cxy = df[[x_3d,y_3d]].corr().iloc[0,1]
                        cxz = df[[x_3d,z_3d]].corr().iloc[0,1]
                        cyz = df[[y_3d,z_3d]].corr().iloc[0,1]
                        eda_memory.log_visualization(st.session_state['eda_session_id'], '3d_scatter',
                            x_col=x_3d, y_col=y_3d, z_col=z_3d, plot_method='plotly_3d')
                        eda_memory.log_insight(st.session_state['eda_session_id'], '3d_relationship',
                            f"3D: {x_3d},{y_3d},{z_3d}. Corr: {x_3d}↔{y_3d}={cxy:.2f}, {x_3d}↔{z_3d}={cxz:.2f}, {y_3d}↔{z_3d}={cyz:.2f}.",
                            [x_3d,y_3d,z_3d], {'corr_xy':float(cxy),'corr_xz':float(cxz),'corr_yz':float(cyz)}, 0.85)
                    except: pass
                with st.expander("📊 3D Statistics", expanded=False):
                    st.dataframe(df[[x_3d,y_3d,z_3d]].describe(), use_container_width=True)

    with tab5:
        st.subheader("📉 Correlation Analysis")
        col_mem, _ = st.columns([1,3])
        with col_mem:
            enable_corr_memory = st.checkbox("💾 Save Correlations", value=False, key="save_corr_memory")
        if enable_corr_memory:
            if 'eda_session_id' not in st.session_state:
                st.session_state['eda_session_id'] = eda_memory.create_eda_session(df, st.session_state.get('file_name','dataset'))
            st.success(f"✅ Session: {st.session_state['eda_session_id']}")
        st.markdown("---")
        if len(numeric_cols) < 2:
            st.markdown('<div class="info-box">Need at least 2 numeric columns</div>', unsafe_allow_html=True)
        else:
            col1, col2 = st.columns([1,3])
            with col1:
                st.info("ℹ️ Using Pearson correlation")
                corr_method   = "pearson"
                show_pairplot = st.checkbox("Show Pairplot", value=False)
                n_cols_pair   = st.slider("Columns in Pairplot", 2, min(6,len(numeric_cols)), min(4,len(numeric_cols))) if show_pairplot else 4
            with col2:
                fig_corr = tools['eda'].plot_correlation_heatmap(df, method=corr_method)
                if fig_corr:
                    st.pyplot(fig_corr)
                    tools['eda'].create_download_button(fig_corr, "correlation_heatmap.png", "📥 Download")
                    if plot_interpretations_available:
                        with st.expander("🤖 AI Interpretation", expanded=True):
                            try:
                                cm = df[numeric_cols].corr(method=corr_method.lower())
                                st.markdown(generate_correlation_heatmap_interpretation_ai(df, cm))
                            except: st.info("Correlation heatmap")
                    if enable_corr_memory and 'eda_session_id' in st.session_state:
                        try:
                            cm = df[numeric_cols].corr(method=corr_method.lower())
                            cpairs = []
                            for i in range(len(cm.columns)):
                                for j in range(i+1,len(cm.columns)):
                                    v = cm.iloc[i,j]
                                    if not np.isnan(v): cpairs.append((cm.columns[i],cm.columns[j],float(v)))
                            cpairs.sort(key=lambda x: abs(x[2]), reverse=True)
                            eda_memory.log_correlation_analysis(st.session_state['eda_session_id'],
                                corr_method.lower(), cm,
                                [c for c in cpairs if c[2]>0][:5],
                                [c for c in cpairs if c[2]<0][:5])
                        except: pass
                cm_disp = df[numeric_cols].corr(method=corr_method)
                cp = [(cm_disp.columns[i],cm_disp.columns[j],cm_disp.iloc[i,j])
                      for i in range(len(cm_disp.columns)) for j in range(i+1,len(cm_disp.columns))
                      if not np.isnan(cm_disp.iloc[i,j])]
                cp.sort(key=lambda x: abs(x[2]), reverse=True)
                if cp:
                    with st.expander("🔝 Top Correlations", expanded=True):
                        st.markdown("#### Top 5 Positive")
                        for c1,c2,v in [c for c in cp if c[2]>0][:5]: st.write(f"**{c1}** ↔ **{c2}**: {v:.3f}")
                        st.markdown("#### Top 5 Negative")
                        for c1,c2,v in [c for c in cp if c[2]<0][:5]: st.write(f"**{c1}** ↔ **{c2}**: {v:.3f}")
            if show_pairplot and len(numeric_cols) >= 2:
                st.markdown("---")
                st.markdown("### 📊 Pairplot")
                sel_pair = numeric_cols[:min(n_cols_pair,len(numeric_cols))]
                with st.spinner("Generating..."):
                    if 'pairplot_fig' not in st.session_state or st.session_state.get('pairplot_cols') != sel_pair:
                        st.session_state['pairplot_fig']  = tools['eda'].plot_pairplot_sample(df, columns=sel_pair)
                        st.session_state['pairplot_cols'] = sel_pair
                    st.pyplot(st.session_state['pairplot_fig'])
                    tools['eda'].create_download_button(st.session_state['pairplot_fig'], "pairplot.png", "📥 Download")

    with tab6:
        st.subheader("📋 Advanced Statistical Summary")
        col_mem, _ = st.columns([1,3])
        with col_mem:
            enable_stats_memory = st.checkbox("💾 Save Stats", value=False, key="save_stats_memory")
        if enable_stats_memory:
            if 'eda_session_id' not in st.session_state:
                st.session_state['eda_session_id'] = eda_memory.create_eda_session(df, st.session_state.get('file_name','dataset'))
            st.success(f"✅ Session: {st.session_state['eda_session_id']}")
        st.markdown("---")
        col1, col2 = st.columns([2,1])
        with col1:
            if st.button("📊 Generate Statistical Report", use_container_width=True, type="primary"):
                with st.spinner("Generating..."):
                    report = tools['eda'].generate_text_summary(df)
                    st.session_state['eda_report'] = report
                    if enable_stats_memory:
                        try:
                            if 'eda_session_id' not in st.session_state:
                                st.session_state['eda_session_id'] = eda_memory.create_eda_session(df, st.session_state.get('file_name','dataset'))
                            eda_memory.log_insight(
                                st.session_state['eda_session_id'], 'statistical_report',
                                f"Report: {len(df)} rows, {len(df.columns)} cols.",
                                df.columns.tolist(),
                                {'row_count': len(df), 'col_count': len(df.columns),
                                 'numeric_cols': len(df.select_dtypes(include=[np.number]).columns),
                                 'categorical_cols': len(df.select_dtypes(include=['object','category']).columns)},
                                1.0)
                            st.success("✅ Report generated and saved!")
                        except Exception as e:
                            st.error(f"❌ Save failed: {str(e)}")
                            st.success("✅ Report generated (not saved)")
                    else:
                        st.success("✅ Report generated!")
        with col2:
            if 'eda_report' in st.session_state:
                st.download_button("📥 Download Report", data=st.session_state['eda_report'],
                                   file_name=f"eda_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                   mime="text/plain", use_container_width=True)
        if 'eda_report' in st.session_state:
            with st.expander("📄 View Full Report", expanded=True):
                st.text(st.session_state['eda_report'])
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Numeric Columns")
            if numeric_cols: st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            else: st.info("No numeric columns")
        with col2:
            st.markdown("#### Categorical Columns")
            if categorical_cols:
                cat_sum = [{'Column': col, 'Unique': df[col].nunique(),
                            'Top Value': df[col].mode()[0] if not df[col].mode().empty else "N/A",
                            'Frequency': (df[col]==df[col].mode()[0]).sum() if not df[col].mode().empty else 0}
                           for col in categorical_cols[:5]]
                st.dataframe(pd.DataFrame(cat_sum), use_container_width=True)
            else: st.info("No categorical columns")
        st.markdown("---")
        st.markdown("### 🔍 Missing Data Pattern")
        if df.isnull().sum().sum() > 0:
            fig_m = tools['eda'].plot_missing_heatmap(df)
            if fig_m:
                st.pyplot(fig_m)
                tools['eda'].create_download_button(fig_m, "missing_heatmap.png", "📥 Download")
        else:
            st.markdown('<div class="success-box">✅ No missing values!</div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 📊 Data Types Distribution")
        dtype_counts = df.dtypes.value_counts()
        fig_dt = plt.figure(figsize=(10,6))
        plt.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Data Types Distribution', fontsize=14, fontweight='bold')
        st.pyplot(fig_dt)
        tools['eda'].create_download_button(fig_dt, "data_types.png", "📥 Download")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING PAGE
# ══════════════════════════════════════════════════════════════════════════════

def show_feature_engineering_page():
    """
    Semi-automated Feature Engineering page.

    Phase 1  — Agent analyses ALL features (numeric + categorical) using
                Random Forest importance (RF only — categorical columns scored as whole, not per-dummy).
    Phase 2  — User reviews: keep/drop checkboxes, new-feature toggles,
                top-K slider, encoding/scaling/imbalance settings.
    Phase 3  — Apply approved operations and save df_fe to session.

    Techniques available:
      Log1p · Yeo-Johnson · Binning · DateTime extraction
      Interaction · Polynomial · Ratio · One-Hot · Label · Target encoding
      SMOTE · Undersampling · StandardScaler · MinMax · RobustScaler
    """
    st.header("⚙️ Feature Engineering")

    if st.session_state.get("validation_rejected", False):
        st.markdown(
            '<div class="error-box">⛔ BLOCKED — Fix validation format mismatch first.</div>',
            unsafe_allow_html=True); return
    if st.session_state.get("df") is None:
        st.markdown('<div class="warning-box">⚠️ No data loaded.</div>',
                    unsafe_allow_html=True); return
    if not st.session_state.get("cleaning_done"):
        st.markdown(
            '<div class="warning-box">⚠️ Run Data Cleaning before Feature Engineering.</div>',
            unsafe_allow_html=True); return

    try:
        from tools.feature_engineering import analyse_features, apply_feature_engineering
        from tools.feature_engineering_memory import FEMemoryManager
    except ImportError as err:
        st.error(f"❌ Import error: {err}"); return

    # use df_cleaned if available, else df
    df = st.session_state.get("df_cleaned")
    if df is None:
        df = st.session_state.get("df")
    if df is None:
        st.markdown('<div class="warning-box">⚠️ No cleaned data found.</div>',
                    unsafe_allow_html=True); return
    df = df.copy()

    st.markdown(
        '<div class="info-box">'
        '🤖 <b>Semi-Automated Mode</b> — Agent scores <b>all features</b> '
        '(numeric + categorical) using <b>Random Forest importance</b>. '
        'Categorical columns are scored as a whole (not per-dummy). '
        'You control every decision before anything is applied.'
        '</div>', unsafe_allow_html=True)
    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1  — TARGET & ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    st.subheader("🎯 Step 1 — Configure Target & Run Analysis")

    c1, c2, c3 = st.columns(3)
    with c1:
        target_col = st.selectbox(
            "Target column (what to predict)",
            options=df.columns.tolist(),
            index=len(df.columns)-1)
    with c2:
        problem_type = st.selectbox(
            "Problem type", ["auto", "regression", "classification"])
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        analyse_btn = st.button("🔍 Analyse & Get Suggestions",
                                type="primary", use_container_width=True)

    if target_col in df.columns:
        t = df[target_col].dropna()
        ta, tb, tc = st.columns(3)
        ta.metric("Unique values", t.nunique())
        tb.metric("Dtype",         str(t.dtype))
        tc.metric("Missing",       int(t.isnull().sum()))

    if analyse_btn:
        with st.spinner("🤖 Agent computing RF importance for ALL features…"):
            inferred = problem_type
            if problem_type == "auto":
                y = df[target_col].dropna()
                inferred = ("classification"
                            if y.dtype == object or y.nunique() <= 20
                            else "regression")
            suggestions = analyse_features(df, target_col, inferred)
            st.session_state.update({
                "fe_suggestions":  suggestions,
                "fe_phase":        2,
                "fe_target_col":   target_col,
                "fe_problem_type": inferred,
                "fe_done":         False,
            })
            st.rerun()

    if st.session_state.get("fe_phase", 1) < 2:
        st.info("👆 Pick target column and click **Analyse & Get Suggestions**.")
        return

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2  — REVIEW & OVERRIDE
    # ══════════════════════════════════════════════════════════════════
    suggestions  = st.session_state["fe_suggestions"]
    target_col   = st.session_state["fe_target_col"]
    problem_type = st.session_state["fe_problem_type"]
    imp_df       = suggestions.get("importance_df", pd.DataFrame())
    skewed_cols  = suggestions.get("skewed_cols",   [])
    imb_info     = suggestions.get("imbalance_info", None)

    st.markdown("---")
    st.subheader("🧑‍💻 Step 2 — Review & Override Agent Suggestions")
    st.markdown(
        f'<div class="success-box">✅ Analysis complete — problem type: <b>{problem_type}</b>. '
        f'Check = include | Uncheck = exclude. Nothing runs until Apply.</div>',
        unsafe_allow_html=True)

    om1, om2, om3, om4 = st.columns(4)
    om1.metric("Total features",        len(df.columns)-1)
    om2.metric("Agent suggests KEEP",   len(suggestions["keep"]))
    om3.metric("Agent suggests DROP",   len(suggestions["drop"]))
    om4.metric("New features available",len(suggestions["create"]))

    tabs = ["📊 Importance Chart", "✅ Features to Keep", "🗑️ Features to Drop",
            "✨ New Features", "⚙️ Settings"]
    if imb_info and imb_info.get("imbalanced"):
        tabs.append("⚖️ Class Imbalance")

    tab_handles = st.tabs(tabs)
    tab_imp, tab_keep, tab_drop, tab_create, tab_settings = tab_handles[:5]
    tab_imb = tab_handles[5] if len(tab_handles) > 5 else None

    # ── Importance Chart ─────────────────────────────────────────────
    with tab_imp:
        if not imp_df.empty:
            # ── RF importance chart — ALL features (numeric=cyan, categorical=purple)
            st.markdown("#### 🌲 RF Importance — all features")
            st.caption("🔵 Cyan = numeric  🟣 Purple = categorical  |  "
                       "Each categorical column scored as a whole (not per-dummy)")
            rf_plot = imp_df.head(25).sort_values("rf_score").copy()
            rf_plot["bar_color"] = rf_plot["col_type"].map(
                {"numeric": "#06b6d4", "categorical": "#a855f7"})
            fig_rf = go.Figure()
            fig_rf.add_trace(go.Bar(
                x=rf_plot["rf_score"],
                y=rf_plot["feature"],
                orientation="h",
                marker_color=rf_plot["bar_color"].tolist(),
                text=rf_plot["col_type"].tolist(),
                textposition="none",
                hovertemplate=(
                    "<b>%{y}</b><br>RF Score: %{x:.4f}"
                    "<br>Type: %{text}<extra></extra>"),
            ))
            fig_rf.update_layout(
                title="RF Importance  (cyan=numeric  purple=categorical)",
                xaxis_title="RF Importance Score",
                yaxis_title="",
                yaxis={"autorange": "reversed"},
                height=max(420, len(rf_plot) * 26),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                showlegend=False)
            st.plotly_chart(fig_rf, use_container_width=True)

            # full ranked table
            with st.expander("📋 Full Ranked Feature Table", expanded=False):
                tbl  = imp_df.copy()
                show = [c for c in ["rank","feature","col_type","rf_score"]
                        if c in tbl.columns]
                st.dataframe(
                    tbl[show].rename(columns={
                        "col_type": "Type",
                        "rf_score": "RF Importance"}),
                    use_container_width=True)

            # skewness table
            if skewed_cols:
                st.markdown("---")
                st.markdown(f"#### 📐 Skewed Columns ({len(skewed_cols)})")
                st.dataframe(pd.DataFrame(skewed_cols),
                             use_container_width=True)
        else:
            st.info("Run analysis first to see importance scores.")

    # ── Features to Keep ─────────────────────────────────────────────
    with tab_keep:
        st.markdown("**Agent recommends keeping these.** Uncheck any to exclude.")
        search_k = st.text_input("🔍 Filter", key="search_keep",
                                  placeholder="column name…")
        filtered = ([f for f in suggestions["keep"]
                     if search_k.lower() in f["name"].lower()]
                    if search_k else suggestions["keep"])

        ka, kb, _ = st.columns([1, 1, 4])
        if ka.button("☑ Select All",   key="sel_all_k"):
            for f in suggestions["keep"]:
                st.session_state[f"keep_{f['name']}"] = True
        if kb.button("☐ Deselect All", key="desel_all_k"):
            for f in suggestions["keep"]:
                st.session_state[f"keep_{f['name']}"] = False

        hc = st.columns([3, 1.5, 1.5, 4])
        for h, lbl in zip(hc, ["Feature","Type","RF Score","Detail"]):
            h.markdown(f"**{lbl}**")
        st.divider()

        for feat in filtered:
            row = st.columns([3, 1.5, 1.5, 4])
            row[0].checkbox(
                feat["name"],
                value=st.session_state.get(f"keep_{feat['name']}", True),
                key=f"keep_{feat['name']}")
            row[1].markdown(
                f"`{feat['dtype']}`"
                f" {'🔢' if feat.get('reason') == 'good_rf_importance' else '🔤'}")
            rf = feat.get("rf_score", 0)
            row[2].markdown(
                f"{'🟢' if rf>0.05 else '🟡' if rf>0.01 else '🔴'} {rf:.4f}")
            row[3].markdown(
                f"<span style='color:#94a3b8;font-size:12px'>"
                f"{feat['detail']}</span>", unsafe_allow_html=True)

    # ── Features to Drop ─────────────────────────────────────────────
    with tab_drop:
        st.markdown("**Agent recommends dropping these.** Check to rescue.")
        REASON_LABELS = {
            "high_missing":    "🔴 High missing",
            "constant":        "⚫ Constant",
            "high_correlation":"🟠 High correlation",
            "low_importance":  "🟡 Low RF importance",
        }
        da, db, _ = st.columns([1, 1, 4])
        if da.button("☑ Rescue All", key="rescue_all"):
            for f in suggestions["drop"]:
                st.session_state[f"rescue_{f['name']}"] = True
        if db.button("☐ Drop All",   key="drop_all"):
            for f in suggestions["drop"]:
                st.session_state[f"rescue_{f['name']}"] = False

        if not suggestions["drop"]:
            st.success("✅ No features flagged for dropping!")
        else:
            hd = st.columns([3, 1.5, 4, 1.2, 1])
            for h, lbl in zip(hd, ["Feature","Dtype","Reason","RF Score","Rescue?"]):
                h.markdown(f"**{lbl}**")
            st.divider()
            for feat in suggestions["drop"]:
                row = st.columns([3, 1.5, 4, 1.2, 1])
                row[0].markdown(f"`{feat['name']}`")
                row[1].markdown(f"`{feat['dtype']}`")
                row[2].markdown(
                    f"{REASON_LABELS.get(feat['reason'],feat['reason'])} "
                    f"<span style='color:#94a3b8;font-size:11px'>"
                    f"({feat['detail']})</span>", unsafe_allow_html=True)
                row[3].markdown(f"{feat.get('rf_score',0):.4f}")
                row[4].checkbox(
                    "Keep",
                    value=st.session_state.get(f"rescue_{feat['name']}", False),
                    key=f"rescue_{feat['name']}")

    # ── New Features ─────────────────────────────────────────────────
    with tab_create:
        st.markdown("**Toggle new features ON/OFF.** Agent pre-enables recommended ones.")
        METHOD_ICONS = {
            "log_transform":       "📐",
            "yeo_johnson":         "📐",
            "datetime_extraction": "📅",
            "interaction":         "🔗",
            "polynomial":          "🔢",
            "ratio":               "➗",
            "bin":                 "📦",
        }
        CATEGORY_LABELS = {
            "skewness":    "📐 Skewness Fix  (log1p · Yeo-Johnson · Binning)",
            "datetime":    "📅 DateTime Extraction",
            "interaction": "🔗 Interaction Features",
            "polynomial":  "🔢 Polynomial Features",
            "ratio":       "➗ Ratio Features",
        }

        # group by category
        categories: Dict = {}
        for i, feat in enumerate(suggestions["create"]):
            cat = feat.get("category", "other")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((i, feat))

        for cat, items in categories.items():
            st.markdown(
                f"**{CATEGORY_LABELS.get(cat, cat.title())} "
                f"— {len(items)} suggestion(s)**")
            for i, feat in items:
                icon = METHOD_ICONS.get(feat["method"], "✨")
                row  = st.columns([0.5, 3, 2, 5])
                row[0].checkbox(
                    "",
                    value=st.session_state.get(
                        f"create_{i}_{feat['name']}", feat["enabled"]),
                    key=f"create_{i}_{feat['name']}")
                row[1].markdown(f"**{icon} {feat['name']}**")
                row[2].markdown(f"`{feat['method']}`")
                row[3].markdown(
                    f"<span style='color:#94a3b8;font-size:12px'>"
                    f"{feat['description']}</span>", unsafe_allow_html=True)
            st.divider()

        # custom feature
        st.markdown("#### ➕ Add Your Own Feature")
        with st.expander("Define a custom feature", expanded=False):
            cn = st.text_input("Feature name", key="custom_name")
            cm = st.selectbox("Method",
                ["log_transform","yeo_johnson","polynomial","interaction",
                 "ratio","bin","datetime_extraction"],
                key="custom_method")
            cs = st.multiselect("Source columns",
                [c for c in df.columns if c != target_col],
                key="custom_sources")
            cd = st.text_input("Description", key="custom_desc")
            nb = st.slider("Bins (bin method only)", 2, 20, 5,
                           key="custom_bins") if cm == "bin" else 5
            if st.button("➕ Add", key="add_custom"):
                if cn and cs:
                    new_entry = {
                        "name": cn, "method": cm, "sources": cs,
                        "description": cd or f"Custom {cm}",
                        "enabled": True, "category": "custom", "n_bins": nb
                    }
                    suggestions["create"].append(new_entry)
                    st.session_state["fe_suggestions"] = suggestions
                    idx = len(suggestions["create"]) - 1
                    st.session_state[f"create_{idx}_{cn}"] = True
                    st.success(f"✅ Added '{cn}'")
                    st.rerun()
                else:
                    st.warning("Enter a name and at least one source column.")

    # ── Settings ─────────────────────────────────────────────────────
    with tab_settings:
        st.markdown("#### 🔢 Encoding")
        st.selectbox(
            "Encoding method",
            ["auto", "onehot", "label", "target"],
            index=["auto","onehot","label","target"].index(
                st.session_state.get("fe_encode_method","auto")),
            key="fe_encode_method",
            help="auto: ≤15 unique → one-hot, >15 → label  |  target: mean target per category")

        st.markdown("---")
        st.markdown("#### 📏 Scaling")
        st.selectbox(
            "Scaling method",
            ["standard", "minmax", "robust", "none"],
            index=["standard","minmax","robust","none"].index(
                st.session_state.get("fe_scale_method","standard")),
            key="fe_scale_method")

        st.markdown("---")
        st.markdown("#### 🎯 Feature Selection — Top-K by RF Importance")
        st.slider(
            "Keep top-K features by RF importance  (0 = keep all approved)",
            min_value=0,
            max_value=max(1, len(df.columns)-1),
            value=st.session_state.get("fe_top_k", min(15, len(df.columns)-1)),
            key="fe_top_k",
            help="Ranked by RF importance. 0 = no automatic cutoff.")

        st.markdown("---")
        st.markdown("#### ✂️ Train / Test Split")
        st.markdown(
            '<div class="info-box">'
            '🔒 <b>Split happens BEFORE encoding and scaling</b> — '
            'encoders and scalers are fitted on training data only, '
            'then applied to test data. This prevents data leakage.'
            '</div>', unsafe_allow_html=True)
        test_size_slider = st.slider(
            "Test set size",
            min_value=0.10, max_value=0.40,
            value=st.session_state.get("fe_test_size", 0.20),
            step=0.05, format="%.2f",
            key="fe_test_size",
            help="e.g. 0.20 = 80% train / 20% test")
        n_rows = len(df)
        train_n = int(n_rows * (1 - test_size_slider))
        test_n  = n_rows - train_n
        ts1, ts2, ts3 = st.columns(3)
        ts1.metric("Total rows",  f"{n_rows:,}")
        ts2.metric("Train rows",  f"{train_n:,}  ({round((1-test_size_slider)*100)}%)")
        ts3.metric("Test rows",   f"{test_n:,}  ({round(test_size_slider*100)}%)")

        st.markdown("---")
        if st.button("🔄 Re-run Analysis", key="rerun_analysis"):
            st.session_state["fe_phase"]       = 1
            st.session_state["fe_suggestions"] = None
            st.session_state["fe_done"]        = False
            st.rerun()

    # ── Class Imbalance tab ───────────────────────────────────────────
    if tab_imb is not None:
        with tab_imb:
            st.markdown("#### ⚖️ Class Imbalance")
            if imb_info:
                st.markdown(
                    f'<div class="warning-box">'
                    f'⚠️ Imbalance detected!  Minority/Majority ratio = '
                    f'<b>{imb_info["ratio"]:.3f}</b><br>'
                    f'Minority: <b>{imb_info["minority_class"]}</b>  |  '
                    f'Majority: <b>{imb_info["majority_class"]}</b></div>',
                    unsafe_allow_html=True)
                st.write("**Class counts:**")
                st.json(imb_info["class_counts"])
            st.checkbox(
                "Apply class balancing",
                value=st.session_state.get("fe_handle_imbalance", False),
                key="fe_handle_imbalance")
            st.selectbox(
                "Method",
                ["smote", "undersample"],
                index=["smote","undersample"].index(
                    st.session_state.get("fe_imbalance_method","smote")),
                key="fe_imbalance_method",
                help="SMOTE: oversample minority  |  undersample: reduce majority")
            st.info("💡 Requires: pip install imbalanced-learn")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3  — APPLY
    # ══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🚀 Step 3 — Apply Approved Feature Engineering")

    approved_keep = [f["name"] for f in suggestions["keep"]
                     if st.session_state.get(f"keep_{f['name']}", True)]
    rescued = []
    for feat in suggestions["drop"]:
        if st.session_state.get(f"rescue_{feat['name']}", False):
            approved_keep.append(feat["name"])
            rescued.append(feat["name"])

    approved_create = [feat for i, feat in enumerate(suggestions["create"])
                       if st.session_state.get(
                           f"create_{i}_{feat['name']}", feat["enabled"])]

    encode_method    = st.session_state.get("fe_encode_method",  "auto")
    scale_method     = st.session_state.get("fe_scale_method",   "standard")
    top_k            = st.session_state.get("fe_top_k",          0)
    handle_imbalance = st.session_state.get("fe_handle_imbalance", False)
    imbalance_method = st.session_state.get("fe_imbalance_method", "smote")

    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    sc1.metric("Features approved",  len(approved_keep))
    sc2.metric("Features to drop",   (len(df.columns)-1) - len(approved_keep))
    sc3.metric("New features",        len(approved_create))
    sc4.metric("Rescued",             len(rescued))
    sc5.metric("Top-K",               str(top_k) if top_k > 0 else "all")

    if rescued:
        st.markdown(
            f'<div class="warning-box">⚠️ Rescued <b>{len(rescued)}</b> ' 
            f'agent-flagged features: {rescued}</div>',
            unsafe_allow_html=True)

    apply_col, _, _ = st.columns([2, 1, 1])
    with apply_col:
        apply_btn = st.button(
            "✅ Apply & Finalise Feature Engineering",
            type="primary", use_container_width=True,
            disabled=len(approved_keep) == 0)

    if apply_btn:
        with st.spinner("⚙️ Applying approved operations — encoding/scaling on train only…"):
            try:
                fe_mem     = FEMemoryManager()
                session_id = fe_mem.create_fe_session(
                    dataset_name=st.session_state.get("file_name","dataset"),
                    problem_type=problem_type,
                    target_column=target_col,
                    input_features=len(df.columns)-1,
                    rows=len(df))

                test_size = st.session_state.get("fe_test_size", 0.2)

                # apply_feature_engineering now returns (splits_dict, report)
                # splits = {X_train, X_test, y_train, y_test, feature_names}
                splits, report = apply_feature_engineering(
                    df=df,
                    target_col=target_col,
                    problem_type=problem_type,
                    keep_cols=approved_keep,
                    create_approved=approved_create,
                    encode_cats=True,
                    encoding_method=encode_method,
                    handle_imbalance=handle_imbalance,
                    imbalance_method=imbalance_method,
                    scale_method=scale_method,
                    top_k=top_k,
                    importance_df=imp_df,
                    test_size=test_size,
                    session_id=session_id,
                    fe_mem=fe_mem,
                )

                X_train = splits["X_train"]
                X_test  = splits["X_test"]
                y_train = splits["y_train"]
                y_test  = splits["y_test"]
                feature_names = splits["feature_names"]

                # log to memory
                fe_mem.log_selected_features(session_id, [
                    {"name": c,
                     "dtype": str(X_train[c].dtype),
                     "rf_score": float(
                         imp_df.loc[imp_df["feature"]==c,"rf_score"].values[0])
                         if not imp_df.empty and c in imp_df["feature"].values
                         else None,
                     "pearson_r": None,
                     "method": "user_approved"}
                    for c in feature_names
                ])

                for col in report["dropped"]:
                    reason = next((f["reason"] for f in suggestions["drop"]
                                   if f["name"]==col), "user_removed")
                    detail = next((f["detail"] for f in suggestions["drop"]
                                   if f["name"]==col), "Excluded by user")
                    rf_s = (float(imp_df.loc[imp_df["feature"]==col,
                                             "rf_score"].values[0])
                            if not imp_df.empty
                            and col in imp_df["feature"].values else None)
                    fe_mem.log_dropped_feature(session_id, col, reason,
                                               detail, rf_score=rf_s)

                if not imp_df.empty:
                    rf_list = list(zip(imp_df["feature"], imp_df["rf_score"]))
                    fe_mem.log_importance_scores(
                        session_id, "random_forest", rf_list, pearson_map={})

                for info in skewed_cols:
                    col = info["name"]
                    for suffix, meth in [("_log1p","log1p"),
                                         ("_yeojohnson","yeo_johnson")]:
                        if any(f.get("name")==f"{col}{suffix}"
                               for f in approved_create):
                            fe_mem.log_skewness_op(
                                session_id, col,
                                info["skewness"], meth, f"{col}{suffix}")

                if handle_imbalance and report.get("imbalance"):
                    fe_mem.log_imbalance_op(
                        session_id=session_id,
                        method=imbalance_method,
                        target_column=target_col,
                        class_counts_before=(imb_info["class_counts"]
                                             if imb_info else {}),
                        class_counts_after={},
                        rows_before=len(df),
                        rows_after=len(X_train)+len(X_test))

                fe_mem.log_insight(
                    session_id, "summary",
                    (f"FE complete. features={len(feature_names)}, "
                     f"dropped={len(report['dropped'])}, "
                     f"created={len(report['created'])}, "
                     f"encoding={encode_method}, scaling={scale_method}, "
                     f"test_size={test_size}, top_k={top_k}. "
                     f"Target='{target_col}', problem='{problem_type}'."),
                    feature_names[:10],
                    {"features": len(feature_names),
                     "dropped":  len(report["dropped"]),
                     "created":  len(report["created"]),
                     "train_rows": len(X_train),
                     "test_rows":  len(X_test)},
                    confidence=1.0)

                fe_mem.update_session_output(
                    session_id,
                    output_features=len(feature_names),
                    rows_after=len(X_train)+len(X_test),
                    notes=(f"features={len(feature_names)}, "
                           f"train={len(X_train)}, test={len(X_test)}"))

                # Save to session state — 4 splits + report
                st.session_state.update({
                    "fe_X_train":  X_train,
                    "fe_X_test":   X_test,
                    "fe_y_train":  y_train,
                    "fe_y_test":   y_test,
                    "fe_report": {
                        "session_id":        session_id,
                        "target_column":     target_col,
                        "problem_type":      problem_type,
                        "selected_features": feature_names,
                        "dropped_features":  report["dropped"],
                        "created_features":  report["created"],
                        "importance_df":     imp_df,
                        "encoded":           report["encoded"],
                        "imbalance":         report.get("imbalance", []),
                        "scaled":            report.get("scaled", []),
                        "errors":            report["errors"],
                        "train_rows":        len(X_train),
                        "test_rows":         len(X_test),
                        "test_size":         test_size,
                    },
                    "fe_done":         True,
                    "fe_session_id":   session_id,
                    "fe_target_col":   target_col,
                    "fe_problem_type": problem_type,
                    "fe_phase":        3,
                })
                st.success(
                    f"✅ Feature engineering complete! "
                    f"Train: {len(X_train)} rows | Test: {len(X_test)} rows | "
                    f"Features: {len(feature_names)}")
                st.rerun()

            except Exception as exc:
                st.markdown(
                    f'<div class="error-box">❌ Error: {exc}</div>',
                    unsafe_allow_html=True)
                import traceback; st.code(traceback.format_exc())

    # ══════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════
    if not st.session_state.get("fe_done"):
        return

    rf_rep  = st.session_state.get("fe_report", {})
    X_train = st.session_state.get("fe_X_train", pd.DataFrame())
    X_test  = st.session_state.get("fe_X_test",  pd.DataFrame())
    y_train = st.session_state.get("fe_y_train", pd.Series(dtype=float))
    y_test  = st.session_state.get("fe_y_test",  pd.Series(dtype=float))

    st.markdown("---")
    st.subheader("📊 Final Results")

    r1, r2, r3, r4, r5, r6 = st.columns(6)
    r1.metric("Input features",  len(df.columns)-1)
    r2.metric("Final features",  len(rf_rep.get("selected_features",[])))
    r3.metric("Created",         len(rf_rep.get("created_features",[])))
    r4.metric("Dropped",         len(rf_rep.get("dropped_features",[])))
    r5.metric("Train rows",      len(X_train))
    r6.metric("Test rows",       len(X_test))

    st.markdown(
        '<div class="success-box">'
        '🔒 <b>Leakage-free split:</b> Encoding and scaling were fitted on '
        'X_train only, then applied to X_test. '
        'The 4 CSVs below are ready for model training.'
        '</div>', unsafe_allow_html=True)

    rt1, rt2, rt3, rt4 = st.tabs(
        ["✅ Final Features", "📈 Importance", "🔍 Preview & Download", "📋 Full Log"])

    with rt1:
        feat_names = rf_rep.get("selected_features", [])
        if feat_names:
            sel_rows = []
            for feat in feat_names:
                row_imp = imp_df[imp_df["feature"]==feat] if not imp_df.empty else pd.DataFrame()
                rf_v = f"{row_imp['rf_score'].values[0]:.4f}" if not row_imp.empty else "—"
                dtype_v = str(X_train[feat].dtype) if feat in X_train.columns else "—"
                sel_rows.append({"Feature": feat, "Dtype": dtype_v, "RF Score": rf_v})
            st.dataframe(pd.DataFrame(sel_rows), use_container_width=True, height=400)

    with rt2:
        imp_f = rf_rep.get("importance_df", pd.DataFrame())
        if imp_f is not None and not imp_f.empty:
            rf_plot2 = imp_f.head(20).sort_values("rf_score").copy()
            rf_plot2["bar_color"] = rf_plot2["col_type"].map(
                {"numeric": "#06b6d4", "categorical": "#a855f7"})
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                x=rf_plot2["rf_score"], y=rf_plot2["feature"],
                orientation="h",
                marker_color=rf_plot2["bar_color"].tolist(),
                text=rf_plot2["col_type"].tolist(), textposition="none",
                hovertemplate="<b>%{y}</b><br>RF: %{x:.4f}<br>Type: %{text}<extra></extra>"))
            fig3.update_layout(
                title="RF Importance (cyan=numeric  purple=categorical)",
                xaxis_title="RF Score", yaxis_title="",
                yaxis={"autorange": "reversed"},
                height=max(380, len(rf_plot2)*26),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0", showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

    with rt3:
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.markdown("#### 📥 Download 4 Split CSVs")
        st.markdown(
            "These are ready for model training. "
            "X_train and X_test are encoded and scaled. "
            "y_train and y_test are the raw target labels.")

        dc1, dc2 = st.columns(2)
        with dc1:
            st.markdown(f"**X_train** — {X_train.shape[0]} rows × {X_train.shape[1]} features")
            st.download_button(
                "📥 Download X_train.csv",
                data=X_train.to_csv(index=False).encode("utf-8"),
                file_name=f"X_train_{ts_str}.csv", mime="text/csv",
                use_container_width=True, type="primary")
            st.markdown(f"**X_test** — {X_test.shape[0]} rows × {X_test.shape[1]} features")
            st.download_button(
                "📥 Download X_test.csv",
                data=X_test.to_csv(index=False).encode("utf-8"),
                file_name=f"X_test_{ts_str}.csv", mime="text/csv",
                use_container_width=True, type="primary")
        with dc2:
            st.markdown(f"**y_train** — {len(y_train)} rows")
            st.download_button(
                "📥 Download y_train.csv",
                data=y_train.to_frame().to_csv(index=False).encode("utf-8"),
                file_name=f"y_train_{ts_str}.csv", mime="text/csv",
                use_container_width=True)
            st.markdown(f"**y_test** — {len(y_test)} rows")
            st.download_button(
                "📥 Download y_test.csv",
                data=y_test.to_frame().to_csv(index=False).encode("utf-8"),
                file_name=f"y_test_{ts_str}.csv", mime="text/csv",
                use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🔍 Preview")
        prev_tab1, prev_tab2 = st.tabs(["X_train (first 10)", "X_test (first 10)"])
        with prev_tab1:
            st.dataframe(X_train.head(10), use_container_width=True)
        with prev_tab2:
            st.dataframe(X_test.head(10), use_container_width=True)

    with rt4:
        if rf_rep.get("encoded"):
            st.markdown("**Encoding (fitted on X_train only):**")
            for e in rf_rep["encoded"]: st.write(f"• {e}")
        if rf_rep.get("imbalance"):
            st.markdown("**Class Balancing (X_train only):**")
            for e in rf_rep["imbalance"]: st.write(f"• {e}")
        if rf_rep.get("scaled"):
            st.markdown(f"**Scaling (fitted on X_train only):** "
                        f"{len(rf_rep['scaled'])} numeric cols")
        if rf_rep.get("errors"):
            st.markdown("**⚠️ Errors:**")
            for e in rf_rep["errors"]: st.warning(e)

# ══════════════════════════════════════════════════════════════════════════════
# REMAINING PAGE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING PAGE
# ══════════════════════════════════════════════════════════════════════════════


def _show_reused_pipeline_page():
    """
    Shows the reused pipeline info and allows:
    1. Viewing the reused model metrics
    2. Saving a new snapshot for the current (new) dataset
    3. Exiting reuse mode to run full pipeline
    """
    pipeline   = st.session_state.get("reused_pipeline", {})
    snap_id    = st.session_state.get("reused_snapshot_id", "")
    model      = pipeline.get("model")
    model_name = pipeline.get("model_name", "")
    target_col = pipeline.get("target_col", "")
    prob_type  = pipeline.get("problem_type", "")
    feat_names = pipeline.get("feature_names", [])

    st.markdown(
        f'<div class="success-box">'
        f'♻️ <b>Reused Pipeline Active</b><br>'
        f'Model: <b>{model_name}</b> | Target: <b>{target_col}</b> | '
        f'Problem: <b>{prob_type}</b><br>'
        f'Source snapshot: <code>{snap_id}</code>'
        f'</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Model info ────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Model",    model_name)
    col2.metric("Target",   target_col)
    col3.metric("Features", len(feat_names))

    st.markdown("**Features used by this pipeline:**")
    st.write(", ".join(feat_names))

    st.markdown("---")
    st.markdown("### 💾 Save New Snapshot for This Dataset")
    st.markdown(
        '<div class="info-box">'
        'You are reusing the pipeline from <b>' + snap_id + '</b>.<br>'
        'You can now save a new snapshot for the <b>current dataset</b> '
        'so future datasets can be compared against both pipelines.'
        '</div>', unsafe_allow_html=True)

    df_current = st.session_state.get("df")
    snap_name  = st.text_input(
        "New snapshot name",
        value=st.session_state.get("file_name", "dataset").replace(".csv", ""),
        key="reuse_snap_name")

    if st.button("💾 Save Snapshot for Current Dataset",
                 key="save_reuse_snapshot", type="primary",
                 use_container_width=True):
        if df_current is None:
            st.error("❌ No dataset in session.")
        else:
            with st.spinner("Saving snapshot…"):
                try:
                    raw_cols = [c for c in df_current.columns if c != target_col]
                    sid = save_snapshot(
                        df_original   = df_current,
                        target_col    = target_col,
                        problem_type  = prob_type,
                        feature_names = raw_cols,
                        encoders      = pipeline.get("encoders", {}),
                        scaler        = pipeline.get("scaler"),
                        model         = model,
                        model_name    = model_name,
                        model_metrics = {},
                        fe_config     = pipeline.get("fe_config", {}),
                        dataset_name  = snap_name or "dataset")
                    st.success(
                        f"✅ New snapshot saved! ID: `{sid}`\n\n"
                        f"Go to **🔄 Pipeline Snapshots** to see both saved pipelines.\n\n"
                        f"Now upload a third dataset → Validation → Run Drift Check "
                        f"to compare against both snapshots.")
                    st.session_state["last_snapshot_id"] = sid
                except Exception as e:
                    import traceback
                    st.error(f"❌ Error: {e}")
                    st.code(traceback.format_exc())

    st.markdown("---")
    st.markdown("### 🔄 Exit Reuse Mode")
    st.markdown(
        '<div class="info-box">'
        'If you want to run the full pipeline on this new dataset '
        '(Feature Engineering + Model Training from scratch), '
        'click below to exit reuse mode.'
        '</div>', unsafe_allow_html=True)

    if st.button("🔄 Exit Reuse — Run Full Pipeline",
                 key="exit_reuse_btn"):
        st.session_state["reuse_active"]       = False
        st.session_state["reused_pipeline"]    = None
        st.session_state["reused_snapshot_id"] = None
        st.session_state["mt_done"]            = False
        st.session_state["mt_results"]         = {}
        st.rerun()


def show_model_training_page():
    """
    Model Training page — 8 steps.
    Handles two modes:
      1. Normal: FE was run, X_train/X_test/y_train/y_test in session state
      2. Reuse:  Pipeline loaded from snapshot, show reused model results
                 and allow saving a new snapshot for the current dataset.
    """
    st.header("🤖 Model Training")

    # ── REUSE MODE ────────────────────────────────────────────────────
    # When user clicks "Reuse Pipeline" in Validation, reuse_active=True
    # and the loaded pipeline is in session state. Show reused model info
    # and allow saving a new snapshot for this new dataset.
    if st.session_state.get("reuse_active"):
        _show_reused_pipeline_page()
        return

    # ── NORMAL MODE ──────────────────────────────────────────────────
    if not st.session_state.get("fe_done"):
        st.markdown(
            '<div class="warning-box">⚠️ Complete <b>Feature Engineering</b> '
            'first to generate the model-ready dataset.</div>',
            unsafe_allow_html=True)
        return

    X_train_fe = st.session_state.get("fe_X_train")
    X_test_fe  = st.session_state.get("fe_X_test")
    y_train_fe = st.session_state.get("fe_y_train")
    y_test_fe  = st.session_state.get("fe_y_test")

    if X_train_fe is None or X_train_fe.empty:
        st.markdown(
            '<div class="warning-box">⚠️ No split data found. '
            'Re-run Feature Engineering → click Apply to generate '
            'X_train, X_test, y_train, y_test.</div>',
            unsafe_allow_html=True)
        return
    df_fe = X_train_fe  # kept for backward compatibility in metrics display

    try:
        from tools.model_training import (
            recommend_models, train_models, retrain_model,
            get_comparison_df, get_best_model_name,
            save_best_model, get_fit_badge, get_fix_suggestions,
            _build_model_registry, MODELS_DIR
        )
        from tools.model_memory import ModelMemoryManager
    except ImportError as err:
        st.error(f"❌ Import error: {err}"); return

    fe_report    = st.session_state.get("fe_report", {})
    target_col   = fe_report.get("target_column") or \
                   st.session_state.get("fe_target_col", "")
    problem_type = fe_report.get("problem_type") or \
                   st.session_state.get("fe_problem_type", "classification")

    # X_train_fe has no target column (it was correctly separated during FE split)
    # so we validate target_col from fe_report, not from df_fe.columns
    if not target_col:
        st.markdown(
            '<div class="error-box">❌ Target column not found. '
            'Please re-run Feature Engineering and ensure a target column is selected.</div>',
            unsafe_allow_html=True)
        return

    st.markdown(
        f'<div class="info-box">'
        f'📊 Dataset: <b>{df_fe.shape[0]:,} rows × {df_fe.shape[1]} cols</b> &nbsp;|&nbsp; '
        f'Target: <b>{target_col}</b> &nbsp;|&nbsp; '
        f'Problem: <b>{problem_type}</b>'
        f'</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # STEP 1 — DATA PREVIEW
    # ══════════════════════════════════════════════════════════════════
    with st.expander("📋 Step 1 — Data Preview", expanded=False):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Train rows",  f"{X_train_fe.shape[0]:,}")
        c2.metric("Test rows",   f"{X_test_fe.shape[0]:,}")
        c3.metric("Features",    f"{X_train_fe.shape[1]}")
        c4.metric("Target",      target_col)
        c5.metric("Problem",     problem_type)

        prev_t1, prev_t2, prev_t3, prev_t4 = st.tabs([
            "X_train", "y_train", "X_test", "y_test"])
        with prev_t1:
            st.markdown(f"**X_train** — {X_train_fe.shape[0]:,} rows × {X_train_fe.shape[1]} features (encoded + scaled)")
            st.dataframe(X_train_fe.head(10), use_container_width=True)
        with prev_t2:
            st.markdown(f"**y_train** — {len(y_train_fe):,} rows  |  target: `{target_col}`")
            st.dataframe(y_train_fe.head(10).to_frame(), use_container_width=True)
        with prev_t3:
            st.markdown(f"**X_test** — {X_test_fe.shape[0]:,} rows × {X_test_fe.shape[1]} features (encoded + scaled)")
            st.dataframe(X_test_fe.head(10), use_container_width=True)
        with prev_t4:
            st.markdown(f"**y_test** — {len(y_test_fe):,} rows  |  target: `{target_col}`")
            st.dataframe(y_test_fe.head(10).to_frame(), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════
    # STEP 2 — AGENT RECOMMENDATION
    # ══════════════════════════════════════════════════════════════════
    st.subheader("🤖 Step 2 — Agent Recommendation")

    if st.button("🔍 Get Model Recommendations", type="primary",
                 key="get_recommendations"):
        with st.spinner("Agent analysing dataset…"):
            # pass X_train + y_train combined for recommendation analysis
            _df_for_rec = X_train_fe.copy()
            _df_for_rec[target_col] = y_train_fe.values
            recs = recommend_models(_df_for_rec, target_col, problem_type)
            st.session_state["mt_recommendations"] = recs
            st.session_state["mt_problem_type"]    = problem_type
            st.session_state["mt_target_col"]      = target_col

    recs = st.session_state.get("mt_recommendations", [])
    if recs:
        st.markdown("**Agent recommends (based on dataset size and type):**")
        for r in recs:
            badge = {"linear":"🔵","ensemble":"🟢","distance":"🟠",
                     "kernel":"🟣","probabilistic":"🟡","tree":"🟤"}.get(
                         r.get("type",""),"⚪")
            st.markdown(
                f'<div class="info-box">'
                f'{badge} <b>#{r["priority"]} — {r["name"]}</b><br>'
                f'{r["reason"]}'
                f'</div>', unsafe_allow_html=True)
    else:
        st.info("👆 Click **Get Model Recommendations** to start.")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # STEP 3 — MODEL SELECTION
    # ══════════════════════════════════════════════════════════════════
    st.subheader("✅ Step 3 — Select Models to Train")

    registry      = _build_model_registry(problem_type)
    all_models    = list(registry.keys())
    rec_names     = [r["name"] for r in recs]
    default_check = rec_names if rec_names else all_models[:3]

    selected_models = []
    cols = st.columns(3)
    for i, model_name in enumerate(all_models):
        with cols[i % 3]:
            checked = st.checkbox(
                model_name,
                value=(model_name in default_check),
                key=f"model_sel_{model_name}")
            if checked:
                selected_models.append(model_name)

    if not selected_models:
        st.warning("Select at least one model to continue.")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # STEP 4 — TRAINING SETTINGS
    # ══════════════════════════════════════════════════════════════════
    st.subheader("⚙️ Step 4 — Training Settings")

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        # Split was done in Feature Engineering — display info only
        _tr = X_train_fe.shape[0]
        _te = X_test_fe.shape[0]
        _tot = _tr + _te
        st.metric("Train / Test split",
                  f"{_tr:,} / {_te:,}",
                  help="Split was configured in Feature Engineering settings")
        st.caption(f"Train: {round(_tr/_tot*100)}% | Test: {round(_te/_tot*100)}%")
        test_size = _te / _tot  # kept for session logging
    with sc2:
        use_cv   = st.checkbox("Enable Cross-Validation (5-fold)",
                                value=True, key="mt_use_cv")
        cv_folds = 5 if use_cv else 0
    with sc3:
        tuning_mode = st.selectbox(
            "Training mode",
            ["quick", "tuned"],
            format_func=lambda x: (
                "⚡ Quick (default params)"
                if x == "quick" else
                "🔧 Tuned (custom hyperparams)"),
            key="mt_tuning_mode",
            help="Quick = default sklearn params | Tuned = you set hyperparams per model")

    # ── Hyperparameter UI (only in tuned mode) ────────────────────────
    # Uses tune_params from registry (not default_params) to decide
    # which sliders to show — this is what the model can actually tune.
    custom_params = {}   # {model_name: {param: value}}

    if tuning_mode == "tuned" and selected_models:
        st.markdown("---")
        st.markdown("#### 🔧 Hyperparameter Configuration")
        st.markdown(
            '<div class="info-box">ℹ️ Set hyperparameters for each selected model. '
            'Leave defaults if unsure.</div>',
            unsafe_allow_html=True)

        for model_name in selected_models:
            spec      = registry.get(model_name, {})
            default_p = spec.get("default_params", {})
            tune_p    = spec.get("tune_params",    {})

            with st.expander(f"⚙️ {model_name} — Hyperparameters",
                             expanded=True):
                # Start from default params — user overrides specific ones
                params = dict(default_p)

                if not tune_p:
                    # model has no tunable params (e.g. Linear Regression,
                    # Naive Bayes) — just inform user
                    st.info(f"ℹ️ **{model_name}** has no tunable hyperparameters. "
                            f"It will train with its optimal fixed settings.")
                    custom_params[model_name] = params
                    continue

                # ── max_depth ─────────────────────────────────────────
                if "max_depth" in tune_p:
                    col_a, col_b = st.columns([1, 3])
                    with col_a:
                        use_depth = st.checkbox(
                            "Limit max_depth", value=True,
                            key=f"hp_use_depth_{model_name}",
                            help="Unchecked = unlimited depth (may overfit)")
                    with col_b:
                        if use_depth:
                            params["max_depth"] = st.slider(
                                "max_depth", 1, 30, 5,
                                key=f"hp_max_depth_{model_name}",
                                help="Max tree depth. Lower = simpler = less overfit.")
                        else:
                            params["max_depth"] = None
                            st.caption("max_depth = None (unlimited)")

                # ── min_samples_leaf ──────────────────────────────────
                if "min_samples_leaf" in tune_p:
                    params["min_samples_leaf"] = st.slider(
                        "min_samples_leaf", 1, 50,
                        int(default_p.get("min_samples_leaf", 1)),
                        key=f"hp_min_leaf_{model_name}",
                        help="Min samples at a leaf node. Higher = less overfit.")

                # ── min_samples_split ─────────────────────────────────
                if "min_samples_split" in tune_p:
                    params["min_samples_split"] = st.slider(
                        "min_samples_split", 2, 50,
                        int(default_p.get("min_samples_split", 2)),
                        key=f"hp_min_split_{model_name}",
                        help="Min samples to split a node.")

                # ── n_estimators ──────────────────────────────────────
                if "n_estimators" in tune_p:
                    params["n_estimators"] = st.slider(
                        "n_estimators", 10, 500,
                        int(default_p.get("n_estimators", 100)),
                        step=10,
                        key=f"hp_n_est_{model_name}",
                        help="Number of trees. More = slower but more stable.")

                # ── learning_rate ─────────────────────────────────────
                if "learning_rate" in tune_p:
                    params["learning_rate"] = st.slider(
                        "learning_rate", 0.001, 0.5,
                        float(default_p.get("learning_rate", 0.1)),
                        step=0.005, format="%.3f",
                        key=f"hp_lr_{model_name}",
                        help="Shrinkage step. Lower = more accurate but slower.")

                # ── subsample (XGBoost) ───────────────────────────────
                if "subsample" in tune_p:
                    params["subsample"] = st.slider(
                        "subsample", 0.3, 1.0,
                        float(default_p.get("subsample", 1.0)),
                        step=0.05, format="%.2f",
                        key=f"hp_subsample_{model_name}",
                        help="Fraction of rows per tree. <1 reduces overfit.")

                # ── colsample_bytree (XGBoost) ────────────────────────
                if "colsample_bytree" in tune_p:
                    params["colsample_bytree"] = st.slider(
                        "colsample_bytree", 0.3, 1.0,
                        float(default_p.get("colsample_bytree", 1.0)),
                        step=0.05, format="%.2f",
                        key=f"hp_colsample_{model_name}",
                        help="Fraction of features per tree.")

                # ── C (LR, SVM, SVR) ──────────────────────────────────
                if "C" in tune_p:
                    c_options = [0.001, 0.01, 0.1, 0.5, 1.0,
                                 2.0, 5.0, 10.0, 50.0, 100.0]
                    cur_c = float(default_p.get("C", 1.0))
                    # snap to nearest option
                    cur_c = min(c_options, key=lambda x: abs(x - cur_c))
                    params["C"] = st.select_slider(
                        "C (regularisation — smaller = stronger regularisation)",
                        options=c_options, value=cur_c,
                        key=f"hp_C_{model_name}",
                        help="Smaller C = more regularisation = less overfit.")

                # ── alpha (Ridge, Lasso, ElasticNet) ──────────────────
                if "alpha" in tune_p:
                    a_options = [0.001, 0.01, 0.1, 0.5, 1.0,
                                 5.0, 10.0, 50.0, 100.0]
                    cur_a = float(default_p.get("alpha", 1.0))
                    cur_a = min(a_options, key=lambda x: abs(x - cur_a))
                    params["alpha"] = st.select_slider(
                        "alpha (regularisation strength — larger = stronger)",
                        options=a_options, value=cur_a,
                        key=f"hp_alpha_{model_name}",
                        help="Larger alpha = more regularisation.")

                # ── l1_ratio (ElasticNet) ─────────────────────────────
                if "l1_ratio" in tune_p:
                    params["l1_ratio"] = st.slider(
                        "l1_ratio  (0 = Ridge, 1 = Lasso, 0.5 = mix)",
                        0.0, 1.0,
                        float(default_p.get("l1_ratio", 0.5)),
                        step=0.05, format="%.2f",
                        key=f"hp_l1ratio_{model_name}")

                # ── n_neighbors (KNN) ─────────────────────────────────
                if "n_neighbors" in tune_p:
                    params["n_neighbors"] = st.slider(
                        "n_neighbors", 1, 50,
                        int(default_p.get("n_neighbors", 5)),
                        key=f"hp_knn_{model_name}",
                        help="More neighbours = smoother boundary = less overfit.")

                # ── kernel (SVM, SVR) ─────────────────────────────────
                if "kernel" in tune_p:
                    params["kernel"] = st.selectbox(
                        "kernel",
                        ["rbf", "linear", "poly", "sigmoid"],
                        index=0,
                        key=f"hp_kernel_{model_name}",
                        help="rbf = best general purpose | linear = fast + interpretable")

                # keep fixed params (random_state, n_jobs, max_iter etc.)
                for k, v in default_p.items():
                    if k not in params:
                        params[k] = v

                # show what will be used
                tunable_only = {k: v for k, v in params.items()
                                if k in tune_p}
                st.caption(f"📋 Tunable params set: {tunable_only}")

                custom_params[model_name] = params

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # STEP 5 — TRAIN
    # ══════════════════════════════════════════════════════════════════
    st.subheader("🚀 Step 5 — Train Models")

    if selected_models:
        mode_label = ("⚡ Quick (default params)"
                      if tuning_mode == "quick"
                      else "🔧 Tuned (custom hyperparams)")
        st.markdown(
            f'<div class="info-box">'
            f'🤖 Agent will train: <b>{", ".join(selected_models)}</b><br>'
            f'Test size: {int(test_size*100)}% | '
            f'CV: {"5-fold" if use_cv else "off"} | '
            f'Mode: {mode_label}'
            f'</div>', unsafe_allow_html=True)

    train_btn = st.button(
        "✅ Confirm & Train Selected Models",
        type="primary", use_container_width=True,
        disabled=len(selected_models) == 0,
        key="train_btn")

    if train_btn and selected_models:
        mem        = ModelMemoryManager()
        session_id = mem.create_session(
            dataset_name  = st.session_state.get("file_name", "dataset"),
            target_column = target_col,
            problem_type  = problem_type,
            n_features    = df_fe.shape[1] - 1,
            n_rows        = df_fe.shape[0],
            test_size     = test_size,
            cv_folds      = cv_folds,
            tuning_mode   = tuning_mode)

        progress_bar = st.progress(0, text="Starting training…")
        results      = {}

        for i, model_name in enumerate(selected_models):
            progress_bar.progress(
                (i + 1) / len(selected_models),
                text=f"Training {model_name}…")

            # pass custom params if tuned mode
            override = custom_params.get(model_name, {}) \
                       if tuning_mode == "tuned" else {}

            r = train_models(
                X_train=X_train_fe,
                X_test=X_test_fe,
                y_train=y_train_fe,
                y_test=y_test_fe,
                target_col=target_col,
                problem_type=problem_type,
                selected_models=[model_name],
                cv_folds=cv_folds,
                tuning_mode=tuning_mode,
                session_id=session_id,
                mem=mem,
                custom_params=override)
            results.update(r)

        # Empty the progress bar BEFORE saving state or rerunning.
        # Leaving an active progress bar widget alive when st.rerun()
        # fires causes Streamlit's "SessionInfo not initialized" error
        # because the widget's internal session context is torn down
        # mid-render.
        progress_bar.empty()

        best_name = get_best_model_name(results, problem_type)
        save_best_model(results, problem_type, session_id)
        mem.update_best_model(session_id, best_name or "")
        mem.log_insight(
            session_id, "training_summary",
            f"Trained {len(results)} models. Best: {best_name}. "
            f"Problem: {problem_type}. "
            f"Train rows: {X_train_fe.shape[0]}, Test rows: {X_test_fe.shape[0]}.",
            best_name)

        # widget-bound keys (mt_test_size, mt_use_cv, mt_tuning_mode)
        # are intentionally NOT written here — Streamlit owns those keys.
        st.session_state["mt_results"]      = results
        st.session_state["mt_session_id"]   = session_id
        st.session_state["mt_best_model"]   = best_name
        st.session_state["mt_problem_type"] = problem_type
        st.session_state["mt_target_col"]   = target_col
        st.session_state["mt_mem"]          = mem
        st.session_state["mt_done"]         = True

        st.success(f"✅ Training complete! Best model: {best_name} "
                   f"(saved to models/best_model.pkl)")
        st.rerun()

    # ══════════════════════════════════════════════════════════════════
    # RESULTS — only shown after training
    # ══════════════════════════════════════════════════════════════════
    if not st.session_state.get("mt_done"):
        return

    results      = st.session_state.get("mt_results",    {})
    session_id   = st.session_state.get("mt_session_id", "")
    best_name    = st.session_state.get("mt_best_model",  "")
    problem_type = st.session_state.get("mt_problem_type", problem_type)
    target_col   = st.session_state.get("mt_target_col",   target_col)

    if not results:
        return

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # STEP 6 — FIT ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    st.subheader("🔍 Step 6 — Fit Analysis")
    st.markdown("Overfitting if train-test gap > 10%. "
                "Underfitting if test score < 60% (classification) "
                "or R² < 0.40 (regression).")

    from tools.model_training import detect_fit_status, get_fit_badge, get_fix_suggestions

    for model_name, res in results.items():
        if "error" in res:
            st.error(f"❌ {model_name}: {res['error']}"); continue

        fit_status  = res.get("fit_status", "good")
        train_score = res.get("train_score", 0)
        test_score  = res.get("test_score",  0)
        badge       = get_fit_badge(fit_status)

        with st.expander(f"{badge}  {model_name}", expanded=True):
            fa1, fa2, fa3, fa4 = st.columns(4)
            fa1.metric("Train Score", f"{train_score:.4f}")
            fa2.metric("Test Score",  f"{test_score:.4f}",
                       delta=f"{test_score-train_score:+.4f}")
            if res.get("cv_score") is not None:
                fa3.metric("CV Score",
                           f"{res['cv_score']:.4f} ± {res.get('cv_std',0):.4f}")
            fa4.metric("Status", badge)

            # ── Leakage warning (detect only — user decides) ──────────
            leaky_cols = res.get("leaky_cols", {})
            if leaky_cols:
                st.markdown(
                    f'<div class="warning-box">'
                    f'⚠️ <b>Potential Data Leakage Detected</b> — '
                    f'{len(leaky_cols)} column(s) may be leaking target information:<br>'
                    + "".join(
                        f"&nbsp;&nbsp;• <b>{c}</b>: {r}<br>"
                        for c, r in leaky_cols.items())
                    + '<br>These columns are <b>NOT automatically removed</b> — you decide. '
                    f'If they are derived from the target (e.g. encoded versions or binned versions '
                    f'of the target column), exclude them in ⚙️ Feature Engineering top-K selection, '
                    f'then re-run training.'
                    f'</div>', unsafe_allow_html=True)
                # user choice: drop and retrain
                if st.checkbox(
                        f"Drop leaky columns and retrain {model_name}",
                        value=False,
                        key=f"drop_leaky_{model_name}"):
                    if st.button(
                            f"🗑️ Drop {len(leaky_cols)} leaky col(s) & Retrain",
                            key=f"drop_retrain_{model_name}",
                            type="primary"):
                        with st.spinner("Dropping leaky columns and retraining…"):
                            _mem = st.session_state.get(
                                "mt_mem", ModelMemoryManager())
                            # build df without leaky cols
                            df_no_leak = df_fe.drop(
                                columns=[c for c in leaky_cols
                                         if c in df_fe.columns],
                                errors="ignore")
                            from tools.model_training import retrain_model
                            # Use pre-split data from FE step, drop leaky cols
                            _X_tr = st.session_state.get("fe_X_train").drop(
                                columns=[c for c in leaky_cols
                                         if c in st.session_state.get("fe_X_train").columns],
                                errors="ignore")
                            _X_te = st.session_state.get("fe_X_test").drop(
                                columns=[c for c in leaky_cols
                                         if c in st.session_state.get("fe_X_test").columns],
                                errors="ignore")
                            r2 = retrain_model(
                                X_train      = _X_tr,
                                X_test       = _X_te,
                                y_train      = st.session_state.get("fe_y_train"),
                                y_test       = st.session_state.get("fe_y_test"),
                                target_col   = target_col,
                                problem_type = problem_type,
                                model_name   = model_name,
                                new_params   = res.get("params", {}),
                                session_id   = session_id,
                                mem          = _mem)
                            if "error" not in r2:
                                results[model_name] = r2
                                st.session_state["mt_results"] = results
                                st.success(
                                    f"✅ Retrained without leaky cols — "
                                    f"train={r2['train_score']:.4f} | "
                                    f"test={r2['test_score']:.4f} | "
                                    f"status: {get_fit_badge(r2['fit_status'])}")
                                st.rerun()
                            else:
                                st.error(f"❌ {r2['error']}")

            def _do_retrain(mname, fstatus, key_suffix):
                """Helper: get fixes, show params, retrain button."""
                # Strip params not supported by this model before passing to fix
                _raw_params = res.get("params", {})
                _supports_depth = mname in {
                    "Random Forest","Decision Tree","Gradient Boosting",
                    "XGBoost","LightGBM","Extra Trees"}
                if not _supports_depth:
                    _raw_params = {k: v for k, v in _raw_params.items()
                                   if k != "max_depth"}
                fix_params = get_fix_suggestions(mname, fstatus, _raw_params)
                st.markdown("**Suggested fix parameters:**")
                st.json(fix_params)
                rb_col, _ = st.columns([2, 3])
                with rb_col:
                    if st.button(
                            f"🔄 Auto-Retrain {mname} with fixes",
                            key=f"retrain_{key_suffix}_{mname}",
                            type="primary",
                            use_container_width=True):
                        with st.spinner(
                                f"Retraining {mname} with "
                                f"{fstatus} fixes…"):
                            _mem = st.session_state.get(
                                "mt_mem", ModelMemoryManager())
                            r2   = retrain_model(
                                X_train      = st.session_state.get("fe_X_train"),
                                X_test       = st.session_state.get("fe_X_test"),
                                y_train      = st.session_state.get("fe_y_train"),
                                y_test       = st.session_state.get("fe_y_test"),
                                target_col   = target_col,
                                problem_type = problem_type,
                                model_name   = mname,
                                new_params   = fix_params,
                                session_id   = session_id,
                                mem          = _mem)
                            if "error" not in r2:
                                results[mname] = r2
                                st.session_state["mt_results"] = results
                                new_badge = get_fit_badge(
                                    r2["fit_status"])
                                st.success(
                                    f"✅ Retrained — "
                                    f"train={r2['train_score']:.4f} | "
                                    f"test={r2['test_score']:.4f} | "
                                    f"status: {new_badge}")
                                st.rerun()
                            else:
                                st.error(f"❌ {r2['error']}")

            if fit_status == "overfit":
                st.markdown(
                    '<div class="error-box">'
                    '🔴 <b>Overfitting detected</b><br>'
                    'Train score is significantly higher than test score (gap > 5%).<br>'
                    '<b>Techniques applied automatically on retrain:</b><br>'
                    '• Reduce max_depth (less complex trees)<br>'
                    '• Increase min_samples_leaf (smoother splits)<br>'
                    '• Stronger regularisation (higher alpha / lower C)<br>'
                    '• Reduce n_estimators and learning_rate'
                    '</div>',
                    unsafe_allow_html=True)
                _do_retrain(model_name, "overfit", "of")

            elif fit_status == "underfit":
                st.markdown(
                    '<div class="warning-box">'
                    '🟡 <b>Underfitting detected</b><br>'
                    'Both train and test scores are too low — model is too simple.<br>'
                    '<b>Techniques applied automatically on retrain:</b><br>'
                    '• Increase max_depth (more complex trees)<br>'
                    '• Decrease min_samples_leaf (finer splits)<br>'
                    '• Weaker regularisation (lower alpha / higher C)<br>'
                    '• Add more estimators<br>'
                    '• Tip: also try a more powerful model (XGBoost, LightGBM)'
                    '</div>',
                    unsafe_allow_html=True)
                _do_retrain(model_name, "underfit", "uf")

            elif fit_status == "suspect":
                st.markdown(
                    '<div class="warning-box">'
                    f'🟠 <b>Suspicious Fit — Possible Data Leakage</b><br>'
                    f'Test score ≥ train score or near-perfect (1.0). '
                    f'This happens when a feature encodes the answer the model is trying to predict.<br>'
                    f'<b>Data leakage</b> = model sees information during training it should not have access to.<br>'
                    f'<b>Suspected leaky columns detected: {", ".join(leaky_cols.keys()) if leaky_cols else "check below"}</b><br>'
                    f'<b>Fix:</b> Go back to ⚙️ Feature Engineering, deselect the suspected columns, '
                    f'then re-run training.'
                    '</div>',
                    unsafe_allow_html=True)

            else:
                st.markdown(
                    '<div class="success-box">'
                    '🟢 <b>Good Fit</b> — train and test scores are close '
                    'and both are strong. No action needed.'
                    '</div>',
                    unsafe_allow_html=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # STEP 7 — EVALUATION + COMPARISON
    # ══════════════════════════════════════════════════════════════════
    st.subheader("📊 Step 7 — Evaluation & Comparison")

    comp_df = get_comparison_df(results, problem_type)
    if not comp_df.empty:
        st.markdown(
            f'<div class="success-box">🏆 Best model: '
            f'<b>{best_name}</b></div>', unsafe_allow_html=True)

        def _highlight_best(row):
            color = "background-color: rgba(6,182,212,0.2)" \
                    if row["Model"] == best_name else ""
            return [color] * len(row)

        st.dataframe(
            comp_df.style.apply(_highlight_best, axis=1),
            use_container_width=True)

    model_names_ok = [n for n in results if "error" not in results[n] and results[n].get("metrics")]

    if not model_names_ok:
        st.error("❌ All models failed to train. Common causes:\n"
                 "• Target column has too many unique classes (e.g. free-text used as target)\n"
                 "• Too few samples per class for cross-validation\n"
                 "• Data leakage or encoding issues\n\n"
                 "**Fix:** Go back to Feature Engineering and select a proper target column "
                 "with a small number of distinct classes.")
        return

    eval_tabs = st.tabs([f"📈 {n}" for n in model_names_ok])

    for tab, model_name in zip(eval_tabs, model_names_ok):
        with tab:
            res = results[model_name]
            m   = res.get("metrics", {})

            if problem_type == "classification":
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Accuracy",  f"{m.get('accuracy',  0):.4f}")
                e2.metric("Precision", f"{m.get('precision', 0):.4f}")
                e3.metric("Recall",    f"{m.get('recall',    0):.4f}")
                e4.metric("F1 Score",  f"{m.get('f1_score',  0):.4f}")
                if m.get("roc_auc") is not None:
                    st.metric("ROC-AUC", f"{m['roc_auc']:.4f}")

                cm = m.get("confusion_matrix")
                if cm:
                    st.markdown("**Confusion Matrix:**")
                    cm_df = pd.DataFrame(
                        cm,
                        index   = [f"Actual {i}"   for i in range(len(cm))],
                        columns = [f"Predicted {i}" for i in range(len(cm))])
                    st.dataframe(cm_df, use_container_width=True)

                cr = m.get("classification_report")
                if cr:
                    with st.expander("📋 Full Classification Report"):
                        cr_rows = {k: v for k, v in cr.items()
                                   if isinstance(v, dict)}
                        if cr_rows:
                            st.dataframe(
                                pd.DataFrame(cr_rows).T.round(4),
                                use_container_width=True)

            else:
                e1, e2, e3, e4 = st.columns(4)

                def _fmt(v):
                    """Format metric — large numbers get commas, small get decimals."""
                    if v is None: return "N/A"
                    if abs(v) >= 1000:
                        return f"{v:,.2f}"
                    return f"{v:.4f}"

                mae_v  = m.get('mae',  0)
                mse_v  = m.get('mse',  0)
                rmse_v = m.get('rmse', 0)
                r2_v   = m.get('r2',   0)

                e1.metric("MAE",  _fmt(mae_v),  help="Mean Absolute Error — avg error in target units")
                e2.metric("MSE",  _fmt(mse_v),  help="Mean Squared Error — penalises large errors")
                e3.metric("RMSE", _fmt(rmse_v), help="Root MSE — same units as target")
                e4.metric("R²",   f"{r2_v:.4f}", help="1.0 = perfect | 0 = predicts mean | <0 = worse than mean")

                # Show context for large errors
                if mae_v > 1000 or mse_v > 1_000_000:
                    target_mean = float(y_train_fe.mean()) if y_train_fe is not None and len(y_train_fe) > 0 else None
                    if target_mean and target_mean > 0:
                        mae_pct = (mae_v / abs(target_mean)) * 100
                        st.markdown(
                            f'<div class="info-box">'
                            f'ℹ️ <b>Large error values are normal</b> when the target variable '
                            f'(e.g. price in PKR) has large magnitude.<br>'
                            f'MAE = <b>{_fmt(mae_v)}</b> = '
                            f'<b>{mae_pct:.1f}%</b> of target mean ({_fmt(target_mean)})<br>'
                            f'R² = <b>{r2_v:.4f}</b> — this is the true performance indicator '
                            f'(1.0 = perfect, 0.85+ = good)<br>'
                            f'💡 To reduce error: apply log1p to target in Feature Engineering, '
                            f'or use more powerful models (XGBoost, LightGBM).'
                            f'</div>', unsafe_allow_html=True)

                y_pred = m.get("y_pred")
                y_test = m.get("y_test")
                if y_pred and y_test:
                    fig_pa = go.Figure()
                    fig_pa.add_trace(go.Scatter(
                        x=y_test, y=y_pred, mode="markers",
                        marker=dict(color="#06b6d4", size=5, opacity=0.6),
                        name="Predictions"))
                    mn = min(min(y_test), min(y_pred))
                    mx = max(max(y_test), max(y_pred))
                    fig_pa.add_trace(go.Scatter(
                        x=[mn, mx], y=[mn, mx], mode="lines",
                        line=dict(color="#f43f5e", dash="dash"),
                        name="Perfect fit"))
                    fig_pa.update_layout(
                        title="Predicted vs Actual",
                        xaxis_title="Actual", yaxis_title="Predicted",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#e2e8f0")
                    st.plotly_chart(fig_pa, use_container_width=True)

                    residuals = [p - a for p, a in zip(y_pred, y_test)]
                    fig_res = go.Figure()
                    fig_res.add_trace(go.Scatter(
                        x=y_pred, y=residuals, mode="markers",
                        marker=dict(color="#a855f7", size=5, opacity=0.6),
                        name="Residuals"))
                    fig_res.add_hline(y=0,
                        line=dict(color="#f43f5e", dash="dash"))
                    fig_res.update_layout(
                        title="Residual Plot",
                        xaxis_title="Predicted",
                        yaxis_title="Residual (Predicted - Actual)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#e2e8f0")
                    st.plotly_chart(fig_res, use_container_width=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # STEP 8 — EXPLAINABILITY
    # ══════════════════════════════════════════════════════════════════
    st.subheader("🔬 Step 8 — Explainability")

    try:
        from tools.explainability import (
            get_shap_values, plot_shap_bar,
            plot_shap_beeswarm, plot_shap_waterfall,
            get_lime_explanation, plot_lime_bar,
            SHAP_AVAILABLE, LIME_AVAILABLE
        )
    except ImportError as e:
        st.error(f"❌ Explainability import error: {e}"); return

    exp_model_name = st.selectbox(
        "Select model to explain",
        model_names_ok,
        index=(model_names_ok.index(best_name)
               if best_name in model_names_ok else 0),
        key="exp_model_select")

    exp_res = results.get(exp_model_name, {})
    if "error" in exp_res:
        st.error(f"❌ Cannot explain {exp_model_name}: {exp_res['error']}")
        return

    exp_model     = exp_res["model"]
    X_train_exp   = exp_res.get("X_train")
    X_test_exp    = exp_res.get("X_test")
    # X_train_fe has no target column (correctly separated in FE split)
    # Use feature_names from the result dict, fallback to X_train_fe columns
    feature_names = (exp_res.get("feature_names")
                     or X_train_fe.columns.tolist())

    st.markdown("#### 🌊 SHAP Explanations")

    def _openai_explain(prompt: str, cache_key: str,
                        insight_type: str = "explainability") -> None:
        """
        Call gpt-4o-mini, display plain-English explanation,
        and save to model_insights SQLite table for AI chat support.
        """
        import os as _os
        ck = f"ai_exp_{cache_key}"
        if ck in st.session_state:
            st.markdown(
                f'<div class="info-box">🤖 <b>AI Explanation</b><br>'
                f'{st.session_state[ck]}</div>',
                unsafe_allow_html=True)
            return
        api_key = _os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            st.info("ℹ️ Add OPENAI_API_KEY to .env to enable AI explanations.")
            return
        try:
            from openai import OpenAI as _OAI
            client = _OAI(api_key=api_key)
            resp   = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=220,
                temperature=0.3,
                messages=[
                    {"role": "system",
                     "content": (
                         "You are a data science explainer. "
                         "Explain model predictions in simple English "
                         "for a non-technical person. "
                         "Be concise — max 4 sentences.")},
                    {"role": "user", "content": prompt}
                ])
            explanation = resp.choices[0].message.content.strip()

            # ── cache in session state — no re-calls on rerun ─────────
            st.session_state[ck] = explanation

            # ── save to SQLite model_insights for AI chat support ──────
            try:
                from tools.model_memory import ModelMemoryManager as _MMM
                _mem = st.session_state.get("mt_mem") or _MMM()
                _sid = st.session_state.get("mt_session_id", "")
                if _sid:
                    _mem.log_insight(
                        session_id   = _sid,
                        insight_type = insight_type,
                        description  = explanation,
                        model_name   = exp_model_name,
                        statistics   = {"prompt_key":     cache_key,
                                        "prompt_summary": prompt[:200]})
            except Exception:
                pass  # memory failure must not break UI

            st.markdown(
                f'<div class="info-box">🤖 <b>AI Explanation</b><br>'
                f'{explanation}</div>',
                unsafe_allow_html=True)
        except Exception as _e:
            st.warning(f"AI explanation unavailable: {_e}")

    if not SHAP_AVAILABLE:
        st.warning("⚠️ SHAP not installed. Run: `pip install shap==0.46.0`")
    elif X_test_exp is not None:
        with st.spinner("Computing SHAP values…"):
            try:
                shap_vals, _ = get_shap_values(
                    exp_model, X_test_exp, exp_model_name)

                import pandas as _pd
                mean_abs_shap = np.abs(shap_vals).mean(axis=0)
                top5 = sorted(zip(feature_names, mean_abs_shap),
                              key=lambda x: x[1], reverse=True)[:5]
                top5_str = ", ".join(f"{f} ({v:.4f})" for f, v in top5)

                shap_tab1, shap_tab2, shap_tab3 = st.tabs([
                    "📊 Global Bar", "🐝 Beeswarm", "💧 Waterfall (Local)"])

                with shap_tab1:
                    fig_bar = plot_shap_bar(shap_vals, feature_names)
                    st.pyplot(fig_bar)
                    buf = __import__("io").BytesIO()
                    fig_bar.savefig(buf, format="png",
                                    bbox_inches="tight", dpi=150)
                    st.download_button(
                        "📥 Download SHAP Bar Chart", buf.getvalue(),
                        f"shap_bar_{exp_model_name}.png", "image/png")
                    _openai_explain(
                        f"A {problem_type} model ({exp_model_name}) predicts "
                        f"'{target_col}'. Top 5 features by SHAP importance: "
                        f"{top5_str}. Explain in simple English what these "
                        f"features mean and why they are important.",
                        f"shap_global_{exp_model_name}",
                        insight_type="shap_global")

                with shap_tab2:
                    X_test_df = (_pd.DataFrame(
                        X_test_exp, columns=feature_names)
                                  if not hasattr(X_test_exp, "columns")
                                  else X_test_exp)
                    fig_bee = plot_shap_beeswarm(shap_vals, X_test_df)
                    st.pyplot(fig_bee)
                    buf2 = __import__("io").BytesIO()
                    fig_bee.savefig(buf2, format="png",
                                    bbox_inches="tight", dpi=150)
                    st.download_button(
                        "📥 Download Beeswarm", buf2.getvalue(),
                        f"shap_bee_{exp_model_name}.png", "image/png")
                    _openai_explain(
                        f"A SHAP beeswarm plot shows how feature values "
                        f"(red=high, blue=low) push predictions up or down "
                        f"for a {problem_type} model predicting '{target_col}'. "
                        f"Top features: {top5_str}. "
                        f"Explain what patterns the beeswarm reveals.",
                        f"shap_beeswarm_{exp_model_name}",
                        insight_type="shap_beeswarm")

                with shap_tab3:
                    row_idx = st.number_input(
                        "Row index to explain",
                        min_value=0,
                        max_value=len(X_test_exp) - 1,
                        value=0, key="shap_row_idx")
                    X_test_df2 = (_pd.DataFrame(
                        X_test_exp, columns=feature_names)
                                   if not hasattr(X_test_exp, "columns")
                                   else X_test_exp)
                    fig_wf = plot_shap_waterfall(
                        shap_vals, X_test_df2, idx=row_idx)
                    st.pyplot(fig_wf)
                    pred = exp_model.predict(
                        X_test_exp[row_idx:row_idx+1])[0]
                    actual = (exp_res["y_test"].iloc[row_idx]
                              if hasattr(exp_res["y_test"], "iloc")
                              else exp_res["y_test"][row_idx])
                    st.markdown(
                        f'<div class="info-box">'
                        f'Row {row_idx}: Predicted = <b>{pred}</b> | '
                        f'Actual = <b>{actual}</b>'
                        f'</div>', unsafe_allow_html=True)
                    # per-row SHAP for OpenAI
                    row_shap = shap_vals[row_idx]
                    row_top5 = sorted(zip(feature_names, row_shap),
                                      key=lambda x: abs(x[1]), reverse=True)[:5]
                    row_top5_str = ", ".join(
                        f"{f} ({v:+.4f})" for f, v in row_top5)
                    _openai_explain(
                        f"For row {row_idx}, the model predicted "
                        f"{target_col}={pred} (actual={actual}). "
                        f"Top features affecting THIS prediction: "
                        f"{row_top5_str} (+ pushed up, - pushed down). "
                        f"Explain in simple English why the model made "
                        f"this specific prediction.",
                        f"shap_local_{exp_model_name}_{row_idx}",
                        insight_type="shap_local")

                # log SHAP to memory
                mem = st.session_state.get("mt_mem", ModelMemoryManager())
                mem.log_feature_importance(
                    session_id, exp_model_name, "shap",
                    [{"feature": f, "value": float(v), "rank": i+1}
                     for i, (f, v) in enumerate(
                         sorted(zip(feature_names, mean_abs_shap),
                                key=lambda x: x[1], reverse=True))])

            except Exception as exc:
                st.error(f"❌ SHAP error: {exc}")
                import traceback; st.code(traceback.format_exc())

    st.markdown("---")
    st.markdown("#### 🍋 LIME Local Explanation")
    st.markdown(
        '<div class="info-box">'
        '🍋 LIME explains a <b>single prediction</b> in simple terms — '
        'which features pushed the prediction up or down for that specific row.'
        '</div>', unsafe_allow_html=True)

    if not LIME_AVAILABLE:
        st.warning("⚠️ LIME not installed. Run: `pip install lime`")
    elif X_train_exp is not None and X_test_exp is not None:
        show_lime = st.checkbox(
            "Generate LIME explanation", value=False,
            key="show_lime")
        if show_lime:
            lime_row = st.number_input(
                "Select test row to explain",
                min_value=0,
                max_value=len(X_test_exp) - 1,
                value=0, key="lime_row_idx",
                help="Each row is one sample from the test set")
            if st.button("🍋 Run LIME", key="run_lime_btn", type="primary"):
                with st.spinner(f"Computing LIME for row {lime_row}…"):
                    try:
                        import pandas as _pd
                        X_train_np = (X_train_exp.values
                                      if hasattr(X_train_exp, "values")
                                      else X_train_exp)
                        X_row = (X_test_exp.iloc[[lime_row]]
                                 if hasattr(X_test_exp, "iloc")
                                 else X_test_exp[lime_row:lime_row+1])

                        exp_lime, html_str = get_lime_explanation(
                            model         = exp_model,
                            X_train       = X_train_np,
                            X_test_row    = X_row,
                            feature_names = feature_names,
                            problem_type  = problem_type)

                        # bar chart
                        fig_lime = plot_lime_bar(
                            exp_lime,
                            f"LIME — {exp_model_name} — Row {lime_row}")
                        st.pyplot(fig_lime)

                        # prediction vs actual
                        lime_pred = exp_model.predict(
                            X_row.values if hasattr(X_row,"values")
                            else X_row)[0]
                        lime_actual = (y_test_fe.iloc[lime_row]
                                       if hasattr(y_test_fe,"iloc")
                                       else y_test_fe[lime_row])
                        lc1, lc2 = st.columns(2)
                        lc1.metric("Predicted", str(lime_pred))
                        lc2.metric("Actual",    str(lime_actual))

                        # How to read the chart
                        with st.expander("📖 How to read this chart"):
                            st.markdown("""
**LIME bar chart guide:**
- Each bar = one feature's contribution to this prediction
- 🟦 **Cyan / positive bar** = feature pushed prediction **higher**
- 🟥 **Red / negative bar** = feature pushed prediction **lower**
- The **condition** (e.g. `Age > 30`) tells you the value range for this row
- The **number** is how much that feature changed the predicted value
- Features are ranked by absolute impact — top = most influential
                            """)

                        # OpenAI explanation
                        try:
                            lime_feats = exp_lime.as_list()
                            lime_top5  = sorted(
                                lime_feats, key=lambda x: abs(x[1]),
                                reverse=True)[:5]
                            lime_str   = ", ".join(
                                f"{f} ({v:+.4f})" for f, v in lime_top5)
                            _openai_explain(
                                f"LIME explains a single prediction: "
                                f"the {problem_type} model ({exp_model_name}) "
                                f"predicted {target_col}={lime_pred} "
                                f"(actual={lime_actual}) for row {lime_row}. "
                                f"Top contributing features: "
                                f"{lime_str} (+ pushed prediction up, "
                                f"- pushed prediction down). "
                                f"Explain in simple English why the model made "
                                f"this specific prediction for this row.",
                                f"lime_{exp_model_name}_{lime_row}",
                                insight_type="lime_local")
                        except Exception:
                            pass

                        with st.expander("📄 Full LIME HTML Report"):
                            st.components.v1.html(html_str, height=400,
                                                  scrolling=True)

                    except Exception as exc:
                        st.error(f"❌ LIME error: {exc}")
                        import traceback; st.code(traceback.format_exc())

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # SAVE / EXPORT
    # ══════════════════════════════════════════════════════════════════
    st.subheader("💾 Save & Download Models")

    import joblib as _jl, io as _io

    # Auto-save best model info box
    st.markdown(
        f'<div class="success-box">'
        f'✅ Best model <b>{best_name}</b> is auto-saved to '
        f'<code>models/best_model.pkl</code> on disk.<br>'
        f'Use the download buttons below to get any trained model as a .pkl file.'
        f'</div>', unsafe_allow_html=True)

    # Download buttons — one per trained model, in a grid
    st.markdown("#### 📥 Download Trained Models")
    dl_cols = st.columns(min(len(model_names_ok), 3))
    for idx, model_name in enumerate(model_names_ok):
        res = results[model_name]
        buf = _io.BytesIO()
        _jl.dump({
            "model":         res["model"],
            "model_name":    model_name,
            "feature_names": res.get("feature_names", []),
            "params":        res.get("params", {}),
            "problem_type":  problem_type,
            "target_col":    target_col,
            "train_score":   res.get("train_score", 0),
            "test_score":    res.get("test_score",  0),
            "fit_status":    res.get("fit_status",  ""),
        }, buf)
        buf.seek(0)
        star = " 🏆" if model_name == best_name else ""
        with dl_cols[idx % min(len(model_names_ok), 3)]:
            st.download_button(
                label=f"📥 {model_name}{star}",
                data=buf.getvalue(),
                file_name=f"{model_name.replace(' ','_')}.pkl",
                mime="application/octet-stream",
                key=f"dl_{model_name}",
                use_container_width=True,
                type="primary" if model_name == best_name else "secondary")
            m = res.get("metrics", {})
            if problem_type == "classification":
                st.caption(f"F1={m.get('f1_score',0):.3f} | "
                           f"Acc={m.get('accuracy',0):.3f} | "
                           f"Test={res.get('test_score',0):.3f}")
            else:
                st.caption(f"R²={m.get('r2',0):.3f} | "
                           f"RMSE={m.get('rmse',0):.3f} | "
                           f"Test={res.get('test_score',0):.3f}")

    st.markdown("---")
    st.markdown("#### 🔄 Set Production Model")
    prod_model = st.selectbox(
        "Select model to save as production (overwrites best_model.pkl):",
        model_names_ok, key="prod_model_sel",
        index=model_names_ok.index(best_name) if best_name in model_names_ok else 0)
    if st.button("💾 Save as Production Model",
                 key="save_prod_btn", type="primary"):
        prod_res  = results[prod_model]
        prod_path = os.path.join(MODELS_DIR, "best_model.pkl")
        _jl.dump({
            "model":         prod_res["model"],
            "model_name":    prod_model,
            "le":            prod_res.get("le"),
            "feature_names": prod_res.get("feature_names", []),
            "params":        prod_res.get("params", {}),
            "session_id":    session_id,
            "problem_type":  problem_type,
            "target_col":    target_col,
        }, prod_path)
        st.success(f"✅ {prod_model} saved as production model at {prod_path}")

    # ── SAVE PIPELINE SNAPSHOT ────────────────────────────────────────
    show_model_training_report_button()
    st.markdown("---")
    st.markdown("#### 💾 Save Pipeline Snapshot")
    st.markdown(
        '<div class="info-box">'
        '💾 <b>Pipeline Snapshot</b> — Saves the complete end-to-end pipeline '
        '(preprocessing config + trained model + reference statistics) for '
        'future drift detection and reuse.</div>',
        unsafe_allow_html=True)

    snap_name = st.text_input(
        "Snapshot name (optional label)",
        value=st.session_state.get("file_name", "dataset").replace(".csv",""),
        key="snap_name_input",
        help="A label to identify this snapshot later")

    if st.button("💾 Save Complete Pipeline Snapshot",
                 key="save_snapshot_btn", type="primary",
                 use_container_width=True):
        with st.spinner("Saving pipeline snapshot…"):
            try:
                # Use df_cleaned (full dataset, raw values, before FE encoding/scaling)
                # This is the correct reference for drift detection:
                # new uploads arrive as raw data, so reference must also be raw
                # Use df_raw (original uploaded data, before any cleaning/outlier capping)
                # This ensures reference stats match what users upload for drift check
                # df_cleaned has outliers capped → different distribution than raw uploads
                ref_df = st.session_state.get("df_raw")
                if ref_df is None:
                    ref_df = st.session_state.get("df_cleaned")
                if ref_df is None:
                    ref_df = st.session_state.get("df")
                if ref_df is None:
                    st.error("❌ Original dataset not in session. Cannot save snapshot.")
                else:
                    # Best model objects
                    best_res      = results.get(best_name, {})
                    snap_model    = best_res.get("model")
                    snap_metrics  = best_res.get("metrics", {})

                    # FE config
                    fe_rep = st.session_state.get("fe_report", {})
                    fe_cfg = {
                        "problem_type":     problem_type,
                        "target_column":    target_col,
                        "created_features": fe_rep.get("created_features", []),
                    }

                    # ── KEY FIX ───────────────────────────────────────────
                    # For drift detection we MUST use the original column names
                    # and RAW values (before encoding/scaling).
                    # ref_df = cleaned df (raw values, original column names).
                    # We compute reference stats on columns that exist in ref_df
                    # AND are meaningful features (not encoded dummies, not Id).
                    # This means: all columns in ref_df except the target column.
                    raw_feature_cols = [c for c in ref_df.columns
                                        if c != target_col]

                    enc   = st.session_state.get("fe_encoders", {})
                    scl   = st.session_state.get("fe_scaler")

                    sid = save_snapshot(
                        df_original   = ref_df,
                        target_col    = target_col,
                        problem_type  = problem_type,
                        feature_names = raw_feature_cols,
                        encoders      = enc,
                        scaler        = scl,
                        model         = snap_model,
                        model_name    = best_name,
                        model_metrics = snap_metrics,
                        fe_config     = fe_cfg,
                        dataset_name  = snap_name or "dataset")

                    st.success(
                        f"✅ Pipeline snapshot saved! ID: `{sid}`\n\n"
                        f"Model: **{best_name}** | Target: **{target_col}** | "
                        f"Features: **{len(raw_feature_cols)}**\n\n"
                        f"Go to 🔄 Pipeline Snapshots to manage saved pipelines.")
                    st.session_state["last_snapshot_id"] = sid
            except Exception as _se:
                import traceback
                st.error(f"❌ Snapshot save error: {_se}")
                st.code(traceback.format_exc())

    # ── Generate Report button ──────────────────────────────────────────────
def show_model_training_report_button():
    """Generate PDF report — saved as dataset filename."""
    st.markdown("---")
    st.markdown("### 📄 Generate Pipeline Report")
    st.caption("Download a PDF summary — data overview, model results, "
               "SHAP importance, feature analysis, and AI interpretation.")
    if st.button("📄 Generate & Download Report", key="gen_report_btn",
                 use_container_width=True):
        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(__file__))
            from tools.report_generator import generate_pipeline_report
            logo_path  = os.path.join(os.path.dirname(__file__), "logo.png")
            memory_dir = os.path.join(os.path.dirname(__file__), "memory")
            with st.spinner("Generating report..."):
                pdf_bytes = generate_pipeline_report(
                    dict(st.session_state),
                    logo_path=logo_path,
                    memory_dir=memory_dir)

            # Save as dataset name
            raw_name = st.session_state.get("file_name","pipeline_report")
            safe_name = os.path.splitext(raw_name)[0]  # strip .csv etc
            safe_name = safe_name.replace(" ","_")
            pdf_filename = f"{safe_name}_report.pdf"

            st.download_button(
                label="⬇️ Download Pipeline Report (PDF)",
                data=pdf_bytes,
                file_name=pdf_filename,
                mime="application/pdf",
                key="download_report_btn")
            st.success(f"✅ Report generated → {pdf_filename}")
        except Exception as _re:
            import traceback
            st.error(f"❌ Report generation failed: {_re}")
            st.code(traceback.format_exc())


def show_model_training_report_button():
    """Generate PDF report button — called at end of show_model_training_page."""
    import os  # ← moved here, available for the whole function

    st.markdown("---")
    st.markdown("### 📄 Generate Pipeline Report")
    st.caption("Download a 3-page PDF summary — data overview, model results, "
               "SHAP importance, and drift detection.")

    # ── User-defined report name ──────────────────────────────────────
    default_name = os.path.splitext(
        st.session_state.get("file_name", "pipeline_report")
    )[0].replace(" ", "_")

    report_name = st.text_input(
        "Report name (used as filename)",
        value=default_name,
        key="report_name_input",
        help="e.g. 'housing_analysis' → saves as 'housing_analysis_report.pdf'")

    st.caption(f"📄 Will save as: **{report_name}_report.pdf**")

    if st.button("📄 Generate & Download Report", key="gen_report_btn",
                 use_container_width=True):
        try:
            import sys
            sys.path.insert(0, os.path.dirname(__file__))
            from tools.report_generator import generate_pipeline_report
            logo_path  = os.path.join(os.path.dirname(__file__), "logo.png")
            memory_dir = os.path.join(os.path.dirname(__file__), "memory")
            with st.spinner("Generating report..."):
                pdf_bytes = generate_pipeline_report(
                    dict(st.session_state),
                    logo_path=logo_path,
                    memory_dir=memory_dir)

            safe_name = (report_name.strip().replace(" ", "_")
                                            .replace("/", "_")
                                            .replace("\\", "_") or "pipeline_report")
            pdf_filename = f"{safe_name}_report.pdf"

            st.download_button(
                label=f"⬇️ Download {pdf_filename}",
                data=pdf_bytes,
                file_name=pdf_filename,
                mime="application/pdf",
                key="download_report_btn")
            st.success(f"✅ Report ready → {pdf_filename}")
        except Exception as _re:
            import traceback
            st.error(f"❌ Report generation failed: {_re}")
            st.code(traceback.format_exc())
def show_chat_page():
    st.header("💬 AI Chat Assistant")
    # Re-create agent whenever pipeline state changes so it picks up
    # new steps (e.g. model training done after agent was first created)
    pipeline_sig = (
        st.session_state.get('cleaning_done', False),
        st.session_state.get('fe_done', False),
        st.session_state.get('mt_done', False),
        st.session_state.get('mt_best_model', ''),
    )
    if ('chat_agent' not in st.session_state or
            st.session_state.get('_chat_pipeline_sig') != pipeline_sig):
        st.session_state['chat_agent']          = ChatSupportAgent(unified_memory)
        st.session_state['_chat_pipeline_sig']  = pipeline_sig
    agent = st.session_state['chat_agent']
    if st.session_state.get('validation_rejected', False):
        st.markdown('<div class="error-box">⚠️ <b>Limited Functionality:</b> Fix format mismatch first.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">🤖 <b>AI-Powered Chat</b> with Full Pipeline Context Awareness</div>', unsafe_allow_html=True)
    for message in st.session_state.get('chat_history', []):
        role, content = message.get('role','user'), message.get('content','')
        if role == 'user':
            st.markdown(f'<div class="chat-message user-message">👤 <b>You:</b><br>{content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message">🤖 <b>Assistant:</b><br>{content}</div>', unsafe_allow_html=True)
    with st.expander("💡 Example Questions", expanded=False):
        st.markdown("""
        **Data:** "What is the shape of my dataset?" | "How many missing values?"
        **Cleaning:** "What operations were performed?" | "How many rows removed?"
        **Feature Engineering:** "Which features were selected?" | "Why was column X dropped?" | "What new features were created?" | "Which feature has highest importance?"
        **Model Training:** "Which model performed best?" | "What was the F1 score of XGBoost?" | "Was the model overfitting?" | "Which feature had highest SHAP importance?"
        **Next Steps:** "What should I do next?" | "Is my data ready for model training?"
        """)
    user_query = st.text_input("Ask anything…", placeholder="E.g., Which features were selected?", key="chat_input")
    col1, col2 = st.columns([4,1])
    with col1:
        send_button = st.button("📤 Send", use_container_width=True, type="primary")
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            agent.clear_history(); clear_agent_memory(); st.rerun()
    if send_button and user_query:
        with st.spinner("🤖 Thinking..."):
            agent.chat(user_query); st.rerun()


def show_history_page():
    """
    Pipeline History — shows ALL completed steps with full details.
    Fix 4: Now includes FE and Model Training steps.
    Fix 5: Added Final Report tab.
    """
    st.header("📜 Pipeline History & Final Report")

    tab_history, tab_ingestion, tab_report = st.tabs([
        "🔄 Pipeline Status",
        "📥 Ingestion History",
        "📋 Final Report"
    ])

    # ── Tab 1: Pipeline Status ────────────────────────────────────────
    with tab_history:
        st.markdown("### 🔄 Complete Pipeline Status")

        steps_done = []
        steps_todo = []
        all_steps  = ["Data Ingestion","Validation","Data Cleaning",
                      "EDA","Feature Engineering","Model Training"]

        if st.session_state.get('df') is not None:        steps_done.append("Data Ingestion")
        if st.session_state.get('validation_report'):     steps_done.append("Validation")
        if st.session_state.get('cleaning_done'):         steps_done.append("Data Cleaning")
        # EDA is done if user visited EDA page (eda_session_id) OR ran full analysis
        if (st.session_state.get('analysis_context') or
                st.session_state.get('eda_session_id') or
                st.session_state.get('eda_report')):      steps_done.append("EDA")
        if st.session_state.get('fe_done'):               steps_done.append("Feature Engineering")
        if st.session_state.get('mt_done'):               steps_done.append("Model Training")
        steps_todo = [s for s in all_steps if s not in steps_done]

        c1, c2 = st.columns(2)
        c1.metric("Steps Completed", len(steps_done))
        c2.metric("Steps Remaining", len(steps_todo))

        for step in all_steps:
            done = step in steps_done
            icon = "✅" if done else "⏳"
            color = "#10b981" if done else "#94a3b8"
            st.markdown(
                f'<div style="padding:0.6rem 1rem;margin:0.3rem 0;border-radius:8px;'
                f'border-left:4px solid {color};background:rgba(15,23,42,0.5);">'
                f'{icon} <b style="color:{color}">{step}</b>'
                f'</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Detailed per-step info
        full_context = unified_memory.get_full_context()

        if full_context.get('ingestion'):
            ing = full_context['ingestion']
            with st.expander("📥 Data Ingestion Details", expanded=False):
                st.write(f"**File:** {ing.get('file_name','?')} | **Format:** {ing.get('data_format','?')}")
                st.write(f"**Size:** {ing.get('row_count', ing.get('rows',0)):,} rows × {ing.get('column_count', ing.get('columns',0))} cols")
                st.write(f"**Time:** {ing.get('timestamp','?')}") 

        if full_context.get('validation'):
            val = full_context['validation']
            with st.expander("🔍 Validation Details", expanded=False):
                st.write(f"**Status:** {val.get('validation_status','?')} | **Issues:** {val.get('total_issues',0)}")

        if full_context.get('cleaning'):
            cln = full_context['cleaning']
            with st.expander("🧹 Cleaning Details", expanded=False):
                st.write(f"**Rows:** {cln.get('rows_before',0):,} → {cln.get('rows_after',0):,}")
                st.write(f"**Operations:** {len(cln.get('operations',[]))}")
                for op in cln.get('operations',[]):
                    st.write(f"  • {op.get('type','?')}: {op.get('details','')}")

        if st.session_state.get('fe_done'):
            fe_rep = st.session_state.get('fe_report', {})
            with st.expander("⚙️ Feature Engineering Details", expanded=False):
                st.write(f"**Target:** {fe_rep.get('target_column','?')}")
                st.write(f"**Problem:** {fe_rep.get('problem_type','?')}")
                sel = fe_rep.get('selected_features',[])
                st.write(f"**Selected features ({len(sel)}):** {', '.join(sel[:10])}")
                drp = fe_rep.get('dropped_features',[])
                st.write(f"**Dropped:** {len(drp)} features")
                cre = fe_rep.get('created_features',[])
                if cre: st.write(f"**Created:** {', '.join(cre)}")

        if st.session_state.get('mt_done'):
            mt_res  = st.session_state.get('mt_results', {})
            best_m  = st.session_state.get('mt_best_model','')
            prob    = st.session_state.get('mt_problem_type','')
            tgt     = st.session_state.get('mt_target_col','')
            with st.expander("🤖 Model Training Details", expanded=False):
                st.write(f"**Target:** {tgt} | **Problem:** {prob}")
                st.write(f"**Best model:** 🏆 {best_m}")
                for mname, res in mt_res.items():
                    if 'error' in res:
                        st.write(f"  ❌ {mname}: {res['error']}")
                    else:
                        m = res.get('metrics',{})
                        best_tag = " 🏆" if mname == best_m else ""
                        if prob == 'classification':
                            st.write(f"  • {mname}{best_tag}: "
                                     f"Acc={m.get('accuracy',0):.4f} "
                                     f"F1={m.get('f1_score',0):.4f} "
                                     f"Train={res.get('train_score',0):.4f} "
                                     f"Test={res.get('test_score',0):.4f} "
                                     f"({res.get('fit_status','?')})")
                        else:
                            st.write(f"  • {mname}{best_tag}: "
                                     f"R²={m.get('r2',0):.4f} "
                                     f"RMSE={m.get('rmse',0):.4f} "
                                     f"Train={res.get('train_score',0):.4f} "
                                     f"Test={res.get('test_score',0):.4f} "
                                     f"({res.get('fit_status','?')})")

    # ── Tab 2: Ingestion History ──────────────────────────────────────
    with tab_ingestion:
        st.markdown("### 📥 Ingestion History")
        try:
            history_df = tools['ingestion'].memory.get_all_ingestions()
            if not history_df.empty:
                st.dataframe(history_df, use_container_width=True)
            else:
                st.markdown('<div class="info-box">No ingestion history yet.</div>',
                            unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {str(e)}")

    # ── Tab 3: Final Report ───────────────────────────────────────────
    with tab_report:
        st.markdown("### 📋 Final Pipeline Report")
        st.markdown(
            '<div class="info-box">📋 Complete summary of everything the agent did — from data upload to model training.</div>',
            unsafe_allow_html=True)

        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("AGENTIC DATA SCIENTIST — FINAL PIPELINE REPORT")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 70)

        full_context = unified_memory.get_full_context()

        # ── 1. Ingestion ──────────────────────────────────────────────
        report_lines.append("\n📥 STEP 1: DATA INGESTION")
        report_lines.append("-" * 40)
        if full_context.get('ingestion'):
            ing = full_context['ingestion']
            report_lines.append(f"  File          : {ing.get('file_name','?')} ({ing.get('data_format','?')})")
            report_lines.append(f"  Original size : {ing.get('row_count',ing.get('rows',0)):,} rows × {ing.get('column_count',ing.get('columns',0))} cols")
            report_lines.append(f"  Timestamp     : {ing.get('timestamp','?')}")
            report_lines.append("  Status        : ✅ COMPLETE")
        else:
            report_lines.append("  Status        : ⏳ NOT DONE")

        # ── 2. Validation ─────────────────────────────────────────────
        report_lines.append("\n🔍 STEP 2: VALIDATION")
        report_lines.append("-" * 40)
        if full_context.get('validation'):
            val = full_context['validation']
            report_lines.append(f"  Status       : {val.get('validation_status','?')}")
            report_lines.append(f"  Issues found : {val.get('total_issues',0)}")
            mv = val.get('missing_values',{})
            report_lines.append(f"  Missing cells: {mv.get('total_missing',0)}")
            dup = val.get('duplicates',{})
            report_lines.append(f"  Duplicates   : {dup.get('total_duplicates',0)} rows")
            report_lines.append("  Result       : ✅ COMPLETE")
        else:
            report_lines.append("  Status       : ⏳ NOT DONE")

        # ── 3. Cleaning ───────────────────────────────────────────────
        report_lines.append("\n🧹 STEP 3: DATA CLEANING")
        report_lines.append("-" * 40)
        if full_context.get('cleaning'):
            cln = full_context['cleaning']
            report_lines.append(f"  Before    : {cln.get('rows_before',0):,} rows × {cln.get('columns_before',0)} cols")
            report_lines.append(f"  After     : {cln.get('rows_after',0):,} rows × {cln.get('columns_after',0)} cols")
            report_lines.append(f"  Rows removed: {cln.get('rows_before',0)-cln.get('rows_after',0):,}")
            report_lines.append(f"  Operations: {len(cln.get('operations',[]))}")
            for op in cln.get('operations',[]):
                report_lines.append(f"    • {op.get('type','?')}: {op.get('details','')}")
            report_lines.append("  Result    : ✅ COMPLETE")
        else:
            report_lines.append("  Status    : ⏳ NOT DONE")

        # ── 4. EDA ────────────────────────────────────────────────────
        report_lines.append("\n📊 STEP 4: EXPLORATORY DATA ANALYSIS")
        report_lines.append("-" * 40)
        if st.session_state.get('analysis_context'):
            ctx = st.session_state['analysis_context']
            shape = ctx.get('dataset_shape',(0,0))
            report_lines.append(f"  Dataset shape : {shape[0]:,} rows × {shape[1]} cols")
            report_lines.append(f"  Numeric cols  : {len(ctx.get('numeric_columns',[]))}")
            report_lines.append(f"  Categ. cols   : {len(ctx.get('categorical_columns',[]))}")
            report_lines.append(f"  Missing values: {sum(ctx.get('missing_values',{}).values())}")
            report_lines.append(f"  Outlier cols  : {len(ctx.get('outliers',{}))}")
            report_lines.append("  Result        : ✅ COMPLETE")
        else:
            report_lines.append("  Status        : ⏳ NOT DONE / not saved to memory")

        # ── 5. Feature Engineering ────────────────────────────────────
        report_lines.append("\n⚙️ STEP 5: FEATURE ENGINEERING")
        report_lines.append("-" * 40)
        if st.session_state.get('fe_done'):
            fe  = st.session_state.get('fe_report', {})
            sel = fe.get('selected_features',[])
            drp = fe.get('dropped_features',[])
            cre = fe.get('created_features',[])
            enc = fe.get('encoded',[])
            scl = fe.get('scaled',[])
            imp = fe.get('importance_df')
            report_lines.append(f"  Target column   : {fe.get('target_column','?')}")
            report_lines.append(f"  Problem type    : {fe.get('problem_type','?')}")
            report_lines.append(f"  Input features  : {len(sel)+len(drp)}")
            report_lines.append(f"  Selected (top-K): {len(sel)}")
            report_lines.append(f"  Dropped         : {len(drp)}")
            report_lines.append(f"  Created         : {len(cre)}")
            report_lines.append(f"  Encoded cols    : {len(enc)}")
            report_lines.append(f"  Scaled cols     : {len(scl)}")
            if sel:
                report_lines.append(f"  Final features  : {', '.join(sel[:10])}{'...' if len(sel)>10 else ''}")
            if imp is not None and not imp.empty:
                report_lines.append("  Top RF scores:")
                for _, row in imp.head(5).iterrows():
                    report_lines.append(f"    {int(row['rank']):>2}. {row['feature']:<25} RF={row['rf_score']:.4f}")
            if fe.get('errors'):
                for e in fe['errors']: report_lines.append(f"  ⚠️ {e}")
            report_lines.append("  Result          : ✅ COMPLETE")
        else:
            report_lines.append("  Status          : ⏳ NOT DONE")

        # ── 6. Model Training ─────────────────────────────────────────
        report_lines.append("\n🤖 STEP 6: MODEL TRAINING")
        report_lines.append("-" * 40)
        if st.session_state.get('mt_done'):
            mt_res = st.session_state.get('mt_results',{})
            best_m = st.session_state.get('mt_best_model','')
            prob   = st.session_state.get('mt_problem_type','')
            tgt    = st.session_state.get('mt_target_col','')
            ok     = [n for n,r in mt_res.items() if 'error' not in r]
            report_lines.append(f"  Target       : {tgt}")
            report_lines.append(f"  Problem type : {prob}")
            report_lines.append(f"  Models trained: {len(ok)}")
            report_lines.append(f"  Best model   : 🏆 {best_m}")
            report_lines.append("")
            for mname, res in mt_res.items():
                if 'error' in res:
                    report_lines.append(f"  ❌ {mname}: {res['error']}")
                    continue
                m    = res.get('metrics',{})
                star = " 🏆 BEST" if mname == best_m else ""
                report_lines.append(f"  {'★' if mname==best_m else '•'} {mname}{star}")
                report_lines.append(f"    Train score : {res.get('train_score',0):.4f}")
                report_lines.append(f"    Test score  : {res.get('test_score',0):.4f}")
                report_lines.append(f"    Fit status  : {res.get('fit_status','?')}")
                if res.get('cv_score') is not None:
                    report_lines.append(f"    CV score    : {res['cv_score']:.4f} ± {res.get('cv_std',0):.4f}")
                if prob == 'classification':
                    report_lines.append(f"    Accuracy    : {m.get('accuracy',0):.4f}")
                    report_lines.append(f"    Precision   : {m.get('precision',0):.4f}")
                    report_lines.append(f"    Recall      : {m.get('recall',0):.4f}")
                    report_lines.append(f"    F1 Score    : {m.get('f1_score',0):.4f}")
                    if m.get('roc_auc'): report_lines.append(f"    ROC-AUC     : {m['roc_auc']:.4f}")
                else:
                    report_lines.append(f"    MAE         : {m.get('mae',0):,.4f}")
                    report_lines.append(f"    RMSE        : {m.get('rmse',0):,.4f}")
                    report_lines.append(f"    R²          : {m.get('r2',0):.4f}")
                # leakage warning
                leaky = res.get('leaky_cols',{})
                if leaky:
                    report_lines.append(f"    ⚠️ Leakage detected in: {', '.join(leaky.keys())}")
            report_lines.append("  Result       : ✅ COMPLETE")

            # SHAP from memory
            sid = st.session_state.get('mt_session_id','')
            if sid:
                try:
                    from tools.model_memory import ModelMemoryManager
                    _mm = ModelMemoryManager()
                    shap_top = _mm.get_top_features(sid, best_m, 8)
                    if shap_top:
                        report_lines.append("\n  SHAP Feature Importance:")
                        for item in shap_top:
                            report_lines.append(
                                f"    {item['rank']:>2}. {item.get('feature_name',''):<25} "
                                f"{item.get('importance_value',0):.4f}")
                except Exception:
                    pass
        else:
            report_lines.append("  Status       : ⏳ NOT DONE")

        # ── Summary ───────────────────────────────────────────────────
        report_lines.append("\n" + "=" * 70)
        report_lines.append("SUMMARY")
        report_lines.append("=" * 70)
        n_done = sum([
            st.session_state.get('df') is not None,
            st.session_state.get('validation_report') is not None,
            st.session_state.get('cleaning_done', False),
            st.session_state.get('analysis_context') is not None,
            st.session_state.get('fe_done', False),
            st.session_state.get('mt_done', False),
        ])
        report_lines.append(f"Steps completed : {n_done} / 6")
        if st.session_state.get('mt_done'):
            report_lines.append(f"Best model      : {st.session_state.get('mt_best_model','?')}")
            report_lines.append(f"Target          : {st.session_state.get('mt_target_col','?')}")
            report_lines.append(f"Problem type    : {st.session_state.get('mt_problem_type','?')}")
        report_lines.append("=" * 70)

        full_report = "\n".join(report_lines)

        # Display
        with st.expander("📄 View Full Report", expanded=True):
            st.code(full_report, language=None)

        # Download
        st.download_button(
            "📥 Download Full Report",
            data=full_report,
            file_name=f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            type="primary",
            use_container_width=True)


def show_snapshots_page():
    """Manage saved pipeline snapshots."""
    st.header("🔄 Pipeline Snapshots")
    st.markdown(
        '<div class="info-box">' +
        '🔄 <b>Pipeline Snapshots</b> — View and manage all saved pipelines. ' +
        'Each snapshot can be reused on new datasets if distributions match.'
        '</div>', unsafe_allow_html=True)

    snapshots = list_snapshots()

    if not snapshots:
        st.markdown(
            '<div class="info-box">ℹ️ No snapshots saved yet.<br>' +
            'Complete the pipeline (Data Upload → Validation → Cleaning → EDA → ' +
            'Feature Engineering → Model Training) then click ' +
            '<b>💾 Save Complete Pipeline Snapshot</b> at the bottom of the ' +
            'Model Training page.</div>',
            unsafe_allow_html=True)
        return

    st.markdown(f"**{len(snapshots)} pipeline snapshot(s) saved**")

    # Summary table
    rows = []
    for s in snapshots:
        m = s.get("metrics", {})
        rows.append({
            "Snapshot ID":  s["snapshot_id"],
            "Dataset":      s["dataset_name"],
            "Target":       s["target_column"],
            "Problem":      s["problem_type"],
            "Model":        s["model_name"],
            "Features":     s["n_features"],
            "Rows":         f"{s['n_rows']:,}",
            "Created":      s["created_at"][:19],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("---")

    # Per-snapshot detail
    for snap in snapshots:
        with st.expander(
                f"📦 {snap['snapshot_id']} — {snap['model_name']} → {snap['target_column']}",
                expanded=False):
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Model",    snap["model_name"])
            sc2.metric("Target",   snap["target_column"])
            sc3.metric("Problem",  snap["problem_type"])
            sc4.metric("Features", snap["n_features"])

            # Metrics
            m = snap.get("metrics", {})
            if m:
                st.markdown("**Model Metrics:**")
                mc1, mc2, mc3 = st.columns(3)
                if snap["problem_type"] == "classification":
                    mc1.metric("Accuracy", f"{m.get('accuracy',0):.4f}")
                    mc2.metric("F1 Score", f"{m.get('f1_score',0):.4f}")
                    mc3.metric("ROC-AUC",  f"{m.get('roc_auc','N/A')}")
                else:
                    mc1.metric("R²",   f"{m.get('r2',0):.4f}")
                    mc2.metric("RMSE", f"{m.get('rmse',0):.4f}")
                    mc3.metric("MAE",  f"{m.get('mae',0):.4f}")

            st.caption(f"Created: {snap['created_at']}")

            # Delete button
            if st.button(f"🗑️ Delete {snap['snapshot_id']}",
                         key=f"del_{snap['snapshot_id']}"):
                if delete_snapshot(snap["snapshot_id"]):
                    st.success(f"✅ Deleted {snap['snapshot_id']}")
                    st.rerun()

    st.markdown("---")
    st.markdown(
        '<div class="info-box">' +
        '💡 <b>How to use snapshots:</b><br>' +
        '1. Upload a new dataset → go to 🔍 Validation<br>' +
        '2. Run Validation → scroll to <b>Pipeline Snapshot Check</b><br>' +
        '3. Click <b>Run Drift Check</b> — it compares new data to all saved pipelines<br>' +
        '4. If PSI &lt; 0.25 → click <b>♻️ Reuse Pipeline</b> — skips retraining<br>' +
        '5. If PSI &gt; 0.25 → run full pipeline and save a new snapshot'
        '</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
# tools/chat_support.py - AI Chat Assistant with Full Memory Integration
"""
AI Chat Support for Agentic Data Scientist
- Accesses all memory databases (SQLite + FAISS)
- Understands full pipeline context including Model Training
- No hallucination (uses only stored facts)
- GPT-4o-mini powered
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
from tools.eda_memory import EDAMemoryManager
from tools.feature_engineering_memory import FEMemoryManager as _FEMem
from tools.model_memory import ModelMemoryManager as _MTMem

load_dotenv()


class ChatSupportAgent:

    def __init__(self, unified_memory_manager):
        self.unified_memory = unified_memory_manager
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model  = "gpt-4o-mini"
        self.eda_memory = EDAMemoryManager()
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

    # ──────────────────────────────────────────────────────────────────
    def get_full_context(self) -> Dict[str, Any]:
        context = {
            'timestamp':      datetime.now().isoformat(),
            'pipeline_state': {},
            'current_data':   {},
            'memory_summary': {},
            'session_state':  {}
        }

        # ── current DataFrame ────────────────────────────────────────
        if st.session_state.get('df') is not None:
            df = st.session_state['df']
            context['current_data'] = {
                'shape':   {'rows': len(df), 'columns': len(df.columns)},
                'columns': df.columns.tolist(),
                'dtypes':  {col: str(dtype) for col, dtype in df.dtypes.items()},
                'statistics': {
                    'missing_values': df.isnull().sum().to_dict(),
                    'memory_usage': f"{df.memory_usage(deep=True).sum()/1024**2:.2f} MB"
                }
            }

        # ── unified memory (ingestion / validation / cleaning / eda) ──
        memory_context = self.unified_memory.get_full_context()

        if memory_context.get('ingestion'):
            ing = memory_context['ingestion']
            context['memory_summary']['ingestion'] = {
                'file_name':   ing.get('file_name',   'Unknown'),
                'data_format': ing.get('data_format', 'Unknown'),
                'rows':        ing.get('row_count',   ing.get('rows', 0)),
                'columns':     ing.get('column_count',ing.get('columns', 0)),
                'timestamp':   ing.get('timestamp',   'Unknown')
            }

        if memory_context.get('validation'):
            val = memory_context['validation']
            context['memory_summary']['validation'] = {
                'status':         val.get('validation_status', 'Unknown'),
                'total_issues':   val.get('total_issues', 0),
                'missing_values': val.get('missing_values', {}),
                'duplicates':     val.get('duplicates', {}),
                'outliers':       val.get('outliers', {}),
                'timestamp':      val.get('timestamp', 'Unknown')
            }

        if memory_context.get('cleaning'):
            cln = memory_context['cleaning']
            context['memory_summary']['cleaning'] = {
                'rows_before':    cln.get('rows_before', 0),
                'rows_after':     cln.get('rows_after',  0),
                'columns_before': cln.get('columns_before', 0),
                'columns_after':  cln.get('columns_after',  0),
                'operations':     cln.get('operations', []),
                'summary':        cln.get('summary', ''),
                'timestamp':      cln.get('timestamp', 'Unknown')
            }

        # ── session state flags ───────────────────────────────────────
        context['session_state'] = {
            'data_loaded':         st.session_state.get('df') is not None,
            'validation_done':     st.session_state.get('validation_report') is not None,
            'validation_rejected': st.session_state.get('validation_rejected', False),
            'cleaning_done':       st.session_state.get('cleaning_done', False),
            'fe_done':             st.session_state.get('fe_done', False),
            'mt_done':             st.session_state.get('mt_done', False),
            'file_name':           st.session_state.get('file_name', 'Unknown')
        }

        steps_completed = []
        if context['session_state']['data_loaded']:       steps_completed.append('Data Ingestion')
        if context['session_state']['validation_done']:   steps_completed.append('Validation')
        if context['session_state']['cleaning_done']:     steps_completed.append('Data Cleaning')

        # ── EDA memory ───────────────────────────────────────────────
        try:
            eda_session = self.eda_memory.get_latest_session()
            if eda_session:
                sid     = eda_session['session_id']
                summary = self.eda_memory.get_session_summary(sid)
                context['memory_summary']['eda'] = {
                    'enabled':       True,
                    'session_id':    sid,
                    'timestamp':     eda_session['timestamp'],
                    'dataset_name':  eda_session['dataset_name'],
                    'row_count':     eda_session['row_count'],
                    'column_count':  eda_session['column_count'],
                    'numeric_columns': (
                        json.loads(eda_session['numeric_columns'])
                        if eda_session.get('numeric_columns') else []),
                    'categorical_columns': (
                        json.loads(eda_session['categorical_columns'])
                        if eda_session.get('categorical_columns') else []),
                    'distributions_analyzed':  len(summary.get('distributions', [])),
                    'distribution_details':    summary.get('distributions', []),
                    'correlations_computed':   len(summary.get('correlations', [])),
                    'correlation_details':     summary.get('correlations', []),
                    'outliers_detected_columns': len(summary.get('outliers', [])),
                    'outlier_details':         summary.get('outliers', []),
                    'visualizations_created':  len(summary.get('visualizations', [])),
                    'insights_generated':      len(summary.get('insights', [])),
                    'insight_details':         summary.get('insights', [])
                }
                steps_completed.append('EDA')
            else:
                context['memory_summary']['eda'] = {
                    'enabled': False,
                    'message': 'No EDA operations saved. Enable "💾 Save to Memory" in the EDA page.'}
        except Exception as e:
            context['memory_summary']['eda'] = {'enabled': False, 'error': str(e)}

        # ── Feature Engineering memory ────────────────────────────────
        try:
            fe_mem     = _FEMem()
            fe_session = fe_mem.get_latest_session()
            if fe_session:
                sid     = fe_session['session_id']
                summary = fe_mem.get_session_summary(sid)
                context['memory_summary']['feature_engineering'] = {
                    'enabled':           True,
                    'session_id':        sid,
                    'timestamp':         fe_session.get('timestamp'),
                    'dataset_name':      fe_session.get('dataset_name'),
                    'problem_type':      fe_session.get('problem_type'),
                    'target_column':     fe_session.get('target_column'),
                    'input_features':    fe_session.get('input_features'),
                    'output_features':   fe_session.get('output_features'),
                    'selected_features': [r['feature_name'] for r in summary.get('selected_features', [])],
                    'dropped_features':  [{'name': r['feature_name'], 'reason': r.get('reason',''), 'detail': r.get('detail','')}
                                          for r in summary.get('dropped_features', [])],
                    'created_features':  [{'name': r['feature_name'], 'method': r.get('creation_method','')}
                                          for r in summary.get('created_features', [])],
                    'top_importance':    sorted(
                        [{'feature': r['feature_name'], 'score': r['score'], 'rank': r['rank']}
                         for r in summary.get('importance_scores', [])
                         if r.get('method') == 'random_forest'],
                        key=lambda x: x['rank'])[:10],
                    'ai_explanations':   [{'type': r['insight_type'], 'text': r['description']}
                                          for r in summary.get('insights', [])
                                          if r.get('insight_type') in
                                          ('shap_global','shap_local','shap_beeswarm','lime_local')]
                }
                steps_completed.append('Feature Engineering')
            else:
                context['memory_summary']['feature_engineering'] = {
                    'enabled': False, 'message': 'Feature Engineering not run yet.'}
        except Exception as exc:
            context['memory_summary']['feature_engineering'] = {'enabled': False, 'error': str(exc)}

        # ── Model Training memory ─────────────────────────────────────
        try:
            mt_mem     = _MTMem()
            mt_session = mt_mem.get_latest_session()

            # also try session state first (more up-to-date)
            ss_mt_done    = st.session_state.get('mt_done', False)
            ss_mt_results = st.session_state.get('mt_results', {})
            ss_best_model = st.session_state.get('mt_best_model', '')
            ss_problem    = st.session_state.get('mt_problem_type', '')
            ss_target     = st.session_state.get('mt_target_col', '')
            ss_session_id = st.session_state.get('mt_session_id', '')

            if ss_mt_done and ss_mt_results:
                # Build context from live session state (most accurate)
                trained_models = []
                for mname, res in ss_mt_results.items():
                    if 'error' in res:
                        trained_models.append({'model_name': mname, 'error': res['error']})
                        continue
                    m = res.get('metrics', {})
                    row = {
                        'model_name':   mname,
                        'train_score':  res.get('train_score', 0),
                        'test_score':   res.get('test_score',  0),
                        'cv_score':     res.get('cv_score'),
                        'fit_status':   res.get('fit_status', 'good'),
                        'is_best':      mname == ss_best_model,
                    }
                    if ss_problem == 'classification':
                        row.update({
                            'accuracy':  m.get('accuracy'),
                            'f1_score':  m.get('f1_score'),
                            'precision': m.get('precision'),
                            'recall':    m.get('recall'),
                            'roc_auc':   m.get('roc_auc'),
                        })
                    else:
                        row.update({
                            'mae':  m.get('mae'),
                            'mse':  m.get('mse'),
                            'rmse': m.get('rmse'),
                            'r2':   m.get('r2'),
                        })
                    trained_models.append(row)

                # get SHAP from memory if available
                shap_features = []
                ai_explanations = []
                if ss_session_id and mt_session:
                    try:
                        shap_features = mt_mem.get_top_features(
                            ss_session_id, ss_best_model, 10)
                        mt_summary = mt_mem.get_session_summary(ss_session_id)
                        ai_explanations = [
                            {'type': r['insight_type'], 'text': r['description']}
                            for r in mt_summary.get('insights', [])
                            if r.get('insight_type') in
                            ('shap_global','shap_local','shap_beeswarm',
                             'lime_local','training_summary','explainability')
                        ]
                    except Exception:
                        pass

                context['memory_summary']['model_training'] = {
                    'enabled':        True,
                    'source':         'session_state',
                    'session_id':     ss_session_id,
                    'target_column':  ss_target,
                    'problem_type':   ss_problem,
                    'best_model':     ss_best_model,
                    'models_trained': trained_models,
                    'top_shap_features': shap_features,
                    'ai_explanations':   ai_explanations,
                }
                steps_completed.append('Model Training')

            elif mt_session:
                # Fallback: load from SQLite
                sid         = mt_session['session_id']
                models      = mt_mem.get_session_models(sid)
                metrics_all = mt_mem.get_session_metrics(sid)
                mt_summary  = mt_mem.get_session_summary(sid)
                best_name   = mt_session.get('best_model_name', '')

                # group metrics by model
                metrics_by_model = {}
                for row in metrics_all:
                    mn = row['model_name']
                    if mn not in metrics_by_model:
                        metrics_by_model[mn] = {}
                    metrics_by_model[mn][row['metric_name']] = row['metric_value']

                trained_models = []
                for m in models:
                    mname = m['model_name']
                    row   = {
                        'model_name':  mname,
                        'train_score': m.get('train_score', 0),
                        'test_score':  m.get('test_score',  0),
                        'cv_score':    m.get('cv_score'),
                        'fit_status':  m.get('fit_status', 'good'),
                        'is_best':     mname == best_name,
                    }
                    row.update(metrics_by_model.get(mname, {}))
                    trained_models.append(row)

                ai_explanations = [
                    {'type': r['insight_type'], 'text': r['description']}
                    for r in mt_summary.get('insights', [])
                    if r.get('insight_type') in
                    ('shap_global','shap_local','shap_beeswarm',
                     'lime_local','training_summary','explainability')
                ]

                context['memory_summary']['model_training'] = {
                    'enabled':        True,
                    'source':         'sqlite',
                    'session_id':     sid,
                    'target_column':  mt_session.get('target_column', ''),
                    'problem_type':   mt_session.get('problem_type', ''),
                    'best_model':     best_name,
                    'models_trained': trained_models,
                    'top_shap_features': mt_mem.get_top_features(sid, best_name, 10),
                    'ai_explanations':   ai_explanations,
                }
                steps_completed.append('Model Training')
            else:
                context['memory_summary']['model_training'] = {
                    'enabled': False,
                    'message': 'Model Training not run yet. Go to 🤖 Model Training page.'}
        except Exception as exc:
            context['memory_summary']['model_training'] = {
                'enabled': False, 'error': str(exc)}

        context['pipeline_state'] = {
            'completed_steps': steps_completed,
            'current_step':    steps_completed[-1] if steps_completed else 'Not Started',
            'next_step':       self._get_next_step(steps_completed)
        }
        return context

    # ──────────────────────────────────────────────────────────────────
    def _get_next_step(self, completed_steps: List[str]) -> str:
        pipeline = ['Data Ingestion', 'Validation', 'Data Cleaning',
                    'EDA', 'Feature Engineering', 'Model Training']
        for step in pipeline:
            if step not in completed_steps:
                return step
        return 'All steps complete! Ready to export.'

    # ──────────────────────────────────────────────────────────────────
    def create_system_prompt(self, context: Dict[str, Any]) -> str:
        ss  = context['session_state']
        ms  = context['memory_summary']
        cd  = context['current_data']
        ps  = context['pipeline_state']

        prompt = f"""You are an expert AI Data Science Assistant for the "Agentic Data Scientist" platform.
CURRENT DATE: {context['timestamp']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔄 PIPELINE STATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Completed Steps : {', '.join(ps['completed_steps']) if ps['completed_steps'] else 'None'}
Current Step    : {ps['current_step']}
Next Step       : {ps['next_step']}
File            : {ss['file_name']}
"""

        if cd:
            prompt += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 CURRENT DATASET
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Shape   : {cd['shape']['rows']:,} rows × {cd['shape']['columns']} cols
Memory  : {cd['statistics']['memory_usage']}
Columns : {', '.join(cd['columns'][:15])}{'...' if len(cd['columns'])>15 else ''}
"""

        # ── ingestion ─────────────────────────────────────────────────
        if ms.get('ingestion'):
            ing = ms['ingestion']
            prompt += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📥 DATA INGESTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
File   : {ing['file_name']} ({ing['data_format']})
Size   : {ing['rows']:,} rows × {ing['columns']} cols
Time   : {ing['timestamp']}
"""

        # ── validation ────────────────────────────────────────────────
        if ms.get('validation'):
            val = ms['validation']
            prompt += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 VALIDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status : {val['status']}
Issues : {val['total_issues']}
Missing: {val['missing_values'].get('total_missing',0)} cells
Dupes  : {val['duplicates'].get('total_duplicates',0)} rows
"""

        # ── cleaning ──────────────────────────────────────────────────
        if ms.get('cleaning'):
            cln = ms['cleaning']
            ops = [f"  {i+1}. {op.get('type','?')}: {op.get('details','')}"
                   for i, op in enumerate(cln['operations'][:5])]
            prompt += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧹 DATA CLEANING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before : {cln['rows_before']:,} rows × {cln['columns_before']} cols
After  : {cln['rows_after']:,} rows × {cln['columns_after']} cols
Removed: {cln['rows_before']-cln['rows_after']:,} rows
Operations ({len(cln['operations'])}):
""" + "\n".join(ops) + "\n"

        # ── EDA ───────────────────────────────────────────────────────
        eda = ms.get('eda', {})
        if eda.get('enabled'):
            prompt += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 EDA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dataset     : {eda['row_count']:,} rows × {eda['column_count']} cols
Numeric cols: {', '.join(eda.get('numeric_columns',[])[:8])}
Categ. cols : {', '.join(eda.get('categorical_columns',[])[:8])}
Distributions analysed : {eda['distributions_analyzed']}
Correlations computed  : {eda['correlations_computed']}
Outlier cols detected  : {eda['outliers_detected_columns']}
Visualizations created : {eda['visualizations_created']}
Insights generated     : {eda['insights_generated']}
"""
        else:
            prompt += "\n📊 EDA: Not saved to memory (enable 💾 Save in EDA page)\n"

        # ── Feature Engineering ───────────────────────────────────────
        fe = ms.get('feature_engineering', {})
        if fe.get('enabled'):
            sel = fe.get('selected_features', [])
            drp = fe.get('dropped_features',  [])
            cre = fe.get('created_features',  [])
            imp = fe.get('top_importance',    [])
            exp = fe.get('ai_explanations',   [])
            prompt += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚙️ FEATURE ENGINEERING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Target  : {fe.get('target_column')}  |  Problem: {fe.get('problem_type')}
Features: {fe.get('input_features')} input → {fe.get('output_features')} output

Selected ({len(sel)}): {', '.join(sel[:10])}{'...' if len(sel)>10 else ''}

Top RF Importance:
""" + "\n".join(f"  {i+1}. {r['feature']} (score={r['score']:.4f})"
               for i, r in enumerate(imp[:8])) + f"""

Created ({len(cre)}): {', '.join(f"{c['name']} [{c['method']}]" for c in cre[:5])}

Dropped ({len(drp)}): {', '.join(f"{d['name']} ({d['reason']})" for d in drp[:5])}
"""
            if exp:
                prompt += "\nAI Explanations from SHAP/LIME:\n"
                for e in exp[:3]:
                    prompt += f"  [{e['type']}] {e['text'][:200]}\n"
        else:
            prompt += f"\n⚙️ FEATURE ENGINEERING: {fe.get('message','Not run yet')}\n"

        # ── Model Training ────────────────────────────────────────────
        mt = ms.get('model_training', {})
        if mt.get('enabled'):
            models = mt.get('models_trained', [])
            best   = mt.get('best_model', '')
            ptype  = mt.get('problem_type', '')
            shap_f = mt.get('top_shap_features', [])
            mt_exp = mt.get('ai_explanations', [])

            prompt += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 MODEL TRAINING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Target       : {mt.get('target_column')}
Problem type : {ptype}
Best model   : {best}
Models trained: {len(models)}

"""
            for m in models:
                if 'error' in m:
                    prompt += f"  ❌ {m['model_name']}: {m['error']}\n"
                    continue
                star = " 🏆" if m.get('is_best') else ""
                prompt += (f"  {'★' if m.get('is_best') else '•'} {m['model_name']}{star}\n"
                           f"    Train={m.get('train_score',0):.4f}  "
                           f"Test={m.get('test_score',0):.4f}  "
                           f"Fit={m.get('fit_status','?')}\n")
                if ptype == 'classification':
                    prompt += (f"    Accuracy={m.get('accuracy',0):.4f}  "
                               f"F1={m.get('f1_score',0):.4f}  "
                               f"ROC-AUC={m.get('roc_auc','N/A')}\n")
                else:
                    prompt += (f"    MAE={m.get('mae',0):.4f}  "
                               f"RMSE={m.get('rmse',0):.4f}  "
                               f"R²={m.get('r2',0):.4f}\n")

            if shap_f:
                prompt += "\nTop SHAP Feature Importances:\n"
                for item in shap_f[:8]:
                    fname = item.get('feature_name', item.get('feature',''))
                    val   = item.get('importance_value', item.get('value', 0))
                    rank  = item.get('rank', '-')
                    prompt += f"  {rank}. {fname}: {val:.4f}\n"

            if mt_exp:
                prompt += "\nAI Explanations (SHAP/LIME):\n"
                for e in mt_exp[:3]:
                    prompt += f"  [{e['type']}] {e['text'][:250]}\n"
        else:
            prompt += f"\n🤖 MODEL TRAINING: {mt.get('message','Not run yet.')}\n"

        prompt += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 YOUR ROLE & STRICT SCOPE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You are an AI assistant EXCLUSIVELY for the Agentic Data Scientist pipeline.
Your knowledge is STRICTLY LIMITED to the pipeline context provided above.

✅ YOU CAN ANSWER:
  • Questions about the uploaded dataset (shape, columns, types, missing values)
  • Questions about validation results (issues found, duplicates, outliers)
  • Questions about cleaning operations performed
  • Questions about EDA results (distributions, correlations, outliers)
  • Questions about feature engineering (selected features, RF scores, encoding, scaling)
  • Questions about model training (which models, accuracy, F1, R², RMSE, fit status)
  • Questions about SHAP/LIME explanations and feature importance
  • Questions about what step to do next in the pipeline
  • Suggestions to improve the model or data quality based on pipeline results

❌ YOU MUST REFUSE anything outside this scope, including:
  • General knowledge questions (history, sports, science, news, geography, etc.)
  • Coding help unrelated to this pipeline
  • Creative writing, jokes, or entertainment
  • Medical, legal, or financial advice
  • Questions about other datasets or projects not in this pipeline
  • Anything not reflected in the pipeline context above

🚫 REFUSAL RESPONSE (use this exact format for out-of-scope questions):
  "I can only answer questions about your Agentic Data Scientist pipeline —
   your dataset, cleaning, EDA, feature engineering, and model results.
   Please ask me something about your data or pipeline."

ADDITIONAL RULES:
  • Use ONLY numbers and facts from the context above — never hallucinate
  • Be specific with numbers (cite exact values from the context)
  • If a pipeline step has not been run yet, say so clearly
  • Explain technical terms in simple English
  • If asked about model performance, use exact values from MODEL TRAINING section
  • If asked about features, use exact values from FEATURE ENGINEERING section
"""
        return prompt

    # ──────────────────────────────────────────────────────────────────
    def chat(self, user_message: str) -> str:
        context       = self.get_full_context()
        system_prompt = self.create_system_prompt(context)
        messages      = [{"role": "system", "content": system_prompt}]
        for msg in st.session_state['chat_history'][-10:]:
            messages.append(msg)
        messages.append({"role": "user", "content": user_message})
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=0.3, max_tokens=1000, top_p=0.9)
            reply = response.choices[0].message.content
            st.session_state['chat_history'].append({"role": "user",      "content": user_message})
            st.session_state['chat_history'].append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def clear_history(self):
        st.session_state['chat_history'] = []
        return "✅ Conversation history cleared!"

    def export_conversation(self) -> str:
        if not st.session_state.get('chat_history'):
            return "No conversation history to export."
        out  = f"# Chat History\nExported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        out += "="*60 + "\n\n"
        for msg in st.session_state['chat_history']:
            role = "🧑 You" if msg['role'] == 'user' else "🤖 Assistant"
            out += f"{role}:\n{msg['content']}\n\n" + "-"*60 + "\n\n"
        return out
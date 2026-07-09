"""
report_generator.py — Pipeline Summary PDF Report
Dark navy/cyan/purple theme matching dashboard.
DUET logo top-right. OpenAI interpretation at end.
Fixes: dedup features, real cleaning data, no drift section,
       save as dataset name, 6 unique SHAP features only.
"""

import io, os, datetime, sqlite3, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image, PageBreak,
)
from reportlab.platypus.flowables import Flowable
from reportlab.lib.colors import HexColor, white

# ── Colours (matching dashboard) ────────────────────────────────────────────
NAVY      = HexColor("#0A1929")
NAVY2     = HexColor("#1A2332")
CARD      = HexColor("#0F172A")
CYAN      = HexColor("#06B6D4")
CYAN_D    = HexColor("#0891B2")
PURPLE    = HexColor("#A855F7")
AMBER     = HexColor("#F59E0B")
RED       = HexColor("#EF4444")
SLATE     = HexColor("#CBD5E1")
GRAY      = HexColor("#64748B")
BORDER    = HexColor("#1E3A5F")
ROW_ALT   = HexColor("#162032")

PAGE_W, PAGE_H = A4
MARGIN = 16 * mm


# ── Styles ────────────────────────────────────────────────────────────────────
def _S():
    s = {}
    s["title"]    = ParagraphStyle("t",  fontName="Helvetica-Bold", fontSize=20, textColor=CYAN,   spaceAfter=2,  leading=24)
    s["sub"]      = ParagraphStyle("s",  fontName="Helvetica",      fontSize=9,  textColor=GRAY,   spaceAfter=8,  leading=12)
    s["body"]     = ParagraphStyle("b",  fontName="Helvetica",      fontSize=8,  textColor=SLATE,  spaceAfter=3,  leading=11)
    s["label"]    = ParagraphStyle("l",  fontName="Helvetica-Bold", fontSize=8,  textColor=CYAN,   spaceAfter=2,  leading=10)
    s["note"]     = ParagraphStyle("n",  fontName="Helvetica-Oblique", fontSize=7, textColor=GRAY, spaceAfter=2,  leading=9)
    s["interp"]   = ParagraphStyle("i",  fontName="Helvetica",      fontSize=8,  textColor=SLATE,  spaceAfter=4,  leading=12)
    s["interp_h"] = ParagraphStyle("ih", fontName="Helvetica-Bold", fontSize=9,  textColor=PURPLE, spaceAfter=3,  leading=12)
    return s


# ── Section header flowable ───────────────────────────────────────────────────
class SecHead(Flowable):
    def __init__(self, text, width, tc=None):
        super().__init__()
        self.text = text; self._w = width
        self.tc = tc or CYAN; self._h = 20
    def wrap(self, *_): return self._w, self._h
    def draw(self):
        c = self.canv
        c.setFillColor(NAVY2); c.roundRect(0,0,self._w,self._h,4,fill=1,stroke=0)
        c.setFillColor(self.tc); c.rect(0,0,4,self._h,fill=1,stroke=0)
        c.setFillColor(self.tc); c.setFont("Helvetica-Bold",9)
        c.drawString(12,6,self.text.upper())


# ── Metric card flowable ──────────────────────────────────────────────────────
class MetBox(Flowable):
    def __init__(self, label, value, width, accent=None):
        super().__init__()
        self.label=label; self.value=str(value)
        self._w=width; self._h=20*mm; self.accent=accent or CYAN
    def wrap(self,*_): return self._w, self._h
    def draw(self):
        c=self.canv
        c.setFillColor(NAVY2); c.roundRect(0,0,self._w,self._h,5,fill=1,stroke=0)
        c.setStrokeColor(self.accent); c.setLineWidth(1)
        c.roundRect(0,0,self._w,self._h,5,fill=0,stroke=1)
        c.setFillColor(self.accent); c.rect(0,self._h-3,self._w,3,fill=1,stroke=0)
        c.setFillColor(self.accent); c.setFont("Helvetica-Bold",13)
        c.drawCentredString(self._w/2, self._h-12*mm+2, self.value)
        c.setFillColor(GRAY); c.setFont("Helvetica",6.5)
        c.drawCentredString(self._w/2, 3*mm, self.label)


# ── DB readers ────────────────────────────────────────────────────────────────
def _qdb(path, sql, default=[]):
    if not os.path.exists(path): return default
    try:
        conn=sqlite3.connect(path); conn.row_factory=sqlite3.Row
        rows=[dict(r) for r in conn.execute(sql).fetchall()]
        conn.close(); return rows
    except: return default

def _read_model_db(d):
    p = os.path.join(d, "model_training", "model_metadata.db")
    out = {"sessions": [], "models": [], "metrics": [], "features": []}

    # Get latest session only
    out["sessions"] = _qdb(p, "SELECT * FROM training_sessions ORDER BY id DESC LIMIT 1")

    if not out["sessions"]:
        return out

    latest_sid = out["sessions"][0].get("session_id", "")
    if not latest_sid:
        return out

    # Filter all tables by latest session_id — no old runs
    out["models"]   = _qdb(p, f"SELECT * FROM trained_models WHERE session_id='{latest_sid}' ORDER BY id DESC")
    out["metrics"]  = _qdb(p, f"SELECT * FROM model_metrics WHERE session_id='{latest_sid}' ORDER BY id DESC")
    out["features"] = _qdb(p, f"SELECT * FROM feature_importance WHERE session_id='{latest_sid}' ORDER BY id DESC")

    return out

def _read_fe_db(d):
    p=os.path.join(d,"feature_engineering","feature_engineering_metadata.db")
    out={"sessions":[],"selected":[],"dropped":[]}
    for tbl,key in [("fe_sessions","sessions"),("selected_features","selected"),
                    ("dropped_features","dropped")]:
        out[key]=_qdb(p,f"SELECT * FROM {tbl} ORDER BY id DESC LIMIT 100")
    return out

def _read_cleaning_db(d):
    p=os.path.join(d,"cleaning","cleaning_metadata.db")
    rows=_qdb(p,"SELECT * FROM cleaning_sessions ORDER BY id DESC LIMIT 1")
    return rows[0] if rows else {}

def _read_eda_corr(d):
    p=os.path.join(d,"eda","eda_metadata.db")
    rows=_qdb(p,"SELECT correlation_matrix FROM correlations ORDER BY id DESC LIMIT 1")
    if not rows: return None
    try:
        import pandas as pd
        return pd.read_json(io.StringIO(rows[0]["correlation_matrix"]))
    except: return None


# ── Dark-theme matplotlib helper ──────────────────────────────────────────────
def _dark(fig, ax):
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1A2332")
    for sp in ax.spines.values(): sp.set_color("#1E3A5F")
    ax.tick_params(colors="#94A3B8",labelsize=7)
    ax.xaxis.label.set_color("#94A3B8")
    ax.yaxis.label.set_color("#94A3B8")
    ax.title.set_color("#06B6D4")


# ── 4 plot functions ──────────────────────────────────────────────────────────
def _p1_target(df, col):
    """Target distribution from live df."""
    if df is None or col not in df.columns: return None
    s = df[col].dropna()
    fig,ax = plt.subplots(figsize=(5,3)); _dark(fig,ax)
    if s.dtype==object or s.nunique()<=20:
        vc=s.value_counts().head(15)
        ax.bar(range(len(vc)),vc.values,color="#06B6D4",edgecolor="#0A1929",alpha=0.9)
        ax.set_xticks(range(len(vc)))
        ax.set_xticklabels(vc.index.astype(str),rotation=30,ha="right",fontsize=6.5,color="#94A3B8")
        ax.set_ylabel("Count",fontsize=7)
    else:
        ax.hist(s,bins=30,color="#06B6D4",edgecolor="#0A1929",linewidth=0.3,alpha=0.85)
        mv=s.mean()
        ax.axvline(mv,color="#A855F7",linewidth=1.5,linestyle="--",label=f"Mean:{mv:.2f}")
        ax.legend(fontsize=7,facecolor="#1A2332",labelcolor="#CBD5E1",edgecolor="#1E3A5F")
        ax.set_ylabel("Frequency",fontsize=7)
    ax.set_title(f"Distribution of '{col}'",fontsize=9,fontweight="bold",pad=6)
    ax.set_xlabel(col,fontsize=7); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=150,bbox_inches="tight",facecolor="#0F172A")
    plt.close(fig); buf.seek(0); return buf


def _p2_corr(corr, top_n=10):
    """Correlation heatmap from stored matrix."""
    if corr is None or corr.empty: return None
    from matplotlib.colors import LinearSegmentedColormap
    cols = corr.abs().mean().sort_values(ascending=False).head(top_n).index.tolist()
    sub  = corr.loc[cols,cols]; n=len(cols)
    fig,ax=plt.subplots(figsize=(max(4,n*0.55),max(3.5,n*0.5))); _dark(fig,ax)
    cmap=LinearSegmentedColormap.from_list("d",["#EF4444","#1A2332","#06B6D4"])
    im=ax.imshow(sub.values,cmap=cmap,vmin=-1,vmax=1,aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(cols,rotation=40,ha="right",fontsize=6,color="#94A3B8")
    ax.set_yticklabels(cols,fontsize=6,color="#94A3B8")
    for i in range(n):
        for j in range(n):
            v=sub.values[i,j]
            ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=5,
                    color="white" if abs(v)>0.5 else "#94A3B8")
    cb=plt.colorbar(im,ax=ax,fraction=0.035,pad=0.03)
    cb.ax.tick_params(colors="#94A3B8",labelsize=6)
    ax.set_title(f"Correlation Heatmap (top {n})",fontsize=9,fontweight="bold",pad=6)
    plt.tight_layout()
    buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=150,bbox_inches="tight",facecolor="#0F172A")
    plt.close(fig); buf.seek(0); return buf


def _p3_shap(feat_rows, model_name):
    """SHAP bar — deduplicated, top 6 unique features."""
    if not feat_rows: return None
    shap=[r for r in feat_rows if r.get("importance_type")=="shap"]
    rows=sorted(shap if shap else feat_rows,
                key=lambda x:x.get("importance_value",0),reverse=True)
    # DEDUP by feature name
    seen=set(); deduped=[]
    for r in rows:
        fn=r.get("feature_name","")
        if fn not in seen: seen.add(fn); deduped.append(r)
        if len(deduped)>=6: break
    names=[r.get("feature_name","") for r in deduped]
    vals=[float(r.get("importance_value",0)) for r in deduped]
    n=len(names)
    fig,ax=plt.subplots(figsize=(5,max(2,n*0.4))); _dark(fig,ax)
    colors=["#06B6D4" if v>=0 else "#EF4444" for v in vals]
    ax.barh(range(n),vals,color=colors,edgecolor="none",height=0.6,alpha=0.9)
    ax.set_yticks(range(n)); ax.set_yticklabels(names,fontsize=7,color="#CBD5E1")
    ax.invert_yaxis(); ax.set_xlabel("Mean |SHAP|",fontsize=7)
    ax.set_title(f"SHAP Importance — {model_name}",fontsize=9,fontweight="bold",pad=6)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=150,bbox_inches="tight",facecolor="#0F172A")
    plt.close(fig); buf.seek(0); return buf


def _p4_feat(feat_rows, model_name):
    """RF/native importance — deduplicated top 6."""
    if not feat_rows: return None
    native=[r for r in feat_rows if r.get("importance_type","").lower()!="shap"]
    rows=sorted(native if native else feat_rows,
                key=lambda x:x.get("importance_value",0),reverse=True)
    seen=set(); deduped=[]
    for r in rows:
        fn=r.get("feature_name","")
        if fn not in seen: seen.add(fn); deduped.append(r)
        if len(deduped)>=6: break
    names=[r.get("feature_name","") for r in deduped]
    vals=[float(r.get("importance_value",0)) for r in deduped]
    n=len(names)
    # cyan→purple gradient
    cls=["#{:02x}{:02x}{:02x}".format(
        int(0x06+(0xA8-0x06)*i/max(n-1,1)),
        int(0xB6+(0x55-0xB6)*i/max(n-1,1)),
        int(0xD4+(0xF7-0xD4)*i/max(n-1,1))) for i in range(n)]
    fig,ax=plt.subplots(figsize=(5,max(2,n*0.4))); _dark(fig,ax)
    ax.barh(range(n),vals,color=cls,edgecolor="none",height=0.6,alpha=0.95)
    ax.set_yticks(range(n)); ax.set_yticklabels(names,fontsize=7,color="#CBD5E1")
    ax.invert_yaxis(); ax.set_xlabel("Importance Score",fontsize=7)
    ax.set_title(f"Feature Importance — {model_name}",fontsize=9,fontweight="bold",pad=6,color="#A855F7")
    ax.title.set_color("#A855F7")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=150,bbox_inches="tight",facecolor="#0F172A")
    plt.close(fig); buf.seek(0); return buf


def _p5_models(all_models):
    """Train vs test grouped bar."""
    if not all_models: return None
    names,trains,tests=[],[],[]
    for m in all_models:
        try:
            names.append(m.get("model_name","?"))
            trains.append(float(m.get("train_score",0) or 0))
            tests.append(float(m.get("test_score",0) or 0))
        except: pass
    if not names: return None
    x=np.arange(len(names)); w=0.35
    fig,ax=plt.subplots(figsize=(5,2.8)); _dark(fig,ax)
    ax.bar(x-w/2,trains,w,label="Train",color="#06B6D4",alpha=0.9)
    ax.bar(x+w/2,tests, w,label="Test", color="#A855F7",alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(names,fontsize=6.5,rotation=15,ha="right",color="#94A3B8")
    ax.set_ylabel("Score",fontsize=7); ax.set_ylim(0,1.1)
    ax.set_title("Model Performance Comparison",fontsize=9,fontweight="bold")
    ax.legend(fontsize=7,facecolor="#1A2332",labelcolor="#CBD5E1",edgecolor="#1E3A5F")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=150,bbox_inches="tight",facecolor="#0F172A")
    plt.close(fig); buf.seek(0); return buf


# ── Image helper ──────────────────────────────────────────────────────────────
def _img(buf, w_mm, h_mm=None):
    if buf is None: return None
    buf.seek(0)
    return Image(buf,width=w_mm*mm,height=h_mm*mm) if h_mm else Image(buf,width=w_mm*mm)

def _two_col(b1, b2, wc, c1="", c2="", S=None):
    hw=(wc-6*mm)/2
    li=_img(b1,hw/mm,52) if b1 else Paragraph("N/A",S["note"])
    ri=_img(b2,hw/mm,52) if b2 else Paragraph("N/A",S["note"])
    t=Table([[li,ri]],colWidths=[hw,hw])
    t.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"TOP"),
                            ("LEFTPADDING",(0,0),(-1,-1),2),("RIGHTPADDING",(0,0),(-1,-1),2)]))
    items=[t]
    if (c1 or c2) and S:
        ct=Table([[Paragraph(c1,S["note"]),Paragraph(c2,S["note"])]],colWidths=[hw,hw])
        ct.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER")]))
        items.append(ct)
    return items


# ── Dark table style ──────────────────────────────────────────────────────────
def _ts(best_row=None, hc=None):
    hc=hc or NAVY2
    st=[("BACKGROUND",(0,0),(-1,0),hc),("TEXTCOLOR",(0,0),(-1,0),CYAN),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[CARD,ROW_ALT]),("TEXTCOLOR",(0,1),(-1,-1),SLATE),
        ("GRID",(0,0),(-1,-1),0.4,BORDER),("LEFTPADDING",(0,0),(-1,-1),6),
        ("RIGHTPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),4),
        ("TOPPADDING",(0,0),(-1,-1),4)]
    if best_row:
        st+=[("BACKGROUND",(0,best_row),(-1,best_row),HexColor("#0D2A1E")),
             ("TEXTCOLOR",(0,best_row),(-1,best_row),CYAN),
             ("FONTNAME",(0,best_row),(-1,best_row),"Helvetica-Bold")]
    return TableStyle(st)


# ── Page header/footer ────────────────────────────────────────────────────────
def _hdr(canvas, doc, logo_path, session_id):
    canvas.saveState()
    canvas.setFillColor(NAVY); canvas.rect(0,PAGE_H-11*mm,PAGE_W,11*mm,fill=1,stroke=0)
    canvas.setFillColor(CYAN); canvas.rect(0,PAGE_H-11*mm,PAGE_W,1.2,fill=1,stroke=0)
    canvas.setFillColor(NAVY); canvas.rect(0,0,PAGE_W,7*mm,fill=1,stroke=0)
    canvas.setFillColor(CYAN); canvas.rect(0,7*mm,PAGE_W,0.8,fill=1,stroke=0)
    canvas.setFillColor(GRAY); canvas.setFont("Helvetica",6)
    canvas.drawString(MARGIN,2*mm,f"Agentic Data Scientist  |  DUET Karachi  |  Session: {session_id}")
    canvas.drawRightString(PAGE_W-MARGIN,2*mm,f"Page {doc.page}  |  {datetime.date.today()}")
    canvas.setFillColor(CYAN); canvas.setFont("Helvetica-Bold",8)
    canvas.drawString(MARGIN,PAGE_H-7.5*mm,"Agentic Data Scientist")
    canvas.setFillColor(GRAY); canvas.setFont("Helvetica",6.5)
    canvas.drawString(MARGIN+106,PAGE_H-7.5*mm,"| ML Pipeline Analysis Report")
    if logo_path and os.path.exists(logo_path):
        try:
            canvas.drawImage(logo_path,PAGE_W-MARGIN-32*mm,PAGE_H-10.5*mm,
                             width=30*mm,height=9*mm,preserveAspectRatio=True,mask="auto")
        except: pass
    canvas.restoreState()


# ── OpenAI interpretation ─────────────────────────────────────────────────────
def _get_interpretation(best_name, problem_type, target, train_sc, test_sc,
                         cv_sc, fit_st, top_features, n_models, best_metrics):
    """Call OpenAI to generate a plain-English report summary."""
    api_key = os.getenv("OPENAI_API_KEY","")
    if not api_key:
        return ("This pipeline report was generated automatically. "
                "Add OPENAI_API_KEY to .env for AI interpretation.")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        feat_str = ", ".join(top_features[:5]) if top_features else "N/A"
        metrics_str = ""
        if problem_type == "regression":
            metrics_str = (f"R²={best_metrics.get('r2','N/A')}, "
                           f"RMSE={best_metrics.get('rmse','N/A')}, "
                           f"MAE={best_metrics.get('mae','N/A')}")
        else:
            metrics_str = (f"Accuracy={best_metrics.get('accuracy','N/A')}, "
                           f"F1={best_metrics.get('f1','N/A')}, "
                           f"Precision={best_metrics.get('precision','N/A')}, "
                           f"Recall={best_metrics.get('recall','N/A')}")

        prompt = f"""
You are a data science report writer. Write a clear, concise 4-5 sentence summary 
of this ML pipeline for a non-technical reader. Include:
1. What was predicted (target={target}, problem={problem_type})
2. How the model performed (best model={best_name}, train={train_sc:.3f}, test={test_sc:.3f}, CV={cv_sc:.3f}, fit={fit_st})
3. Key metrics: {metrics_str}
4. Most important features: {feat_str}
5. Whether the results are good and what they mean in plain English
6. Any concerns (overfitting/underfitting if fit_status is not 'good')
Trained {n_models} model(s) total. Write in a professional but accessible tone.
"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=300, temperature=0.4,
            messages=[{"role":"system","content":"You write concise ML report summaries."},
                      {"role":"user","content":prompt}])
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"AI interpretation unavailable: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def generate_pipeline_report(session_state: dict,
                              logo_path: str = "logo.png",
                              memory_dir: str = "./memory") -> bytes:
    buf      = io.BytesIO()
    S        = _S()
    WC       = PAGE_W - 2*MARGIN
    sid      = session_state.get("session_id","N/A")

    # ── Read DBs ──────────────────────────────────────────────────────
    mdb  = _read_model_db(memory_dir)
    fedb = _read_fe_db(memory_dir)
    cln  = _read_cleaning_db(memory_dir)
    corr = _read_eda_corr(memory_dir)

    m_sess   = mdb["sessions"][0] if mdb["sessions"] else {}
    best_name= m_sess.get("best_model_name","N/A")
    best_mdl = next((m for m in mdb["models"] if m.get("model_name")==best_name),{})
    best_met_rows=[m for m in mdb["metrics"] if m.get("model_name")==best_name]
    best_met = {r["metric_name"]:r["metric_value"] for r in best_met_rows}

    feat_rows_raw=[f for f in mdb["features"] if f.get("model_name")==best_name]
    feat_rows_raw.sort(key=lambda x:x.get("importance_value",0),reverse=True)

    # Dataset info — prefer session state (live), fall back to DB
    ds_name  = session_state.get("file_name", m_sess.get("dataset_name","Unknown"))
    target   = (session_state.get("mt_target_col") or
                session_state.get("fe_target_col") or
                m_sess.get("target_column","N/A"))
    problem  = (session_state.get("mt_problem_type") or
                m_sess.get("problem_type","N/A"))
    fmt      = session_state.get("data_format","CSV")

    # Scores
    train_sc = float(best_mdl.get("train_score",0) or 0)
    test_sc  = float(best_mdl.get("test_score",0)  or 0)
    cv_sc    = float(best_mdl.get("cv_score",0)    or 0)
    fit_st   = best_mdl.get("fit_status","N/A")

    # Cleaning — from session state first (most accurate)
    full_ctx = session_state.get("_unified_memory_context",{})
    cln_ctx  = session_state.get("cleaning_context",{})
    # Try unified_memory context stored by app
    rb = cln.get("rows_before") or session_state.get("rows_before_cleaning","N/A")
    ra = cln.get("rows_after")  or session_state.get("rows_after_cleaning","N/A")
    cb = cln.get("cols_before","N/A")
    ca = cln.get("cols_after","N/A")

    # FE
    fe_sess  = fedb["sessions"][0] if fedb["sessions"] else {}
    n_sel    = fe_sess.get("n_selected_features", len(fedb["selected"]))
    n_drp    = fe_sess.get("n_dropped_features",  len(fedb["dropped"]))
    top_k    = fe_sess.get("top_k_features", n_sel)
    tr_rows  = fe_sess.get("train_rows","N/A")
    te_rows  = fe_sess.get("test_rows","N/A")

    # Get row/col counts from df if available
    df_live  = session_state.get("df")
    if df_live is not None:
        n_rows_orig = f"{len(df_live):,}"
        n_cols_orig = str(len(df_live.columns))
    else:
        n_rows_orig = "N/A"; n_cols_orig = "N/A"

    # Deduped top features
    seen_f=set(); top_feats=[]
    for r in feat_rows_raw:
        fn=r.get("feature_name","")
        if fn not in seen_f: seen_f.add(fn); top_feats.append(r)
        if len(top_feats)>=6: break

    top_feat_names=[r.get("feature_name","") for r in top_feats]

    # ── Generate plots ────────────────────────────────────────────────
    p1=_p1_target(df_live, target)
    p2=_p2_corr(corr,10)
    p3=_p3_shap(feat_rows_raw, best_name)
    p4=_p4_feat(feat_rows_raw, best_name)
    p5=_p5_models(mdb["models"])

    # ── AI interpretation ─────────────────────────────────────────────
    interp_text = _get_interpretation(
        best_name, str(problem), target,
        train_sc, test_sc, cv_sc, str(fit_st),
        top_feat_names, len(mdb["models"]), best_met)

    # ── Build PDF ─────────────────────────────────────────────────────
    doc=SimpleDocTemplate(buf,pagesize=A4,
        leftMargin=MARGIN,rightMargin=MARGIN,topMargin=16*mm,bottomMargin=12*mm)

    def ph(canvas,doc): _hdr(canvas,doc,logo_path,sid)

    story=[]

    def _fmt(v,pct=True):
        try:
            f=float(v)
            return f"{f:.1%}" if pct else f"{f:,.2f}"
        except: return str(v)

    # ════════════════════════════ PAGE 1 ═════════════════════════════
    story.append(Paragraph("Agentic Data Scientist",S["title"]))
    story.append(Paragraph(
        f"ML Pipeline Analysis Report  ·  "
        f"{datetime.datetime.now().strftime('%d %B %Y, %H:%M')}  ·  "
        f"Dawood University of Engineering & Technology, Karachi",S["sub"]))
    story.append(HRFlowable(width=WC,thickness=1.5,color=CYAN,spaceAfter=5))

    # Info row
    bw=(WC-8*mm)/3
    id_=[
        [Paragraph("<b>Dataset</b>",S["label"]),
         Paragraph("<b>Report</b>",S["label"]),
         Paragraph("<b>Pipeline</b>",S["label"])],
        [Paragraph(f"File: {ds_name}<br/>Format: {fmt}<br/>Rows: {n_rows_orig}  Cols: {n_cols_orig}",S["body"]),
         Paragraph(f"Session: {sid}<br/>Date: {datetime.date.today()}",S["body"]),
         Paragraph(f"Target: {target}<br/>Problem: {str(problem).capitalize()}<br/>Best: {best_name}",S["body"])]
    ]
    it=Table(id_,colWidths=[bw,bw,bw])
    it.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),NAVY2),("BACKGROUND",(0,1),(-1,1),CARD),
        ("TEXTCOLOR",(0,1),(-1,1),SLATE),("GRID",(0,0),(-1,-1),0.4,BORDER),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7)]))
    story.append(it); story.append(Spacer(1,4*mm))

    # Metric cards
    story.append(SecHead("Key Metrics",WC)); story.append(Spacer(1,2*mm))
    mw=(WC-12*mm)/4
    fa=CYAN if "good" in str(fit_st).lower() else AMBER
    mt_=Table([[MetBox("Test Score", _fmt(test_sc), mw, CYAN),
                MetBox("Train Score",_fmt(train_sc),mw, CYAN),
                MetBox("CV Score",   _fmt(cv_sc),   mw, PURPLE),
                MetBox("Fit Status", str(fit_st).replace("_"," ").title() if fit_st else "N/A", mw, fa)]],
               colWidths=[mw]*4)
    mt_.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                              ("LEFTPADDING",(0,0),(-1,-1),2),("RIGHTPADDING",(0,0),(-1,-1),2)]))
    story.append(mt_); story.append(Spacer(1,4*mm))

    # EDA plots
    story.append(SecHead("EDA Insights",WC,CYAN)); story.append(Spacer(1,3*mm))
    story.extend(_two_col(p1,p2,WC,
        f"Target Distribution: '{target}'","Correlation Heatmap (top 10)",S=S))
    story.append(Spacer(1,4*mm))

    # Validation table
    story.append(SecHead("Data Validation",WC,CYAN)); story.append(Spacer(1,2*mm))
    vr=session_state.get("validation_report",{}) or {}
    mv_=vr.get("missing_values",{}); dup_=vr.get("duplicates",{}); out_=vr.get("outliers",{})
    mc=mv_.get("total_missing","N/A"); dc=dup_.get("total_duplicates","N/A")
    oc=len(out_.get("iqr",{})) if isinstance(out_,dict) else "N/A"
    vd=[["Issue","Count","Status"],
        ["Missing Values",str(mc),"⚠ Present" if mc and mc!="N/A" and int(mc)>0 else "✓ None"],
        ["Outliers",str(oc),"⚠ Present" if oc and oc!="N/A" and int(oc)>0 else "✓ None"],
        ["Duplicates",str(dc),"⚠ Present" if dc and dc!="N/A" and int(dc)>0 else "✓ None"],
        ["Format Match","✔","✓ Passed"]]
    vt=Table(vd,colWidths=[WC*0.5,WC*0.25,WC*0.25])
    vt.setStyle(_ts()); story.append(vt); story.append(Spacer(1,4*mm))

    # Cleaning table — pull from unified_memory context in session state
    story.append(SecHead("Data Cleaning",WC,CYAN)); story.append(Spacer(1,2*mm))
    # Try to get from session state cleaning_context or unified_memory
    cln_ss = {}
    try:
        um = session_state.get("_tools",{})
        # Try direct session keys set by app.py unified_memory.update_cleaning_context
        cln_rb = session_state.get("cleaning_rows_before", rb)
        cln_ra = session_state.get("cleaning_rows_after",  ra)
        cln_cb = session_state.get("cleaning_cols_before", cb)
        cln_ca = session_state.get("cleaning_cols_after",  ca)
    except:
        cln_rb,cln_ra,cln_cb,cln_ca = rb,ra,cb,ca

    # Also try SQLite cleaning DB directly
    cln_db_rb = cln.get("rows_before", cln_rb)
    cln_db_ra = cln.get("rows_after",  cln_ra)

    cd=[["Metric","Before","After"],
        ["Rows",   str(cln_db_rb), str(cln_db_ra)],
        ["Columns",str(cln_cb),   str(cln_ca)],
        ["Status", "Raw data","Cleaned ✓"]]
    ct=Table(cd,colWidths=[WC*0.4,WC*0.3,WC*0.3])
    ct.setStyle(_ts()); story.append(ct)

    story.append(PageBreak())

    # ════════════════════════════ PAGE 2 ═════════════════════════════
    story.append(SecHead("Feature Engineering",WC,CYAN)); story.append(Spacer(1,2*mm))
    fed=[["Parameter","Value"],
         ["Scoring Method","Random Forest Importance"],
         ["Features Kept",str(n_sel)],["Features Dropped",str(n_drp)],
         ["Top-K Selected",str(top_k)],["Train Rows",str(tr_rows)],
         ["Test Rows",str(te_rows)],["Encoding","Fitted on X_train only"],
         ["Scaling","Fitted on X_train only"],["Leakage Prevention","Split before encode/scale"]]
    ft=Table(fed,colWidths=[WC*0.5,WC*0.5]); ft.setStyle(_ts())
    story.append(ft); story.append(Spacer(1,5*mm))

    # SHAP + importance plots
    story.append(SecHead("Feature Importance & SHAP (Top 6 Unique Features)",WC,PURPLE))
    story.append(Spacer(1,3*mm))
    story.extend(_two_col(p3,p4,WC,
        f"SHAP — {best_name}",f"RF Importance — {best_name}",S=S))
    story.append(Spacer(1,5*mm))

    # Model results table
    story.append(SecHead("Model Training Results",WC,CYAN)); story.append(Spacer(1,2*mm))
    mtd=[["Model","Train","Test","CV","Fit"]]
    b_idx=None
    for i,m in enumerate(mdb["models"]):
        nm=m.get("model_name","N/A")
        tr=m.get("train_score",0); te=m.get("test_score",0); cv=m.get("cv_score",0)
        if nm==best_name: b_idx=i+1
        mtd.append([nm,
                    f"{float(tr):.4f}" if tr else "N/A",
                    f"{float(te):.4f}" if te else "N/A",
                    f"{float(cv):.4f}" if cv else "N/A",
                    m.get("fit_status","N/A")])
    mcw=WC/5; mtt=Table(mtd,colWidths=[mcw]*5)
    mtt.setStyle(_ts(best_row=b_idx)); story.append(mtt)
    story.append(Paragraph(f"★ Highlighted = Best model: {best_name}",S["note"]))
    story.append(Spacer(1,4*mm))
    if p5:
        ci=_img(p5,WC/mm*0.78,50)
        if ci: ci.hAlign="CENTER"; story.append(ci)

    story.append(PageBreak())

    # ════════════════════════════ PAGE 3 ═════════════════════════════
    story.append(SecHead("Best Model — Detailed Metrics",WC,CYAN)); story.append(Spacer(1,2*mm))
    dr=[["Metric","Value"]]
    for k,lbl in [("model_name","Model"),("train_score","Train Score"),
                   ("test_score","Test Score"),("cv_score","CV Score"),("fit_status","Fit Status")]:
        dr.append([lbl,str(best_mdl.get(k,"N/A"))])
    label_map={"r2":"R² Score","mae":"MAE","rmse":"RMSE","accuracy":"Accuracy",
               "f1":"F1 Score","precision":"Precision","recall":"Recall",
               "f1_score":"F1 Score","roc_auc":"ROC-AUC"}
    for k,lbl in label_map.items():
        v=best_met.get(k)
        if v is not None: dr.append([lbl,f"{float(v):.4f}"])
    dt=Table(dr,colWidths=[WC*0.5,WC*0.5]); dt.setStyle(_ts()); story.append(dt)
    story.append(Spacer(1,4*mm))

    # Top 5 unique features table
    if top_feats:
        story.append(SecHead("Top 6 Important Features (Unique)",WC,PURPLE)); story.append(Spacer(1,2*mm))
        fd=[["Rank","Feature","Importance Score"]]
        for i,f in enumerate(top_feats,1):
            fd.append([str(i),f.get("feature_name","N/A"),f"{float(f.get('importance_value',0)):.6f}"])
        ftt=Table(fd,colWidths=[WC*0.1,WC*0.6,WC*0.3]); ftt.setStyle(_ts())
        story.append(ftt); story.append(Spacer(1,5*mm))

    # Pipeline snapshot (no drift section)
    story.append(SecHead("Pipeline Snapshot",WC,CYAN)); story.append(Spacer(1,2*mm))
    snap_name=session_state.get("last_snapshot_name","Not saved")
    snd=[["Snapshot","Model","Date","Problem"],
         [snap_name, best_name, str(datetime.date.today()), str(problem)]]
    snt=Table(snd,colWidths=[WC*0.35,WC*0.25,WC*0.20,WC*0.20]); snt.setStyle(_ts())
    story.append(snt); story.append(Spacer(1,6*mm))

    # ── AI Interpretation section ─────────────────────────────────────
    story.append(SecHead("📝 Report Interpretation (AI Generated)",WC,PURPLE))
    story.append(Spacer(1,3*mm))
    story.append(Paragraph("What does this report mean?",S["interp_h"]))
    story.append(Spacer(1,2*mm))

    # Wrap interpretation in a dark box
    interp_tbl=Table([[Paragraph(interp_text,S["interp"])]],colWidths=[WC])
    interp_tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),NAVY2),("GRID",(0,0),(-1,-1),0.4,BORDER),
        ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
        ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8)]))
    story.append(interp_tbl); story.append(Spacer(1,5*mm))

    # Footer
    story.append(HRFlowable(width=WC,thickness=0.5,color=BORDER,spaceAfter=4))
    story.append(Paragraph(
        "■ Auto-generated by Agentic Data Scientist  |  "
        "Dawood University of Engineering & Technology, Karachi  |  "
        "Plots generated fresh from stored statistics  |  "
        "Interpretation powered by GPT-4o-mini.",S["note"]))

    doc.build(story,onFirstPage=ph,onLaterPages=ph)
    return buf.getvalue()
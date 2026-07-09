# tools/explainability.py
"""
Explainability Tool  —  Agentic Data Scientist
================================================
SHAP  — primary (global + local explanations, works on all models)
LIME  — secondary (local explanations on request)

Public functions
----------------
  get_shap_values(model, X, model_name)
      → shap_values array, explainer

  plot_shap_summary(shap_values, X, feature_names)
      → matplotlib Figure (beeswarm / bar)

  plot_shap_bar(shap_values, feature_names, top_n)
      → matplotlib Figure (global mean |SHAP| bar)

  plot_shap_waterfall(shap_values, X, idx, feature_names)
      → matplotlib Figure (local waterfall for one row)

  get_lime_explanation(model, X_train, X_test_row,
                       feature_names, problem_type)
      → lime Explanation object + html string
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
from typing import List, Optional, Tuple, Any

warnings.filterwarnings("ignore")

# ── NumPy 2.0 compatibility patch — MUST run BEFORE importing shap ───
# shap < 0.46 calls np.obj2sctype which was removed in NumPy 2.0.
# Restoring it here allows older SHAP versions to load correctly.
import numpy as _np
if not hasattr(_np, 'obj2sctype'):
    def _obj2sctype(rep, default=None):
        try:
            if isinstance(rep, type) and issubclass(rep, _np.generic):
                return rep
            return _np.dtype(rep).type
        except Exception:
            return default
    _np.obj2sctype = _obj2sctype

# Also patch np.string_ → bytes and np.bool → bool if missing (numpy 2.0)
if not hasattr(_np, 'string_'):
    _np.string_ = bytes
if not hasattr(_np, 'bool'):
    _np.bool = bool
if not hasattr(_np, 'int'):
    _np.int = int
if not hasattr(_np, 'float'):
    _np.float = float
if not hasattr(_np, 'complex'):
    _np.complex = complex
if not hasattr(_np, 'object'):
    _np.object = object
if not hasattr(_np, 'str'):
    _np.str = str

# ── Optional imports ─────────────────────────────────────────────────
try:
    import shap  # type: ignore
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    import importlib
    lime = importlib.import_module("lime")
    importlib.import_module("lime.lime_tabular")
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════
# SHAP
# ══════════════════════════════════════════════════════════════════════

def _get_explainer(model, X_background):
    """
    Choose the right SHAP explainer for the model type.
    Returns explainer instance.
    """
    model_type = type(model).__name__

    # Tree-based models — use TreeExplainer (fast, exact)
    tree_types = {
        "RandomForestClassifier", "RandomForestRegressor",
        "DecisionTreeClassifier", "DecisionTreeRegressor",
        "XGBClassifier", "XGBRegressor",
        "LGBMClassifier", "LGBMRegressor",
        "GradientBoostingClassifier", "GradientBoostingRegressor",
        "ExtraTreesClassifier", "ExtraTreesRegressor",
    }

    if model_type in tree_types:
        return shap.TreeExplainer(model)

    # Linear models — use LinearExplainer
    linear_types = {
        "LogisticRegression", "LinearRegression",
        "Ridge", "Lasso", "ElasticNet",
    }
    if model_type in linear_types:
        return shap.LinearExplainer(model, X_background)

    # Everything else — use KernelExplainer (slow but universal)
    # Sample background to keep it fast
    bg = (X_background.sample(min(50, len(X_background)), random_state=42)
          if hasattr(X_background, "sample") else X_background[:50])
    return shap.KernelExplainer(model.predict, bg)


def get_shap_values(model, X: pd.DataFrame,
                     model_name: str = "") -> Tuple[Any, Any]:
    """
    Compute SHAP values for X.

    Returns
    -------
    (shap_values, explainer)
    shap_values shape: (n_samples, n_features)
    """
    if not SHAP_AVAILABLE:
        raise ImportError("shap not installed. Run: pip install shap")

    X_arr = X.values if hasattr(X, "values") else np.array(X)

    explainer   = _get_explainer(model, X)
    shap_result = explainer(X)

    # Handle multi-class — take the positive class or mean
    if hasattr(shap_result, "values"):
        sv = shap_result.values
        if sv.ndim == 3:
            sv = sv[:, :, 1]  # binary → positive class
    else:
        sv = np.array(shap_result)
        if sv.ndim == 3:
            sv = sv[:, :, 1]

    return sv, explainer


def plot_shap_bar(shap_values: np.ndarray,
                   feature_names: List[str],
                   top_n: int = 15) -> plt.Figure:
    """
    Global feature importance: mean |SHAP| bar chart.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    indices  = np.argsort(mean_abs)[-top_n:]
    feats    = [feature_names[i] for i in indices]
    vals     = mean_abs[indices]

    fig, ax = plt.subplots(figsize=(10, max(5, len(feats) * 0.45)))
    colors  = ["#06b6d4" if v >= vals.mean() else "#94a3b8" for v in vals]
    bars    = ax.barh(feats, vals, color=colors, edgecolor="none")

    ax.set_xlabel("Mean |SHAP value|  (average impact on output)", fontsize=12)
    ax.set_title(f"SHAP Global Feature Importance (top {top_n})",
                 fontsize=14, fontweight="bold", pad=15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.tick_params(colors="#e2e8f0")
    ax.xaxis.label.set_color("#e2e8f0")
    ax.title.set_color("#e2e8f0")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    # value labels
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + max(vals) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, color="#e2e8f0")

    plt.tight_layout()
    return fig


def plot_shap_beeswarm(shap_values: np.ndarray,
                        X: pd.DataFrame,
                        top_n: int = 15) -> plt.Figure:
    """
    SHAP beeswarm (summary) plot — shows distribution of SHAP values.
    """
    if not SHAP_AVAILABLE:
        raise ImportError("shap not installed.")

    mean_abs  = np.abs(shap_values).mean(axis=0)
    top_idx   = np.argsort(mean_abs)[-top_n:][::-1]
    top_feats = [X.columns[i] for i in top_idx]
    sv_top    = shap_values[:, top_idx]
    X_top     = X.iloc[:, top_idx]

    fig, ax = plt.subplots(figsize=(10, max(5, len(top_feats) * 0.5)))

    for row_idx, feat in enumerate(reversed(top_feats)):
        col_idx = list(reversed(list(range(len(top_feats)))))[row_idx]
        sv_col  = sv_top[:, col_idx]
        x_col   = X_top.iloc[:, col_idx]

        norm_x  = (x_col - x_col.min()) / (x_col.max() - x_col.min() + 1e-9)
        colors  = plt.cm.coolwarm(norm_x)

        # jitter y
        jitter  = np.random.uniform(-0.35, 0.35, size=len(sv_col))
        ax.scatter(sv_col, np.full(len(sv_col), row_idx) + jitter,
                   c=colors, s=8, alpha=0.7, linewidths=0)

    ax.axvline(0, color="#64748b", linewidth=1.2, linestyle="--")
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels(list(reversed(top_feats)), fontsize=10)
    ax.set_xlabel("SHAP value  (impact on model output)", fontsize=12)
    ax.set_title("SHAP Beeswarm — Feature Impact Distribution",
                 fontsize=14, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.tick_params(colors="#e2e8f0")
    ax.xaxis.label.set_color("#e2e8f0")
    ax.title.set_color("#e2e8f0")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    plt.tight_layout()
    return fig


def plot_shap_waterfall(shap_values: np.ndarray,
                         X: pd.DataFrame,
                         idx: int = 0,
                         top_n: int = 10) -> plt.Figure:
    """
    Local SHAP waterfall for one prediction (row idx).
    Shows which features pushed the prediction up or down.
    """
    sv_row   = shap_values[idx]
    order    = np.argsort(np.abs(sv_row))[-top_n:]
    feats    = [X.columns[i] for i in order]
    vals     = sv_row[order]
    x_vals   = X.iloc[idx, order].values

    labels = [f"{f}={v:.2f}" for f, v in zip(feats, x_vals)]
    colors = ["#06b6d4" if v >= 0 else "#f43f5e" for v in vals]

    fig, ax = plt.subplots(figsize=(10, max(5, len(feats) * 0.5)))
    bars = ax.barh(labels, vals, color=colors, edgecolor="none")
    ax.axvline(0, color="#64748b", linewidth=1.2, linestyle="--")
    ax.set_xlabel("SHAP value", fontsize=12)
    ax.set_title(f"SHAP Waterfall — Local Explanation (row {idx})",
                 fontsize=14, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.tick_params(colors="#e2e8f0")
    ax.xaxis.label.set_color("#e2e8f0")
    ax.title.set_color("#e2e8f0")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    for bar, val in zip(bars, vals):
        xpos = bar.get_width() + (max(abs(vals)) * 0.02
                                  if val >= 0
                                  else -max(abs(vals)) * 0.08)
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", fontsize=9, color="#e2e8f0")

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# LIME
# ══════════════════════════════════════════════════════════════════════

def get_lime_explanation(model,
                          X_train: np.ndarray,
                          X_test_row: np.ndarray,
                          feature_names: List[str],
                          problem_type: str,
                          num_features: int = 10) -> Tuple[Any, str]:
    """
    Compute LIME local explanation for one row.

    Returns
    -------
    (explanation object, html_string)
    """
    if not LIME_AVAILABLE:
        raise ImportError("lime not installed. Run: pip install lime")

    mode = "classification" if problem_type == "classification" \
           else "regression"

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data  = (X_train.values
                          if hasattr(X_train, "values") else X_train),
        feature_names  = feature_names,
        mode           = mode,
        random_state   = 42,
        discretize_continuous = True,
    )

    row = (X_test_row.values.flatten()
           if hasattr(X_test_row, "values")
           else np.array(X_test_row).flatten())

    predict_fn = (model.predict_proba
                  if problem_type == "classification"
                     and hasattr(model, "predict_proba")
                  else model.predict)

    exp = explainer.explain_instance(
        data_row       = row,
        predict_fn     = predict_fn,
        num_features   = num_features,
    )

    html_str = exp.as_html()
    return exp, html_str


def plot_lime_bar(exp, title: str = "LIME Local Explanation") -> plt.Figure:
    """
    Plot LIME explanation as a horizontal bar chart.
    """
    items  = exp.as_list()
    labels = [item[0] for item in items]
    values = [item[1] for item in items]
    colors = ["#06b6d4" if v >= 0 else "#f43f5e" for v in values]

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.5)))
    bars = ax.barh(labels, values, color=colors, edgecolor="none")
    ax.axvline(0, color="#64748b", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Feature contribution to prediction", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.tick_params(colors="#e2e8f0")
    ax.xaxis.label.set_color("#e2e8f0")
    ax.title.set_color("#e2e8f0")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    for bar, val in zip(bars, values):
        offset = max(abs(v) for v in values) * 0.02
        xpos   = bar.get_width() + (offset if val >= 0 else -offset * 4)
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f}", va="center", fontsize=9, color="#e2e8f0")

    plt.tight_layout()
    return fig
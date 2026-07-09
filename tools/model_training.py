# tools/model_training.py
"""
Model Training Tool  —  Agentic Data Scientist
================================================
Uses the engineered df_fe CSV from Feature Engineering.

Public functions
----------------
  recommend_models(df, target_col, problem_type)
      → list of recommended model dicts with reasons

  train_models(df, target_col, problem_type,
               selected_models, test_size, cv_folds,
               tuning_mode, session_id, mem)
      → {model_name: result_dict}

  detect_fit_status(train_score, test_score, problem_type)
      → "overfit" | "underfit" | "good"

  get_retrain_params(model_name, fit_status, current_params)
      → new_params dict

  evaluate_model(model, X_test, y_test, problem_type)
      → metrics dict

  get_comparison_df(results)
      → pd.DataFrame sorted by primary metric
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, KFold
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

warnings.filterwarnings("ignore")

# ── Models folder ──────────────────────────────────────────────────────
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════

def _build_model_registry(problem_type: str) -> Dict[str, Dict]:
    """Return all available models for the given problem type."""
    classification_models = {
        "Logistic Regression": {
            "class": LogisticRegression,
            "default_params": {"max_iter": 1000, "random_state": 42},
            "tune_params": {"C": [0.01, 0.1, 1, 10]},
            "type": "linear",
        },
        "KNN": {
            "class": KNeighborsClassifier,
            "default_params": {"n_neighbors": 5},
            "tune_params": {"n_neighbors": [3, 5, 7, 11]},
            "type": "distance",
        },
        "SVM": {
            "class": SVC,
            "default_params": {
                "probability": True, "random_state": 42, "max_iter": 2000},
            "tune_params": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
            "type": "kernel",
        },
        "Naive Bayes": {
            "class": GaussianNB,
            "default_params": {},
            "tune_params": {},
            "type": "probabilistic",
        },
        "Decision Tree": {
            "class": DecisionTreeClassifier,
            "default_params": {"random_state": 42},
            "tune_params": {"max_depth": [3, 5, 7, None],
                            "min_samples_leaf": [1, 5, 10]},
            "type": "tree",
        },
        "Random Forest": {
            "class": RandomForestClassifier,
            "default_params": {"n_estimators": 100, "random_state": 42,
                               "n_jobs": -1},
            "tune_params": {"n_estimators": [50, 100, 200],
                            "max_depth": [5, 10, None]},
            "type": "ensemble",
        },
    }

    regression_models = {
        "Linear Regression": {
            "class": LinearRegression,
            "default_params": {},
            "tune_params": {},
            "type": "linear",
        },
        "Ridge": {
            "class": Ridge,
            "default_params": {"random_state": 42},
            "tune_params": {"alpha": [0.1, 1.0, 10.0, 100.0]},
            "type": "linear",
        },
        "Lasso": {
            "class": Lasso,
            "default_params": {"random_state": 42, "max_iter": 5000},
            "tune_params": {"alpha": [0.01, 0.1, 1.0, 10.0]},
            "type": "linear",
        },
        "ElasticNet": {
            "class": ElasticNet,
            "default_params": {"random_state": 42, "max_iter": 5000},
            "tune_params": {"alpha": [0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]},
            "type": "linear",
        },
        "KNN Regressor": {
            "class": KNeighborsRegressor,
            "default_params": {"n_neighbors": 5},
            "tune_params": {"n_neighbors": [3, 5, 7, 11]},
            "type": "distance",
        },
        "SVR": {
            "class": SVR,
            "default_params": {"max_iter": 2000},
            "tune_params": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
            "type": "kernel",
        },
        "Decision Tree Regressor": {
            "class": DecisionTreeRegressor,
            "default_params": {"random_state": 42},
            "tune_params": {"max_depth": [3, 5, 7, None],
                            "min_samples_leaf": [1, 5, 10]},
            "type": "tree",
        },
        "Random Forest Regressor": {
            "class": RandomForestRegressor,
            "default_params": {"n_estimators": 100, "random_state": 42,
                               "n_jobs": -1},
            "tune_params": {"n_estimators": [50, 100, 200],
                            "max_depth": [5, 10, None]},
            "type": "ensemble",
        },
    }

    # Tree-based available for both
    if XGBOOST_AVAILABLE:
        if problem_type == "classification":
            classification_models["XGBoost"] = {
                "class": XGBClassifier,
                "default_params": {"use_label_encoder": False,
                                   "eval_metric": "logloss",
                                   "random_state": 42, "n_jobs": -1,
                                   "verbosity": 0},
                "tune_params": {"n_estimators": [50, 100],
                                "max_depth": [3, 5, 7],
                                "learning_rate": [0.05, 0.1, 0.2]},
                "type": "ensemble",
            }
        else:
            regression_models["XGBoost Regressor"] = {
                "class": XGBRegressor,
                "default_params": {"random_state": 42, "n_jobs": -1,
                                   "verbosity": 0},
                "tune_params": {"n_estimators": [50, 100],
                                "max_depth": [3, 5, 7],
                                "learning_rate": [0.05, 0.1, 0.2]},
                "type": "ensemble",
            }

    if LGBM_AVAILABLE:
        if problem_type == "classification":
            classification_models["LightGBM"] = {
                "class": LGBMClassifier,
                "default_params": {"random_state": 42, "n_jobs": -1,
                                   "verbose": -1},
                "tune_params": {"n_estimators": [50, 100],
                                "max_depth": [3, 5, 7],
                                "learning_rate": [0.05, 0.1]},
                "type": "ensemble",
            }
        else:
            regression_models["LightGBM Regressor"] = {
                "class": LGBMRegressor,
                "default_params": {"random_state": 42, "n_jobs": -1,
                                   "verbose": -1},
                "tune_params": {"n_estimators": [50, 100],
                                "max_depth": [3, 5, 7],
                                "learning_rate": [0.05, 0.1]},
                "type": "ensemble",
            }

    if problem_type == "classification":
        return classification_models
    else:
        return regression_models


# ══════════════════════════════════════════════════════════════════════
# RECOMMEND MODELS
# ══════════════════════════════════════════════════════════════════════

def recommend_models(df: pd.DataFrame, target_col: str,
                     problem_type: str) -> List[Dict]:
    """
    Analyse dataset and recommend top-3 models with reasons.

    Returns list of dicts:
      [{"name": str, "reason": str, "priority": int, "type": str}]
    """
    n_rows    = len(df)
    n_features = len(df.columns) - 1
    registry  = _build_model_registry(problem_type)
    recommendations = []

    if problem_type == "classification":
        # Check class balance
        vc = df[target_col].value_counts()
        n_classes = len(vc)
        imbalanced = (vc.min() / vc.max()) < 0.4 if len(vc) >= 2 else False

        if n_rows < 1000:
            # Small dataset — simple models generalize better
            recommendations = [
                {"name": "Logistic Regression",
                 "reason": f"Small dataset ({n_rows} rows). Linear model "
                            "generalizes well and is interpretable.",
                 "priority": 1, "type": "linear"},
                {"name": "Random Forest",
                 "reason": "Handles small datasets well, robust to noise, "
                            "provides feature importance.",
                 "priority": 2, "type": "ensemble"},
                {"name": "KNN",
                 "reason": "Effective on small datasets with clear "
                            "decision boundaries.",
                 "priority": 3, "type": "distance"},
            ]
        elif n_rows < 10000:
            # Medium dataset
            primary = "XGBoost" if XGBOOST_AVAILABLE else "Random Forest"
            secondary = "LightGBM" if LGBM_AVAILABLE else "Random Forest"
            recommendations = [
                {"name": primary,
                 "reason": f"Medium dataset ({n_rows} rows). Gradient "
                            "boosting excels at tabular data.",
                 "priority": 1, "type": "ensemble"},
                {"name": secondary,
                 "reason": "Fast gradient boosting — excellent "
                            "accuracy/speed trade-off.",
                 "priority": 2, "type": "ensemble"},
                {"name": "Random Forest",
                 "reason": "Robust ensemble method, handles mixed feature "
                            "types well.",
                 "priority": 3, "type": "ensemble"},
            ]
        else:
            # Large dataset
            primary = "LightGBM" if LGBM_AVAILABLE else "Random Forest"
            secondary = "XGBoost" if XGBOOST_AVAILABLE else "Random Forest"
            recommendations = [
                {"name": primary,
                 "reason": f"Large dataset ({n_rows} rows). LightGBM is "
                            "optimised for speed at scale.",
                 "priority": 1, "type": "ensemble"},
                {"name": secondary,
                 "reason": "High accuracy on large tabular datasets.",
                 "priority": 2, "type": "ensemble"},
                {"name": "Logistic Regression",
                 "reason": "Fast baseline. Good reference point for "
                            "comparing complex models.",
                 "priority": 3, "type": "linear"},
            ]

        if imbalanced:
            for r in recommendations:
                r["reason"] += (
                    " ⚠️ Class imbalance detected — consider enabling "
                    "SMOTE in Feature Engineering.")

    else:  # regression
        if n_rows < 1000:
            recommendations = [
                {"name": "Ridge",
                 "reason": f"Small dataset ({n_rows} rows). Ridge "
                            "regularisation prevents overfitting.",
                 "priority": 1, "type": "linear"},
                {"name": "Random Forest Regressor",
                 "reason": "Handles non-linear relationships, "
                            "robust on small datasets.",
                 "priority": 2, "type": "ensemble"},
                {"name": "ElasticNet",
                 "reason": "Combines L1+L2 regularisation — good "
                            "when features may be correlated.",
                 "priority": 3, "type": "linear"},
            ]
        elif n_rows < 10000:
            primary = "XGBoost Regressor" if XGBOOST_AVAILABLE \
                      else "Random Forest Regressor"
            secondary = "LightGBM Regressor" if LGBM_AVAILABLE \
                        else "Random Forest Regressor"
            recommendations = [
                {"name": primary,
                 "reason": f"Medium dataset ({n_rows} rows). XGBoost "
                            "leads on tabular regression tasks.",
                 "priority": 1, "type": "ensemble"},
                {"name": secondary,
                 "reason": "Fast and accurate gradient boosting.",
                 "priority": 2, "type": "ensemble"},
                {"name": "Ridge",
                 "reason": "Fast linear baseline for comparison.",
                 "priority": 3, "type": "linear"},
            ]
        else:
            primary = "LightGBM Regressor" if LGBM_AVAILABLE \
                      else "Random Forest Regressor"
            secondary = "XGBoost Regressor" if XGBOOST_AVAILABLE \
                        else "Random Forest Regressor"
            recommendations = [
                {"name": primary,
                 "reason": f"Large dataset ({n_rows} rows). LightGBM "
                            "is the fastest at scale.",
                 "priority": 1, "type": "ensemble"},
                {"name": secondary,
                 "reason": "High accuracy gradient boosting.",
                 "priority": 2, "type": "ensemble"},
                {"name": "Linear Regression",
                 "reason": "Fastest baseline — check if linear "
                            "relationship holds.",
                 "priority": 3, "type": "linear"},
            ]

    # Filter to only models that are actually available in registry
    available = [r for r in recommendations if r["name"] in registry]
    return available


# ══════════════════════════════════════════════════════════════════════
# FIT STATUS DETECTION
# ══════════════════════════════════════════════════════════════════════

def detect_fit_status(train_score: float, test_score: float,
                       problem_type: str) -> str:
    """
    Detect overfitting / underfitting / suspicious fit.

    Rules:
    ┌─────────────────────────────────────────────────────────────────┐
    │ OVERFIT   : train - test > 0.05  (gap > 5%)                    │
    │             Model memorised training data, fails on new data.   │
    │                                                                 │
    │ UNDERFIT  : classification → train < 0.60 AND test < 0.60      │
    │             regression     → train R² < 0.45 AND test R² < 0.45│
    │             Model too simple, not learning the pattern.         │
    │                                                                 │
    │ SUSPECT   : test > train + 0.05 (test suspiciously higher)     │
    │             Possible data leakage or very small test set.       │
    │             test = 1.0 exactly is almost always suspicious.     │
    │                                                                 │
    │ GOOD FIT  : gap ≤ 0.05 AND both scores above thresholds        │
    └─────────────────────────────────────────────────────────────────┘

    Returns "overfit" | "underfit" | "suspect" | "good"
    """
    gap = train_score - test_score   # positive = overfit, negative = test > train

    # ── Underfitting ───────────────────────────────────────────────
    # Both train AND test are low — model is too simple
    if problem_type == "classification":
        underfit_threshold = 0.60
    else:
        underfit_threshold = 0.45   # R² < 0.45 = model barely better than mean

    if train_score < underfit_threshold and test_score < underfit_threshold:
        return "underfit"

    # ── Overfitting ────────────────────────────────────────────────
    # Train score significantly higher than test score (gap > 5%)
    if gap > 0.05:
        return "overfit"

    # ── Suspicious fit ─────────────────────────────────────────────
    # Test score is higher than train by more than 5%, OR
    # test score is perfect (1.0) — almost never happens in real data.
    # Both indicate possible data leakage or too-small test set.
    if test_score > train_score + 0.05 or test_score >= 0.999:
        return "suspect"

    # ── Good fit ───────────────────────────────────────────────────
    return "good"


def get_fit_badge(fit_status: str) -> str:
    return {
        "overfit":  "🔴 Overfitting",
        "underfit": "🟡 Underfitting",
        "suspect":  "🟠 Suspicious Fit",
        "good":     "🟢 Good Fit",
    }.get(fit_status, "⚪ Unknown")


def get_fix_suggestions(model_name: str,
                         fit_status: str,
                         current_params: Dict) -> Dict:
    """
    Return suggested hyperparameter changes to fix overfit/underfit.
    Returns dict of new params to use when retraining.
    """
    if fit_status == "good":
        return current_params

    fixes = dict(current_params)

    if fit_status == "overfit":
        if "max_depth" in fixes and fixes["max_depth"] is not None:
            fixes["max_depth"] = max(2, fixes["max_depth"] - 2)
        elif "max_depth" not in fixes:
            fixes["max_depth"] = 5          # add constraint
        if "min_samples_leaf" in fixes:
            fixes["min_samples_leaf"] = min(20, fixes["min_samples_leaf"] * 2)
        if "C" in fixes:
            fixes["C"] = fixes["C"] / 10   # stronger regularisation
        if "alpha" in fixes:
            fixes["alpha"] = fixes["alpha"] * 10
        if "n_estimators" in fixes:
            fixes["n_estimators"] = max(50, fixes["n_estimators"] - 50)
        if "learning_rate" in fixes:
            fixes["learning_rate"] = max(0.01,
                                         fixes["learning_rate"] * 0.5)

    elif fit_status == "underfit":
        if "max_depth" in fixes:
            if fixes["max_depth"] is None:
                pass  # already unlimited
            else:
                fixes["max_depth"] = fixes["max_depth"] + 3
        if "min_samples_leaf" in fixes:
            fixes["min_samples_leaf"] = max(1,
                fixes["min_samples_leaf"] // 2)
        if "C" in fixes:
            fixes["C"] = fixes["C"] * 10   # less regularisation
        if "alpha" in fixes:
            fixes["alpha"] = max(0.001, fixes["alpha"] / 10)
        if "n_estimators" in fixes:
            fixes["n_estimators"] = fixes["n_estimators"] + 100
        if "n_neighbors" in fixes:
            fixes["n_neighbors"] = max(1, fixes["n_neighbors"] - 2)

    return fixes


# ══════════════════════════════════════════════════════════════════════
# EVALUATE MODEL
# ══════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test,
                   problem_type: str,
                   X_train=None, y_train=None) -> Dict:
    """
    Return full metrics dict for a trained model.
    """
    metrics = {}
    y_pred  = model.predict(X_test)

    if problem_type == "classification":
        metrics["accuracy"]  = float(accuracy_score(y_test, y_pred))
        metrics["precision"] = float(
            precision_score(y_test, y_pred, average="weighted",
                            zero_division=0))
        metrics["recall"]    = float(
            recall_score(y_test, y_pred, average="weighted",
                         zero_division=0))
        metrics["f1_score"]  = float(
            f1_score(y_test, y_pred, average="weighted",
                     zero_division=0))
        # ROC-AUC (binary only)
        n_classes = len(set(y_test))
        if n_classes == 2 and hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
            except Exception:
                metrics["roc_auc"] = None
        else:
            metrics["roc_auc"] = None

        metrics["confusion_matrix"] = confusion_matrix(
            y_test, y_pred).tolist()
        metrics["classification_report"] = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0)

    else:  # regression
        metrics["mae"]  = float(mean_absolute_error(y_test, y_pred))
        metrics["mse"]  = float(mean_squared_error(y_test, y_pred))
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))
        metrics["r2"]   = float(r2_score(y_test, y_pred))
        metrics["y_pred"] = y_pred.tolist()
        metrics["y_test"] = (y_test.tolist()
                             if hasattr(y_test, "tolist") else list(y_test))

    if X_train is not None and y_train is not None:
        metrics["train_score"] = float(model.score(X_train, y_train))
    metrics["test_score"]  = float(model.score(X_test, y_test))

    return metrics


# ══════════════════════════════════════════════════════════════════════
# TRAIN MODELS
# ══════════════════════════════════════════════════════════════════════

def train_models(X_train: pd.DataFrame,
                 X_test: pd.DataFrame,
                 y_train: pd.Series,
                 y_test: pd.Series,
                 target_col: str,
                 problem_type: str,
                 selected_models: List[str],
                 cv_folds: int,
                 tuning_mode: str,
                 session_id: str,
                 mem,
                 custom_params: Dict = None) -> Dict[str, Dict]:
    """
    Train all selected models using pre-split, pre-encoded, pre-scaled data.
    Encoding and scaling are done in Feature Engineering (fit on train only).
    NO train_test_split here — data arrives already split from FE step.
    """
    registry = _build_model_registry(problem_type)
    results  = {}

    # Fill any remaining NaN safely using train statistics
    X_train = X_train.fillna(X_train.median(numeric_only=True))
    X_test  = X_test.fillna(X_train.median(numeric_only=True))

    # Leakage detection for UI warning (detect only, no drop)
    leaky_cols   = {}
    target_lower = target_col.lower()
    for col in X_train.columns:
        col_lower = col.lower()
        if target_lower in col_lower or col_lower in target_lower:
            leaky_cols[col] = f"name contains '{target_col}'"
            continue
        if pd.api.types.is_numeric_dtype(X_train[col]) and \
           pd.api.types.is_numeric_dtype(y_train):
            try:
                corr = abs(float(X_train[col].corr(y_train)))
                if corr > 0.95:
                    leaky_cols[col] = f"|corr|={corr:.3f} with target"
            except Exception:
                pass
    le = None  # target already encoded in FE step

    # CV splitter — on X_train, y_train only
    if cv_folds > 1:
        cv_splitter = (StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                        random_state=42)
                       if problem_type == "classification"
                       else KFold(n_splits=cv_folds, shuffle=True,
                                   random_state=42))
        cv_scoring = "f1_weighted" if problem_type == "classification" \
                     else "r2"
    else:
        cv_splitter = None

    # ── Train each model ─────────────────────────────────────────────
    for model_name in selected_models:
        if model_name not in registry:
            continue

        spec           = registry[model_name]
        default_params = dict(spec["default_params"])

        try:
            # user-provided custom params override defaults
            user_params = (custom_params or {}) if tuning_mode == "tuned" else {}
            if user_params:
                # merge: start from defaults, override with user values
                merged_params = {**default_params, **user_params}
                # remove None values for params like max_depth=None
                # (keep them — sklearn accepts None for unlimited depth)
                model       = spec["class"](**merged_params)
                model.fit(X_train, y_train)
                best_params = merged_params
            elif tuning_mode == "tuned" and spec["tune_params"]:
                from sklearn.model_selection import GridSearchCV
                base_model = spec["class"](**default_params)
                grid = GridSearchCV(
                    base_model, spec["tune_params"],
                    cv=min(3, cv_folds if cv_folds > 1 else 3),
                    scoring=(cv_scoring if cv_splitter else None),
                    n_jobs=-1, refit=True, error_score="raise")
                grid.fit(X_train, y_train)
                model        = grid.best_estimator_
                best_params  = {**default_params, **grid.best_params_}
            else:
                model = spec["class"](**default_params)
                model.fit(X_train, y_train)
                best_params = default_params

            train_score = float(model.score(X_train, y_train))
            test_score  = float(model.score(X_test,  y_test))

            # cross-validation on train set only
            cv_score = cv_std = None
            if cv_splitter is not None:
                cv_scores = cross_val_score(
                    model, X_train, y_train, cv=cv_splitter,
                    scoring=cv_scoring, n_jobs=-1)
                cv_score = float(cv_scores.mean())
                cv_std   = float(cv_scores.std())

            fit_status = detect_fit_status(train_score, test_score,
                                            problem_type)
            metrics    = evaluate_model(model, X_test, y_test,
                                         problem_type, X_train, y_train)

            # save model to disk
            model_path = os.path.join(
                MODELS_DIR,
                f"{session_id}_{model_name.replace(' ', '_')}.pkl")
            joblib.dump({"model": model, "le": le,
                         "feature_names": X_train.columns.tolist(),
                         "params": best_params}, model_path)

            # log to memory
            mem.log_model(
                session_id, model_name, best_params,
                train_score, test_score, cv_score, cv_std, fit_status)
            mem.log_metrics(
                session_id, model_name,
                {k: v for k, v in metrics.items()
                 if isinstance(v, (int, float)) and v is not None})

            results[model_name] = {
                "model":       model,
                "metrics":     metrics,
                "train_score": train_score,
                "test_score":  test_score,
                "cv_score":    cv_score,
                "cv_std":      cv_std,
                "fit_status":  fit_status,
                "params":      best_params,
                "model_path":  model_path,
                "le":          le,
                "feature_names": X_train.columns.tolist(),
                "X_train": X_train, "X_test": X_test,
                "y_train": y_train, "y_test":  y_test,
                "leaky_cols":  leaky_cols,
            }

        except Exception as exc:
            results[model_name] = {"error": str(exc)}

    return results


# ══════════════════════════════════════════════════════════════════════
# RETRAIN WITH FIXES
# ══════════════════════════════════════════════════════════════════════

def retrain_model(X_train: pd.DataFrame,
                  X_test: pd.DataFrame,
                  y_train: pd.Series,
                  y_test: pd.Series,
                  target_col: str,
                  problem_type: str,
                  model_name: str,
                  new_params: Dict,
                  session_id: str,
                  mem) -> Dict:
    """
    Retrain a single model with updated hyperparameters.
    Accepts pre-split, pre-encoded, pre-scaled data from Feature Engineering.
    No internal train_test_split — data arrives already split.
    Returns updated result dict.
    """
    registry = _build_model_registry(problem_type)
    if model_name not in registry:
        return {"error": f"Model {model_name} not found in registry."}

    # Fill any remaining NaN using train statistics only
    X_train = X_train.copy().fillna(X_train.median(numeric_only=True))
    X_test  = X_test.copy().fillna(X_train.median(numeric_only=True))

    # Leakage detection (detect only — user decides)
    leaky_cols = {}
    target_lower = target_col.lower()
    for col in X_train.columns:
        col_lower = col.lower()
        if target_lower in col_lower or col_lower in target_lower:
            leaky_cols[col] = f"name contains '{target_col}'"
            continue
        if pd.api.types.is_numeric_dtype(X_train[col]) and            pd.api.types.is_numeric_dtype(y_train):
            try:
                corr = abs(float(X_train[col].corr(y_train)))
                if corr > 0.95:
                    leaky_cols[col] = f"|corr|={corr:.3f} with target"
            except Exception:
                pass

    try:
        spec  = registry[model_name]
        model = spec["class"](**new_params)
        model.fit(X_train, y_train)

        train_score = float(model.score(X_train, y_train))
        test_score  = float(model.score(X_test,  y_test))
        fit_status  = detect_fit_status(train_score, test_score,
                                         problem_type)
        metrics     = evaluate_model(model, X_test, y_test,
                                      problem_type, X_train, y_train)

        # save retrained model
        model_path = os.path.join(
            MODELS_DIR,
            f"{session_id}_{model_name.replace(' ', '_')}_retrained.pkl")
        joblib.dump({"model": model, "le": le,
                     "feature_names": X.columns.tolist(),
                     "params": new_params}, model_path)

        mem.log_retrain(session_id, model_name, new_params, test_score)

        return {
            "model":       model,
            "metrics":     metrics,
            "train_score": train_score,
            "test_score":  test_score,
            "fit_status":  fit_status,
            "params":      new_params,
            "model_path":  model_path,
            "le":          le,
            "feature_names": X.columns.tolist(),
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test":  y_test,
            "leaky_cols":  leaky_cols,
        }

    except Exception as exc:
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════

def get_comparison_df(results: Dict,
                       problem_type: str) -> pd.DataFrame:
    """Build a comparison DataFrame from results dict."""
    rows = []
    for model_name, res in results.items():
        if "error" in res:
            rows.append({"Model": model_name, "Error": res["error"]})
            continue
        m = res.get("metrics", {})
        row = {
            "Model":       model_name,
            "Train Score": round(res.get("train_score", 0), 4),
            "Test Score":  round(res.get("test_score",  0), 4),
            "Fit Status":  get_fit_badge(res.get("fit_status", "good")),
        }
        if res.get("cv_score") is not None:
            row["CV Score"] = round(res["cv_score"], 4)
            row["CV Std"]   = round(res.get("cv_std", 0), 4)

        if problem_type == "classification":
            row["Accuracy"]  = round(m.get("accuracy",  0), 4)
            row["Precision"] = round(m.get("precision", 0), 4)
            row["Recall"]    = round(m.get("recall",    0), 4)
            row["F1 Score"]  = round(m.get("f1_score",  0), 4)
            if m.get("roc_auc") is not None:
                row["ROC-AUC"] = round(m["roc_auc"], 4)
            sort_col = "F1 Score"
        else:
            row["MAE"]  = round(m.get("mae",  0), 4)
            row["MSE"]  = round(m.get("mse",  0), 4)
            row["RMSE"] = round(m.get("rmse", 0), 4)
            row["R²"]   = round(m.get("r2",   0), 4)
            sort_col = "R²"

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df_comp = pd.DataFrame(rows)
    # sort_col may be unset if all models errored — guard with locals() check
    _sort = locals().get("sort_col", None)
    if _sort and _sort in df_comp.columns:
        df_comp = df_comp.sort_values(_sort, ascending=False)
    return df_comp.reset_index(drop=True)


def get_best_model_name(results: Dict, problem_type: str) -> str:
    """Return the name of the best model."""
    best_name  = None
    best_score = -np.inf

    for model_name, res in results.items():
        if "error" in res:
            continue
        m = res.get("metrics", {})
        score = (m.get("f1_score", 0) if problem_type == "classification"
                 else m.get("r2", -np.inf))
        if score > best_score:
            best_score = score
            best_name  = model_name

    return best_name


def save_best_model(results: Dict, problem_type: str,
                     session_id: str) -> Optional[str]:
    """Save the best model as models/best_model.pkl."""
    best_name = get_best_model_name(results, problem_type)
    if not best_name or "error" in results.get(best_name, {}):
        return None

    res       = results[best_name]
    best_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump({
        "model":         res["model"],
        "model_name":    best_name,
        "le":            res.get("le"),
        "feature_names": res.get("feature_names", []),
        "params":        res.get("params", {}),
        "session_id":    session_id,
        "problem_type":  problem_type,
        "saved_at":      datetime.now().isoformat(),
    }, best_path)
    return best_path
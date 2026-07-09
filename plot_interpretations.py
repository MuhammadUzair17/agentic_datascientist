"""
AI-powered plot interpretation functions for EDA
Uses OpenAI to generate natural language explanations for all plot types
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Optional

def generate_histogram_interpretation_ai(df: pd.DataFrame, column: str) -> str:
    """
    Generate AI-powered interpretation for histogram/distribution plot
    
    Args:
        df: DataFrame
        column: Column name being plotted
        
    Returns:
        Natural language interpretation
    """
    try:
        from tools.plot_explainer import PlotExplainer
        
        # Calculate statistics
        stats = {
            'mean': float(df[column].mean()),
            'median': float(df[column].median()),
            'std': float(df[column].std()),
            'min': float(df[column].min()),
            'max': float(df[column].max()),
            'skewness': float(df[column].skew()),
            'kurtosis': float(df[column].kurtosis()),
            'n_points': len(df[column].dropna())
        }
        
        # Create prompt for AI
        prompt = f"""
Explain this histogram/distribution plot in simple terms for non-technical users:

Variable: {column}
Data points: {stats['n_points']:,}
Mean: {stats['mean']:.2f}
Median: {stats['median']:.2f}
Std Dev: {stats['std']:.2f}
Range: [{stats['min']:.2f}, {stats['max']:.2f}]
Skewness: {stats['skewness']:.3f}
Kurtosis: {stats['kurtosis']:.3f}

Provide:
1. What the distribution shape tells us (2-3 sentences)
2. What this means practically (2-3 sentences)
3. One key insight

Keep it concise, friendly, and avoid jargon.
"""
        
        # Try AI-enhanced explanation
        try:
            explainer = PlotExplainer(openai_api_key=os.getenv('OPENAI_API_KEY'))
            if explainer.openai_api_key:
                response = explainer.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a data visualization expert who explains plots in simple, user-friendly language."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                
                ai_explanation = response.choices[0].message.content
                
                # Format with header
                result = f"## 📊 Distribution Interpretation\n\n{ai_explanation}\n\n"
                result += f"**Statistics:** Mean={stats['mean']:.2f}, Median={stats['median']:.2f}, Std={stats['std']:.2f}"
                
                return result
        except:
            pass
        
        # Fallback to rule-based
        return generate_histogram_interpretation_fallback(column, stats)
        
    except Exception as e:
        return f"Distribution plot of {column}"

def generate_histogram_interpretation_fallback(column: str, stats: Dict) -> str:
    """Fallback interpretation without AI"""
    skew = stats['skewness']
    
    interpretation = f"## 📊 Distribution Interpretation\n\n"
    
    # Shape description
    if abs(skew) < 0.5:
        interpretation += f"**Shape:** The distribution of {column} is approximately symmetric (bell-shaped). "
        interpretation += f"Values are evenly distributed around the mean of {stats['mean']:.2f}.\n\n"
    elif skew > 0.5:
        interpretation += f"**Shape:** The distribution is right-skewed (positively skewed). "
        interpretation += f"Most values cluster toward {stats['min']:.2f}, with a long tail extending to {stats['max']:.2f}.\n\n"
    else:
        interpretation += f"**Shape:** The distribution is left-skewed (negatively skewed). "
        interpretation += f"Most values cluster toward {stats['max']:.2f}, with a long tail extending to {stats['min']:.2f}.\n\n"
    
    # Spread description
    cv = (stats['std'] / stats['mean']) * 100 if stats['mean'] != 0 else 0
    if cv > 50:
        interpretation += f"**Spread:** High variability (CV={cv:.1f}%). Data points are widely dispersed.\n\n"
    elif cv > 20:
        interpretation += f"**Spread:** Moderate variability (CV={cv:.1f}%). Reasonable spread around the mean.\n\n"
    else:
        interpretation += f"**Spread:** Low variability (CV={cv:.1f}%). Data points cluster tightly around the mean.\n\n"
    
    interpretation += f"**Range:** Values span from {stats['min']:.2f} to {stats['max']:.2f}."
    
    return interpretation

def generate_boxplot_interpretation_ai(df: pd.DataFrame, column: str) -> str:
    """Generate AI-powered interpretation for boxplot"""
    try:
        from tools.plot_explainer import PlotExplainer
        
        # Calculate quartiles
        Q1 = df[column].quantile(0.25)
        Q2 = df[column].quantile(0.50)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(df)) * 100
        
        prompt = f"""
Explain this boxplot in simple terms for non-technical users:

Variable: {column}
Median (Q2): {Q2:.2f}
Q1 (25th percentile): {Q1:.2f}
Q3 (75th percentile): {Q3:.2f}
IQR: {IQR:.2f}
Outliers: {outlier_count} ({outlier_pct:.1f}%)
Range: [{df[column].min():.2f}, {df[column].max():.2f}]

Provide:
1. What the boxplot shape reveals (2-3 sentences)
2. What the outliers mean (1-2 sentences)
3. One practical insight

Keep it simple and friendly.
"""
        
        try:
            explainer = PlotExplainer(openai_api_key=os.getenv('OPENAI_API_KEY'))
            if explainer.openai_api_key:
                response = explainer.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a data visualization expert who explains plots in simple terms."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=400
                )
                
                return f"## 📊 Boxplot Interpretation\n\n{response.choices[0].message.content}"
        except:
            pass
        
        # Fallback
        interp = f"## 📊 Boxplot Interpretation\n\n"
        interp += f"**Central Tendency:** The median value is {Q2:.2f}. Half the data falls below this value.\n\n"
        interp += f"**Spread:** The middle 50% of data (IQR) ranges from {Q1:.2f} to {Q3:.2f}.\n\n"
        
        if outlier_count > 0:
            interp += f"**Outliers:** {outlier_count} outliers detected ({outlier_pct:.1f}%). These are unusual values that fall outside the typical range."
        else:
            interp += "**Outliers:** No outliers detected. All values fall within the expected range."
        
        return interp
        
    except Exception as e:
        return f"Boxplot of {column}"

def generate_categorical_plot_interpretation_ai(df: pd.DataFrame, column: str, plot_type: str = "count") -> str:
    """Generate AI interpretation for categorical plots (count, bar, pie)"""
    try:
        from tools.plot_explainer import PlotExplainer
        
        value_counts = df[column].value_counts()
        n_categories = len(value_counts)
        top_5 = value_counts.head(5)
        
        prompt = f"""
Explain this {plot_type} plot in simple terms:

Variable: {column}
Total categories: {n_categories}
Total data points: {len(df)}
Top 5 categories:
{chr(10).join([f"  • {cat}: {count} ({count/len(df)*100:.1f}%)" for cat, count in top_5.items()])}

Provide:
1. What the distribution shows (2-3 sentences)
2. Key patterns or imbalances (1-2 sentences)
3. One practical insight

Be concise and user-friendly.
"""
        
        try:
            explainer = PlotExplainer(openai_api_key=os.getenv('OPENAI_API_KEY'))
            if explainer.openai_api_key:
                response = explainer.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a data visualization expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=400
                )
                
                return f"## 📊 {plot_type.title()} Plot Interpretation\n\n{response.choices[0].message.content}"
        except:
            pass
        
        # Fallback
        interp = f"## 📊 {plot_type.title()} Plot Interpretation\n\n"
        interp += f"**Distribution:** {column} has {n_categories} unique categories across {len(df):,} records.\n\n"
        interp += f"**Top Category:** '{top_5.index[0]}' is most frequent with {top_5.iloc[0]:,} occurrences ({top_5.iloc[0]/len(df)*100:.1f}%).\n\n"
        
        if n_categories > 10:
            interp += f"**Note:** Large number of categories ({n_categories}). Consider grouping rare categories."
        
        return interp
        
    except Exception as e:
        return f"{plot_type.title()} plot of {column}"

def generate_3d_plot_interpretation_ai(df: pd.DataFrame, x_col: str, y_col: str, z_col: str) -> str:
    """Generate AI interpretation for 3D scatter plot"""
    try:
        from tools.plot_explainer import PlotExplainer
        
        # Calculate pairwise correlations
        corr_xy = df[[x_col, y_col]].corr().iloc[0, 1]
        corr_xz = df[[x_col, z_col]].corr().iloc[0, 1]
        corr_yz = df[[y_col, z_col]].corr().iloc[0, 1]
        
        prompt = f"""
Explain this 3D scatter plot in simple terms:

Variables: {x_col}, {y_col}, {z_col}
Data points: {len(df):,}
Correlations:
  • {x_col} ↔ {y_col}: {corr_xy:.3f}
  • {x_col} ↔ {z_col}: {corr_xz:.3f}
  • {y_col} ↔ {z_col}: {corr_yz:.3f}

Provide:
1. What the 3D relationship shows (2-3 sentences)
2. Which variables are most related (1-2 sentences)
3. One key insight for decision-making

Keep it practical and clear.
"""
        
        try:
            explainer = PlotExplainer(openai_api_key=os.getenv('OPENAI_API_KEY'))
            if explainer.openai_api_key:
                response = explainer.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a data visualization expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                
                return f"## 📊 3D Plot Interpretation\n\n{response.choices[0].message.content}"
        except:
            pass
        
        # Fallback
        correlations = [(x_col, y_col, corr_xy), (x_col, z_col, corr_xz), (y_col, z_col, corr_yz)]
        correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        interp = f"## 📊 3D Plot Interpretation\n\n"
        interp += f"**Strongest Relationship:** {correlations[0][0]} ↔ {correlations[0][1]} (r={correlations[0][2]:.3f})\n\n"
        interp += f"**3D Pattern:** This plot shows how {z_col} relates to both {x_col} and {y_col} simultaneously.\n\n"
        
        avg_corr = (abs(corr_xy) + abs(corr_xz) + abs(corr_yz)) / 3
        if avg_corr > 0.7:
            interp += "**Insight:** Strong interconnection detected. These variables form a highly related system."
        elif avg_corr > 0.4:
            interp += "**Insight:** Moderate relationships present. Variables show partial interdependence."
        else:
            interp += "**Insight:** Weak overall correlation. Variables capture different aspects of the data."
        
        return interp
        
    except Exception as e:
        return f"3D plot of {x_col}, {y_col}, {z_col}"

def generate_correlation_heatmap_interpretation_ai(df: pd.DataFrame, corr_matrix: pd.DataFrame) -> str:
    """Generate AI interpretation for correlation heatmap"""
    try:
        from tools.plot_explainer import PlotExplainer
        
        # Find top correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                val = corr_matrix.iloc[i, j]
                if not np.isnan(val):
                    corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], float(val)))
        
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        top_5 = corr_pairs[:5]
        
        prompt = f"""
Explain this correlation heatmap in simple terms:

Total variables: {len(corr_matrix.columns)}
Total correlations analyzed: {len(corr_pairs)}

Top 5 strongest correlations:
{chr(10).join([f"  • {c1} ↔ {c2}: {corr:.3f}" for c1, c2, corr in top_5])}

Provide:
1. What the heatmap reveals overall (2-3 sentences)
2. Key patterns or clusters (1-2 sentences)
3. One actionable insight

Keep it clear and practical.
"""
        
        try:
            explainer = PlotExplainer(openai_api_key=os.getenv('OPENAI_API_KEY'))
            if explainer.openai_api_key:
                response = explainer.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a data visualization expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                
                return f"## 📊 Correlation Heatmap Interpretation\n\n{response.choices[0].message.content}"
        except:
            pass
        
        # Fallback
        interp = f"## 📊 Correlation Heatmap Interpretation\n\n"
        interp += f"**Overview:** Analyzing relationships between {len(corr_matrix.columns)} variables.\n\n"
        
        if top_5:
            interp += f"**Strongest Correlation:** {top_5[0][0]} ↔ {top_5[0][1]} (r={top_5[0][2]:.3f})\n\n"
        
        strong_corrs = [c for c in corr_pairs if abs(c[2]) > 0.7]
        if strong_corrs:
            interp += f"**Insight:** {len(strong_corrs)} strong correlations detected. Consider multicollinearity for modeling."
        else:
            interp += "**Insight:** No strong correlations (|r| > 0.7). Variables are relatively independent."
        
        return interp
        
    except Exception as e:
        return "Correlation heatmap showing variable relationships"

if __name__ == "__main__":
    print("AI Plot Interpretation Functions Loaded!")
    print("\nAvailable functions:")
    print("  • generate_histogram_interpretation_ai()")
    print("  • generate_boxplot_interpretation_ai()")
    print("  • generate_categorical_plot_interpretation_ai()")
    print("  • generate_3d_plot_interpretation_ai()")
    print("  • generate_correlation_heatmap_interpretation_ai()")
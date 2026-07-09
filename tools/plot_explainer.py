# Enhanced Dynamic Plot Explanation Generator
# Automatically generates user-friendly explanations for ANY plot type

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
import json
import os

class PlotExplainer:
    """
    Generates dynamic, user-friendly explanations for any plot
    Supports: scatter, line, bar, histogram, boxplot, 3D, correlation heatmap
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the plot explainer with optional OpenAI API key"""
        self.openai_api_key = openai_api_key
        if openai_api_key:
            from openai import OpenAI
            self.client = OpenAI(api_key=openai_api_key)
    
    # ==================== EXISTING FUNCTIONS (UNCHANGED) ====================
    
    def generate_scatter_explanation(self, df: pd.DataFrame, x_col: str, y_col: str, plot_title: str = None) -> str:
        """Generate explanation for scatter plot"""
        stats_data = self._calculate_statistics(df, x_col, y_col)
        explanation = self._build_explanation(x_col=x_col, y_col=y_col, stats=stats_data, plot_type="Scatter Plot", plot_title=plot_title)
        return explanation
    
    def generate_line_explanation(self, df: pd.DataFrame, x_col: str, y_col: str, plot_title: str = None) -> str:
        """Generate explanation for line plot"""
        stats_data = self._calculate_statistics(df, x_col, y_col)
        explanation = self._build_explanation(x_col=x_col, y_col=y_col, stats=stats_data, plot_type="Line Plot", plot_title=plot_title)
        return explanation
    
    def generate_bar_explanation(self, df: pd.DataFrame, x_col: str, y_col: str, plot_title: str = None) -> str:
        """Generate explanation for bar chart"""
        stats_data = self._calculate_bar_statistics(df, x_col, y_col)
        explanation = self._build_bar_explanation(x_col=x_col, y_col=y_col, stats=stats_data, plot_title=plot_title)
        return explanation
    
    # ==================== NEW FUNCTIONS FOR ALL PLOT TYPES ====================
    
    def generate_histogram_explanation(self, df: pd.DataFrame, column: str) -> str:
        """Generate AI-powered explanation for histogram"""
        if not self.openai_api_key:
            return self._generate_histogram_explanation_basic(df, column)
        
        try:
            # Calculate statistics
            data = df[column].dropna()
            stats_info = {
                'count': len(data),
                'mean': float(data.mean()),
                'median': float(data.median()),
                'std': float(data.std()),
                'min': float(data.min()),
                'max': float(data.max()),
                'skewness': float(data.skew()),
                'kurtosis': float(data.kurtosis())
            }
            
            # Create AI prompt
            prompt = f"""Explain this histogram in simple terms for non-technical users:

Column: {column}
Data points: {stats_info['count']:,}
Mean: {stats_info['mean']:.2f}
Median: {stats_info['median']:.2f}
Std Dev: {stats_info['std']:.2f}
Range: [{stats_info['min']:.2f}, {stats_info['max']:.2f}]
Skewness: {stats_info['skewness']:.2f}
Kurtosis: {stats_info['kurtosis']:.2f}

Provide:
1. What the distribution shape shows (2-3 sentences)
2. What this means in plain English (2-3 sentences)
3. One key insight about the data spread

Keep it concise, friendly, and avoid jargon."""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a data visualization expert who explains plots in simple, user-friendly language."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=600
            )
            
            ai_explanation = response.choices[0].message.content
            
            return f"""# 📊 Histogram: {column}

{ai_explanation}

---

## 📊 Quick Stats
• **Data Points:** {stats_info['count']:,}
• **Mean:** {stats_info['mean']:.2f}
• **Median:** {stats_info['median']:.2f}
• **Std Dev:** {stats_info['std']:.2f}
"""
        except Exception as e:
            print(f"AI enhancement failed: {e}")
            return self._generate_histogram_explanation_basic(df, column)
    
    def generate_boxplot_explanation(self, df: pd.DataFrame, column: str) -> str:
        """Generate AI-powered explanation for boxplot"""
        if not self.openai_api_key:
            return self._generate_boxplot_explanation_basic(df, column)
        
        try:
            # Calculate statistics and outliers
            data = df[column].dropna()
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            outlier_count = len(outliers)
            outlier_pct = (outlier_count / len(data)) * 100
            
            stats_info = {
                'count': len(data),
                'median': float(data.median()),
                'Q1': float(Q1),
                'Q3': float(Q3),
                'IQR': float(IQR),
                'min': float(data.min()),
                'max': float(data.max()),
                'outlier_count': outlier_count,
                'outlier_pct': outlier_pct
            }
            
            # Create AI prompt
            prompt = f"""Explain this boxplot in simple terms for non-technical users:

Column: {column}
Data points: {stats_info['count']:,}
Median (middle line): {stats_info['median']:.2f}
Q1 (25th percentile): {stats_info['Q1']:.2f}
Q3 (75th percentile): {stats_info['Q3']:.2f}
IQR (box height): {stats_info['IQR']:.2f}
Range: [{stats_info['min']:.2f}, {stats_info['max']:.2f}]
Outliers: {stats_info['outlier_count']} ({stats_info['outlier_pct']:.1f}%)

Provide:
1. What the box and whiskers show (2-3 sentences)
2. What the outliers mean (1-2 sentences)
3. One key insight about data spread

Keep it concise, friendly, and avoid jargon."""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a data visualization expert who explains plots in simple, user-friendly language."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=600
            )
            
            ai_explanation = response.choices[0].message.content
            
            return f"""# 📦 Boxplot: {column}

{ai_explanation}

---

## 📊 Quick Stats
• **Median:** {stats_info['median']:.2f}
• **IQR:** {stats_info['IQR']:.2f}
• **Outliers:** {stats_info['outlier_count']} ({stats_info['outlier_pct']:.1f}%)
"""
        except Exception as e:
            print(f"AI enhancement failed: {e}")
            return self._generate_boxplot_explanation_basic(df, column)
    
    def generate_countplot_explanation(self, df: pd.DataFrame, column: str, top_n: int = 10) -> str:
        """Generate AI-powered explanation for count plot"""
        if not self.openai_api_key:
            return self._generate_countplot_explanation_basic(df, column, top_n)
        
        try:
            value_counts = df[column].value_counts().head(top_n)
            total_count = len(df)
            
            # Create AI prompt
            prompt = f"""Explain this count plot in simple terms for non-technical users:

Column: {column}
Total records: {total_count:,}
Unique categories: {df[column].nunique()}
Showing top {top_n} categories

Top categories:
"""
            for i, (cat, count) in enumerate(value_counts.items(), 1):
                pct = (count / total_count) * 100
                prompt += f"{i}. {cat}: {count:,} ({pct:.1f}%)\n"
            
            prompt += """
Provide:
1. What the distribution shows (2-3 sentences)
2. Key findings about category distribution (2-3 sentences)
3. One actionable insight

Keep it concise, friendly, and avoid jargon."""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a data visualization expert who explains plots in simple, user-friendly language."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=600
            )
            
            ai_explanation = response.choices[0].message.content
            
            return f"""# 📊 Count Plot: {column}

{ai_explanation}

---

## 📊 Quick Stats
• **Total Records:** {total_count:,}
• **Unique Categories:** {df[column].nunique()}
• **Top Category:** {value_counts.index[0]} ({(value_counts.iloc[0]/total_count)*100:.1f}%)
"""
        except Exception as e:
            print(f"AI enhancement failed: {e}")
            return self._generate_countplot_explanation_basic(df, column, top_n)
    
    def generate_piechart_explanation(self, df: pd.DataFrame, column: str, top_n: int = 10) -> str:
        """Generate AI-powered explanation for pie chart"""
        if not self.openai_api_key:
            return self._generate_piechart_explanation_basic(df, column, top_n)
        
        try:
            value_counts = df[column].value_counts().head(top_n)
            total_count = len(df)
            
            # Create AI prompt
            prompt = f"""Explain this pie chart in simple terms for non-technical users:

Column: {column}
Total records: {total_count:,}
Showing top {top_n} categories

Distribution:
"""
            for i, (cat, count) in enumerate(value_counts.items(), 1):
                pct = (count / total_count) * 100
                prompt += f"{i}. {cat}: {pct:.1f}%\n"
            
            prompt += """
Provide:
1. What the proportions show (2-3 sentences)
2. Key insights about the distribution (2-3 sentences)
3. One actionable takeaway

Keep it concise, friendly, and avoid jargon."""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a data visualization expert who explains plots in simple, user-friendly language."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=600
            )
            
            ai_explanation = response.choices[0].message.content
            
            return f"""# 🥧 Pie Chart: {column}

{ai_explanation}

---

## 📊 Quick Stats
• **Total Records:** {total_count:,}
• **Largest Slice:** {value_counts.index[0]} ({(value_counts.iloc[0]/total_count)*100:.1f}%)
"""
        except Exception as e:
            print(f"AI enhancement failed: {e}")
            return self._generate_piechart_explanation_basic(df, column, top_n)
    
    def generate_barplot_explanation(self, df: pd.DataFrame, cat_col: str, num_col: str, top_n: int = 10) -> str:
        """Generate AI-powered explanation for bar plot (categorical vs numeric)"""
        if not self.openai_api_key:
            return self._generate_barplot_explanation_basic(df, cat_col, num_col, top_n)
        
        try:
            grouped = df.groupby(cat_col)[num_col].agg(['mean', 'sum', 'count']).sort_values('sum', ascending=False).head(top_n)
            
            # Create AI prompt
            prompt = f"""Explain this bar plot in simple terms for non-technical users:

Chart: {num_col} by {cat_col}
Total categories: {df[cat_col].nunique()}
Showing top {top_n} categories

Top categories by total {num_col}:
"""
            for i, (cat, row) in enumerate(grouped.iterrows(), 1):
                prompt += f"{i}. {cat}: Total={row['sum']:.2f}, Average={row['mean']:.2f}, Count={int(row['count'])}\n"
            
            prompt += """
Provide:
1. What the comparison shows (2-3 sentences)
2. Key findings about category performance (2-3 sentences)
3. One actionable insight

Keep it concise, friendly, and avoid jargon."""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a data visualization expert who explains plots in simple, user-friendly language."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=600
            )
            
            ai_explanation = response.choices[0].message.content
            
            return f"""# 📊 Bar Plot: {num_col} by {cat_col}

{ai_explanation}

---

## 📊 Quick Stats
• **Top Category:** {grouped.index[0]}
• **Highest Total:** {grouped['sum'].iloc[0]:.2f}
"""
        except Exception as e:
            print(f"AI enhancement failed: {e}")
            return self._generate_barplot_explanation_basic(df, cat_col, num_col, top_n)
    
    def generate_3d_explanation(self, df: pd.DataFrame, x_col: str, y_col: str, z_col: str) -> str:
        """Generate AI-powered explanation for 3D scatter plot"""
        if not self.openai_api_key:
            return self._generate_3d_explanation_basic(df, x_col, y_col, z_col)
        
        try:
            # Calculate pairwise correlations
            data_clean = df[[x_col, y_col, z_col]].dropna()
            corr_xy = data_clean[[x_col, y_col]].corr().iloc[0, 1]
            corr_xz = data_clean[[x_col, z_col]].corr().iloc[0, 1]
            corr_yz = data_clean[[y_col, z_col]].corr().iloc[0, 1]
            
            # Create AI prompt
            prompt = f"""Explain this 3D scatter plot in simple terms for non-technical users:

3D Plot: {x_col}, {y_col}, {z_col}
Data points: {len(data_clean):,}

Pairwise correlations:
- {x_col} ↔ {y_col}: {corr_xy:.3f}
- {x_col} ↔ {z_col}: {corr_xz:.3f}
- {y_col} ↔ {z_col}: {corr_yz:.3f}

Provide:
1. What the 3D relationship shows (2-3 sentences)
2. How the three variables interact (2-3 sentences)
3. One key insight about the multi-dimensional relationship

Keep it concise, friendly, and avoid jargon."""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a data visualization expert who explains plots in simple, user-friendly language."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=700
            )
            
            ai_explanation = response.choices[0].message.content
            
            return f"""# 🎯 3D Scatter Plot

{ai_explanation}

---

## 📊 Quick Stats
• **Data Points:** {len(data_clean):,}
• **Strongest Correlation:** {max([(x_col, y_col, corr_xy), (x_col, z_col, corr_xz), (y_col, z_col, corr_yz)], key=lambda x: abs(x[2]))[0]} ↔ {max([(x_col, y_col, corr_xy), (x_col, z_col, corr_xz), (y_col, z_col, corr_yz)], key=lambda x: abs(x[2]))[1]} ({max(abs(corr_xy), abs(corr_xz), abs(corr_yz)):.3f})
"""
        except Exception as e:
            print(f"AI enhancement failed: {e}")
            return self._generate_3d_explanation_basic(df, x_col, y_col, z_col)
    
    def generate_correlation_heatmap_explanation(self, df: pd.DataFrame, numeric_cols: List[str]) -> str:
        """Generate AI-powered explanation for correlation heatmap"""
        if not self.openai_api_key:
            return self._generate_correlation_explanation_basic(df, numeric_cols)
        
        try:
            corr_matrix = df[numeric_cols].corr()
            
            # Find strongest correlations
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    val = corr_matrix.iloc[i, j]
                    if not np.isnan(val):
                        corr_pairs.append((col1, col2, val))
            
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            top_positive = [c for c in corr_pairs if c[2] > 0][:3]
            top_negative = [c for c in corr_pairs if c[2] < 0][:3]
            
            # Create AI prompt
            prompt = f"""Explain this correlation heatmap in simple terms for non-technical users:

Variables analyzed: {len(numeric_cols)}
Total correlations: {len(corr_pairs)}

Top positive correlations:
"""
            for i, (col1, col2, corr) in enumerate(top_positive, 1):
                prompt += f"{i}. {col1} ↔ {col2}: {corr:.3f}\n"
            
            prompt += "\nTop negative correlations:\n"
            for i, (col1, col2, corr) in enumerate(top_negative, 1):
                prompt += f"{i}. {col1} ↔ {col2}: {corr:.3f}\n"
            
            prompt += """
Provide:
1. What the correlation patterns show (2-3 sentences)
2. Key insights about variable relationships (2-3 sentences)
3. One actionable recommendation

Keep it concise, friendly, and avoid jargon."""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a data visualization expert who explains plots in simple, user-friendly language."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=700
            )
            
            ai_explanation = response.choices[0].message.content
            
            return f"""# 🔥 Correlation Heatmap

{ai_explanation}

---

## 📊 Quick Stats
• **Variables:** {len(numeric_cols)}
• **Strongest Correlation:** {corr_pairs[0][0]} ↔ {corr_pairs[0][1]} ({corr_pairs[0][2]:.3f})
"""
        except Exception as e:
            print(f"AI enhancement failed: {e}")
            return self._generate_correlation_explanation_basic(df, numeric_cols)
    
    # ==================== BASIC FALLBACK FUNCTIONS ====================
    
    def _generate_histogram_explanation_basic(self, df: pd.DataFrame, column: str) -> str:
        """Basic histogram explanation without AI"""
        data = df[column].dropna()
        return f"""# 📊 Histogram: {column}

This histogram shows the distribution of {column} across {len(data):,} data points.

**Statistics:**
• Mean: {data.mean():.2f}
• Median: {data.median():.2f}
• Std Dev: {data.std():.2f}
• Range: [{data.min():.2f}, {data.max():.2f}]
"""
    
    def _generate_boxplot_explanation_basic(self, df: pd.DataFrame, column: str) -> str:
        """Basic boxplot explanation without AI"""
        data = df[column].dropna()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        return f"""# 📦 Boxplot: {column}

This boxplot shows the distribution and outliers in {column}.

**Statistics:**
• Median: {data.median():.2f}
• IQR: {IQR:.2f}
• Range: [{data.min():.2f}, {data.max():.2f}]
"""
    
    def _generate_countplot_explanation_basic(self, df: pd.DataFrame, column: str, top_n: int) -> str:
        """Basic count plot explanation without AI"""
        value_counts = df[column].value_counts().head(top_n)
        return f"""# 📊 Count Plot: {column}

Shows the frequency distribution across {df[column].nunique()} unique categories.

**Top Category:** {value_counts.index[0]} ({value_counts.iloc[0]:,} occurrences)
"""
    
    def _generate_piechart_explanation_basic(self, df: pd.DataFrame, column: str, top_n: int) -> str:
        """Basic pie chart explanation without AI"""
        value_counts = df[column].value_counts().head(top_n)
        total = len(df)
        return f"""# 🥧 Pie Chart: {column}

Shows the proportional distribution across categories.

**Largest Slice:** {value_counts.index[0]} ({(value_counts.iloc[0]/total)*100:.1f}%)
"""
    
    def _generate_barplot_explanation_basic(self, df: pd.DataFrame, cat_col: str, num_col: str, top_n: int) -> str:
        """Basic bar plot explanation without AI"""
        grouped = df.groupby(cat_col)[num_col].agg(['mean', 'sum']).sort_values('sum', ascending=False).head(top_n)
        return f"""# 📊 Bar Plot: {num_col} by {cat_col}

Comparing {num_col} across different {cat_col} categories.

**Top Category:** {grouped.index[0]} (Total: {grouped['sum'].iloc[0]:.2f})
"""
    
    def _generate_3d_explanation_basic(self, df: pd.DataFrame, x_col: str, y_col: str, z_col: str) -> str:
        """Basic 3D explanation without AI"""
        data_clean = df[[x_col, y_col, z_col]].dropna()
        return f"""# 🎯 3D Scatter Plot

3D visualization of the relationship between {x_col}, {y_col}, and {z_col}.

**Data Points:** {len(data_clean):,}
"""
    
    def _generate_correlation_explanation_basic(self, df: pd.DataFrame, numeric_cols: List[str]) -> str:
        """Basic correlation explanation without AI"""
        return f"""# 🔥 Correlation Heatmap

Showing correlations between {len(numeric_cols)} numeric variables.

Examine the heatmap to identify strong positive (red) and negative (blue) correlations.
"""
    
    # ==================== EXISTING HELPER FUNCTIONS (UNCHANGED) ====================
    
    def _calculate_statistics(self, df: pd.DataFrame, x_col: str, y_col: str) -> Dict:
        """Calculate correlation and other statistics"""
        clean_df = df[[x_col, y_col]].dropna()
        
        if len(clean_df) == 0:
            return self._empty_stats()
        
        x_data = clean_df[x_col].values
        y_data = clean_df[y_col].values
        
        correlation, p_value = stats.pearsonr(x_data, y_data)
        r_squared = correlation ** 2
        
        x_range = (x_data.min(), x_data.max())
        y_range = (y_data.min(), y_data.max())
        
        trend = self._determine_trend(correlation)
        strength = self._determine_strength(abs(correlation))
        
        return {
            'correlation': correlation,
            'r_squared': r_squared,
            'p_value': p_value,
            'x_range': x_range,
            'y_range': y_range,
            'x_mean': np.mean(x_data),
            'y_mean': np.mean(y_data),
            'x_std': np.std(x_data),
            'y_std': np.std(y_data),
            'n_points': len(clean_df),
            'trend': trend,
            'strength': strength
        }
    
    def _calculate_bar_statistics(self, df: pd.DataFrame, x_col: str, y_col: str) -> Dict:
        """Calculate statistics for bar charts"""
        clean_df = df[[x_col, y_col]].dropna()
        
        if len(clean_df) == 0:
            return self._empty_stats()
        
        grouped = clean_df.groupby(x_col)[y_col].agg(['mean', 'sum', 'count'])
        top_categories = grouped.nlargest(3, 'sum')
        
        return {
            'n_categories': len(grouped),
            'total': grouped['sum'].sum(),
            'average': grouped['mean'].mean(),
            'top_categories': top_categories.to_dict('index'),
            'n_points': len(clean_df)
        }
    
    def _determine_trend(self, correlation: float) -> str:
        """Determine trend direction"""
        if correlation > 0.05:
            return "positive"
        elif correlation < -0.05:
            return "negative"
        else:
            return "no clear"
    
    def _determine_strength(self, abs_correlation: float) -> str:
        """Determine relationship strength"""
        if abs_correlation >= 0.7:
            return "STRONG"
        elif abs_correlation >= 0.4:
            return "MODERATE"
        elif abs_correlation >= 0.2:
            return "WEAK"
        else:
            return "VERY WEAK"
    
    def _build_explanation(self, x_col: str, y_col: str, stats: Dict, plot_type: str, plot_title: str = None) -> str:
        """Build complete explanation for scatter/line plots"""
        title = plot_title or f"{y_col} vs {x_col}"
        
        explanation = f"""
# 📊 {title}

## 🎯 Quick Summary
**Plot Type:** {plot_type}
**Data Points:** {stats['n_points']:,} records
**Relationship:** {stats['strength']} {stats['trend']} correlation

---

## 📖 What This Plot Shows

### The Pattern:
"""
        
        if stats['trend'] == 'positive':
            explanation += f"""
• **Upward trend**: As {x_col} increases, {y_col} tends to increase
• **Strength**: {stats['strength']} relationship (r = {stats['correlation']:.3f})
• **Meaning**: Higher values of {x_col} are generally associated with higher {y_col}
"""
        elif stats['trend'] == 'negative':
            explanation += f"""
• **Downward trend**: As {x_col} increases, {y_col} tends to decrease
• **Strength**: {stats['strength']} relationship (r = {stats['correlation']:.3f})
• **Meaning**: Higher values of {x_col} are generally associated with lower {y_col}
"""
        else:
            explanation += f"""
• **No clear pattern**: {x_col} and {y_col} don't show a clear relationship
• **Strength**: {stats['strength']} (r = {stats['correlation']:.3f})
• **Meaning**: {x_col} doesn't predict {y_col} reliably
"""
        
        explanation += f"""
### Data Ranges:
• **{x_col}**: {stats['x_range'][0]:.2f} to {stats['x_range'][1]:.2f}
• **{y_col}**: {stats['y_range'][0]:.2f} to {stats['y_range'][1]:.2f}
"""
        
        return explanation
    
    def _build_bar_explanation(self, x_col: str, y_col: str, stats: Dict, plot_title: str = None) -> str:
        """Build explanation for bar charts"""
        title = plot_title or f"{y_col} by {x_col}"
        
        explanation = f"""
# 📊 {title}

## 🎯 Quick Summary
**Plot Type:** Bar Chart
**Categories:** {stats['n_categories']}
**Total {y_col}:** {stats['total']:,.2f}
**Average {y_col}:** {stats['average']:.2f}

---

## 📖 What This Plot Shows

### The Pattern:
• Comparing {y_col} across different {x_col} categories
• Shows which categories have the highest/lowest values
• Total data points: {stats['n_points']:,}

### Top Categories:
"""
        
        for i, (category, data) in enumerate(stats['top_categories'].items(), 1):
            explanation += f"{i}. **{category}**: {data['sum']:,.2f} (from {data['count']} items)\n"
        
        return explanation
    
    def _empty_stats(self) -> Dict:
        """Return empty stats for invalid data"""
        return {
            'correlation': 0,
            'r_squared': 0,
            'p_value': 1,
            'x_range': (0, 0),
            'y_range': (0, 0),
            'x_mean': 0,
            'y_mean': 0,
            'x_std': 0,
            'y_std': 0,
            'n_points': 0,
            'trend': 'no',
            'strength': 'NONE'
        }
    
    def generate_with_ai_enhancement(self, df: pd.DataFrame, x_col: str, y_col: str, plot_type: str = "scatter") -> str:
        """Generate explanation with AI enhancement"""
        if not self.openai_api_key:
            if plot_type == "scatter":
                return self.generate_scatter_explanation(df, x_col, y_col)
            elif plot_type == "bar":
                return self.generate_bar_explanation(df, x_col, y_col)
        
        if plot_type in ["scatter", "line"]:
            stats = self._calculate_statistics(df, x_col, y_col)
        else:
            stats = self._calculate_bar_statistics(df, x_col, y_col)
        
        prompt = self._create_ai_prompt(x_col, y_col, stats, plot_type)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a data visualization expert who explains plots in simple, user-friendly language. Be concise and clear."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            ai_explanation = response.choices[0].message.content
            
            final_explanation = f"""
# 📊 {y_col} vs {x_col}

{ai_explanation}

---

## 📊 Quick Stats
• **Data Points:** {stats['n_points']:,}
• **Correlation:** {stats.get('correlation', 'N/A')}
• **Strength:** {stats.get('strength', 'N/A')}
"""
            
            return final_explanation
            
        except Exception as e:
            print(f"AI enhancement failed: {e}")
            if plot_type == "scatter":
                return self.generate_scatter_explanation(df, x_col, y_col)
            else:
                return self.generate_bar_explanation(df, x_col, y_col)
    
    def _create_ai_prompt(self, x_col: str, y_col: str, stats: Dict, plot_type: str) -> str:
        """Create prompt for AI enhancement"""
        if plot_type in ["scatter", "line"]:
            prompt = f"""
Explain this {plot_type} plot in simple terms for non-technical users:

Plot: {y_col} vs {x_col}
Data points: {stats['n_points']:,}
Correlation: {stats['correlation']:.3f} ({stats['strength']} {stats['trend']})
R-squared: {stats['r_squared']:.3f}

Provide:
1. What pattern the plot shows (2-3 sentences)
2. What this means in plain English (2-3 sentences)
3. One key takeaway

Keep it concise, friendly, and avoid jargon. Use emojis sparingly.
"""
        else:
            prompt = f"""
Explain this bar chart in simple terms for non-technical users:

Chart: {y_col} by {x_col}
Categories: {stats['n_categories']}
Total {y_col}: {stats['total']:.2f}

Provide:
1. What the chart compares (1-2 sentences)
2. Key findings (2-3 sentences)
3. One key takeaway

Keep it concise, friendly, and avoid jargon.
"""
        
        return prompt


# ==================== CONVENIENCE FUNCTIONS ====================

def create_plot_explanation(df: pd.DataFrame, x_col: str, y_col: str, plot_type: str = "scatter", 
                           use_ai: bool = False, openai_api_key: str = None) -> str:
    """Main function to create plot explanations"""
    explainer = PlotExplainer(openai_api_key=openai_api_key)
    
    if use_ai and openai_api_key:
        return explainer.generate_with_ai_enhancement(df, x_col, y_col, plot_type)
    else:
        if plot_type == "scatter" or plot_type == "line":
            return explainer.generate_scatter_explanation(df, x_col, y_col)
        elif plot_type == "bar":
            return explainer.generate_bar_explanation(df, x_col, y_col)
        else:
            return explainer.generate_scatter_explanation(df, x_col, y_col)


if __name__ == "__main__":
    print("Enhanced Plot Explainer Module Loaded!")
    print("\nNew functions available:")
    print("  • generate_histogram_explanation()")
    print("  • generate_boxplot_explanation()")
    print("  • generate_countplot_explanation()")
    print("  • generate_piechart_explanation()")
    print("  • generate_barplot_explanation()")
    print("  • generate_3d_explanation()")
    print("  • generate_correlation_heatmap_explanation()")
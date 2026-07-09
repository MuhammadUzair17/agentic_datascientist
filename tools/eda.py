# tools/eda.py - UPDATED VERSION with Count/Bar plots, Download, Fixed Sizes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from typing import Dict, Any, List, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style and FIXED PLOT SIZES
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# FIXED SIZES FOR ALL PLOTS
PLOT_WIDTH = 10
PLOT_HEIGHT = 6

class EDAAnalyzer:
    """
    Comprehensive EDA Tool with Dynamic Visualizations
    NOW WITH: Count plots, Bar plots, Download functionality, Fixed sizes
    """
    
    def __init__(self):
        self.analysis_context = {}
    
    # ==================== PLOT DOWNLOAD HELPER ====================
    
    def get_download_link(self, fig, filename: str = "plot.png") -> str:
        """
        Generate download link for matplotlib figure
        
        Args:
            fig: Matplotlib figure
            filename: Name for downloaded file
            
        Returns:
            Base64 encoded image for download
        """
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        return img_base64
    
    def create_download_button(self, fig, filename: str = "plot.png", button_text: str = "📥 Download Plot"):
        """
        Create Streamlit download button for plot
        
        Args:
            fig: Matplotlib figure
            filename: Name for downloaded file
            button_text: Text for button
        """
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
        buffer.seek(0)
        
        st.download_button(
            label=button_text,
            data=buffer,
            file_name=filename,
            mime="image/png",
            use_container_width=True
        )
    
    # ==================== TEXTUAL EDA ====================
    
    def generate_text_summary(self, df: pd.DataFrame) -> str:
        """Generate comprehensive textual EDA summary"""
        summary = []
        summary.append("=" * 60)
        summary.append("📊 EXPLORATORY DATA ANALYSIS REPORT")
        summary.append("=" * 60)
        summary.append(f"\n📈 Dataset Overview:")
        summary.append(f"  • Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        summary.append(f"  • Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Column types
        summary.append(f"\n🏷️ Column Types:")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            summary.append(f"  • {dtype}: {count} columns")
        
        # Missing values
        summary.append(f"\n🔍 Missing Values:")
        total_missing = df.isnull().sum().sum()
        if total_missing > 0:
            missing_pct = (total_missing / (df.shape[0] * df.shape[1])) * 100
            summary.append(f"  • Total: {total_missing:,} ({missing_pct:.2f}%)")
            summary.append(f"\n  Columns with missing values:")
            for col in df.columns[df.isnull().any()]:
                missing_count = df[col].isnull().sum()
                missing_col_pct = (missing_count / len(df)) * 100
                summary.append(f"    - {col}: {missing_count:,} ({missing_col_pct:.2f}%)")
        else:
            summary.append(f"  • No missing values found ✓")
        
        # Duplicates
        summary.append(f"\n🔄 Duplicate Rows:")
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            dup_pct = (dup_count / len(df)) * 100
            summary.append(f"  • {dup_count:,} duplicate rows ({dup_pct:.2f}%)")
        else:
            summary.append(f"  • No duplicate rows found ✓")
        
        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            summary.append(f"\n📊 Numeric Columns Analysis ({len(numeric_cols)} columns):")
            summary.append(df[numeric_cols].describe().to_string())
            
            # Skewness analysis
            summary.append(f"\n📐 Skewness Analysis:")
            for col in numeric_cols:
                skew = df[col].skew()
                if abs(skew) > 1:
                    summary.append(f"  • {col}: {skew:.2f} (Highly skewed)")
                elif abs(skew) > 0.5:
                    summary.append(f"  • {col}: {skew:.2f} (Moderately skewed)")
            
            # Outliers
            summary.append(f"\n🎯 Outliers Detection (IQR method):")
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)][col]
                    if len(outliers) > 0:
                        outlier_pct = (len(outliers) / len(df)) * 100
                        summary.append(f"  • {col}: {len(outliers)} outliers ({outlier_pct:.2f}%)")
        
        # Categorical columns analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            summary.append(f"\n📋 Categorical Columns Analysis ({len(categorical_cols)} columns):")
            for col in categorical_cols[:10]:
                n_unique = df[col].nunique()
                top_val = df[col].mode()[0] if not df[col].mode().empty else "N/A"
                top_count = (df[col] == top_val).sum() if top_val != "N/A" else 0
                summary.append(f"\n  • {col}:")
                summary.append(f"    - Unique values: {n_unique}")
                summary.append(f"    - Most frequent: '{top_val}' ({top_count} times)")
                
                if n_unique <= 10:
                    summary.append(f"    - Value counts:")
                    for val, count in df[col].value_counts().head(5).items():
                        summary.append(f"      · {val}: {count}")
        
        # Correlations (top 10)
        if len(numeric_cols) >= 2:
            summary.append(f"\n🔗 Top Correlations:")
            corr_matrix = df[numeric_cols].corr()
            
            # Get correlation pairs
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    corr_val = corr_matrix.iloc[i, j]
                    if not np.isnan(corr_val) and abs(corr_val) > 0.3:
                        corr_pairs.append((col1, col2, corr_val))
            
            # Sort by absolute correlation
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            for col1, col2, corr in corr_pairs[:10]:
                summary.append(f"  • {col1} ↔ {col2}: {corr:.3f}")
        
        summary.append("\n" + "=" * 60)
        
        return "\n".join(summary)
    
    # ==================== VISUALIZATION HELPERS ====================
    
    def fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode()
    
    # ==================== DISTRIBUTION PLOTS (FIXED SIZE) ====================
    
    def plot_distribution(self, df: pd.DataFrame, column: str, plot_type: str = 'auto'):
        """
        Create distribution plot (histogram + KDE) - FIXED SIZE
        
        Args:
            df: DataFrame
            column: Column name
            plot_type: 'hist' (histogram only), 'both' (histogram + KDE overlay)
        
        Returns:
            Matplotlib figure with histogram and optional KDE overlay
        """
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        data = df[column].dropna()
        
        # Always plot histogram on primary axis
        if plot_type in ['auto', 'both', 'hist']:
            n, bins, patches = ax1.hist(data, bins=30, alpha=0.7, color='skyblue', 
                                        edgecolor='black', label='Histogram')
            ax1.set_ylabel('Frequency', fontsize=12, color='steelblue')
            ax1.tick_params(axis='y', labelcolor='steelblue')
        
        # Plot KDE on SECONDARY Y-AXIS if requested
        if plot_type in ['auto', 'both'] and len(data) > 1:
            # Check if data has variation
            if data.std() > 0 and data.nunique() > 1:
                try:
                    # Create secondary y-axis for KDE
                    ax2 = ax1.twinx()
                    
                    # Plot KDE on secondary axis
                    data.plot(kind='kde', ax=ax2, color='red', linewidth=2.5, label='KDE')
                    ax2.set_ylabel('Density', fontsize=12, color='red')
                    ax2.tick_params(axis='y', labelcolor='red')
                    
                    # Add KDE to legend
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    
                    # Combine legends
                    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                    
                except Exception as e:
                    # If KDE fails, just show histogram
                    print(f"⚠️ KDE plot skipped for {column}: {str(e)}")
            else:
                print(f"ℹ️ KDE skipped for {column}: constant values")
        
        ax1.set_xlabel(column, fontsize=12)
        ax1.set_title(f'Distribution of {column}', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics (mean and median lines)
        mean_val = data.mean()
        median_val = data.median()
        ax1.axvline(mean_val, color='green', linestyle='--', linewidth=2, 
                    label=f'Mean: {mean_val:.2f}')
        ax1.axvline(median_val, color='orange', linestyle='--', linewidth=2, 
                    label=f'Median: {median_val:.2f}')
        
        # Update legend to include mean/median
        if plot_type not in ['auto', 'both']:
            ax1.legend()
        
        plt.tight_layout()
        
        return fig
    
    def plot_boxplot(self, df: pd.DataFrame, column: str, by: str = None):
        """
        Create boxplot with outlier detection - FIXED SIZE
        
        Args:
            df: DataFrame
            column: Numeric column
            by: Optional categorical column for grouping
        """
        fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
        
        if by and by in df.columns:
            df.boxplot(column=column, by=by, ax=ax, patch_artist=True)
            ax.set_title(f'Boxplot of {column} by {by}', fontsize=14, fontweight='bold')
        else:
            bp = ax.boxplot(df[column].dropna(), patch_artist=True, vert=True)
            bp['boxes'][0].set_facecolor('lightblue')
            ax.set_ylabel(column, fontsize=12)
            ax.set_title(f'Boxplot of {column}', fontsize=14, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_violin(self, df: pd.DataFrame, column: str, by: str = None):
        """Create violin plot - FIXED SIZE"""
        fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
        
        if by and by in df.columns:
            sns.violinplot(data=df, x=by, y=column, ax=ax, palette='Set2')
            ax.set_title(f'Violin Plot of {column} by {by}', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
        else:
            sns.violinplot(y=df[column], ax=ax, color='lightblue')
            ax.set_title(f'Violin Plot of {column}', fontsize=14, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        
        return fig
    
    # ==================== CATEGORICAL PLOTS (FIXED SIZE) ====================
    
    def plot_countplot(self, df: pd.DataFrame, column: str, top_n: int = 15):
        """
        Create count plot for categorical variable - FIXED SIZE
        
        Args:
            df: DataFrame
            column: Categorical column name
            top_n: Number of top categories to show
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
        
        # Get top N categories
        value_counts = df[column].value_counts().head(top_n)
        
        # Create count plot
        sns.countplot(data=df[df[column].isin(value_counts.index)], 
                     x=column, ax=ax, palette='viridis', order=value_counts.index)
        
        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Count Plot of {column} (Top {top_n})', fontsize=14, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(value_counts.values):
            ax.text(i, v + max(value_counts.values) * 0.01, str(v), 
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        return fig
    
    def plot_barplot(self, df: pd.DataFrame, x_column: str, y_column: str, top_n: int = 15):
        """
        Create bar plot for showing relationship between categorical and numeric - FIXED SIZE
        
        Args:
            df: DataFrame
            x_column: Categorical column (X-axis)
            y_column: Numeric column (Y-axis)
            top_n: Number of top categories to show
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
        
        # Get top N categories by mean value
        top_categories = df.groupby(x_column)[y_column].mean().nlargest(top_n).index
        plot_data = df[df[x_column].isin(top_categories)]
        
        # Create bar plot
        sns.barplot(data=plot_data, x=x_column, y=y_column, 
                   ax=ax, palette='coolwarm', estimator='mean', ci=None,
                   order=top_categories)
        
        ax.set_xlabel(x_column, fontsize=12)
        ax.set_ylabel(f'Mean {y_column}', fontsize=12)
        ax.set_title(f'Bar Plot: {y_column} by {x_column} (Top {top_n})', fontsize=14, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', fontsize=9)
        
        plt.tight_layout()
        
        return fig
    
    def plot_pie_chart(self, df: pd.DataFrame, column: str, top_n: int = 10):
        """Create pie chart for categorical variable - FIXED SIZE"""
        fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
        
        value_counts = df[column].value_counts().head(top_n)
        
        colors = plt.cm.Set3(range(len(value_counts)))
        
        wedges, texts, autotexts = ax.pie(
            value_counts.values,
            labels=value_counts.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title(f'Distribution of {column} (Top {top_n})', fontsize=14, fontweight='bold')
        
        return fig
    
    # ==================== CORRELATION PLOTS (FIXED SIZE) ====================
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, method: str = 'pearson'):
        """
        Create correlation heatmap - FIXED SIZE
        
        Args:
            df: DataFrame
            method: 'pearson', 'spearman', or 'kendall'
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            st.warning("⚠️ Need at least 2 numeric columns for correlation heatmap")
            return None
        
        corr = numeric_df.corr(method=method)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(
            corr, 
            mask=mask,
            annot=True, 
            fmt='.2f', 
            cmap='coolwarm', 
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title(f'Correlation Heatmap ({method.capitalize()})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_pairplot_sample(self, df: pd.DataFrame, columns: List[str] = None, n_cols: int = 4):
        """Create pairplot for selected columns - FIXED SIZE"""
        if columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            columns = numeric_cols[:min(n_cols, len(numeric_cols))]
        
        if len(columns) < 2:
            st.warning("⚠️ Need at least 2 columns for pairplot")
            return None
        
        # Sample data if too large
        sample_df = df[columns].sample(min(1000, len(df)))
        
        fig = sns.pairplot(sample_df, diag_kind='kde', plot_kws={'alpha': 0.6})
        fig.fig.suptitle('Pairplot of Selected Features', y=1.02, fontsize=14, fontweight='bold')
        
        return fig.fig
    
    # ==================== RELATIONSHIP PLOTS (FIXED SIZE) ====================
    
    def plot_scatter(self, df: pd.DataFrame, x: str, y: str, hue: str = None):
        """Create scatter plot - FIXED SIZE"""
        fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
        
        if hue and hue in df.columns:
            sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax, alpha=0.6, s=50)
        else:
            ax.scatter(df[x], df[y], alpha=0.6, s=50, c='steelblue')
        
        # Add regression line
        if df[x].notna().sum() > 1 and df[y].notna().sum() > 1:
            z = np.polyfit(df[x].dropna(), df[y].dropna(), 1)
            p = np.poly1d(z)
            ax.plot(df[x].sort_values(), p(df[x].sort_values()), 
                   "r--", alpha=0.8, linewidth=2, label='Trend line')
        
        ax.set_xlabel(x, fontsize=12)
        ax.set_ylabel(y, fontsize=12)
        ax.set_title(f'{y} vs {x}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig
    
    def plot_line(self, df: pd.DataFrame, x: str, y: str):
        """Create line plot - FIXED SIZE"""
        fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
        
        ax.plot(df[x], df[y], marker='o', linewidth=2, markersize=4, alpha=0.7)
        
        ax.set_xlabel(x, fontsize=12)
        ax.set_ylabel(y, fontsize=12)
        ax.set_title(f'{y} over {x}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    # ==================== ADVANCED VISUALIZATIONS (FIXED SIZE) ====================
    
    def plot_qq(self, df: pd.DataFrame, column: str):
        """Create Q-Q plot for normality check - FIXED SIZE"""
        fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
        
        stats.probplot(df[column].dropna(), dist="norm", plot=ax)
        
        ax.set_title(f'Q-Q Plot of {column}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_kde_comparison(self, df: pd.DataFrame, columns: List[str]):
        """Compare distributions using KDE - FIXED SIZE"""
        fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
        
        plotted_any = False
        skipped_cols = []
        
        for col in columns:
            if col in df.columns and df[col].dtype in [np.float64, np.int64]:
                # Get clean data (no NaN)
                clean_data = df[col].dropna()
                
                # Check if we have enough data points
                if len(clean_data) < 2:
                    skipped_cols.append((col, "insufficient data (< 2 points)"))
                    continue
                
                # Check if data has any variation
                if clean_data.std() == 0 or clean_data.nunique() == 1:
                    skipped_cols.append((col, "constant values (no variation)"))
                    continue
                
                try:
                    # Try to plot KDE
                    clean_data.plot(kind='kde', ax=ax, label=col, linewidth=2)
                    plotted_any = True
                except Exception as e:
                    # If KDE fails for any reason, skip this column
                    skipped_cols.append((col, f"KDE error"))
                    continue
        
        if not plotted_any:
            # If no columns could be plotted, create a message plot
            ax.text(0.5, 0.5, 
                   "Cannot generate KDE plot\n\nAll selected columns have:\n• Constant values\n• Insufficient data points\n• No variation",
                   ha='center', va='center', fontsize=14, color='red',
                   transform=ax.transAxes)
            ax.set_title('KDE Comparison - Error', fontsize=14, fontweight='bold')
        else:
            ax.set_xlabel('Value', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.set_title('KDE Comparison', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Add note about skipped columns if any
        if skipped_cols:
            note_text = "Skipped columns:\n"
            for col_name, reason in skipped_cols[:3]:  # Show first 3
                note_text += f"• {col_name}: {reason}\n"
            if len(skipped_cols) > 3:
                note_text += f"... and {len(skipped_cols) - 3} more"
            
            ax.text(0.02, 0.98, note_text,
                   transform=ax.transAxes,
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        return fig

    
    def plot_missing_heatmap(self, df: pd.DataFrame):
        """Visualize missing data pattern - FIXED SIZE"""
        if df.isnull().sum().sum() == 0:
            st.info("ℹ️ No missing values to visualize")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create binary matrix (1 = missing, 0 = present)
        missing_matrix = df.isnull().astype(int)
        
        sns.heatmap(
            missing_matrix.T,
            cbar=True,
            cmap='RdYlGn_r',
            ax=ax,
            yticklabels=True
        )
        
        ax.set_title('Missing Data Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Row Index', fontsize=12)
        ax.set_ylabel('Columns', fontsize=12)
        
        return fig
    
    # ==================== 3D VISUALIZATIONS (SIMPLE - NO 4D/5D) ====================
    
    def create_3d_scatter(self, df: pd.DataFrame, x: str, y: str, z: str, title: str = None):
        """
        Create simple 3D scatter plot (X, Y, Z only - NO 4D/5D)
        
        Args:
            df: DataFrame
            x: X-axis column name
            y: Y-axis column name
            z: Z-axis column name
            title: Optional plot title
            
        Returns:
            Plotly Figure object
        """
        # Create simple 3D scatter
        fig = go.Figure(data=[go.Scatter3d(
            x=df[x],
            y=df[y],
            z=df[z],
            mode='markers',
            marker=dict(
                size=5,
                color=df[z],  # Color by Z values
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=z),
                line=dict(width=0.5, color='white')
            ),
            text=df.index,
            hovertemplate=f'<b>{x}</b>: %{{x}}<br>' +
                          f'<b>{y}</b>: %{{y}}<br>' +
                          f'<b>{z}</b>: %{{z}}<br>' +
                          '<extra></extra>'
        )])
        
        # Update layout - FIXED SIZE
        fig.update_layout(
            title=title or f'3D Scatter: {x} vs {y} vs {z}',
            scene=dict(
                xaxis_title=x,
                yaxis_title=y,
                zaxis_title=z,
                bgcolor='rgba(15, 23, 42, 0.9)',
                xaxis=dict(gridcolor='rgba(6, 182, 212, 0.2)'),
                yaxis=dict(gridcolor='rgba(6, 182, 212, 0.2)'),
                zaxis=dict(gridcolor='rgba(6, 182, 212, 0.2)')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            width=900,   # FIXED WIDTH
            height=600   # FIXED HEIGHT
        )
        
        return fig
    
    # ==================== INTERACTIVE PLOTLY PLOTS ====================
    
    def create_interactive_scatter(self, df: pd.DataFrame, x: str, y: str, color: str = None):
        """Create interactive scatter plot using Plotly"""
        fig = px.scatter(
            df, 
            x=x, 
            y=y, 
            color=color if color else None,
            title=f'{y} vs {x}',
            template='plotly_white',
            opacity=0.7
        )
        
        fig.update_layout(
            font=dict(size=12),
            title_font=dict(size=16, family='Arial Black')
        )
        
        return fig
    
    def create_interactive_histogram(self, df: pd.DataFrame, column: str):
        """Create interactive histogram using Plotly"""
        fig = px.histogram(
            df,
            x=column,
            nbins=30,
            title=f'Distribution of {column}',
            template='plotly_white'
        )
        
        fig.update_layout(
            font=dict(size=12),
            title_font=dict(size=16, family='Arial Black')
        )
        
        return fig
    
    # ==================== COMPREHENSIVE ANALYSIS ====================
    
    def perform_comprehensive_eda(self, df: pd.DataFrame = None) -> Dict[str, Any]:
        """Perform comprehensive EDA and return analysis context"""
        if df is None:
            df = st.session_state.get("df")
        
        if df is None:
            st.error("❌ No DataFrame found for EDA.")
            st.info("💡 Please upload or ingest data first")
            return {}
        
        st.info("🔍 Performing comprehensive EDA...")
        
        # Initialize context
        context = {
            'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            'dataset_shape': df.shape,
            'column_names': df.columns.tolist(),
            'data_types': df.dtypes.astype(str).to_dict()
        }
        
        # Identify column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        context['numeric_columns'] = numeric_cols
        context['categorical_columns'] = categorical_cols
        
        # Missing values analysis
        missing_data = df.isnull().sum()
        context['missing_values'] = {col: int(count) for col, count in missing_data.items() if count > 0}
        
        # Analyze numeric columns
        if numeric_cols:
            distributions = {}
            outliers = {}
            
            for col in numeric_cols:
                # Distribution stats
                distributions[col] = {
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'skewness': float(df[col].skew()),
                    'kurtosis': float(df[col].kurtosis())
                }
                
                # Outlier detection
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                    outlier_count = outlier_mask.sum()
                    
                    if outlier_count > 0:
                        outliers[col] = {
                            'count': int(outlier_count),
                            'percentage': float((outlier_count / len(df)) * 100),
                            'lower_bound': float(lower_bound),
                            'upper_bound': float(upper_bound),
                            'Q1': float(Q1),
                            'Q3': float(Q3)
                        }
            
            context['distributions'] = distributions
            context['outliers'] = outliers
            
            # Correlations
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                correlations = {}
                
                for col in numeric_cols:
                    correlations[col] = {
                        other_col: float(corr_matrix.loc[col, other_col])
                        for other_col in numeric_cols
                        if col != other_col and not np.isnan(corr_matrix.loc[col, other_col])
                    }
                
                context['correlations'] = correlations
        
        # Analyze categorical columns
        if categorical_cols:
            categorical_analysis = {}
            
            for col in categorical_cols[:10]:
                categorical_analysis[col] = {
                    'unique_count': int(df[col].nunique()),
                    'top_value': str(df[col].mode()[0]) if not df[col].mode().empty else None,
                    'top_value_count': int((df[col] == df[col].mode()[0]).sum()) if not df[col].mode().empty else 0
                }
            
            context['categorical_analysis'] = categorical_analysis
        
        # Store in session state
        st.session_state['analysis_context'] = context
        self.analysis_context = context
        
        st.success("✅ EDA completed successfully!")
        
        return context


# ==================== LEGACY FUNCTIONS ====================

def eda_summary(df: pd.DataFrame = None) -> str:
    """Legacy function for backward compatibility"""
    if df is None:
        df = st.session_state.get("df")
    
    if df is None:
        st.error("❌ No DataFrame found for EDA.")
        return "Error: No DataFrame found"
    
    try:
        analyzer = EDAAnalyzer()
        return analyzer.generate_text_summary(df)
    except Exception as e:
        st.error(f"❌ EDA error: {str(e)}")
        return f"Error: {str(e)}"


def plot_histogram(df: pd.DataFrame, col: str):
    """Legacy histogram function"""
    analyzer = EDAAnalyzer()
    return analyzer.plot_distribution(df, col, plot_type='hist')


def plot_boxplot(df: pd.DataFrame, col: str):
    """Legacy boxplot function"""
    analyzer = EDAAnalyzer()
    return analyzer.plot_boxplot(df, col)


def plot_correlation_heatmap(df: pd.DataFrame):
    """Legacy correlation heatmap function"""
    analyzer = EDAAnalyzer()
    return analyzer.plot_correlation_heatmap(df)


def fig_to_base64(fig) -> str:
    """Legacy fig to base64 converter"""
    analyzer = EDAAnalyzer()
    return analyzer.fig_to_base64(fig)
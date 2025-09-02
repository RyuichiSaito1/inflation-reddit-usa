#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_inflation_sentiment_data(file_path, sheet_name='total'):

    try:
        # Expand user path
        file_path = os.path.expanduser(file_path)
        
        # Read Excel file
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # Ensure required columns exist
        required_cols = ['year_month', 'moving_average']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Filter data for 2012-03 to 2022-12
        df['year_month'] = pd.to_datetime(df['year_month'], format='%Y-%m')
        start_date = pd.to_datetime('2012-03')
        end_date = pd.to_datetime('2022-12')
        
        df = df[(df['year_month'] >= start_date) & (df['year_month'] <= end_date)]
        
        # Sort by date
        df = df.sort_values('year_month').reset_index(drop=True)
        
        print(f"Loaded {len(df)} sentiment records from {file_path}")
        print(f"Date range: {df['year_month'].min()} to {df['year_month'].max()}")
        
        return df
        
    except Exception as e:
        print(f"Error loading sentiment data: {e}")
        return None

def load_mich_data(file_path):

    try:
        # Expand user path
        file_path = os.path.expanduser(file_path)
        
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Ensure required columns exist
        required_cols = ['observation_date', 'MICH']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Parse observation_date as datetime
        df['year_month'] = pd.to_datetime(df['observation_date'])
        
        # Filter data for 2012-03 to 2022-12
        start_date = pd.to_datetime('2012-03')
        end_date = pd.to_datetime('2022-12')
        
        df = df[(df['year_month'] >= start_date) & (df['year_month'] <= end_date)]
        
        # Sort by date and select relevant columns
        df = df[['year_month', 'MICH']].sort_values('year_month').reset_index(drop=True)
        
        print(f"Loaded {len(df)} MICH records from {file_path}")
        print(f"Date range: {df['year_month'].min()} to {df['year_month'].max()}")
        
        return df
        
    except Exception as e:
        print(f"Error loading MICH data: {e}")
        return None

def merge_datasets(sentiment_df, mich_df):

    try:
        # Merge on year_month
        merged_df = pd.merge(sentiment_df, mich_df, on='year_month', how='inner')
        
        # Rename columns for clarity
        merged_df = merged_df.rename(columns={
            'moving_average': 'sentiment_score',
            'MICH': 'mich_value'
        })
        
        print(f"Merged dataset contains {len(merged_df)} records")
        print(f"Date range: {merged_df['year_month'].min()} to {merged_df['year_month'].max()}")
        
        return merged_df
        
    except Exception as e:
        print(f"Error merging datasets: {e}")
        return None

def calculate_correlations(df):

    try:
        # Remove any rows with missing values
        clean_df = df.dropna(subset=['sentiment_score', 'mich_value'])
        
        x = clean_df['sentiment_score']
        y = clean_df['mich_value']
        
        # Calculate Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(x, y)
        
        # Calculate Spearman correlation
        spearman_rho, spearman_p = stats.spearmanr(x, y)
        
        # Sample size
        n = len(clean_df)

        # Calculate coefficient of determination (R-squared) for Pearson
        r_squared = pearson_r ** 2
        
        correlations = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'pearson_r_squared': r_squared,
            'spearman_rho': spearman_rho,
            'spearman_p': spearman_p,
            'sample_size': n
        }
        
        return correlations
        
    except Exception as e:
        print(f"Error calculating correlations: {e}")
        return None

def create_visualization(df, correlations):
    
    plt.rcParams.update({
        'font.size': 20,          # Base font size
        'axes.titlesize': 20,     # Title font size
        'axes.labelsize': 20,     # X and Y labels
        'xtick.labelsize': 20,    # X tick labels
        'ytick.labelsize': 20,    # Y tick labels
        'legend.fontsize': 20,    # Legend font size
        'figure.titlesize': 20    # Figure title
    })

    try:
        from matplotlib.ticker import FuncFormatter
        import numpy as np

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

        # Define formatting function for two decimal places
        def format_two_decimals(x, pos):
            return f'{x:.2f}'
        formatter = FuncFormatter(format_two_decimals)
        
        # Plot 1: Dual y-axis time series
        ax1_twin = ax1.twinx()
        
        # Plot sentiment scores
        line1 = ax1.plot(df['year_month'], df['sentiment_score'], 
                        color='#2E86AB', linewidth=2.5, marker='s', markersize=3, 
                        label='Reddit Inflation Score', alpha=0.8)
        
        # Plot MICH values
        line2 = ax1_twin.plot(df['year_month'], df['mich_value'], 
                             color='#A23B72', linewidth=2.5, marker='o', markersize=3, 
                             label='Michigan: Inflation Expectation (MICH)', alpha=0.8)
        
        ax1.yaxis.set_major_formatter(formatter)
        
        # Formatting for plot 1
        ax1.set_xlabel('Time Period', fontsize=20, fontweight='bold')
        ax1.set_ylabel('Reddit Inflation Score', fontsize=20, fontweight='bold', color='#2E86AB')
        ax1_twin.set_ylabel('Michigan: Inflation Exp.', fontsize=20, fontweight='bold', color='#A23B72')
        
        # Color y-axis labels
        ax1.tick_params(axis='y', labelcolor='#2E86AB')
        ax1_twin.tick_params(axis='y', labelcolor='#A23B72')
        
        # Rotate x-axis labels
        ax1.tick_params(axis='x', rotation=45)
        
        # Add grid
        ax1.grid(True, alpha=0.3)
        
        # Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
                  bbox_to_anchor=(0.02, 0.98), frameon=True, fancybox=True, shadow=True)
        
        # Plot 2: Scatter plot with regression line
        ax2.scatter(df['sentiment_score'], df['mich_value'], 
                   color='#8C8C8C', alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
        
        # Add regression line with R² calculation
        z = np.polyfit(df['sentiment_score'], df['mich_value'], 1)
        p = np.poly1d(z)
        y_pred = p(df['sentiment_score'])
        
        # Calculate R² manually using numpy
        y_mean = np.mean(df['mich_value'])
        ss_tot = np.sum((df['mich_value'] - y_mean) ** 2)
        ss_res = np.sum((df['mich_value'] - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Plot regression line with label
        ax2.plot(df['sentiment_score'], y_pred, 
                color='#008080', linewidth=2.5, linestyle='--', alpha=0.8, 
                label=f'Linear Fit (R² = {r2:.2f})')
        
        # Formatting for plot 2
        ax2.set_xlabel('Reddit Inflation Score', fontsize=20, fontweight='bold')
        ax2.set_ylabel('Michigan: Inflation Exp.', fontsize=20, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        ax2.xaxis.set_major_formatter(formatter)

        # Add legend for the regression line
        ax2.legend(loc='lower right', fontsize=18, frameon=True, fancybox=True, shadow=True)

        textstr = f'''Pearson $r$ = {correlations['pearson_r']:.2f} ($p$ = {correlations['pearson_p']:.2f})
Spearman $ρ$ = {correlations['spearman_rho']:.2f} ($p$ = {correlations['spearman_p']:.2f})
n = {correlations['sample_size']}'''
        
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=20,
                verticalalignment='top', bbox=props, fontfamily='monospace')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save high-resolution figure
        plt.savefig('peason_spearman_mich.png', dpi=300, bbox_inches='tight')
        
        plt.show()
        
        print("Visualization saved as 'peason_spearman_mich.png''")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")

def print_statistical_summary(correlations, df):

    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\nSample Characteristics:")
    print(f"  • Sample size (n): {correlations['sample_size']}")
    print(f"  • Time period: {df['year_month'].min().strftime('%Y-%m')} to {df['year_month'].max().strftime('%Y-%m')}")
    
    print(f"\nDescriptive Statistics:")
    print(f"  Sentiment Score:")
    print(f"    • Mean: {df['sentiment_score'].mean():.4f}")
    print(f"    • Std Dev: {df['sentiment_score'].std():.4f}")
    print(f"    • Range: [{df['sentiment_score'].min():.4f}, {df['sentiment_score'].max():.4f}]")
    
    print(f"  MICH Inflation Expectation:")
    print(f"    • Mean: {df['mich_value'].mean():.4f}")
    print(f"    • Std Dev: {df['mich_value'].std():.4f}")
    print(f"    • Range: [{df['mich_value'].min():.4f}, {df['mich_value'].max():.4f}]")
    
    print(f"\nCorrelation Analysis:")
    print(f"  Pearson Product-Moment Correlation:")
    print(f"    • r = {correlations['pearson_r']:.4f}")
    print(f"    • p-value = {correlations['pearson_p']:.4f}")
    print(f"    • Significance: {'***' if correlations['pearson_p'] < 0.001 else '**' if correlations['pearson_p'] < 0.01 else '*' if correlations['pearson_p'] < 0.05 else 'ns'}")
    
    print(f"  Spearman Rank-Order Correlation:")
    print(f"    • ρ = {correlations['spearman_rho']:.4f}")
    print(f"    • p-value = {correlations['spearman_p']:.4f}")
    print(f"    • Significance: {'***' if correlations['spearman_p'] < 0.001 else '**' if correlations['spearman_p'] < 0.01 else '*' if correlations['spearman_p'] < 0.05 else 'ns'}")
    
    print(f"\nSignificance levels: *** p < 0.001, ** p < 0.01, * p < 0.05, ns = not significant")
    print("="*60)

def main():

    print("Inflation Sentiment Analysis Pipeline")
    print("====================================")
    
    # File paths
    EXCEL_FILE_PATH = '~/monthly_classification_result.xlsx'
    CSV_FILE_PATH = '~/MICH.csv'
    SHEET_NAME = 'total'
    
    # Load datasets
    print("\n1. Loading datasets...")
    sentiment_df = load_inflation_sentiment_data(EXCEL_FILE_PATH, SHEET_NAME)
    mich_df = load_mich_data(CSV_FILE_PATH)
    
    if sentiment_df is None or mich_df is None:
        print("Error: Could not load required datasets.")
        return
    
    # Merge datasets
    print("\n2. Merging datasets...")
    merged_df = merge_datasets(sentiment_df, mich_df)
    
    if merged_df is None:
        print("Error: Could not merge datasets.")
        return
    
    # Calculate correlations
    print("\n3. Calculating correlations...")
    correlations = calculate_correlations(merged_df)
    
    if correlations is None:
        print("Error: Could not calculate correlations.")
        return
    
    # Create visualization
    print("\n4. Creating visualization...")
    create_visualization(merged_df, correlations)
    
    # Print statistical summary
    print("\n5. Statistical Summary:")
    print_statistical_summary(correlations, merged_df)
    
    # Save results to CSV
    output_file = 'inflation_sentiment_mich_analysis.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
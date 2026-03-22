#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')
from matplotlib.ticker import FuncFormatter, MultipleLocator

OUTPUT_DIR = '/Users/ryuichi/Documents/research/statistical_test'

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

        # Filter data for 2012-03 to 2022-12 (inclusive)
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

def load_cpi_data(file_path):
    try:
        # Expand user path
        file_path = os.path.expanduser(file_path)

        # Read TSV file
        df = pd.read_csv(file_path, sep='\t')

        # Ensure required columns exist
        required_cols = ['Year', 'Period', 'Value']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Clean Period column (remove 'M' prefix if present)
        df['Period_clean'] = df['Period'].astype(str).str.replace('M', '', regex=False)

        # Create year_month column
        df['year_month'] = pd.to_datetime(
            df['Year'].astype(str) + '-' + df['Period_clean'],
            format='%Y-%m'
        )

        # Filter data for 2012-03 to 2022-12 (inclusive)
        start_date = pd.to_datetime('2012-03')
        end_date = pd.to_datetime('2022-12')

        df = df[(df['year_month'] >= start_date) & (df['year_month'] <= end_date)]

        # Sort by date and select relevant columns
        df = df[['year_month', 'Value']].sort_values('year_month').reset_index(drop=True)

        print(f"Loaded {len(df)} CPI records from {file_path}")
        print(f"Date range: {df['year_month'].min()} to {df['year_month'].max()}")

        return df

    except Exception as e:
        print(f"Error loading CPI data: {e}")
        return None

def merge_datasets(sentiment_df, cpi_df):
    try:
        # Merge on year_month
        merged_df = pd.merge(sentiment_df, cpi_df, on='year_month', how='inner')

        # Rename columns for clarity
        merged_df = merged_df.rename(columns={
            'moving_average': 'sentiment_score',
            'Value': 'cpi_value'
        })

        print(f"Merged dataset contains {len(merged_df)} records")
        print(f"Date range: {merged_df['year_month'].min()} to {merged_df['year_month'].max()}")

        return merged_df

    except Exception as e:
        print(f"Error merging datasets: {e}")
        return None

def calculate_correlations(df):
    try:
        # Drop rows with missing values
        clean_df = df.dropna(subset=['sentiment_score', 'cpi_value']).copy()

        x = clean_df['sentiment_score'].astype(float)
        y = clean_df['cpi_value'].astype(float)

        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(x, y)

        # Spearman correlation
        spearman_rho, spearman_p = stats.spearmanr(x, y)

        return {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_rho': spearman_rho,
            'spearman_p': spearman_p,
            'sample_size': len(clean_df)
        }

    except Exception as e:
        print(f"Error calculating correlations: {e}")
        return None

def create_visualization(df, left_ylim=None, left_ytick_step=None, output_filename='volume_cpi.png'):
    """
    Create a single time-series figure with dual y-axes.

    Parameters
    ----------
    left_ylim : tuple(min, max) or None
        Y-axis range for the left axis (Price-related Posts Share).
    left_ytick_step : float or None
        Major tick step for the left axis.
    output_filename : str
        Output file name (saved under OUTPUT_DIR).
    """
    plt.rcParams.update({
        'font.size': 20,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'figure.titlesize': 20
    })

    try:
        # Formatter for two decimal places on the left axis
        def format_two_decimals(x, pos):
            return f'{x:.2f}'
        left_formatter = FuncFormatter(format_two_decimals)

        # Create a single-panel figure
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 5.5))
        ax2 = ax1.twinx()

        # Plot left axis (Price-related Posts Share)
        ax1.plot(
            df['year_month'], df['sentiment_score'],
            color='#CC79A7', linewidth=2.5, marker='s', markersize=3,
            label='Price-related Posts Share', alpha=0.85
        )

        # Plot right axis (CPI)
        ax2.plot(
            df['year_month'], df['cpi_value'],
            color='#E69F00', linewidth=2.5, marker='o', markersize=3,
            label='Consumer Price Index (CPI)', alpha=0.85
        )

        # Apply left-axis scaling options
        if left_ylim is not None:
            ax1.set_ylim(left_ylim[0], left_ylim[1])
        if left_ytick_step is not None:
            ax1.yaxis.set_major_locator(MultipleLocator(left_ytick_step))

        ax1.yaxis.set_major_formatter(left_formatter)

        # Labels and styling
        ax1.set_xlabel('Time Period', fontsize=20, fontweight='bold')
        ax1.set_ylabel('Price-related Posts Share', fontsize=20, fontweight='bold', color='#CC79A7')
        ax2.set_ylabel('Consumer Price Index', fontsize=20, fontweight='bold', color='#E69F00')

        ax1.tick_params(axis='y', labelcolor='#CC79A7')
        ax2.tick_params(axis='y', labelcolor='#E69F00')
        ax1.tick_params(axis='x', rotation=45)

        ax1.grid(True, alpha=0.3)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines1 + lines2, labels1 + labels2,
            loc='upper left', bbox_to_anchor=(0.02, 0.98),
            frameon=True, fancybox=True, shadow=True
        )

        plt.tight_layout()

        # Save figure
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Visualization saved as: {output_path}")

    except Exception as e:
        print(f"Error creating visualization: {e}")

def print_correlation_values(correlations):
    """Print Pearson and Spearman correlations as plain text (console output)."""
    print("\nCorrelation results (Price-related Posts Share vs CPI):")
    print(f"  Pearson r = {correlations['pearson_r']:.4f} (p = {correlations['pearson_p']:.4g})")
    print(f"  Spearman ρ = {correlations['spearman_rho']:.4f} (p = {correlations['spearman_p']:.4g})")
    print(f"  n = {correlations['sample_size']}")

def main():
    print("Inflation Sentiment Analysis Pipeline")
    print("====================================")

    # File paths (edit as needed)
    EXCEL_FILE_PATH = '/Users/ryuichi/Documents/research/statistical_test/monthly_count_results.xlsx'
    TSV_FILE_PATH = '/Users/ryuichi/Documents/research/statistical_test/cpi-u-2012-2022.tsv'
    SHEET_NAME = 'total'

    # Load datasets
    print("\n1. Loading datasets...")
    sentiment_df = load_inflation_sentiment_data(EXCEL_FILE_PATH, SHEET_NAME)
    cpi_df = load_cpi_data(TSV_FILE_PATH)

    if sentiment_df is None or cpi_df is None:
        print("Error: Could not load required datasets.")
        return

    # Merge datasets
    print("\n2. Merging datasets...")
    merged_df = merge_datasets(sentiment_df, cpi_df)

    if merged_df is None:
        print("Error: Could not merge datasets.")
        return

    # Calculate correlations
    print("\n3. Calculating correlations...")
    correlations = calculate_correlations(merged_df)

    if correlations is None:
        print("Error: Could not calculate correlations.")
        return

    # Print correlation values (plain text output)
    print_correlation_values(correlations)

    # Save correlation results to a .txt file (optional)
    txt_path = os.path.join(OUTPUT_DIR, "correlations_volume_cpi.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Correlation results (Price-related Posts Share vs CPI):\n")
        f.write(f"Pearson r = {correlations['pearson_r']:.4f} (p = {correlations['pearson_p']:.4g})\n")
        f.write(f"Spearman ρ = {correlations['spearman_rho']:.4f} (p = {correlations['spearman_p']:.4g})\n")
        f.write(f"n = {correlations['sample_size']}\n")
    print(f"Correlation text saved as: {txt_path}")


    # Create visualization (time series only)
    print("\n4. Creating visualization (time series only)...")
    create_visualization(
        merged_df,
        left_ylim=(0.08, 0.14),
        left_ytick_step=0.01,
        output_filename='volume_cpi.png'
    )

    # Save merged data to CSV (optional, keeps your previous behavior)
    output_file = os.path.join(OUTPUT_DIR, 'volume_vs_cpi.csv')
    merged_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()

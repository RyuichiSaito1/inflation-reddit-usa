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

def load_michigan_expectations_data(file_path):
    
    try:
        # Expand user path
        file_path = os.path.expanduser(file_path)
        
        # Determine file type and read accordingly
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        
        print(f"Original columns: {df.columns.tolist()}")
        print(f"Data shape: {df.shape}")
        print(f"First few rows:")
        print(df.head())
        
        # Check for expected column names
        expected_date_col = 'observation_date'
        expected_mich_col = 'MICH'
        
        if expected_date_col not in df.columns:
            # Try alternative date column names
            date_candidates = [col for col in df.columns if any(term in col.lower() 
                              for term in ['date', 'time', 'period'])]
            if date_candidates:
                expected_date_col = date_candidates[0]
                print(f"Using '{expected_date_col}' as date column")
            else:
                raise ValueError(f"Could not find date column. Available columns: {df.columns.tolist()}")
        
        if expected_mich_col not in df.columns:
            # Try alternative MICH column names
            mich_candidates = [col for col in df.columns if any(term in col.upper() 
                              for term in ['MICH', 'MICHIGAN', 'INFLATION', 'EXPECT'])]
            if mich_candidates:
                expected_mich_col = mich_candidates[0]
                print(f"Using '{expected_mich_col}' as expectations column")
            else:
                raise ValueError(f"Could not find MICH/expectations column. Available columns: {df.columns.tolist()}")
        
        # Create working dataframe with standardized column names
        work_df = df[[expected_date_col, expected_mich_col]].copy()
        work_df.columns = ['date_raw', 'inflation_expectations']
        
        # Convert date column to datetime
        try:
            work_df['year_month'] = pd.to_datetime(work_df['date_raw'])
            print("Successfully parsed dates using pandas datetime parser")
        except Exception as e:
            print(f"Error parsing dates: {e}")
            raise ValueError("Could not parse date column")
        
        # Ensure we have monthly data (convert to month-start)
        work_df['year_month'] = work_df['year_month'].dt.to_period('M').dt.start_time
        
        # Convert MICH values to numeric, handling any non-numeric entries
        work_df['inflation_expectations'] = pd.to_numeric(work_df['inflation_expectations'], errors='coerce')
        
        # Filter data for 2012-03 to 2022-12
        start_date = pd.to_datetime('2012-03')
        end_date = pd.to_datetime('2022-12')
        
        work_df = work_df[(work_df['year_month'] >= start_date) & 
                         (work_df['year_month'] <= end_date)]
        
        # Remove missing values and sort
        initial_count = len(work_df)
        work_df = work_df.dropna().sort_values('year_month').reset_index(drop=True)
        final_count = len(work_df)
        
        if initial_count != final_count:
            print(f"Removed {initial_count - final_count} rows with missing values")
        
        # Select final columns
        result_df = work_df[['year_month', 'inflation_expectations']]
        
        print(f"Loaded {len(result_df)} Michigan expectations records")
        print(f"Date range: {result_df['year_month'].min()} to {result_df['year_month'].max()}")
        print(f"Expectations range: {result_df['inflation_expectations'].min():.2f} to {result_df['inflation_expectations'].max():.2f}")
        
        return result_df
        
    except Exception as e:
        print(f"Error loading Michigan expectations data: {e}")
        print(f"Please ensure your file has columns: 'observation_date' and 'MICH'")
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
        df['year_month'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Period_clean'], format='%Y-%m')
        
        # Filter data for 2012-03 to 2022-12
        start_date = pd.to_datetime('2012-03')
        end_date = pd.to_datetime('2022-12')
        
        df = df[(df['year_month'] >= start_date) & (df['year_month'] <= end_date)]
        
        # Sort by date and select relevant columns
        df = df[['year_month', 'Value']].sort_values('year_month').reset_index(drop=True)
        df = df.rename(columns={'Value': 'cpi_value'})
        
        print(f"Loaded {len(df)} CPI records from {file_path}")
        print(f"Date range: {df['year_month'].min()} to {df['year_month'].max()}")
        print(f"CPI range: {df['cpi_value'].min():.2f} to {df['cpi_value'].max():.2f}")
        
        return df
        
    except Exception as e:
        print(f"Error loading CPI data: {e}")
        return None

def merge_datasets(michigan_df, cpi_df):
    
    try:
        # Merge on year_month
        merged_df = pd.merge(michigan_df, cpi_df, on='year_month', how='inner')
        
        print(f"Merged dataset contains {len(merged_df)} records")
        print(f"Date range: {merged_df['year_month'].min()} to {merged_df['year_month'].max()}")
        
        # Check for missing values
        missing_michigan = merged_df['inflation_expectations'].isna().sum()
        missing_cpi = merged_df['cpi_value'].isna().sum()
        
        if missing_michigan > 0 or missing_cpi > 0:
            print(f"Warning: Missing values - Michigan: {missing_michigan}, CPI: {missing_cpi}")
            merged_df = merged_df.dropna()
            print(f"After removing missing values: {len(merged_df)} records")
        
        return merged_df
        
    except Exception as e:
        print(f"Error merging datasets: {e}")
        return None

def calculate_correlations(df):
    
    try:
        # Remove any rows with missing values
        clean_df = df.dropna(subset=['inflation_expectations', 'cpi_value'])
        
        x = clean_df['inflation_expectations']
        y = clean_df['cpi_value']
        
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

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
        
        # Plot 1: Dual y-axis time series
        ax1_twin = ax1.twinx()
        
        # Plot Michigan expectations
        line1 = ax1.plot(df['year_month'], df['inflation_expectations'], 
                        color='#A23B72', linewidth=2.5, marker='s', markersize=3, 
                        label='Michigan: Inflation Expectation (MICH)', alpha=0.8)
        
        # Plot CPI values
        line2 = ax1_twin.plot(df['year_month'], df['cpi_value'], 
                             color='#E69F00', linewidth=2.5, marker='o', markersize=3, 
                             label='Consumer Price Index (CPI)', alpha=0.8)
        
        # Formatting for plot 1
        ax1.set_xlabel('Time Period', fontsize=20, fontweight='bold')
        ax1.set_ylabel('Michigan: Inflation Exp.', fontsize=20, fontweight='bold', color='#A23B72')
        ax1_twin.set_ylabel('Consumer Price Index', fontsize=20, fontweight='bold', color='#E69F00')
        
        # Color y-axis labels
        ax1.tick_params(axis='y', labelcolor='#A23B72', labelsize=20)
        ax1_twin.tick_params(axis='y', labelcolor='#E69F00', labelsize=20)
        
        # Rotate x-axis labels
        ax1.tick_params(axis='x', rotation=45, labelsize=20)
        
        # Add grid
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
                  bbox_to_anchor=(0.02, 0.98), frameon=True, fancybox=True, shadow=True,
                  fontsize=20)
        
        # Plot 2: Scatter plot with regression line
        ax2.scatter(df['inflation_expectations'], df['cpi_value'], 
                   color='#8C8C8C', alpha=0.7, s=60, edgecolors='black', linewidth=0.8)
        
        # Add regression line
        z = np.polyfit(df['inflation_expectations'], df['cpi_value'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df['inflation_expectations'].min(), df['inflation_expectations'].max(), 100)
        ax2.plot(x_line, p(x_line), 
                color='#008080', linewidth=3, linestyle='--', alpha=0.8,
                label=f'Linear Fit (R² = {correlations["pearson_r_squared"]:.2f})')
        
        # Formatting for plot 2
        ax2.set_xlabel('Michigan: Inflation Exp.', fontsize=20, fontweight='bold')
        ax2.set_ylabel('Consumer Price Index', fontsize=20, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(fontsize=20)
        
        textstr = f"""Pearson $r$ = {correlations['pearson_r']:.2f} ($p$ = {correlations['pearson_p']:.2f})
Spearman $ρ$ = {correlations['spearman_rho']:.2f} ($p$ = {correlations['spearman_p']:.2f})
n = {correlations['sample_size']}"""
        
        props = dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8)
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=20,
                verticalalignment='top', bbox=props, fontfamily='monospace')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save high-resolution figure
        plt.savefig('peason_spearman_cpi_mich.png', dpi=300, bbox_inches='tight')
        
        plt.show()
        
        print("Visualization saved as 'michigan_cpi_correlation_analysis.png'")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")

def print_statistical_summary(correlations, df):
    
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("Michigan Inflation Expectations vs Consumer Price Index")
    print("="*70)
    
    print(f"\nSample Characteristics:")
    print(f"  • Sample size (n): {correlations['sample_size']}")
    print(f"  • Time period: {df['year_month'].min().strftime('%Y-%m')} to {df['year_month'].max().strftime('%Y-%m')}")
    print(f"  • Duration: {len(df)} months ({len(df)/12:.1f} years)")
    
    print(f"\nDescriptive Statistics:")
    print(f"  Michigan Inflation Expectations (%):")
    print(f"    • Mean: {df['inflation_expectations'].mean():.4f}")
    print(f"    • Median: {df['inflation_expectations'].median():.4f}")
    print(f"    • Std Dev: {df['inflation_expectations'].std():.4f}")
    print(f"    • Range: [{df['inflation_expectations'].min():.4f}, {df['inflation_expectations'].max():.4f}]")
    
    print(f"  Consumer Price Index:")
    print(f"    • Mean: {df['cpi_value'].mean():.4f}")
    print(f"    • Median: {df['cpi_value'].median():.4f}")
    print(f"    • Std Dev: {df['cpi_value'].std():.4f}")
    print(f"    • Range: [{df['cpi_value'].min():.4f}, {df['cpi_value'].max():.4f}]")
    
    print(f"\nCorrelation Analysis:")
    print(f"  Pearson Product-Moment Correlation:")
    print(f"    • r = {correlations['pearson_r']:.4f}")
    print(f"    • R² = {correlations['pearson_r_squared']:.4f} ({correlations['pearson_r_squared']*100:.1f}% shared variance)")
    print(f"    • p-value = {correlations['pearson_p']:.6f}")
    print(f"    • Significance: {'***' if correlations['pearson_p'] < 0.001 else '**' if correlations['pearson_p'] < 0.01 else '*' if correlations['pearson_p'] < 0.05 else 'ns'}")
    
    print(f"  Spearman Rank-Order Correlation:")
    print(f"    • ρ = {correlations['spearman_rho']:.4f}")
    print(f"    • p-value = {correlations['spearman_p']:.6f}")
    print(f"    • Significance: {'***' if correlations['spearman_p'] < 0.001 else '**' if correlations['spearman_p'] < 0.01 else '*' if correlations['spearman_p'] < 0.05 else 'ns'}")
    
    # Effect size interpretation
    pearson_abs = abs(correlations['pearson_r'])
    if pearson_abs >= 0.7:
        effect_size = "Large"
    elif pearson_abs >= 0.3:
        effect_size = "Medium"
    elif pearson_abs >= 0.1:
        effect_size = "Small"
    else:
        effect_size = "Negligible"
    
    print(f"\nEffect Size Interpretation (Cohen's guidelines):")
    print(f"  • Pearson correlation magnitude: {effect_size} effect size")
    
    print(f"\nSignificance levels: *** p < 0.001, ** p < 0.01, * p < 0.05, ns = not significant")
    print("="*70)

def main():
    
    print("Michigan Inflation Expectations vs CPI Correlation Analysis")
    print("=" * 60)
    
    # File paths - Update these to match your data files
    MICHIGAN_FILE_PATH = '~/MICH.csv'  # Your MICH data file
    CPI_FILE_PATH = '~/cpi-u-2012-2022.tsv'  # Your existing CPI file
    
    # Load datasets
    print("\n1. Loading datasets...")
    michigan_df = load_michigan_expectations_data(MICHIGAN_FILE_PATH)
    cpi_df = load_cpi_data(CPI_FILE_PATH)
    
    if michigan_df is None or cpi_df is None:
        print("Error: Could not load required datasets.")
        print("Please ensure:")
        print("- Michigan data file exists and has date + expectations columns")
        print("- CPI data file exists with Year, Period, Value columns")
        return
    
    # Merge datasets
    print("\n2. Merging datasets...")
    merged_df = merge_datasets(michigan_df, cpi_df)
    
    if merged_df is None or len(merged_df) == 0:
        print("Error: Could not merge datasets or no overlapping data found.")
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
    output_file = 'michigan_cpi_correlation_analysis.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    print("\nAnalysis complete!")
    
    # Additional insights
    print("\n" + "="*70)
    print("RESEARCH INSIGHTS")
    print("="*70)
    print(f"This analysis examines the relationship between consumer inflation")
    print(f"expectations (Michigan Survey) and actual price levels (CPI).")
    print(f"Strong correlations suggest that consumer expectations are")
    print(f"closely aligned with realized inflation, supporting economic theory")
    print(f"about the role of expectations in price formation.")

if __name__ == "__main__":
    main()
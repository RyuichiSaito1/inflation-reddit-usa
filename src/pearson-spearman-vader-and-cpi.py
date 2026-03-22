#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings("ignore")
from matplotlib.ticker import FuncFormatter, MultipleLocator

OUTPUT_DIR = "/Users/ryuichi/Documents/research/statistical_test"

# Set style for publication-quality plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def load_vader_monthly_data(file_path, sheet_name="total"):
    """
    Load monthly VADER sentiment results from Excel.

    Expected columns:
      - year_month: "YYYY-MM"
      - moving_average: monthly sentiment series
    """
    try:
        file_path = os.path.expanduser(file_path)
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        required_cols = ["year_month", "moving_average"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Filter to 2012-03 .. 2022-12 (inclusive)
        df["year_month"] = pd.to_datetime(df["year_month"], format="%Y-%m")
        start_date = pd.to_datetime("2012-03")
        end_date = pd.to_datetime("2022-12")

        df = df[(df["year_month"] >= start_date) & (df["year_month"] <= end_date)]
        df = df.sort_values("year_month").reset_index(drop=True)

        print(f"Loaded {len(df)} sentiment records from {file_path}")
        print(f"Date range: {df['year_month'].min()} to {df['year_month'].max()}")

        return df

    except Exception as e:
        print(f"Error loading sentiment data: {e}")
        return None


def load_cpi_data(file_path):
    """Load CPI-U monthly series from a TSV file (BLS format)."""
    try:
        file_path = os.path.expanduser(file_path)
        df = pd.read_csv(file_path, sep="\t")

        required_cols = ["Year", "Period", "Value"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Clean "M01" -> "01"
        df["Period_clean"] = df["Period"].astype(str).str.replace("M", "", regex=False)

        df["year_month"] = pd.to_datetime(
            df["Year"].astype(str) + "-" + df["Period_clean"],
            format="%Y-%m",
        )

        # Filter to 2012-03 .. 2022-12 (inclusive)
        start_date = pd.to_datetime("2012-03")
        end_date = pd.to_datetime("2022-12")
        df = df[(df["year_month"] >= start_date) & (df["year_month"] <= end_date)]

        df = df[["year_month", "Value"]].sort_values("year_month").reset_index(drop=True)

        print(f"Loaded {len(df)} CPI records from {file_path}")
        print(f"Date range: {df['year_month'].min()} to {df['year_month'].max()}")

        return df

    except Exception as e:
        print(f"Error loading CPI data: {e}")
        return None


def merge_datasets(vader_df, cpi_df):
    """Merge VADER monthly series and CPI series on year_month."""
    try:
        merged_df = pd.merge(vader_df, cpi_df, on="year_month", how="inner")

        merged_df = merged_df.rename(
            columns={
                "moving_average": "vader_sentiment_score",
                "Value": "cpi_value",
            }
        )

        print(f"Merged dataset contains {len(merged_df)} records")
        print(f"Date range: {merged_df['year_month'].min()} to {merged_df['year_month'].max()}")

        return merged_df

    except Exception as e:
        print(f"Error merging datasets: {e}")
        return None


def calculate_correlations(df):
    """Compute Pearson and Spearman correlations between VADER sentiment score and CPI."""
    try:
        clean_df = df.dropna(subset=["vader_sentiment_score", "cpi_value"]).copy()

        x = clean_df["vader_sentiment_score"].astype(float)
        y = clean_df["cpi_value"].astype(float)

        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_rho, spearman_p = stats.spearmanr(x, y)

        return {
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_rho": spearman_rho,
            "spearman_p": spearman_p,
            "sample_size": len(clean_df),
        }

    except Exception as e:
        print(f"Error calculating correlations: {e}")
        return None


def create_visualization(df, left_ylim=None, left_ytick_step=None, output_filename="vader_cpi.png"):
    """
    Create a single time-series figure with dual y-axes.

    Left axis: Vader Sentiment Score
    Right axis: CPI
    """
    plt.rcParams.update(
        {
            "font.size": 20,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
            "figure.titlesize": 20,
        }
    )

    try:
        def format_two_decimals(x, pos):
            return f"{x:.2f}"

        left_formatter = FuncFormatter(format_two_decimals)

        fig, ax1 = plt.subplots(1, 1, figsize=(12, 5.5))
        ax2 = ax1.twinx()

        # Left axis: Vader Sentiment Score (requested color)
        ax1.plot(
            df["year_month"],
            df["vader_sentiment_score"],
            color="#785EF0",
            linewidth=2.5,
            marker="s",
            markersize=3,
            label="VADER Sentiment Score",
            alpha=0.85,
        )

        # Right axis: CPI
        ax2.plot(
            df["year_month"],
            df["cpi_value"],
            color="#E69F00",
            linewidth=2.5,
            marker="o",
            markersize=3,
            label="Consumer Price Index (CPI)",
            alpha=0.85,
        )

        # Apply left-axis scaling options
        if left_ylim is not None:
            ax1.set_ylim(left_ylim[0], left_ylim[1])
        if left_ytick_step is not None:
            ax1.yaxis.set_major_locator(MultipleLocator(left_ytick_step))

        ax1.yaxis.set_major_formatter(left_formatter)

        ax1.set_xlabel("Time Period", fontsize=20, fontweight="bold")
        ax1.set_ylabel("VADER Sentiment Score", fontsize=20, fontweight="bold", color="#785EF0")
        ax2.set_ylabel("Consumer Price Index", fontsize=20, fontweight="bold", color="#E69F00")

        ax1.tick_params(axis="y", labelcolor="#785EF0")
        ax2.tick_params(axis="y", labelcolor="#E69F00")
        ax1.tick_params(axis="x", rotation=45)

        ax1.grid(True, alpha=0.3)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            frameon=True,
            fancybox=True,
            shadow=True,
        )

        plt.tight_layout()

        output_path = os.path.join(OUTPUT_DIR, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Visualization saved as: {output_path}")

    except Exception as e:
        print(f"Error creating visualization: {e}")


def print_correlation_values(correlations):
    """Print Pearson and Spearman correlations as plain text (console output)."""
    print("\nCorrelation results (Vader Sentiment Score vs CPI):")
    print(f"  Pearson r = {correlations['pearson_r']:.4f} (p = {correlations['pearson_p']:.4g})")
    print(f"  Spearman ρ = {correlations['spearman_rho']:.4f} (p = {correlations['spearman_p']:.4g})")
    print(f"  n = {correlations['sample_size']}")


def main():
    print("Vader Sentiment Score vs CPI Analysis Pipeline")
    print("=============================================")

    EXCEL_FILE_PATH = "/Users/ryuichi/Documents/research/statistical_test/monthly_sentiment_results.xlsx"
    TSV_FILE_PATH = "/Users/ryuichi/Documents/research/statistical_test/cpi-u-2012-2022.tsv"
    SHEET_NAME = "total"

    # Load datasets
    print("\n1. Loading datasets...")
    vader_df = load_vader_monthly_data(EXCEL_FILE_PATH, SHEET_NAME)
    cpi_df = load_cpi_data(TSV_FILE_PATH)

    if vader_df is None or cpi_df is None:
        print("Error: Could not load required datasets.")
        return

    # Merge datasets
    print("\n2. Merging datasets...")
    merged_df = merge_datasets(vader_df, cpi_df)

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

    # Save correlation results to a .txt file
    txt_path = os.path.join(OUTPUT_DIR, "correlations_vader_cpi.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Correlation results (Vader Sentiment Score vs CPI):\n")
        f.write(f"Pearson r = {correlations['pearson_r']:.4f} (p = {correlations['pearson_p']:.4g})\n")
        f.write(f"Spearman ρ = {correlations['spearman_rho']:.4f} (p = {correlations['spearman_p']:.4g})\n")
        f.write(f"n = {correlations['sample_size']}\n")
    print(f"Correlation text saved as: {txt_path}")

    # Create visualization (time series only)
    print("\n4. Creating visualization (time series only)...")
    # create_visualization(
    #     merged_df,
    #     left_ylim=(0.0, -0.45),     # Enforce upper bound at 0.45
    #     left_ytick_step=0.05,      # Reasonable tick spacing for 0.00..0.45
    #     output_filename="vader_cpi.png",
    # )
    create_visualization(
    merged_df,
    left_ylim=None,
    left_ytick_step=None,
    output_filename="vader_cpi.png",
)

    # Save merged data to CSV
    output_file = os.path.join(OUTPUT_DIR, "vader_vs_cpi.csv")
    merged_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

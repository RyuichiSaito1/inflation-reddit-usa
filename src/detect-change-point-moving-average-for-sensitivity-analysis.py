import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Install required packages - run this in terminal first:
# pip install ruptures openpyxl

import ruptures as rpt

# Configuration
EXCEL_FILE_PATH = os.path.expanduser(
    '/Users/ryuichi/Documents/research/detect_change_point/rerun_monthly_classification_result.xlsx'
)
SHEET_NAMES = ['tfood', 'tcars', 'treal_estate', 'ttravel', 'tfrugal']

# Output directory: same as input file directory
OUTPUT_DIR = os.path.dirname(EXCEL_FILE_PATH)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_excel_data(file_path, sheet_names):
    """Load all sheets from the Excel file and perform basic cleaning."""
    all_data = {}

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Excel file not found: {file_path}")

    print(f"Reading data from {file_path}...")

    for sheet_name in sheet_names:
        try:
            # Read the Excel sheet
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            # Convert year_month to datetime and clean data
            df['date'] = pd.to_datetime(df['year_month'])
            df = df.sort_values('date').reset_index(drop=True)

            # Ensure moving_average is numeric
            df['moving_average'] = pd.to_numeric(df['moving_average'], errors='coerce')

            # Drop rows with missing moving_average
            df = df.dropna(subset=['moving_average'])

            all_data[sheet_name] = df

            print(f"✓ {sheet_name}: {len(df)} records from {df['date'].min()} to {df['date'].max()}")

        except Exception as e:
            print(f"✗ Error reading {sheet_name}: {str(e)}")

    print(f"\nSuccessfully loaded {len(all_data)} sheets")
    return all_data


def detect_changepoints_l2(data, min_size=2, jump=1, pen=None):
    """Run PELT with L2 cost on a 1D time series."""
    # Initialize the detection algorithm with L2 model
    algo = rpt.Pelt(model="l2", min_size=min_size, jump=jump)

    # Fit the algorithm to the data
    algo.fit(data)

    # If no penalty is supplied, use a default BIC-type penalty
    if pen is None:
        pen = np.log(len(data)) * np.var(data)

    # Predict change points
    change_points = algo.predict(pen=pen)

    return change_points, pen


def run_sensitivity_analysis(all_data):
    """Run sensitivity analysis over penalty factors and min_size values."""
    penalty_factors = [0.5, 1.0, 2.0]   # c values for c * log(n) * var
    min_sizes = [1, 2, 3]               # minimum segment lengths (in observations)

    rows = []  # Collect results for CSV

    print("\n" + "=" * 60)
    print("PELT SENSITIVITY ANALYSIS")
    print("=" * 60)

    for sheet_name, df in all_data.items():
        ts = df['moving_average'].values
        n_obs = len(ts)

        # Base penalty (BIC-type)
        base_penalty = np.log(n_obs) * np.var(ts)

        print(f"\n=== {sheet_name} ===")
        print(f"n = {n_obs}, base_penalty = {base_penalty:.6f}")

        for ms in min_sizes:
            for c in penalty_factors:
                pen = c * base_penalty

                cps, _ = detect_changepoints_l2(ts, min_size=ms, pen=pen)
                # Remove the last point (n_obs) if returned by ruptures
                cps = [cp for cp in cps if cp < n_obs]

                # Convert indices to dates (YYYY-MM)
                cp_dates = [df.iloc[cp]['date'].strftime('%Y-%m') for cp in cps]

                # Print to stdout
                print(
                    f"{sheet_name} | min_size={ms}, c={c}: "
                    f"{len(cps)} CPs at indices {cps} (dates {cp_dates})"
                )

                # Append to result rows
                rows.append({
                    "sheet": sheet_name,
                    "min_size": ms,
                    "penalty_factor": c,
                    "n_obs": n_obs,
                    "base_penalty": base_penalty,
                    "penalty": pen,
                    "n_changepoints": len(cps),
                    "cp_indices": ";".join(str(x) for x in cps),
                    "cp_dates": ";".join(cp_dates),
                })

    # Convert to DataFrame and save as CSV
    results_df = pd.DataFrame(rows)
    output_path = os.path.join(OUTPUT_DIR, "pelt_sensitivity_results.csv")
    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("Sensitivity analysis results saved as:")
    print(f"- {output_path}")
    print("=" * 60)

    return results_df


def main():
    """Main function to run sensitivity analysis on all sheets."""
    try:
        # Load data from Excel file
        all_data = load_excel_data(EXCEL_FILE_PATH, SHEET_NAMES)

        # Display basic statistics for all sheets
        print("\n" + "=" * 60)
        print("DATA OVERVIEW")
        print("=" * 60)

        for sheet_name, df in all_data.items():
            print(f"\n{sheet_name}:")
            print(f"  Shape: {df.shape}")
            print(f"  Average score range: {df['moving_average'].min():.3f} to {df['moving_average'].max():.3f}")
            print(f"  Mean: {df['moving_average'].mean():.3f}, Std: {df['moving_average'].std():.3f}")

        # Run sensitivity analysis (no visualization)
        results_df = run_sensitivity_analysis(all_data)

        return results_df

    except Exception as e:
        print(f"Error: {str(e)}")
        return None


if __name__ == "__main__":
    # Run the analysis
    results_df = main()

    if results_df is not None:
        print("\nAnalysis completed successfully!")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')
from matplotlib.ticker import FormatStrFormatter

# Install required packages - run this in terminal first:
# pip install ruptures openpyxl

import ruptures as rpt

# Configuration
EXCEL_FILE_PATH = os.path.expanduser('~/monthly_classification_result.xlsx')  # File in home directory
SHEET_NAMES = ['tfood', 'tcars', 'treal_estate', 'ttravel', 'tfrugal']  # Update these with your actual sheet names

def load_excel_data(file_path, sheet_names):

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
            
            # Handle any missing values
            df = df.dropna(subset=['moving_average'])
            
            all_data[sheet_name] = df
            
            print(f"✓ {sheet_name}: {len(df)} records from {df['date'].min()} to {df['date'].max()}")
            
        except Exception as e:
            print(f"✗ Error reading {sheet_name}: {str(e)}")
    
    print(f"\nSuccessfully loaded {len(all_data)} sheets")
    return all_data

def detect_changepoints_l2(data, min_size=2, jump=1, pen=None):
    
    # Initialize the detection algorithm with L2 model
    algo = rpt.Pelt(model="l2", min_size=min_size, jump=jump)
    
    # Fit the algorithm to the data
    algo.fit(data)
    
    # Predict change points
    if pen is None:
        # Use automatic penalty selection
        pen = np.log(len(data)) * np.var(data)
    
    change_points = algo.predict(pen=pen)
    
    return change_points, pen

def analyze_changepoints(all_data):

    results = {}
    
    print("\n" + "="*60)
    print("CHANGE POINT DETECTION - L2 METHOD")
    print("="*60)
    
    for sheet_name, df in all_data.items():
        print(f"\n--- Analyzing {sheet_name} ---")
        
        time_series = df['moving_average'].values
        
        # Detect change points using L2 method
        change_points, penalty = detect_changepoints_l2(time_series)
        
        # Remove the last point (which is always n_observations by default)
        change_points = [cp for cp in change_points if cp < len(time_series)]
        
        results[sheet_name] = {
            'data': df,
            'time_series': time_series,
            'change_points': change_points,
            'penalty': penalty,
            'n_segments': len(change_points) + 1
        }
        
        print(f"Penalty used: {penalty:.3f}")
        print(f"Number of change points: {len(change_points)}")
        print(f"Change points at indices: {change_points}")
        
        if change_points:
            print("Change points at dates:")
            for cp in change_points:
                if cp < len(df):
                    print(f"  Index {cp}: {df.iloc[cp]['date'].strftime('%Y-%m')} (score: {df.iloc[cp]['moving_average']:.3f})")
    
    return results

def create_visualizations(results):

    colors = ['#0072B2', '#0072B2', '#0072B2', '#0072B2', '#0072B2']  # Professional color palette
    
    # Define which change points should have different colors
    special_cp_config = {
        'tfood': {3: '#FF00FF'},        # CP3 -> dark green
        'tcars': {2: '#FF00FF'},           # CP2 -> purple
        'treal_estate': {2: '#FF00FF'},    # CP2 -> orange
        'ttravel': {3: '#FF00FF'},           # CP3 -> navy blue
        'tfrugal': {2: '#FF00FF'}           # CP2 -> brown
    }
    
    # Create a single figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))  # 2 rows, 3 columns
    axes = axes.flatten()  # Flatten to make indexing easier
    
    # Sheet name mapping for cleaner titles
    sheet_titles = {
        'tfood': 'r/food',
        'tcars': 'r/cars', 
        'treal_estate': 'r/RealEstate',
        'ttravel': 'r/travel',
        'tfrugal': 'r/Frugal'
    }
    
    for i, (sheet_name, result) in enumerate(results.items()):
        ax = axes[i]
        df = result['data']
        time_series = result['time_series']
        change_points = result['change_points']
        
        # Plot the time series
        ax.plot(df['date'], time_series, color=colors[i], 
                linewidth=1.5, label='Inflation Score', marker='o', markersize=2, alpha=0.8)
        
        # In the subplot formatting section, add this line after setting up the axis:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # Plot change points
        for j, cp in enumerate(change_points):
            if cp < len(df):
                cp_number = j + 1  # Change point number (1-indexed)
                
                # Determine color for this change point
                if sheet_name in special_cp_config and cp_number in special_cp_config[sheet_name]:
                    line_color = special_cp_config[sheet_name][cp_number]
                    linewidth = 2  # Emphasize special change points
                else:
                    line_color = '#FFA500'  # Default color
                    linewidth = 2
                
                ax.axvline(x=df.iloc[cp]['date'], color=line_color, linestyle='--', 
                        alpha=0.8, linewidth=linewidth)
                
                # Add change point labels (keeping original format)
                y_pos = ax.get_ylim()[1] * 0.85
                ax.text(df.iloc[cp]['date'], y_pos, f'CP{cp_number}',
                    rotation=90, verticalalignment='bottom', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFA500", alpha=0.6))
        
        # Formatting each subplot
        ax.set_title(f'{sheet_titles[sheet_name]}', fontsize=16, fontweight='bold', pad=10)
        ax.set_xlabel('Period', fontsize=12)
        ax.set_ylabel('Inflation Score', fontsize=12)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        
        # Add statistics as subtitle
        mean_val = np.mean(time_series)
        std_val = np.std(time_series)
        n_cp = len(change_points)
        subtitle = f'μ={mean_val:.2f}, σ={std_val:.2f}, CPs={n_cp}'
        ax.text(0.20, 0.9, subtitle, transform=ax.transAxes, ha='center', 
                fontsize=12, style='italic', color='black')
        
        print(f"Chart {i+1}: {sheet_name} - {len(change_points)} change points detected")
    
    # Hide the empty subplot (6th position)
    if len(results) == 5:
        axes[5].set_visible(False)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])  # Leave space for main title
    
    # Save the figure in high resolution for publication
    plt.savefig('changepoint.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')

    plt.show()
    print("\n" + "="*60)
    print("Publication-ready figure saved as:")
    print("- sentiment_changepoint_analysis.png (300 DPI)")
    print("="*60)

def perform_segment_analysis(results):
    """Perform segment analysis for all sheets"""
    print("\n" + "="*60)
    print("SEGMENT ANALYSIS")
    print("="*60)
    
    all_segments = {}
    
    for sheet_name, result in results.items():
        time_series = result['time_series']
        df = result['data']
        change_points = result['change_points']
        
        print(f"\n{sheet_name} Segments:")
        print("-" * 40)
        
        segments = []
        segment_start = 0
        
        for i, cp in enumerate(change_points + [len(time_series)]):
            segment_end = cp
            segment_data = time_series[segment_start:segment_end]
            segment_dates = df.iloc[segment_start:segment_end]['date']
            
            segments.append({
                'sheet': sheet_name,
                'segment': i + 1,
                'start_idx': segment_start,
                'end_idx': segment_end - 1,
                'start_date': segment_dates.iloc[0],
                'end_date': segment_dates.iloc[-1],
                'length': len(segment_data),
                'mean_score': np.mean(segment_data),
                'std_score': np.std(segment_data),
                'trend': 'increasing' if segment_data[-1] > segment_data[0] else 'decreasing'
            })
            
            segment_start = cp
        
        all_segments[sheet_name] = segments
        
        # Display segment statistics
        segments_df = pd.DataFrame(segments)
        print(segments_df[['segment', 'start_date', 'end_date', 'length', 'mean_score', 'std_score', 'trend']].to_string(index=False))
    
    return all_segments

def main():
    """Main function to run the analysis"""
    try:
        # Load data from Excel file
        all_data = load_excel_data(EXCEL_FILE_PATH, SHEET_NAMES)
        
        # Display basic statistics for all sheets
        print("\n" + "="*60)
        print("DATA OVERVIEW")
        print("="*60)
        
        for sheet_name, df in all_data.items():
            print(f"\n{sheet_name}:")
            print(f"  Shape: {df.shape}")
            print(f"  Average score range: {df['moving_average'].min():.3f} to {df['moving_average'].max():.3f}")
            print(f"  Mean: {df['moving_average'].mean():.3f}, Std: {df['moving_average'].std():.3f}")
        
        # Analyze change points
        results = analyze_changepoints(all_data)
        
        # Create visualizations
        create_visualizations(results)
        
        # Perform segment analysis
        all_segments = perform_segment_analysis(results)
        
        return results, all_segments
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None, None

if __name__ == "__main__":
    # Run the analysis
    results, segments = main()
    
    # Optional: Save results to files
    if results is not None:
        print("\nAnalysis completed successfully!")
        print(f"To save results, you can export the data to CSV or pickle files.")

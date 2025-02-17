#Futures_curve.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import numpy as np
import re
import os
from datetime import datetime, timedelta


# ======================= CONFIGURATION ========================
HISTORY_CSV_PATH = r"C:/Users/User/Desktop/UPF/TGF/Data/Stats/stats.csv"
COLOR_CONTANGO = '#2ecc71'  # Green spectrum
COLOR_BACKWARDATION = '#e74c3c'  # Red spectrum
COLOR_NEUTRAL = '#f1c40f'
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
# ===============================================================

plt.style.use(PLOT_STYLE)

MONTH_CODES = {
    'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
    'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
}

def save_index_to_csv(input_path, index_value):
    """Save index value with metadata to history file"""
    new_entry = pd.DataFrame([{
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source_file': os.path.basename(input_path),
        'term_structure_index': round(index_value, 4),
        'calculation_date': datetime.now().strftime('%Y-%m-%d'),
        'market_state': 'Contango' if index_value >= 0 else 'Backwardation'
    }])
    
    if os.path.exists(HISTORY_CSV_PATH):
        new_entry.to_csv(HISTORY_CSV_PATH, mode='a', header=False, index=False)
    else:
        new_entry.to_csv(HISTORY_CSV_PATH, index=False)
    print(f"Index {index_value:.2f} saved to history file")

def parse_contract(contract_str):
    """Robust futures contract parser with error handling"""
    try:
        clean_contract = re.split(r'[ .()]', contract_str)[0]
        if len(clean_contract) < 5 or not clean_contract.startswith('CL'):
            return pd.NaT
            
        month_code = clean_contract[2]
        if month_code not in MONTH_CODES:
            return pd.NaT
            
        year = 2000 + int(clean_contract[3:5]) if int(clean_contract[3:5]) < 50 else 1900 + int(clean_contract[3:5])
        return pd.Timestamp(year=year, month=MONTH_CODES[month_code], day=1)
    except Exception as e:
        print(f"Parse error for {contract_str}: {str(e)}")
        return pd.NaT

def validate_inputs(df):
    """Comprehensive data validation checks"""
    if df.empty:
        raise ValueError("Empty dataframe after filtering")
        
    if not df['Expiration'].is_monotonic_increasing:
        raise ValueError("Contract dates not chronological")
        
    price_changes = df['Last'].pct_change().dropna().abs()
    if (price_changes > 0.25).any():
        raise ValueError(f"Implausible price jump: {price_changes.idxmax()}")
        
    if (df['Last'] < 5).any():
        raise ValueError("Prices below $5 detected")

def calculate_term_structure_index(df):
    """
    Enhanced term structure index calculation calibrated for commodities
    - Uses commodity-appropriate spread normalization (5-10% as significant)
    - Incorporates directional persistence weighting
    - Detailed component analysis
    """
    if len(df) < 2:
        return 0.0  # Need at least two contracts
    
    front_price = df['Last'].iloc[0]
    front_date = df['Expiration'].iloc[0]

    # Find temporal anchors using exact day counts
    year1_idx = next((i for i, row in df.iterrows() 
                     if (row['Expiration'] - front_date).days >= 365), len(df)-1)
    year3_idx = next((i for i, row in df.iterrows() 
                     if (row['Expiration'] - front_date).days >= 3*365), len(df)-1)

    year1_price = df['Last'].iloc[year1_idx]
    year3_price = df['Last'].iloc[year3_idx]

    # Calculate percentage spreads
    year1_spread = (year1_price - front_price) / front_price
    year3_spread = (year3_price - front_price) / front_price

    # Normalize spreads for commodity markets
    year1_severity = year1_spread / 0.05  # 5% = 1.0
    year3_severity = year3_spread / 0.10  # 10% = 1.0

    # Calculate directional persistence
    consecutive_changes = np.diff(df['Last']) < 0
    persistence = np.mean(consecutive_changes)

    # Apply nonlinear normalization
    near_term = np.tanh(year1_severity)
    long_term = np.tanh(year3_severity)
    persistence_impact = (2 * persistence - 1) * min(abs(year3_spread / 0.05), 1)

    # Diagnostic output
    print(f"\nCommodity-calibrated components:")
    print(f"1Y Spread: {year1_spread:.4f} ({year1_price:.2f} vs {front_price:.2f})")
    print(f"1Y Severity: {year1_severity:.4f} -> norm: {near_term:.4f}")
    print(f"3Y Spread: {year3_spread:.4f} ({year3_price:.2f} vs {front_price:.2f})")
    print(f"3Y Severity: {year3_severity:.4f} -> norm: {long_term:.4f}")
    print(f"Persistence: {persistence:.4f} -> impact: {persistence_impact:.4f}")

    # Weighted composite index
    composite = (
        0.35 * near_term +
        0.45 * long_term +
        0.20 * persistence_impact
    )
    
    return np.clip(composite, -1, 1)

def create_visualization(df, index_value):
    """Create professional-grade visualization"""
    plt.rcParams['font.family'] = 'Segoe UI'
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor('#f5f6fa')
    
    # Main price curve
    line_color = COLOR_CONTANGO if index_value >= 0 else COLOR_BACKWARDATION
    ax.plot(df['Expiration'], df['Last'], 
            marker='o', markersize=8, markerfacecolor='white',
            markeredgewidth=2, markeredgecolor=line_color,
            color=line_color, linewidth=2.5, alpha=0.9,
            path_effects=[pe.withStroke(linewidth=4, foreground='#ffffff55')])  # Fixed here
    
    
    # Spread bars
    if 'Annualized_Spread' in df:
        ax2 = ax.twinx()
        bar_color = np.where(df['Annualized_Spread'] >= 0, COLOR_CONTANGO, COLOR_BACKWARDATION)
        bars = ax2.bar(df['Expiration'], df['Annualized_Spread'], 
                      width=20, color=bar_color, alpha=0.15)
        ax2.axhline(0, color='#7f8c8d', linewidth=1, linestyle='--')
        ax2.set_ylabel('Annualized Spread (%)', fontsize=10, labelpad=15)
        ax2.spines['right'].set_position(('outward', 60))
    
    # Index gauge
    gauge_color = COLOR_CONTANGO if index_value >= 0 else COLOR_BACKWARDATION
    gauge_text = f"Term Structure Index\n{index_value:.2f}"
    ax.annotate(gauge_text, xy=(0.03, 0.88), xycoords='axes fraction',
               fontsize=14, color=gauge_color, ha='left', va='center',
               bbox=dict(boxstyle='round', facecolor='#ffffff', 
                         edgecolor=gauge_color, pad=0.5))
    
    # Date formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # Price axis styling
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')
    ax.tick_params(axis='both', colors='#7f8c8d', labelsize=10)
    
    # Dynamic Y-axis scaling
    y_min, y_max = df['Last'].min(), df['Last'].max()
    ax.set_ylim(y_min - (y_max - y_min)*0.05, y_max + (y_max - y_min)*0.1)
    ax.set_ylabel('Futures Price (USD)', fontsize=12, color=line_color, labelpad=15)
    
    # Annotations
    ax.set_title(f"WTI Crude Oil Term Structure - {datetime.now().strftime('%d %b %Y')}",
                fontsize=16, pad=20, color='#2c3e50', fontweight='semibold')
    
    plt.tight_layout()
    plt.show()

def main(csv_path):
    """Main processing pipeline"""
    try:
        # Load and pre-process
        df = pd.read_csv(csv_path)
        df = df[df['Contract'].str.contains(r'^CL[A-Z]\d{2}', regex=True, na=False)]
        df['Expiration'] = df['Contract'].apply(parse_contract)
        df = df.dropna(subset=['Expiration']).sort_values('Expiration')
        
        # Validate and calculate
        validate_inputs(df)
        df['Days_to_Next'] = (df['Expiration'].shift(-1) - df['Expiration']).dt.days
        df['Annualized_Spread'] = (df['Last'].shift(-1) - df['Last']) / df['Last'] * (365/df['Days_to_Next']) * 100
        df = df.iloc[:-1]  # Remove last contract
        
        ts_index = calculate_term_structure_index(df)
        print(f"\nCalculated Index: {ts_index:.2f}")
        
        # Save and visualize
        save_index_to_csv(csv_path, ts_index)
        create_visualization(df, ts_index)
        
    except Exception as e:
        print(f"Processing failed: {str(e)}")

if __name__ == "__main__":

    test_csv = r"C:/Users/User/Desktop/UPF/TGF/Data/Futures/CL_DataSample.csv"
    main(test_csv)
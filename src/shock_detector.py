"""
Online Shock & Anomaly Detection Engine (Stage 7).

This module implements an online, leakage-free anomaly detector for intermittent
demand streams. It computes rolling 14-day z-scores on lagged demand (never observing
contemporaneous demand) and expands upper quantile bounds (p90) when anomalous
surges occur, maintaining the expansion across a 7-day momentum window.

Inputs:
    - data/spare_parts_demand_with_quantiles.csv: Panel dataset with LightGBM quantiles.

Outputs:
    - data/predictions_shock_adjusted.csv: Dataset augmented with 'is_shock_flagged'
      and 'p90_adjusted'.

Pipeline Context:
    Serves as an online mitigation mechanism evaluated alongside the direct LightGBM
    and Hurdle models in evaluation_report.md.
"""

import pandas as pd
import numpy as np
import os

def detect_and_adjust_shocks(df):
    """
    Detect demand anomalies using lagged rolling statistics and widen p90 bounds.

    Computes a 14-day rolling mean and standard deviation on lagged demand (t-1)
    per store-SKU group to ensure no future information leaks into the detector.
    Flags an anomaly when:
        Z_score > 3.0 OR lagged_demand > 5 * (rolling_mean + epsilon)
    When triggered, persists the shock state across a trailing 7-day momentum window
    and scales the 90th percentile prediction by 3.0 (p90_adjusted = 3.0 * p90).

    Parameters:
        df (pd.DataFrame): Panel DataFrame containing 'store_id', 'sku_id', 'date',
            'demand', and 'p90'.

    Returns:
        pd.DataFrame: Augmented DataFrame containing:
            - 'rolling_mean_14' (float): 14-day lagged moving average.
            - 'rolling_std_14' (float): 14-day lagged moving standard deviation.
            - 'z_score_current' (float): Standardized anomaly score of previous-day demand.
            - 'is_anomaly_today' (bool): Binary indicator of single-day trigger.
            - 'is_shock_flagged' (bool): 7-day rolling momentum indicator.
            - 'p90_adjusted' (float): Adjusted upper quantile forecast.
    """
    df = df.copy()
    
    # Sort by store, sku, date just to be sure
    df = df.sort_values(['store_id', 'sku_id', 'date'])
    
    # Compute rolling metrics per store-SKU using lagged values to avoid data leakage
    # We want to use a 14-day rolling window on past data.
    # Group by store_id and sku_id
    grouped = df.groupby(['store_id', 'sku_id'])
    
    # We want rolling mean and std over the past 14 days (exclusive of current day for forecasting)
    df['rolling_mean_14'] = grouped['demand'].transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())
    df['rolling_std_14'] = grouped['demand'].transform(lambda x: x.shift(1).rolling(14, min_periods=1).std())
    
    # Handle NaNs
    df['rolling_mean_14'] = df['rolling_mean_14'].fillna(0)
    df['rolling_std_14'] = df['rolling_std_14'].fillna(0)
    
    epsilon = 0.1
    # Use LAGGED demand (previous day) for z-score to ensure we only use past information
    df['lagged_demand'] = grouped['demand'].shift(1).fillna(0)
    df['z_score_current'] = (df['lagged_demand'] - df['rolling_mean_14']) / (df['rolling_std_14'] + epsilon)
    
    # Flag anomaly based on lagged demand (previous day was anomalous)
    df['is_anomaly_today'] = (df['z_score_current'] > 3.0) | (df['lagged_demand'] > 5 * (df['rolling_mean_14'] + epsilon))
    
    # "widen for the next 7 days after a detected anomaly (momentum effect)"
    df['is_shock_flagged'] = grouped['is_anomaly_today'].transform(lambda x: x.rolling(7, min_periods=1).max())
    
    # Fill NaN with 0
    df['is_shock_flagged'] = df['is_shock_flagged'].fillna(0).astype(bool)
    
    # "When flagged, apply multiplicative adjustment to p90: p90_adjusted = p90 * 3.0"
    if 'p90' in df.columns:
        df['p90_adjusted'] = df['p90'].copy()
        df.loc[df['is_shock_flagged'], 'p90_adjusted'] = df.loc[df['is_shock_flagged'], 'p90'] * 3.0
    else:
        print("Warning: 'p90' column not found in data. No adjustment applied.")
        df['p90_adjusted'] = np.nan
        
    return df

def main():
    """
    Execute online shock detection across the quantile forecast dataset.

    Loads 'data/spare_parts_demand_with_quantiles.csv', applies rolling z-score
    anomaly detection, widens p90 on flagged rows, exports the resulting panel
    to 'data/predictions_shock_adjusted.csv', and prints summary diagnostics.
    """
    input_path = 'data/spare_parts_demand_with_quantiles.csv'
    output_path = 'data/predictions_shock_adjusted.csv'
    
    print("Loading data...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Could not find {input_path}")
        return
        
    # parse dates
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        
    print("Applying shock detection...")
    df_adjusted = detect_and_adjust_shocks(df)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_adjusted.to_csv(output_path, index=False)
    
    num_flagged = df_adjusted['is_shock_flagged'].sum()
    if 'p90' in df_adjusted.columns:
        mean_adj = (df_adjusted['p90_adjusted'] - df_adjusted['p90']).mean()
    else:
        mean_adj = 0.0
        
    print("\n--- Shock Detection Summary ---")
    print(f"Total rows: {len(df_adjusted)}")
    print(f"Rows flagged for shock adjustment: {num_flagged} ({(num_flagged/len(df_adjusted))*100:.2f}%)")
    print(f"Mean p90 adjustment across all rows: +{mean_adj:.4f}")
    if num_flagged > 0 and 'p90' in df_adjusted.columns:
        flagged_adj = (df_adjusted.loc[df_adjusted['is_shock_flagged'], 'p90_adjusted'] - df_adjusted.loc[df_adjusted['is_shock_flagged'], 'p90']).mean()
        print(f"Mean p90 adjustment for flagged rows: +{flagged_adj:.4f}")
    print(f"Adjusted predictions saved to {output_path}")

if __name__ == "__main__":
    main()

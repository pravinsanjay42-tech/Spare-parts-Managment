"""
Croston's Method Intermittent Demand Forecaster (Stage 2).

This module implements the industry-standard benchmark for intermittent demand:
Croston's Method. It separates non-zero demand size and inter-demand arrival
intervals, updating both via single exponential smoothing to produce a point forecast
ratio (q_t / a_t).

Inputs:
    - data/spare_parts_demand.csv: Raw synthetic intermittent demand master panel.

Outputs:
    - data/spare_parts_demand_with_baseline.csv: Master dataset augmented with 'baseline_forecast'.
    - data/baseline_comparison.png: Diagnostic comparison plot illustrating point forecast flatlining.

Pipeline Context:
    Executes in Stage 2 after data_gen.py to provide the baseline point estimate
    against which all subsequent probabilistic and quantile models are compared.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def croston_forecast(ts, alpha=0.1):
    """
    Compute daily intermittent demand point forecasts using Croston's method.

    Maintains two exponentially smoothed state variables:
    1. Smoothed non-zero demand size (q)
    2. Smoothed inter-arrival duration between demands (a)
    The point forecast for any day is the smoothed ratio q / a.

    Parameters:
        ts (np.ndarray or list or pd.Series): 1D array of daily demand quantities.
        alpha (float, default=0.1): Exponential smoothing factor for demand size and intervals.

    Returns:
        np.ndarray: Array of daily point forecasts of identical length to the input series.
    """
    ts = np.array(ts)
    res = np.zeros(len(ts))
    
    q = 1.0 # Smoothed demand size
    a = 1.0 # Smoothed inter-arrival time
    time_since_last = 1
    
    for i in range(len(ts)):
        # Forecast for period i
        res[i] = q / a
        
        # Update based on actual observation at period i
        if ts[i] > 0:
            q = alpha * ts[i] + (1 - alpha) * q
            a = alpha * time_since_last + (1 - alpha) * a
            time_since_last = 1
        else:
            time_since_last += 1
            
    return res

def main():
    """
    Run Croston's baseline model across all store-SKU combinations.

    Loads the raw demand panel, computes Croston's forecasts per time series,
    exports 'data/spare_parts_demand_with_baseline.csv', and plots diagnostic
    comparisons showing point-forecast limitations during normal, shock, and
    cold-start periods ('data/baseline_comparison.png').
    """
    print("Loading data...")
    df = pd.read_csv("data/spare_parts_demand.csv")
    df["date"] = pd.to_datetime(df["date"])
    
    print("Running Croston's baseline model...")
    # Apply Croston's to each store-SKU
    df["baseline_forecast"] = df.groupby(["store_id", "sku_id"])["demand"].transform(lambda x: croston_forecast(x, alpha=0.1))
    
    # Save predictions
    df.to_csv("data/spare_parts_demand_with_baseline.csv", index=False)
    print("Baseline forecasts saved.")
    
    # Analyze a normal store, cold-start, and shock event
    
    # 1. Normal period
    normal_store = "Store_01"
    normal_sku = "SKU_02"
    mask_normal = (df["store_id"] == normal_store) & (df["sku_id"] == normal_sku) & (df["date"] >= "2021-01-01") & (df["date"] <= "2021-03-31")
    df_normal = df[mask_normal]
    
    # 2. Shock event period
    shock_store = "Store_08"
    shock_sku = "SKU_05"
    # Shock date is 2021-08-10, so window around it
    mask_shock = (df["store_id"] == shock_store) & (df["sku_id"] == shock_sku) & (df["date"] >= "2021-07-20") & (df["date"] <= "2021-08-31")
    df_shock = df[mask_shock]
    
    # 3. Cold start store
    cold_store = "Store_14"
    cold_sku = "SKU_01"
    mask_cold = (df["store_id"] == cold_store) & (df["sku_id"] == cold_sku)
    df_cold = df[mask_cold]
    
    # Print sample comparison
    print("\n=== SAMPLE: Normal Store (Store_01, SKU_02) First 10 Days ===")
    print(df_normal[["date", "demand", "baseline_forecast"]].head(10))
    
    print("\n=== SAMPLE: Shock Event (Store_08, SKU_05) Around Spike ===")
    spike_window = df_shock[(df_shock["date"] >= "2021-08-08") & (df_shock["date"] <= "2021-08-12")]
    print(spike_window[["date", "is_shock_event", "demand", "baseline_forecast"]])
    
    print("\n=== SAMPLE: Cold Start (Store_14, SKU_01) First 10 Days ===")
    print(df_cold[["date", "demand", "baseline_forecast"]].head(10))

    # Plot to show where it breaks
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharey=False)
    
    # Normal
    axes[0].plot(df_normal["date"], df_normal["demand"], label="Actual Demand", marker="o", linestyle="")
    axes[0].plot(df_normal["date"], df_normal["baseline_forecast"], label="Croston's Forecast", color="red")
    axes[0].set_title(f"Normal Period: {normal_store} {normal_sku}")
    axes[0].legend()
    
    # Shock
    axes[1].plot(df_shock["date"], df_shock["demand"], label="Actual Demand", marker="o", linestyle="")
    axes[1].plot(df_shock["date"], df_shock["baseline_forecast"], label="Croston's Forecast", color="red")
    axes[1].axvline(pd.Timestamp("2021-08-10"), color="orange", linestyle="--", label="Shock Event Day")
    axes[1].set_title(f"Shock Event: {shock_store} {shock_sku}")
    axes[1].legend()
    
    # Cold start
    axes[2].plot(df_cold["date"], df_cold["demand"], label="Actual Demand", marker="o", linestyle="")
    axes[2].plot(df_cold["date"], df_cold["baseline_forecast"], label="Croston's Forecast", color="red")
    axes[2].set_title(f"Cold Start: {cold_store} {cold_sku}")
    axes[2].legend()
    
    plt.tight_layout()
    plot_path = "data/baseline_comparison.png"
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")

if __name__ == "__main__":
    main()

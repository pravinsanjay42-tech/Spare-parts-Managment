import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

def create_features(df):
    df = df.copy()
    df = df.sort_values(["store_id", "sku_id", "date"])
    
    # Lag features
    df['lag_1'] = df.groupby(["store_id", "sku_id"])['demand'].shift(1)
    df['lag_7'] = df.groupby(["store_id", "sku_id"])['demand'].shift(7)
    df['rolling_mean_7'] = df.groupby(["store_id", "sku_id"])['demand'].transform(lambda x: x.shift(1).rolling(7).mean())
    df['rolling_std_7'] = df.groupby(["store_id", "sku_id"])['demand'].transform(lambda x: x.shift(1).rolling(7).std())
    
    # Encode categorical
    df['store_region'] = df['store_region'].astype('category')
    
    return df

def fix_quantile_crossing(df):
    """
    LightGBM trains quantiles independently, which can lead to crossing (e.g., p10 > p50).
    We fix this by sorting the predictions.
    """
    crossing_mask = (df['p10'] > df['p50']) | (df['p50'] > df['p90'])
    num_crossed = crossing_mask.sum()
    
    if num_crossed > 0:
        # Fix by taking the sorted values across the three columns
        fixed = np.sort(df[['p10', 'p50', 'p90']].values, axis=1)
        df['p10'] = fixed[:, 0]
        df['p50'] = fixed[:, 1]
        df['p90'] = fixed[:, 2]
        
    return df, num_crossed

def train_quantile_model(df, target='demand'):
    # Features to use
    features = [
        'store_size', 'equipment_age_years', 'weather_severity_index', 
        'price', 'is_festival', 'store_region',
        'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7'
    ]
    
    # Walk-forward split: train ONLY on data before the split date.
    # This matches evaluate.py's split and prevents data leakage.
    SPLIT_DATE = '2022-07-01'
    train_mask = df['date'] < SPLIT_DATE
    X_train = df.loc[train_mask, features]
    y_train = df.loc[train_mask, target]
    
    print(f"Walk-forward split: Training on {len(X_train)} rows (date < {SPLIT_DATE})")
    print(f"  Test set (for prediction): {len(df) - len(X_train)} rows (date >= {SPLIT_DATE})")
    
    models = {}
    quantiles = [0.1, 0.5, 0.9]
    
    for q in quantiles:
        # lightgbm handles NaNs automatically
        model = lgb.LGBMRegressor(
            objective='quantile', 
            alpha=q, 
            n_estimators=100, 
            learning_rate=0.1, 
            random_state=42
        )
        model.fit(X_train, y_train)
        models[f'p{int(q*100)}'] = model
        
    return models, features

def scenario_generator(base_row, models, features):
    """
    Given a base forecast row, produce 3 alternative forecasts by toggling features.
    """
    scenarios = []
    
    # 1. Normal
    normal_row = base_row.copy()
    normal_row['is_festival'] = 0
    normal_row['weather_severity_index'] = 0.3 # Average
    scenarios.append(('Normal', normal_row))
    
    # 2. Heatwave
    heatwave_row = base_row.copy()
    heatwave_row['weather_severity_index'] = 1.0
    scenarios.append(('Heatwave', heatwave_row))
    
    # 3. Festival
    festival_row = base_row.copy()
    festival_row['is_festival'] = 1
    scenarios.append(('Festival', festival_row))
    
    results = []
    for name, row in scenarios:
        row_df = pd.DataFrame([row])[features]
        # Ensure categorical type is preserved for lightgbm
        if 'store_region' in features:
            row_df['store_region'] = row_df['store_region'].astype('category')
            
        p10 = models['p10'].predict(row_df)[0]
        p50 = models['p50'].predict(row_df)[0]
        p90 = models['p90'].predict(row_df)[0]
        results.append({
            'Scenario': name,
            'p10': max(0, p10),
            'p50': max(0, p50),
            'p90': max(0, p90)
        })
        
    return pd.DataFrame(results)

def main():
    print("Loading data...")
    df = pd.read_csv("data/spare_parts_demand.csv")
    df["date"] = pd.to_datetime(df["date"])
    
    # Create features
    df = create_features(df)
    
    print("Training LightGBM Quantile Models...")
    models, features = train_quantile_model(df)
    
    # Generate predictions
    print("Generating predictions...")
    X_all = df[features]
    df['p10'] = np.maximum(0, models['p10'].predict(X_all)) # Demand can't be negative
    df['p50'] = np.maximum(0, models['p50'].predict(X_all))
    df['p90'] = np.maximum(0, models['p90'].predict(X_all))
    
    # Fix quantile crossing
    df, num_crossed = fix_quantile_crossing(df)
    print(f"Quantile Crossing Check: {num_crossed} rows needed correction (fixed via sorting).")
    
    # Save predictions
    df.to_csv("data/spare_parts_demand_with_quantiles.csv", index=False)
    
    # Feature Importance for Interval Width
    # We can infer drivers of the upper bound (p90) which dictates the interval width
    importance = models['p90'].feature_importances_
    feat_imp = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values(by='Importance', ascending=False)
    print("\n=== Top Feature Drivers for p90 (Upper Bound Uncertainty) ===")
    print(feat_imp.head(5).to_string(index=False))
    
    # Scenario Generation on a specific row
    sample_row = df.iloc[5000].copy()
    print("\n=== Scenario Generator Output ===")
    scenarios_df = scenario_generator(sample_row, models, features)
    print(scenarios_df)
    
    # Plotting
    print("\nPlotting comparisons...")
    # Load baseline forecasts to overlay
    df_base = pd.read_csv("data/spare_parts_demand_with_baseline.csv")
    df_base["date"] = pd.to_datetime(df_base["date"])
    df = df.merge(df_base[['date', 'store_id', 'sku_id', 'baseline_forecast']], on=['date', 'store_id', 'sku_id'], how='left')
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharey=False)
    
    # 1. Normal period
    normal_store, normal_sku = "Store_01", "SKU_02"
    df_normal = df[(df["store_id"] == normal_store) & (df["sku_id"] == normal_sku) & (df["date"] >= "2021-01-01") & (df["date"] <= "2021-03-31")]
    
    # 2. Shock event period
    shock_store, shock_sku = "Store_08", "SKU_05"
    df_shock = df[(df["store_id"] == shock_store) & (df["sku_id"] == shock_sku) & (df["date"] >= "2021-07-20") & (df["date"] <= "2021-08-31")]
    
    # 3. Cold start store
    cold_store, cold_sku = "Store_14", "SKU_01"
    df_cold = df[(df["store_id"] == cold_store) & (df["sku_id"] == cold_sku)]
    
    for i, (ax, data, title) in enumerate(zip(axes, [df_normal, df_shock, df_cold], [f"Normal Period: {normal_store} {normal_sku}", f"Shock Event: {shock_store} {shock_sku}", f"Cold Start: {cold_store} {cold_sku}"])):
        ax.plot(data["date"], data["demand"], label="Actual Demand", marker="o", linestyle="", color="black", zorder=5)
        ax.plot(data["date"], data["baseline_forecast"], label="Baseline (Flatline)", color="red", linestyle="--")
        
        # Plot quantiles
        ax.plot(data["date"], data["p50"], label="LGBM p50", color="blue")
        ax.fill_between(data["date"], data["p10"], data["p90"], color="blue", alpha=0.2, label="80% Uncertainty Band (p10-p90)")
        
        if "Shock" in title:
            ax.axvline(pd.Timestamp("2021-08-10"), color="orange", linestyle="--", label="Shock Event Day")
            
        ax.set_title(title)
        ax.legend()
        
    plt.tight_layout()
    plot_path = "data/quantile_comparison.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()

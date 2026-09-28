import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_data(
    num_years=3, 
    num_stores=15, 
    num_skus=10, 
    seed=42
):
    np.random.seed(seed)
    
    # Base Setup
    dates = pd.date_range(start="2020-01-01", periods=num_years*365, freq="D")
    stores = [f"Store_{i:02d}" for i in range(1, num_stores + 1)]
    skus = [f"SKU_{i:02d}" for i in range(1, num_skus + 1)]
    
    # Inject edge cases definitions
    cold_start_stores = ["Store_14", "Store_15"]
    shock_dates = [pd.Timestamp("2021-02-15"), pd.Timestamp("2021-08-10"), pd.Timestamp("2022-11-05")]
    
    # Store Attributes
    store_df = pd.DataFrame({
        "store_id": stores,
        "store_size": np.random.uniform(0.5, 1.5, size=num_stores),
        "store_region": np.random.choice(["North", "South", "East", "West"], size=num_stores),
        "is_cold_start_store": [1 if s in cold_start_stores else 0 for s in stores]
    })
    
    # SKU Attributes
    base_prices = np.random.uniform(50, 500, size=num_skus)
    sku_df = pd.DataFrame({
        "sku_id": skus,
        "base_price": base_prices,
        "base_failure_rate": np.random.uniform(0.01, 0.15, size=num_skus), # Probability of demand on a given day
        "lambda_multiplier": np.random.uniform(1.0, 3.0, size=num_skus), # Expected quantity when failure occurs
        "base_lead_time": np.where(base_prices < 250, np.random.randint(3, 8, size=num_skus), np.random.randint(10, 22, size=num_skus))
    })
    
    # Cross Join to create the full panel
    panel = pd.MultiIndex.from_product([dates, stores, skus], names=["date", "store_id", "sku_id"]).to_frame(index=False)
    df = panel.merge(store_df, on="store_id").merge(sku_df, on="sku_id")
    df = df.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True)
    
    df["lead_time_days"] = np.clip(df["base_lead_time"] + np.random.randint(-3, 4, size=len(df)), 1, 30)
    
    # Dynamic Features
    # Equipment age increases over time
    days_since_start = (df["date"] - df["date"].min()).dt.days
    df["equipment_age_years"] = np.random.uniform(1.0, 5.0, size=len(df)) + (days_since_start / 365.0)
    
    # Weather severity and temp
    df["weather_severity_index"] = np.random.beta(a=2, b=5, size=len(df))  # Skewed towards lower severity
    # Add seasonal temperature pattern
    day_of_year = df["date"].dt.dayofyear
    df["temperature"] = 20 + 15 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 3, size=len(df))
    
    # Price variations
    df["price"] = df["base_price"] * np.random.normal(1.0, 0.05, size=len(df))
    
    # Festivals (~5% of days)
    festival_dates = np.random.choice(dates, size=int(len(dates)*0.05), replace=False)
    df["is_festival"] = df["date"].isin(festival_dates).astype(int)
    
    # Shock events
    df["is_shock_event"] = df["date"].isin(shock_dates).astype(int)
    
    # -- DEMAND GENERATION --
    # Calculate occurrence probability
    # Base probability + weather effect + age effect
    weather_effect = np.where(df["weather_severity_index"] > 0.8, 0.15, 0)
    age_effect = df["equipment_age_years"] * 0.015
    price_elasticity = -0.1 * ((df["price"] - df["base_price"]) / df["base_price"])
    shock_effect = np.where(df["is_shock_event"] == 1, 0.8, 0)
    
    p_occurrence = (df["base_failure_rate"] * df["store_size"] + weather_effect + age_effect + price_elasticity + shock_effect)
    p_occurrence = np.clip(p_occurrence, 0.001, 0.99)
    
    # Occurs? (Bernoulli)
    occurrence = np.random.binomial(1, p_occurrence)
    
    # Quantity? (Poisson)
    # Average quantity depends on SKU base, store size, and whether it's a festival
    festival_multiplier = np.where(df["is_festival"] == 1, np.random.choice([0.5, 1.5]), 1.0)
    shock_multiplier = np.where(df["is_shock_event"] == 1, 8.0, 1.0)
    weather_multiplier = np.where(df["weather_severity_index"] > 0.8, 2.0, 1.0)
    
    lambda_qty = df["lambda_multiplier"] * df["store_size"] * festival_multiplier * shock_multiplier * weather_multiplier
    quantity = np.random.poisson(lambda_qty)
    quantity = np.clip(quantity, 1, None) # At least 1 if failure occurs
    
    df["true_demand"] = occurrence * quantity
    
    # Stockout censoring logic
    # If demand yesterday > 5, assume stockout today and true demand is censored to 0
    df["shifted_demand"] = df.groupby(["store_id", "sku_id"])["true_demand"].shift(1).fillna(0)
    df["stockout_flag"] = (df["shifted_demand"] > 5).astype(int)
    
    df["demand"] = np.where(df["stockout_flag"] == 1, 0, df["true_demand"])
    
    # Simulate on_hand_inventory
    def simulate_inventory(d):
        m_demand = d.groupby(["store_id", "sku_id"])["demand"].transform("mean").values
        b_lt = d["base_lead_time"].values
        reorder_points = b_lt * m_demand * 1.5
        reorder_quants = b_lt * m_demand * 2
        
        demands = d["demand"].values
        lead_times = d["lead_time_days"].values
        
        inventory = np.zeros(len(d))
        
        store_sku = d["store_id"] + "_" + d["sku_id"]
        # Find group boundaries
        boundaries = np.where(store_sku.values[:-1] != store_sku.values[1:])[0]
        start_indices = np.concatenate(([0], boundaries + 1))
        end_indices = np.concatenate((boundaries + 1, [len(d)]))
        
        for start, end in zip(start_indices, end_indices):
            cur_m = m_demand[start]
            cur_r_pt = reorder_points[start]
            cur_r_qty = reorder_quants[start]
            
            cur_stock = int(cur_m * 20)
            pending = {}
            
            for i in range(start, end):
                if i in pending:
                    cur_stock += pending.pop(i)
                
                cur_stock -= demands[i]
                if cur_stock < 0:
                    cur_stock = 0
                    
                inventory[i] = cur_stock
                
                if cur_stock < cur_r_pt:
                    arr_idx = i + lead_times[i]
                    if arr_idx < end:
                        pending[arr_idx] = pending.get(arr_idx, 0) + cur_r_qty
                        
        d["on_hand_inventory"] = inventory
        return d

    df = simulate_inventory(df)
    
    # Clean up cold start stores (drop early data)
    cutoff_date = df["date"].max() - pd.Timedelta(days=28)
    mask = (df["is_cold_start_store"] == 1) & (df["date"] < cutoff_date)
    df = df[~mask].reset_index(drop=True)
    
    # Clean up columns
    drop_cols = ["base_price", "base_failure_rate", "lambda_multiplier", "true_demand", "shifted_demand", "base_lead_time"]
    df = df.drop(columns=drop_cols)
    
    return df

def main():
    print("Generating synthetic data...")
    df = generate_data()
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    out_path = "data/spare_parts_demand.csv"
    df.to_csv(out_path, index=False)
    print(f"Data saved to {out_path}\n")
    
    print("=== DATA SUMMARY ===")
    print(f"Date Range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Total Rows: {len(df)}")
    print(f"Num Stores: {df['store_id'].nunique()} (2 cold-start)")
    print(f"Num SKUs: {df['sku_id'].nunique()}")
    print(f"Shock events: {df['is_shock_event'].sum()} rows flagged")
    print(f"Censored days (stockouts): {df['stockout_flag'].sum()}")
    print("\nOverall Demand Stats:")
    print(df["demand"].describe())
    
    print("\nOverall Lead Time Stats (Days):")
    print(df["lead_time_days"].describe())
    
    print("\nOverall Inventory Stats:")
    print(df["on_hand_inventory"].describe())
    
    print("\nProportion of Zero Demand (Intermittency check):")
    zeros = (df["demand"] == 0).sum()
    print(f"{zeros} out of {len(df)} rows are 0 ({zeros/len(df)*100:.1f}%)")
    
    print("\nDemand by SKU (Mean):")
    print(df.groupby("sku_id")["demand"].mean())
    
    print("\nDemand by Store (Mean):")
    print(df.groupby("store_id")["demand"].mean())
    
    # Value counts of demand
    print("\nTop 10 Value Counts of Demand:")
    print(df["demand"].value_counts().head(10))
    
    # Plotting Histogram
    plt.figure(figsize=(10, 5))
    plt.hist(df['demand'], bins=range(0, int(df['demand'].max())+2), align='left', rwidth=0.8)
    plt.title("Distribution of Demand (Highly Intermittent)")
    plt.xlabel("Demand Quantity")
    plt.ylabel("Frequency")
    plt.yscale('log') # Log scale to see the tail
    plt.grid(axis='y', alpha=0.75)
    plot_path = "data/demand_histogram.png"
    plt.savefig(plot_path)
    print(f"\nHistogram saved to {plot_path}")

if __name__ == "__main__":
    main()

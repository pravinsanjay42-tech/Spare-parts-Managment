import pandas as pd
import numpy as np
import lightgbm as lgb
import os

def create_features(df):
    df = df.copy()
    df = df.sort_values(['store_id', 'sku_id', 'date'])
    
    # Basic features
    df['is_festival'] = df['is_festival'].astype(int)
    df['store_region'] = df['store_region'].astype('category')
    
    # Lag features
    grouped = df.groupby(['store_id', 'sku_id'])
    df['lag_1'] = grouped['demand'].shift(1)
    df['lag_7'] = grouped['demand'].shift(7)
    df['rolling_mean_7'] = grouped['demand'].transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    df['rolling_std_7'] = grouped['demand'].transform(lambda x: x.shift(1).rolling(7, min_periods=1).std())
    
    return df

def pinball_loss(y_true, y_pred, tau):
    err = y_true - y_pred
    return np.mean(np.maximum(tau * err, (tau - 1) * err))

def main():
    input_demand = 'data/spare_parts_demand.csv'
    input_quantiles = 'data/spare_parts_demand_with_quantiles.csv'
    output_path = 'data/predictions_hurdle.csv'
    
    print("Loading data...")
    try:
        df = pd.read_csv(input_demand)
    except FileNotFoundError:
        print(f"Error: Could not find {input_demand}")
        return
        
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    df = create_features(df)
    
    # Walk-forward split
    split_date = pd.to_datetime('2022-07-01')
    train_df = df[df['date'] < split_date].copy()
    test_df = df[df['date'] >= split_date].copy()
    
    features = [
        'store_size', 'equipment_age_years', 'weather_severity_index',
        'price', 'is_festival', 'store_region',
        'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7'
    ]
    
    # Stage 1: Binary classifier P(demand > 0)
    print("Training Stage 1 (Binary Classification)...")
    train_df['is_nonzero'] = (train_df['demand'] > 0).astype(int)
    
    clf = lgb.LGBMClassifier(n_estimators=100, random_state=42)
    clf.fit(train_df[features], train_df['is_nonzero'])
    
    # Stage 2: Quantile regression on NONZERO demand
    print("Training Stage 2 (Quantile Regression)...")
    nonzero_train_df = train_df[train_df['demand'] > 0].copy()
    
    quantile_models = {}
    quantiles = [0.1, 0.5, 0.9]
    for q in quantiles:
        print(f"  Training q={q}...")
        reg = lgb.LGBMRegressor(
            objective='quantile', 
            alpha=q, 
            n_estimators=100, 
            random_state=42
        )
        reg.fit(nonzero_train_df[features], nonzero_train_df['demand'])
        quantile_models[q] = reg
        
    print("Generating predictions...")
    p_nonzero = clf.predict_proba(test_df[features])[:, 1]
    test_df['p_nonzero'] = p_nonzero
    
    for q in quantiles:
        test_df[f'cond_p{int(q*100)}'] = quantile_models[q].predict(test_df[features])
        
    # Combined forecast
    # p50
    test_df['hurdle_p50'] = test_df['p_nonzero'] * test_df['cond_p50']
    
    # p90
    test_df['hurdle_p90'] = test_df['p_nonzero'] * test_df['cond_p90']
    
    # p10
    test_df['hurdle_p10'] = np.where(test_df['p_nonzero'] < 0.5, 0, test_df['cond_p10'])
    
    # Save predictions
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    test_df.to_csv(output_path, index=False)
    
    print("Evaluating models...")
    # Load direct quantile predictions to compare
    if os.path.exists(input_quantiles):
        df_quant = pd.read_csv(input_quantiles)
        if 'date' in df_quant.columns:
            df_quant['date'] = pd.to_datetime(df_quant['date'])
        
        # Merge test_df with direct quantiles
        eval_df = test_df.merge(
            df_quant[['store_id', 'sku_id', 'date', 'p10', 'p50', 'p90']], 
            on=['store_id', 'sku_id', 'date'],
            how='inner',
            suffixes=('', '_direct')
        )
        
        # Coverage
        cov_direct = ((eval_df['demand'] >= eval_df['p10']) & (eval_df['demand'] <= eval_df['p90'])).mean()
        cov_hurdle = ((eval_df['demand'] >= eval_df['hurdle_p10']) & (eval_df['demand'] <= eval_df['hurdle_p90'])).mean()
        
        # Pinball Loss
        pb_direct_10 = pinball_loss(eval_df['demand'], eval_df['p10'], 0.1)
        pb_direct_50 = pinball_loss(eval_df['demand'], eval_df['p50'], 0.5)
        pb_direct_90 = pinball_loss(eval_df['demand'], eval_df['p90'], 0.9)
        
        pb_hurdle_10 = pinball_loss(eval_df['demand'], eval_df['hurdle_p10'], 0.1)
        pb_hurdle_50 = pinball_loss(eval_df['demand'], eval_df['hurdle_p50'], 0.5)
        pb_hurdle_90 = pinball_loss(eval_df['demand'], eval_df['hurdle_p90'], 0.9)
        
        # Discrimination (Are p10 and p50 different?)
        diff_direct = (eval_df['p50'] - eval_df['p10']).mean()
        diff_hurdle = (eval_df['hurdle_p50'] - eval_df['hurdle_p10']).mean()
        
        print("\n--- Evaluation Comparison ---")
        print(f"{'Metric':<20} | {'Direct Quantile':<15} | {'Hurdle Model':<15}")
        print("-" * 56)
        print(f"{'Coverage (p10-p90)':<20} | {cov_direct:.4f}          | {cov_hurdle:.4f}")
        print(f"{'Pinball p10':<20} | {pb_direct_10:.4f}          | {pb_hurdle_10:.4f}")
        print(f"{'Pinball p50':<20} | {pb_direct_50:.4f}          | {pb_hurdle_50:.4f}")
        print(f"{'Pinball p90':<20} | {pb_direct_90:.4f}          | {pb_hurdle_90:.4f}")
        print(f"{'Mean(p50 - p10)':<20} | {diff_direct:.4f}          | {diff_hurdle:.4f}")
        
    else:
        print(f"Warning: {input_quantiles} not found. Cannot compare.")

if __name__ == "__main__":
    main()

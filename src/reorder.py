import pandas as pd
import numpy as np

def run_reorder_simulation(df_test, policy='naive', daily_cap=None):
    """
    Simulate reorder decisions across the test period.
    
    For each store-SKU-day:
    - Track on_hand inventory and on_order (in-transit) inventory
    - Decide whether to place a reorder based on the policy
    - Process arrivals when lead_time_days have elapsed since order placement
    
    Policies:
    - 'naive': Reorder when on_hand + on_order < croston_forecast * lead_time_days
    - 'uncertainty': Reorder when on_hand + on_order < p90 * lead_time_days
    
    Returns metrics: stockout_rate, service_level, avg_inventory, num_orders
    """
    df = df_test.sort_values(['store_id', 'sku_id', 'date']).copy()
    
    # Just in case they are missing, fallback to defaults
    if 'lead_time_days' not in df.columns:
        print("Warning: lead_time_days not found in data. Using default 7.")
        df['lead_time_days'] = 7
    if 'on_hand_inventory' not in df.columns:
        print("Warning: on_hand_inventory not found in data. Using default 50.")
        df['on_hand_inventory'] = 50

    results = []
    
    for (store, sku), group in df.groupby(['store_id', 'sku_id']):
        group = group.reset_index(drop=True)
        
        on_hand = group.loc[0, 'on_hand_inventory']
        on_order_dict = {}
        
        stockouts = 0
        total_demand = 0
        fulfilled_demand = 0
        inventory_sum = 0
        reorders = 0
        
        out_rows = []
        
        for i in range(len(group)):
            row = group.loc[i]
            demand = row['demand']
            lead_time = int(row['lead_time_days'])
            
            # 1. Process arrivals
            if i in on_order_dict:
                on_hand += on_order_dict[i]
                
            # 2. Demand and Stockouts
            total_demand += demand
            if demand > 0 and on_hand == 0:
                stockouts += 1
                
            fulfilled = min(on_hand, demand)
            fulfilled_demand += fulfilled
            
            on_hand -= demand
            if on_hand < 0:
                on_hand = 0
                
            inventory_sum += on_hand
            
            # 3. Reorder Logic
            in_transit = sum(amt for arr_idx, amt in on_order_dict.items() if arr_idx > i)
            
            if policy == 'naive':
                forecast = row['baseline_forecast']
            else:
                forecast = row['p90']
                
            reorder_point = forecast * lead_time
            
            if (on_hand + in_transit) < reorder_point:
                order_qty = reorder_point
                if order_qty > 0:
                    arr_time = i + lead_time
                    on_order_dict[arr_time] = on_order_dict.get(arr_time, 0) + order_qty
                    reorders += 1
                    
            out_rows.append({
                'date': row['date'],
                'store_id': store,
                'sku_id': sku,
                'demand': demand,
                'on_hand': on_hand,
                'policy': policy
            })
            
        res = {
            'stockouts': stockouts,
            'total_demand': total_demand,
            'fulfilled_demand': fulfilled_demand,
            'inventory_sum': inventory_sum,
            'reorders': reorders,
            'days_with_demand': sum(group['demand'] > 0),
            'days': len(group)
        }
        results.append((res, out_rows))
        
    all_out_rows = []
    
    total_stockouts = sum(r[0]['stockouts'] for r in results)
    total_days_with_demand = sum(r[0]['days_with_demand'] for r in results)
    total_fulfilled = sum(r[0]['fulfilled_demand'] for r in results)
    total_demand_all = sum(r[0]['total_demand'] for r in results)
    total_inventory = sum(r[0]['inventory_sum'] for r in results)
    total_days = sum(r[0]['days'] for r in results)
    total_orders = sum(r[0]['reorders'] for r in results)
    
    for r in results:
        all_out_rows.extend(r[1])
        
    return {
        'stockout_rate': total_stockouts / total_days_with_demand if total_days_with_demand > 0 else 0,
        'service_level': total_fulfilled / total_demand_all if total_demand_all > 0 else 0,
        'avg_inventory': total_inventory / total_days if total_days > 0 else 0,
        'num_orders': total_orders,
        'daily_data': all_out_rows
    }

def main():
    print("Loading data...")
    df_q = pd.read_csv("data/spare_parts_demand_with_quantiles.csv")
    df_b = pd.read_csv("data/spare_parts_demand_with_baseline.csv")
    
    print("Merging data...")
    merge_cols = ['date', 'store_id', 'sku_id']
    df = pd.merge(df_q, df_b[['date', 'store_id', 'sku_id', 'baseline_forecast']], on=merge_cols, how='inner')
    
    print("Filtering to test period...")
    df['date'] = pd.to_datetime(df['date'])
    df_test = df[df['date'] >= '2022-07-01'].copy()
    
    policies = ['naive', 'uncertainty']
    results = {}
    all_daily_data = []
    
    for pol in policies:
        print(f"Running simulation for {pol} policy...")
        res = run_reorder_simulation(df_test, policy=pol)
        results[pol] = res
        all_daily_data.extend(res['daily_data'])
        print(f"Metrics for {pol}:")
        print(f"  Stockout Rate: {res['stockout_rate']:.2%}")
        print(f"  Service Level: {res['service_level']:.2%}")
        print(f"  Average Inventory: {res['avg_inventory']:.2f}")
        print(f"  Num Orders: {res['num_orders']}")
        
    print("Saving daily simulation data to reorder_results.csv...")
    pd.DataFrame(all_daily_data).to_csv('reorder_results.csv', index=False)
    
    md_lines = []
    md_lines.append("# Reorder Policy Comparison")
    md_lines.append("")
    md_lines.append("| Policy | Stockout Rate (%) | Service Level (%) | Average Inventory Held | Number of Reorder Triggers |")
    md_lines.append("|--------|-------------------|-------------------|------------------------|----------------------------|")
    
    for pol in policies:
        res = results[pol]
        md_lines.append(f"| {pol} | {res['stockout_rate']*100:.2f}% | {res['service_level']*100:.2f}% | {res['avg_inventory']:.2f} | {res['num_orders']} |")
        
    md_content = "\n".join(md_lines)
    with open('reorder_comparison.md', 'w') as f:
        f.write(md_content)
        
    print("Saved comparison to reorder_comparison.md")

if __name__ == "__main__":
    main()

import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.getcwd())


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def raw_data():
    return pd.read_csv("data/spare_parts_demand.csv", parse_dates=['date'])

@pytest.fixture(scope="module")
def df_quantiles():
    return pd.read_csv("data/spare_parts_demand_with_quantiles.csv", parse_dates=['date'])

@pytest.fixture(scope="module")
def df_baseline():
    return pd.read_csv("data/spare_parts_demand_with_baseline.csv", parse_dates=['date'])

@pytest.fixture(scope="module")
def eval_report():
    with open("evaluation_report.md", "r") as f:
        return f.read()


# ============================================================
# 1. Data Generation Tests
# ============================================================

class TestDataGeneration:
    def test_required_columns_exist(self, raw_data):
        required = ['date', 'store_id', 'sku_id', 'demand', 'store_size',
                     'equipment_age_years', 'weather_severity_index', 'price',
                     'is_festival', 'is_shock_event', 'stockout_flag',
                     'lead_time_days', 'on_hand_inventory']
        for col in required:
            assert col in raw_data.columns, f"Missing column: {col}"

    def test_no_negative_demand(self, raw_data):
        assert (raw_data['demand'] >= 0).all(), "Demand must be non-negative"

    def test_lead_time_range(self, raw_data):
        assert raw_data['lead_time_days'].min() >= 1, "Lead time must be >= 1 day"
        assert raw_data['lead_time_days'].max() <= 30, "Lead time must be <= 30 days"

    def test_inventory_never_negative(self, raw_data):
        assert (raw_data['on_hand_inventory'] >= 0).all(), "Inventory must never go negative"

    def test_row_count_reasonable(self, raw_data):
        assert len(raw_data) > 100000, "Expected > 100K rows of panel data"

    def test_cold_start_stores_have_limited_history(self, raw_data):
        cold = raw_data[raw_data['is_cold_start_store'] == 1]
        assert len(cold) > 0, "Cold start stores should exist"
        days = cold['date'].nunique()
        assert days <= 35, f"Cold start stores should have <= ~28 days of data, got {days}"


# ============================================================
# 2. Croston's Baseline Tests
# ============================================================

class TestCrostonBaseline:
    def test_baseline_non_negative(self, df_baseline):
        assert (df_baseline['baseline_forecast'] >= 0).all(), "Croston forecast must be non-negative"

    def test_handles_all_zero_series(self):
        from src.baseline import croston_forecast
        result = croston_forecast(np.zeros(100))
        assert not np.any(np.isnan(result)), "Croston should handle all-zero series without NaN"
        assert not np.any(np.isinf(result)), "Croston should handle all-zero series without Inf"

    def test_responds_to_demand(self):
        from src.baseline import croston_forecast
        ts = np.zeros(50)
        ts[10] = 5
        ts[20] = 3
        result = croston_forecast(ts)
        # After seeing demand, forecast should be positive and non-trivial
        assert result[25] > 0, "Forecast should be positive after observing demand"
        # Forecast should change from initial value after seeing data
        assert result[25] != result[0], "Forecast should update after observing demand"


# ============================================================
# 3. Quantile Model Tests
# ============================================================

class TestQuantileModel:
    def test_quantile_crossing_fixed(self, df_quantiles):
        """Assert p10 <= p50 <= p90 for all rows (no crossing)."""
        assert (df_quantiles['p10'] <= df_quantiles['p50'] + 1e-6).all(), "p10 must be <= p50"
        assert (df_quantiles['p50'] <= df_quantiles['p90'] + 1e-6).all(), "p50 must be <= p90"

    def test_predictions_non_negative(self, df_quantiles):
        assert (df_quantiles['p10'] >= 0).all(), "p10 must be non-negative"
        assert (df_quantiles['p50'] >= 0).all(), "p50 must be non-negative"
        assert (df_quantiles['p90'] >= 0).all(), "p90 must be non-negative"

    def test_quantile_crossing_fix_function(self):
        from src.quantile_model import fix_quantile_crossing
        df = pd.DataFrame({'p10': [3, 1, 5], 'p50': [1, 2, 3], 'p90': [2, 3, 1]})
        fixed, count = fix_quantile_crossing(df)
        assert (fixed['p10'] <= fixed['p50']).all()
        assert (fixed['p50'] <= fixed['p90']).all()
        assert count > 0, "Should have detected and fixed crossings"


# ============================================================
# 4. Evaluation Tests
# ============================================================

class TestEvaluation:
    def test_censored_rows_excluded_from_headline(self, eval_report):
        assert "Overall (Clean" in eval_report or "Excluding Censored" in eval_report

    def test_stockout_segment_reported_separately(self, eval_report):
        assert "Stockout Censored" in eval_report or "Censored" in eval_report

    def test_split_date_respected(self):
        preds = pd.read_csv("data/predictions_test_set.csv", parse_dates=['date'])
        assert preds['date'].min() >= pd.Timestamp('2022-07-01'), \
            "Test predictions must start at or after the split date"

    def test_predictions_file_has_required_columns(self):
        preds = pd.read_csv("data/predictions_test_set.csv")
        for col in ['date', 'store_id', 'sku_id', 'demand', 'baseline_forecast',
                     'p10', 'p50', 'p90', 'is_shock_event']:
            assert col in preds.columns, f"Missing column in predictions: {col}"


# ============================================================
# 5. Reorder Policy Tests
# ============================================================

class TestReorderPolicy:
    def test_lead_time_changes_decisions(self):
        """Verify that different lead times produce different reorder behavior."""
        from src.reorder import run_reorder_simulation
        dates = pd.date_range('2022-07-01', periods=30)
        rows = []
        for d in dates:
            rows.append({'date': d, 'store_id': 'S1', 'sku_id': 'K1',
                         'demand': 1, 'baseline_forecast': 0.5, 'p90': 2.0,
                         'lead_time_days': 5, 'on_hand_inventory': 10})
        df_short = pd.DataFrame(rows)

        rows2 = []
        for d in dates:
            rows2.append({'date': d, 'store_id': 'S1', 'sku_id': 'K1',
                          'demand': 1, 'baseline_forecast': 0.5, 'p90': 2.0,
                          'lead_time_days': 20, 'on_hand_inventory': 10})
        df_long = pd.DataFrame(rows2)

        res_short = run_reorder_simulation(df_short, policy='uncertainty')
        res_long = run_reorder_simulation(df_long, policy='uncertainty')
        # Different lead times must produce different reorder behavior
        assert res_long['num_orders'] != res_short['num_orders'] or \
               res_long['avg_inventory'] != res_short['avg_inventory'], \
            "Lead time must change reorder decisions (different orders or inventory levels)"

    def test_uncertainty_policy_fewer_stockouts(self):
        """p90-based reorder should have fewer stockouts than naive."""
        from src.reorder import run_reorder_simulation
        dates = pd.date_range('2022-07-01', periods=60)
        np.random.seed(42)
        rows = []
        for d in dates:
            rows.append({'date': d, 'store_id': 'S1', 'sku_id': 'K1',
                         'demand': np.random.poisson(2), 'baseline_forecast': 0.5,
                         'p90': 4.0, 'lead_time_days': 7, 'on_hand_inventory': 20})
        df = pd.DataFrame(rows)
        naive = run_reorder_simulation(df, policy='naive')
        unc = run_reorder_simulation(df, policy='uncertainty')
        assert unc['stockout_rate'] <= naive['stockout_rate'], \
            "Uncertainty policy should have <= stockout rate vs naive"

    def test_reorder_handles_missing_lead_time_or_inventory(self):
        """Confirm reorder simulation handles missing columns and row-level NaNs using defaults."""
        from src.reorder import run_reorder_simulation
        dates = pd.date_range('2022-07-01', periods=60)

        # Case 1: Columns missing entirely (defaults to lead_time=7, on_hand=50)
        rows_no_cols = []
        for d in dates:
            rows_no_cols.append({
                'date': d, 'store_id': 'S1', 'sku_id': 'K1',
                'demand': 1, 'baseline_forecast': 0.5, 'p90': 2.0
            })
        df_no_cols = pd.DataFrame(rows_no_cols)
        res_no_cols = run_reorder_simulation(df_no_cols, policy='uncertainty')
        assert res_no_cols['num_orders'] > 0, "Simulation should trigger reorders once default stock of 50 depletes"
        assert res_no_cols['avg_inventory'] > 0
        assert not np.isnan(res_no_cols['service_level'])

        # Case 2: Row-level NaNs and nulls in columns
        rows_nans = []
        for i, d in enumerate(dates):
            rows_nans.append({
                'date': d, 'store_id': 'S1', 'sku_id': 'K1',
                'demand': 1, 'baseline_forecast': 0.5, 'p90': 2.0,
                'lead_time_days': np.nan if i % 2 == 0 else 7,
                'on_hand_inventory': np.nan if i == 0 else 30
            })
        df_nans = pd.DataFrame(rows_nans)
        res_nans = run_reorder_simulation(df_nans, policy='uncertainty')
        assert res_nans['num_orders'] > 0
        assert not np.isnan(res_nans['stockout_rate'])

    def test_croston_safety_stock_improves_service_level(self):
        """Confirm croston_safety_stock produces a different, higher service level than naive Croston."""
        from src.reorder import run_reorder_simulation
        dates = pd.date_range('2022-07-01', periods=60)
        np.random.seed(42)
        rows = []
        for d in dates:
            rows.append({
                'date': d, 'store_id': 'S1', 'sku_id': 'K1',
                'demand': np.random.poisson(2), 'baseline_forecast': 0.5,
                'p90': 4.0, 'lead_time_days': 7, 'on_hand_inventory': 15
            })
        df = pd.DataFrame(rows)
        naive = run_reorder_simulation(df, policy='naive')
        ss = run_reorder_simulation(df, policy='croston_safety_stock')
        assert ss['service_level'] >= naive['service_level'], (
            f"Safety stock service level ({ss['service_level']:.2%}) should be >= naive ({naive['service_level']:.2%})"
        )
        assert ss['service_level'] != naive['service_level'], (
            "Safety stock policy should produce a different service level than naive Croston"
        )


# ============================================================
# 6. Edge Case Tests (original 4)
# ============================================================

class TestEdgeCases:
    def test_shock_event_widens_interval(self, df_quantiles):
        df = df_quantiles.copy()
        df['interval_width'] = df['p90'] - df['p10']
        store, sku = "Store_08", "SKU_05"
        normal = df[(df['store_id'] == store) & (df['sku_id'] == sku) & (df['date'] < '2021-08-01')]
        post_shock = df[(df['store_id'] == store) & (df['sku_id'] == sku)
                        & (df['date'] >= '2021-08-11') & (df['date'] <= '2021-08-20')]
        if len(normal) > 0 and len(post_shock) > 0:
            assert post_shock['interval_width'].mean() > normal['interval_width'].mean()
        shock_days = df[df['is_shock_event'] == 1]
        misses = (shock_days['demand'] > shock_days['p90']).sum()
        assert misses / len(shock_days) > 0.8

    def test_cold_start_fallback(self, df_quantiles):
        cold_start = df_quantiles[df_quantiles['is_cold_start_store'] == 1]
        assert not cold_start['p50'].isna().any()
        first_few = cold_start.sort_values('date').head(100)
        assert first_few['p90'].mean() > 0

    def test_stockout_censored_demand_flagging(self, eval_report):
        assert "Stockout Censored" in eval_report or "Censored" in eval_report
        assert "Overall (Clean" in eval_report or "Excluding Censored" in eval_report

    def test_dispatch_simulator_shock_recovery(self):
        from src.dispatch import run_dispatch_sim
        df_preds = pd.read_csv("data/spare_parts_demand_with_quantiles.csv")
        df_base = pd.read_csv("data/spare_parts_demand_with_baseline.csv")
        df_all = pd.merge(df_preds, df_base[['date', 'store_id', 'sku_id', 'baseline_forecast']],
                          on=['date', 'store_id', 'sku_id'])
        df_test = df_all[df_all['date'] >= '2022-07-01'].copy()
        metrics_A, metrics_B, ts_A, ts_B = run_dispatch_sim(df_test, daily_cap=80)
        assert metrics_B['unsafe_assignments'] == 0.0
        assert metrics_A['unsafe_assignments'] > 1000.0

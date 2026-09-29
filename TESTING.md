# Testing & Quality Assurance Documentation

This document provides a granular reference for the automated test suite in `tests/test_edge_cases.py`, instructions for local execution, and an explanation of the Continuous Integration (CI) configuration.

---

## 1. Test Suite Summary & Granular Mapping

The test suite consists of **22 automated unit and integration tests** implemented with `pytest`. Each test asserts mathematical, structural, or operational invariants across the forecasting and decision pipeline.

### 1.1 Data Generation Tests (`TestDataGeneration`)
* **`test_required_columns_exist`**: Asserts that `data/spare_parts_demand.csv` contains all 13 required feature and target columns (`date`, `store_id`, `sku_id`, `demand`, `store_size`, `equipment_age_years`, `weather_severity_index`, `price`, `is_festival`, `is_shock_event`, `stockout_flag`, `lead_time_days`, `on_hand_inventory`).
* **`test_no_negative_demand`**: Asserts that demand values are strictly non-negative ($Y \ge 0$) across all rows in the dataset.
* **`test_lead_time_range`**: Validates that simulated supplier delivery delays are strictly within the realistic physical bounds of 1 to 30 days ($1 \le L \le 30$).
* **`test_inventory_never_negative`**: Validates that the stateful on-hand inventory balance never drops below zero ($I \ge 0$) even under demand exceeding available stock.
* **`test_row_count_reasonable`**: Asserts that the generated panel scale contains $> 100,000$ daily observations (3 years $\times$ 15 stores $\times$ 10 SKUs with cold-start truncation).
* **`test_cold_start_stores_have_limited_history`**: Asserts that cold-start stores (Store 14 and Store 15) have their historical depth truncated to $\le 35$ days of data prior to the cutoff.

### 1.2 Croston's Baseline Tests (`TestCrostonBaseline`)
* **`test_baseline_non_negative`**: Asserts that Croston point forecasts are non-negative across all historical timestamps.
* **`test_handles_all_zero_series`**: Validates numerical stability of `src/baseline.py:croston_forecast` when evaluated on a series consisting entirely of zeros, ensuring no `NaN` or `Inf` division-by-zero errors.
* **`test_responds_to_demand`**: Asserts that the smoothed demand magnitude $q$ and inter-arrival parameter $a$ update correctly after non-zero demand events occur, altering the forecast away from the initial prior.

### 1.3 Quantile Model Tests (`TestQuantileModel`)
* **`test_quantile_crossing_fixed`**: Asserts that predicted quantile trajectories are strictly monotonic ($p10 \le p50 \le p90$) across all rows in `data/spare_parts_demand_with_quantiles.csv`.
* **`test_predictions_non_negative`**: Validates that all predicted quantile boundaries are non-negative ($\hat{p}_{10}, \hat{p}_{50}, \hat{p}_{90} \ge 0$), since physical spare-parts demand cannot be negative.
* **`test_quantile_crossing_fix_function`**: Directly tests the sorting operator in `src/quantile_model.py:fix_quantile_crossing` on synthetic inverted matrices (e.g., $p10 > p50$), asserting that all crossing inversions are rectified.

### 1.4 Evaluation Engine Tests (`TestEvaluation`)
* **`test_censored_rows_excluded_from_headline`**: Asserts that `evaluation_report.md` segregates stockout-censored rows and evaluates headline metrics on clean, uncensored data.
* **`test_stockout_segment_reported_separately`**: Asserts that stockout-censored observations are explicitly isolated and reported in a separate diagnostic row.
* **`test_split_date_respected`**: Confirms strict walk-forward temporal integrity by asserting that `data/predictions_test_set.csv` contains exclusively rows on or after `2022-07-01` (preventing test data leakage).
* **`test_predictions_file_has_required_columns`**: Asserts that `data/predictions_test_set.csv` contains all necessary out-of-sample fields (`date`, `store_id`, `sku_id`, `demand`, `baseline_forecast`, `p10`, `p50`, `p90`, `is_shock_event`).

### 1.5 Reorder Policy Tests (`TestReorderPolicy`)
* **`test_lead_time_changes_decisions`**: Asserts that changing supplier lead times from short (5 days) to long (20 days) directly alters reorder point triggers and order frequency in `src/reorder.py:run_reorder_simulation`.
* **`test_uncertainty_policy_fewer_stockouts`**: Validates the core inventory thesis by asserting that the tail-risk $p90$ policy achieves an equal or lower stockout rate compared to the naive Croston baseline under identical simulated demand.

### 1.6 Edge Cases & Dispatch Simulation Tests (`TestEdgeCases`)
* **`test_shock_event_widens_interval`**: Asserts that following an injected shock event, the model's uncertainty width ($p90 - p10$) expands due to volatility tracking in rolling statistics, and asserts that unaugmented models miss unprecedented black-swan spikes $>80\%$ of the time.
* **`test_cold_start_fallback`**: Asserts that newly launched stores with minimal history fall back to non-zero, population-level uncertainty bounds rather than failing or outputting trivial zeros.
* **`test_stockout_censored_demand_flagging`**: Asserts that evaluation documentation flags distorted ground truth on stockout days.
* **`test_dispatch_simulator_shock_recovery`**: Asserts that in `src/dispatch.py`, Policy B (uncertainty-aware) incurs strictly zero workload violations ($0.0$) under hard daily delivery caps, while Policy A (naive) incurs $>1,000$ unsafe overtime assignments.

---

## 2. Running Tests Locally

To run the complete test suite locally:

```bash
# Run pytest with verbose output
pytest tests/test_edge_cases.py -v

# Run with concise summary
pytest tests/test_edge_cases.py -q
```

### Expected Output
```text
============================= test session starts =============================
collected 22 items

tests/test_edge_cases.py::TestDataGeneration::test_required_columns_exist PASSED [  4%]
tests/test_edge_cases.py::TestDataGeneration::test_no_negative_demand PASSED [  9%]
tests/test_edge_cases.py::TestDataGeneration::test_lead_time_range PASSED [ 13%]
tests/test_edge_cases.py::TestDataGeneration::test_inventory_never_negative PASSED [ 18%]
tests/test_edge_cases.py::TestDataGeneration::test_row_count_reasonable PASSED [ 22%]
tests/test_edge_cases.py::TestDataGeneration::test_cold_start_stores_have_limited_history PASSED [ 27%]
tests/test_edge_cases.py::TestCrostonBaseline::test_baseline_non_negative PASSED [ 31%]
tests/test_edge_cases.py::TestCrostonBaseline::test_handles_all_zero_series PASSED [ 36%]
tests/test_edge_cases.py::TestCrostonBaseline::test_responds_to_demand PASSED [ 40%]
tests/test_edge_cases.py::TestQuantileModel::test_quantile_crossing_fixed PASSED [ 45%]
tests/test_edge_cases.py::TestQuantileModel::test_predictions_non_negative PASSED [ 50%]
tests/test_edge_cases.py::TestQuantileModel::test_quantile_crossing_fix_function PASSED [ 54%]
tests/test_edge_cases.py::TestEvaluation::test_censored_rows_excluded_from_headline PASSED [ 59%]
tests/test_edge_cases.py::TestEvaluation::test_stockout_segment_reported_separately PASSED [ 63%]
tests/test_edge_cases.py::TestEvaluation::test_split_date_respected PASSED [ 68%]
tests/test_edge_cases.py::TestEvaluation::test_predictions_file_has_required_columns PASSED [ 72%]
tests/test_edge_cases.py::TestReorderPolicy::test_lead_time_changes_decisions PASSED [ 77%]
tests/test_edge_cases.py::TestReorderPolicy::test_uncertainty_policy_fewer_stockouts PASSED [ 81%]
tests/test_edge_cases.py::TestEdgeCases::test_shock_event_widens_interval PASSED [ 86%]
tests/test_edge_cases.py::TestEdgeCases::test_cold_start_fallback PASSED [ 90%]
tests/test_edge_cases.py::TestEdgeCases::test_stockout_censored_demand_flagging PASSED [ 95%]
tests/test_edge_cases.py::TestEdgeCases::test_dispatch_simulator_shock_recovery PASSED [100%]

======================= 22 passed in 3.97s =======================
```

---

## 3. Continuous Integration Architecture

The test suite runs automatically on GitHub Actions via `.github/workflows/tests.yml`.

### Workflow Configuration
* **Trigger Events**: Pushes and pull requests to branches `main`, `review-1`, and `review-2`.
* **Runner Environment**: `ubuntu-latest` with Python 3.11.
* **Pipeline Stages**:
  1. **Checkout Code**: Checks out the repository using `actions/checkout@v4`.
  2. **Set up Python**: Configures Python 3.11 environment using `actions/setup-python@v5`.
  3. **Dependency Resolution**: Installs packages listed in `requirements.txt`.
  4. **Data Verification**: Checks for `data/spare_parts_demand.csv` (runs `src/data_gen.py` if missing).
  5. **Model Pipeline**: Runs `src/baseline.py`, `src/quantile_model.py`, and `src/evaluate.py`.
  6. **Automated Verification**: Executes `pytest tests/test_edge_cases.py -v`.

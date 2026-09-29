# Error Handling & Edge Cases Specification

This document details how **UncertainSpares** handles operational edge cases, mathematical boundary conditions, and degraded input data. For each failure mode, we specify the root issue, the handling function in code, the runtime fallback behavior, and the corresponding automated test that verifies it.

---

## 1. Summary Matrix of Edge Cases

| Edge Case Scenario | Failure Risk | Handling Function | Fallback Mechanism | Verifying Test |
|---|---|---|---|---|
| **1. Cold-Start Store (No History)** | Model failure or default to constant 0 forecast | `src/quantile_model.py:create_features` & LightGBM default tree routing | LightGBM routes `NaN` lag/rolling features to population-level default split paths, producing wide uncertainty bounds rather than 0 | `tests/test_edge_cases.py::TestEdgeCases::test_cold_start_fallback` |
| **2. Zero-Inflated Demand (Quantile Collapse)** | Lower quantiles collapse ($p10 = p50 = 0$) | `src/hurdle_model.py:main` | Decomposes process into two stages: $P(Y > 0)$ classifier + positive-only quantile regressors, recovering $p50 - p10 > 0$ | `tests/test_edge_cases.py::TestQuantileModel::test_quantile_crossing_fixed` & Hurdle evaluation |
| **3. Black-Swan Demand Shock** | Severe under-prediction during rare catastrophes | `src/shock_detector.py:detect_and_adjust_shocks` & `app/dashboard.py` scenario overrides | Computes rolling 14-day z-scores on lagged demand; when $Z > 3.0$, applies a 3x multiplier to $p90$ across a 7-day momentum window | `tests/test_edge_cases.py::TestEdgeCases::test_shock_event_widens_interval` |
| **4. Stockout-Censored Demand** | False penalty for predicting true demand when stock is 0 | `src/evaluate.py:main` & `src/data_gen.py` | Flags days where stock was depleted (`stockout_flag = 1`); explicitly isolates them into a separate segment in evaluation | `tests/test_edge_cases.py::TestEvaluation::test_censored_rows_excluded_from_headline` |
| **5. Missing / Malformed Lead Time or Inventory** | Simulation crash on missing columns or row-level NaNs | `src/reorder.py:run_reorder_simulation` | Defensive parameter defaults: automatically falls back to `lead_time_days = 7` and `on_hand_inventory = 50` for missing columns or row-level `NaN`/nulls | `tests/test_edge_cases.py::TestReorderPolicy::test_reorder_handles_missing_lead_time_or_inventory` |
| **6. Quantile Crossing Inversions** | Non-monotonic intervals ($p10 > p50$ or $p50 > p90$) | `src/quantile_model.py:fix_quantile_crossing` | Vectorized row-wise sorting (`np.sort(axis=1)`) guarantees monotonic order $p10 \le p50 \le p90$ across all rows | `tests/test_edge_cases.py::TestQuantileModel::test_quantile_crossing_fix_function` |

---

## 2. Detailed Technical Breakdown

### 2.1 Cold-Start Store with Near-Zero History
* **Problem**: New stores (Store 14 and Store 15) enter the network with less than 30 days of operation. Lag features (`lag_1`, `lag_7`) and rolling statistics (`rolling_mean_7`, `rolling_std_7`) are populated with `NaN` values for initial observations.
* **Handling Implementation**: 
  - `src/quantile_model.py` leverages LightGBM's native handling of missing values. LightGBM automatically assigns missing values to the optimal branch split determined during training on the rest of the network.
  - The model does not crash and does not return a trivial zero forecast. Instead, it utilizes cross-store operational covariates (`store_size`, `equipment_age_years`, `weather_severity_index`, `price`, `store_region`) to generate a generalized, wide uncertainty interval.
* **Test Verification**: `test_cold_start_fallback` asserts that `p50` is never `NaN` and that initial predictions produce a wide right-tail estimate ($\hat{p}_{90} > 0$).

### 2.2 Zero-Inflated Demand and Lower-Quantile Collapse
* **Problem**: Across the network, demand is exactly zero on ~85% of days. Minimizing continuous pinball loss on heavily zero-inflated distributions causes both the 10th and 50th percentiles to converge to 0, providing zero discrimination between the lower bounds.
* **Handling Implementation**:
  - Direct Model: In `src/quantile_model.py`, the system acknowledges this mathematical reality: operational safety is driven primarily by the right-tail bound ($p90$), while $p10$ and $p50$ represent the stable zero-demand state.
  - Hurdle Architecture: In `src/hurdle_model.py`, a two-stage hurdle formulation decouples occurrence from quantity:
    $$\hat{p}_{50}^{\text{hurdle}} = \hat{P}(Y > 0 \mid X) \cdot \hat{Q}_{Y \mid X, Y>0}(0.50 \mid X)$$
    This restores positive discrimination ($\text{Mean}(p50 - p10) = 0.295$) and sharpens the overall prediction interval.
* **Test Verification**: Tested in `test_quantile_crossing_fixed` and monitored via the comparative metrics in `evaluation_report.md`.

### 2.3 Black-Swan Shock Events
* **Problem**: Unprecedented macro-events (such as major equipment design defects or severe natural disasters) cause sudden 8–16x demand surges. Historical regression models trained on regular data fail to predict these shifts in advance.
* **Handling Implementation**:
  - Statistical Widening: `src/shock_detector.py` computes an online rolling 14-day z-score on lagged demand. When anomalous consumption ($Z > 3.0$ or $y_{t-1} > 5\mu$) is observed, it flags a shock state and expands the upper quantile bound ($\hat{p}_{90}^{\text{adj}} = 3.0 \times \hat{p}_{90}$) for a 7-day momentum window.
  - Human-in-the-Loop Override: In `app/dashboard.py`, planners can toggle interactive scenario stress tests (e.g., Heatwave 4x multiplier, Festival 2x multiplier) to explicitly widen right-tail boundaries when impending external risks are known.
* **Test Verification**: `test_shock_event_widens_interval` verifies that volatility following shock days expands the predicted interval width ($p90 - p10$).

### 2.4 Stockout-Censored Demand
* **Problem**: On days when inventory drops to zero, customer demand is unmet and recorded as zero demand. If treated as genuine zero demand during evaluation, the forecasting model is unfairly penalized for correctly predicting demand based on underlying risk factors.
* **Handling Implementation**:
  - `src/data_gen.py` flags days where previous demand depleted inventory as `stockout_flag = 1`.
  - `src/evaluate.py` excludes censored rows from the headline accuracy metrics (`Overall Clean`), reporting them in a dedicated diagnostic segment (`Stockout Censored Days (Distorted Ground Truth)`).
* **Test Verification**: `test_censored_rows_excluded_from_headline` and `test_stockout_segment_reported_separately` assert that evaluation reports segregate censored rows from clean benchmarks.

### 2.5 Missing or Malformed Lead Time and Inventory Data
* **Problem**: In practical deployment, external ERP feeds or CSV updates might omit `lead_time_days` or `on_hand_inventory` due to transmission errors or format mismatches.
* **Handling Implementation**:
  - In `src/reorder.py:run_reorder_simulation`, defensive fallback checks verify column existence and coerce row-level `NaN` or non-positive values to sensible operational defaults:
    ```python
    if 'lead_time_days' not in df.columns:
        df['lead_time_days'] = 7  # Standard physical supplier median default
    else:
        df['lead_time_days'] = pd.to_numeric(df['lead_time_days'], errors='coerce').fillna(7)
        df['lead_time_days'] = df['lead_time_days'].apply(lambda x: 7 if x <= 0 else x)

    if 'on_hand_inventory' not in df.columns:
        df['on_hand_inventory'] = 50  # Operational buffer fallback
    else:
        df['on_hand_inventory'] = pd.to_numeric(df['on_hand_inventory'], errors='coerce').fillna(50)
        df['on_hand_inventory'] = df['on_hand_inventory'].apply(lambda x: 50 if x < 0 else x)
    ```
  - This prevents unexpected simulation crashes, handles missing columns, handles individual row-level `NaN` values, and ensures valid integer lead times $\ge 1$.
* **Test Verification**: `test_reorder_handles_missing_lead_time_or_inventory` asserts that simulation completes gracefully and outputs valid metrics under both missing columns and row-level `NaN` inputs.

### 2.6 Quantile Crossing Inversions
* **Problem**: Because LightGBM fits independent gradient-boosted trees for $p10$, $p50$, and $p90$, individual tree splits can occasionally predict non-monotonic boundaries ($\hat{p}_{10} > \hat{p}_{50}$ or $\hat{p}_{50} > \hat{p}_{90}$), violating the definition of cumulative distribution functions.
* **Handling Implementation**:
  - In `src/quantile_model.py:fix_quantile_crossing`, predicted arrays are passed through an iso-monotonic sorting operator:
    ```python
    fixed = np.sort(df[['p10', 'p50', 'p90']].values, axis=1)
    df['p10'], df['p50'], df['p90'] = fixed[:, 0], fixed[:, 1], fixed[:, 2]
    ```
  - This post-processing step ensures mathematical consistency without degrading individual quantile accuracy.
* **Test Verification**: `test_quantile_crossing_fixed` and `test_quantile_crossing_fix_function` assert strict monotonic order across all predicted outputs.

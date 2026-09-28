# UncertainSpares

A probabilistic demand forecasting system for intermittent spare-parts logistics.
Demonstrates how uncertainty-aware forecasting (Quantile Regression) and lead-time-aware procurement outperform traditional point-forecast baselines.

## Results Summary

| Metric | Baseline (Croston's) | Uncertainty-Aware (LightGBM p90) |
|--------|---------------------|----------------------------------|
| MAE (Overall Clean) | 0.669 | 0.418 |
| Coverage (80% target) | — | 89.9% |
| Stockout Rate (with lead time) | 6.33% | 0.10% |
| Service Level | 85.05% | 99.53% |
| Shock Event Coverage | — | 3.1% (honest limitation) |

## Architecture

1. **Synthetic Data Generator** (`src/data_gen.py`): 3 years × 15 stores × 10 SKUs (~143K rows). Includes lead times (3–24 days), on-hand inventory, shock events, cold-start stores, stockout censoring.
2. **Baseline Model** (`src/baseline.py`): Croston's Method for intermittent demand.
3. **Uncertainty-Aware Model** (`src/quantile_model.py`): LightGBM quantile regression (p10/p50/p90). Walk-forward split: trains only on data < 2022-07-01.
4. **Evaluation Engine** (`src/evaluate.py`): Segment-level MAE, RMSE, pinball loss, coverage, sharpness. Generates calibration plot.
5. **Dispatch Simulator** (`src/dispatch.py`): Policy A (naive) vs Policy B (uncertainty-aware) under workload constraints.
6. **Reorder Simulator** (`src/reorder.py`): Lead-time-aware reorder policies. Compares naive vs p90-based procurement.
7. **Shock Detector** (`src/shock_detector.py`): Rolling z-score anomaly detection with p90 widening.
8. **Hurdle Model** (`src/hurdle_model.py`): Two-stage zero-inflated model (P(demand>0) + conditional quantiles).
9. **Test Suite** (`tests/test_edge_cases.py`): 17 pytest cases covering data gen, baseline, quantile model, evaluation, reorder, and edge cases.
10. **Dashboard** (`app/dashboard.py`): Streamlit app with KPI bar, risk table (lead-time-aware), scenario toggle, calibration tab.

## Setup

```bash
pip install -r requirements.txt
```

## Run Order (End-to-End)

```bash
# 1. Generate dataset (with lead times and inventory)
python src/data_gen.py

# 2. Baseline model
python src/baseline.py

# 3. Quantile model (walk-forward split)
python src/quantile_model.py

# 4. Evaluation report + calibration plot
python src/evaluate.py

# 5. Dispatch simulation
python src/dispatch.py

# 6. Reorder simulation
python src/reorder.py

# 7. Shock detection + adjustment
python src/shock_detector.py

# 8. Hurdle model comparison
python src/hurdle_model.py

# 9. Run tests
pytest tests/test_edge_cases.py -v

# 10. Launch dashboard
streamlit run app/dashboard.py
```

## Key Output Files

| File | Description |
|------|-------------|
| `evaluation_report.md` | Full evaluation with all model comparisons |
| `dispatch_comparison.md` | Policy A vs B dispatch safety results |
| `reorder_comparison.md` | Naive vs uncertainty-aware reorder results |
| `data/calibration_plot.png` | Quantile calibration plot |
| `data/predictions_test_set.csv` | Test-period predictions for verification |
| `STAKEHOLDER_VALIDATION.md` | Usability walkthrough template |

## Train/Test Split

All models use a strict walk-forward split: train on data before `2022-07-01`, test on data from `2022-07-01` onward. No leakage.

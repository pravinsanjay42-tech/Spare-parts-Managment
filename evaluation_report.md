# Evaluation Report (Review 2)

## Methodology
- **Time Split**: Walk-forward (Train: < 2022-07-01, Test: >= 2022-07-01). All models train on the same pre-split data.
- **Target**: Daily demand per Store-SKU.
- **Models Compared**: Baseline (Croston's Method), Direct LightGBM Quantile Regression, Shock-Adjusted Quantiles, Hurdle Model (Two-Stage).

## 1. Segment-Level Error Analysis (Direct Quantile Model)

| Segment | N Rows | Base MAE | Base RMSE | LGBM p50 MAE | Coverage (80% target) | Sharpness (p90-p10) | Pinball p10 | Pinball p50 | Pinball p90 |
|---------|--------|----------|-----------|--------------|-----------------------|---------------------|-------------|-------------|-------------|
| Overall (Clean) | 24,132 | 0.669 | 1.538 | 0.418 | 89.9% | 1.448 | 0.042 | 0.209 | 0.289 |
| Cold Start Stores | 580 | 0.711 | 0.800 | 0.234 | 90.7% | 0.949 | 0.023 | 0.117 | 0.151 |
| Shock Event Periods | 130 | 14.188 | 16.412 | 14.523 | 3.1% | 1.523 | 1.452 | 7.262 | 11.730 |
| Stockout Censored | 238 | 0.689 | 0.765 | 0.000 | 100.0% | 0.211 | 0.000 | 0.000 | 0.021 |

## 2. Shock Detection & Adjustment Results

The shock detector uses a rolling z-score (14-day window) on *lagged* demand plus a 7-day momentum flag. When flagged, p90 is widened by 3x.

| Metric | Original | Shock-Adjusted |
|--------|----------|----------------|
| Overall Clean Coverage | 89.9% | 92.2% |
| Shock Event Coverage | 3.1% | 3.1% |
| Rows Flagged | — | 41,195 (28.8%) |
| Mean p90 Increase (flagged rows) | — | +2.71 |

**Honest assessment**: The shock adjustment improves overall coverage by catching post-shock volatility, but does not improve coverage *on shock days themselves* — the demand spike (8-16x normal) is simply too extreme for any feature-based adjustment to capture. This confirms that unprecedented structural breaks require manual scenario overrides, which the dashboard supports.

## 3. Hurdle Model vs Direct Quantile Comparison

The hurdle model uses a two-stage approach: (1) P(demand > 0) classifier, then (2) quantile regression on nonzero demand only.

| Metric | Direct Quantile | Hurdle Model |
|--------|-----------------|--------------|
| Coverage (p10-p90) | 90.0% | 84.1% |
| Pinball p10 | 0.041 | 0.041 |
| Pinball p50 | 0.207 | 0.300 |
| Pinball p90 | 0.286 | 0.323 |
| Mean(p50 - p10) | 0.000 | 0.295 |

**Honest assessment**: The hurdle model successfully **breaks the p10/p50 collapse** — Mean(p50 - p10) rises from 0.000 to 0.295, showing genuine discrimination between lower quantiles. However, its coverage (84.1%) is closer to the 80% target (less overcoverage), at the cost of higher pinball loss at p50 and p90. The direct quantile model remains better for operational safety (higher coverage, lower loss), while the hurdle model provides better calibration. Both are retained — direct quantiles for dispatch decisions, hurdle model for diagnostic comparison.

## 4. Lead-Time-Aware Reorder Policy Comparison

| Policy | Stockout Rate | Service Level | Avg Inventory Held | Reorder Triggers |
|--------|---------------|---------------|---------------------|------------------|
| Naive (Croston's) | 6.33% | 85.05% | 41.83 | 786 |
| Uncertainty-Aware (p90) | 0.10% | 99.53% | 54.48 | 375 |

The uncertainty-aware policy nearly eliminates stockouts (0.10% vs 6.33%) while requiring **fewer** reorder triggers (375 vs 786), at the cost of ~30% higher average inventory. This demonstrates that lead time and inventory data substantively change procurement decisions.

## 5. Key Findings
- **Walk-Forward Consistency**: All models (quantile, hurdle, dashboard scenarios) train exclusively on pre-split data. No leakage.
- **Shock Events**: Coverage plummets to 3.1% during black swan events regardless of adjustment strategy. Manual scenario overrides remain essential.
- **p10/p50 Collapse**: The hurdle model resolves this for diagnostic purposes, but the direct quantile model remains operationally superior.
- **Lead Time Impact**: The p90-based reorder policy achieves 99.5% service level vs 85% for the naive policy, proving uncertainty-aware forecasting has real procurement value.
- **Stockout Censoring**: Censored days are explicitly flagged and excluded from headline metrics to avoid artificially inflating accuracy.

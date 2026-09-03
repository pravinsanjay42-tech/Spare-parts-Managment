# Data Generation Assumptions

## Overview
This synthetic dataset simulates intermittent demand for a spare-parts supplier. The demand behaves irregularly (mostly zeros with occasional spikes), resembling real-world machinery failure rates where parts are needed unpredictably.

## Parameters
- **Dates**: 3 years of daily data (1095 days).
- **Stores**: 15 stores (2 of which are marked as "cold start", containing <30 days of history).
- **SKUs**: 10 distinct spare parts, each with a varying base failure rate.

## Feature Logic
- `store_size`: Randomly assigned multiplier (0.5 to 1.5). Larger stores have proportionally higher failure rates.
- `equipment_age_years`: Generated with a slight upward drift over time. Older equipment increases the baseline failure probability.
- `price`: Mild negative elasticity on demand (simulated randomly around a base price).
- `weather_severity_index`: Uniform random (0 to 1). High severity (>0.8) drastically increases failure rates for certain parts.
- `is_festival`: Randomly assigned binary flag (e.g., 5% probability). Drives demand up for some SKUs and down for others.
- `stockout_flag`: If demand on day T-1 exceeded a threshold (e.g., > 5 units), we assume a stockout occurred. When stockout_flag=1, the true demand on day T is censored (recorded as 0 or artificially low).

## Demand Generation Process (Intermittent)
To guarantee intermittent demand:
1. **Failure Occurrence (Bernoulli Trial)**: A probability $p$ is calculated for each store-sku-day based on seasonality, age, weather, and store size. 
2. **Quantity (Poisson)**: If a failure occurs, the actual demand quantity is drawn from a Poisson distribution with a rate $\lambda$. 
This results in a Zero-Inflated Poisson-like distribution.

## Edge Cases (Injected)
- **Shock Events**: 3 explicitly modeled days where demand spikes 5-10x the normal rate due to a simulated "catastrophe." Marked by the `is_shock_event` flag.
- **Cold Starts**: 2 specific stores have their data truncated, leaving only the most recent 28 days of history. Marked by the `is_cold_start_store` flag.

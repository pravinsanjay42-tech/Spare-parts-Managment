# UncertainSpares (Review 1 Scope)

A probabilistic demand forecasting MVP designed for intermittent spare-parts demand. 
This project demonstrates how uncertainty-aware forecasting (Quantile Regression) can out-perform traditional point-forecast baselines in highly volatile environments.

## Scope for Review 1 (Core Forecasting Engine)

This milestone focuses exclusively on generating the synthetic dataset and building the core forecasting models. The dispatch simulator, tests, and planner dashboard are under development and will be included in future reviews.

1. **Synthetic Data Generator** (`src/data_gen.py`): Generates 3 years of daily store-SKU intermittent demand data. Injects features like weather severity, equipment age, festivals, and explicitly modeled "black swan" shock events and stockout censoring.
2. **Baseline Model** (`src/baseline.py`): A traditional point-forecast baseline using Croston's Method, specifically designed for intermittent demand.
3. **Uncertainty-Aware Model** (`src/quantile_model.py`): A LightGBM Quantile Regressor predicting p10, p50, and p90 demand bounds to explicitly quantify right-tail risk.
4. **Evaluation Engine** (`src/evaluate.py`): Initial backtest logic.

## Setup & Installation

1. Ensure Python 3.10+ is installed.
2. Clone this repository and navigate into the root directory.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run 

Run the following commands in order from the project root to reproduce the pipeline:

```bash
# 1. Generate the dataset (Assumptions documented in DATA_ASSUMPTIONS.md)
python src/data_gen.py

# 2. Train the baseline model (Croston's Method)
python src/baseline.py

# 3. Train the uncertainty model (LightGBM Quantiles)
python src/quantile_model.py

# 4. Generate the initial Evaluation metrics
python src/evaluate.py
```

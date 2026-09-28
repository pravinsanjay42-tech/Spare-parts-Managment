# Reorder Policy Comparison

## Evaluated Policies

1. **Policy 1: Naive Baseline (Croston's)**: Reorder point $ROP = \hat{y}_{\text{croston}} \times L$.
2. **Policy 2: Croston + Safety Stock**: $ROP = \hat{y}_{\text{croston}} \times L + z \times \sigma \times \sqrt{L}$, where $z = 1.28$ (standard safety stock factor corresponding to ~90% target non-stockout probability under Gaussian assumptions) and $\sigma$ is the historical standard deviation of daily demand per store-SKU.
3. **Policy 3: Uncertainty-Aware (p90)**: Reorder point $ROP = p90 \times L$, where $p90$ is the dynamically estimated 90th percentile demand bound from LightGBM quantile regression.

## Performance Comparison (Walk-Forward Test Period)

| Policy | Stockout Rate (%) | Service Level (%) | Average Inventory Held | Number of Reorder Triggers |
|--------|-------------------|-------------------|------------------------|----------------------------|
| **Policy 1: Naive (Croston's)** | 6.33% | 85.05% | 41.83 | 786 |
| **Policy 2: Croston + Safety Stock (z=1.28)** | 1.69% | 94.28% | 44.18 | 572 |
| **Policy 3: Uncertainty-Aware (p90)** | **0.10%** | **99.53%** | 54.48 | **375** |

## Key Findings & Reorder Order-Quantity Fairness Analysis

### 1. Fair and Consistent Order-Quantity Logic
All three policies use the exact same replenishment rule structure:
$$\text{If } (\text{On-Hand} + \text{In-Transit}) < ROP \implies \text{Order Quantity } Q = ROP$$
No policy is given an artificial quantity multiplier or favored batch rules. The difference in operational behavior stems entirely from the **statistical definition of the reorder point $ROP$**.

### 2. Why Does the Uncertainty-Aware (p90) Policy Place Fewer Orders than Naive?
- **The Naive Churn Trap**: The naive Croston forecast predicts an average daily demand of fractional units (e.g., 0.3 parts/day). Over an 8-day lead time, its $ROP$ is only $\approx 2.4$ units. Because it orders in tiny batch quantities ($Q \approx 2.4$), any single lumpy demand spike (e.g., 3–5 parts) immediately wipes out the newly arrived stock. This triggers an unending cycle of frequent, panicked reorders (**786 orders placed**) while still suffering a **6.33% stockout rate**.
- **p90 Batching Efficiency**: The uncertainty-aware model reflects the right-tail risk ($p90 \approx 2.0$), yielding $ROP \approx 16$ units. Each replenishment order arrives with sufficient buffer to absorb stochastic bursts without immediately re-triggering procurement. As a result, the p90 policy places only **375 orders** (a 52% reduction in purchasing transactions) while delivering a near-perfect **99.53% service level**.
- **Croston + Safety Stock Middle Ground**: Adding traditional Gaussian safety stock ($z=1.28$) improves service level from 85.05% to 94.28% and cuts orders from 786 to 572. However, because intermittent demand violates Gaussian normality (having heavy right skew and zero-inflation), traditional safety stock still yields 16x more stockouts than the quantile-derived p90 policy (1.69% vs 0.10%).

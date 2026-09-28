# Problem Analysis: The Illusion of Certainty in Intermittent Demand

## The Core Thesis
In intermittent spare-parts environments, **point forecasts hide uncertainty, leading planners to over-trust a single number**. 
When demand is zero 85% of the time, the statistical "expected value" (p50 or mean) is almost always zero or a fractional number close to zero. If a planner only sees this single number, they are flying blind to the right-tail risk (the small but critical probability of a massive spike). 

When black-swan shocks occur, point forecasts fail silently. The planner is entirely unprepared, leading to stockouts, massive operational strain, and unsafe workload environments.

## Evidence from the Data

### 1. The Baseline Flatline
Our evaluation of Croston's Method (the industry standard for intermittent demand) perfectly illustrates this illusion. Croston's method produced a completely flatlined point forecast slightly above zero. While mathematically correct for minimizing average error (MAE = 0.698), it provides absolutely zero operational value to a planner trying to prepare for a sudden spike of 200 units during a heatwave. It is a single number that is almost always wrong.

### 2. Calibration Failures During Shocks
During our synthetic black-swan shock event, the evaluation showed that the models' target coverage plummeted from ~90% down to **4.6%**. 
This is a critical finding: **Machine learning models cannot reliably predict unprecedented structural breaks from historical features alone**. 
If a planner relies solely on the model's output without the ability to inject human judgment via scenario overrides, they will fail spectacularly during these rare events.

### 3. The True Cost of Point Forecasts (Dispatch Overtime)
In the Dispatch Simulator, we tested **Policy A (Naive Baseline)**, which blindly attempts to fulfill the exact expected point forecast plus any open backorders immediately.
During the shock event, the backlog exploded to 1,700+ units. Because Policy A trusts the naive mandate to clear the backlog instantly, it forced a dispatch effort of **1,811 orders in a single day** against a physical warehouse constraint of 80/day.
This translates to **1,731 unsafe workload violations**, requiring massive mandatory overtime, warehouse chaos, and likely physical injury. Point forecasts do not just cause stockouts; they break operational constraints.

## The Solution
By shifting from point forecasts to **probabilistic uncertainty bands** (p10 to p90), planners can see the tail-risk. By implementing a **Workload-Constrained Dispatch Policy (Policy B)**, the system honors the physical limits of the warehouse (0 violations, capped at 80/day), organically paying down the shock backlog over 51 days while maintaining a safe, sustainable operation.

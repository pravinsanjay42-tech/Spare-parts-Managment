# Stage 5: Workload-Constrained Dispatch Simulator (Including Shock Event)

## Configuration
- **Network Avg Daily Demand**: 55.1 parts
- **Shock Event Included**: A black swan event occurs on 2022-11-05, driving demand > 1800 parts.
- **Metric Definition: Cumulative Backorder-Days**: The sum across all days of units still outstanding at the end of each day. (A single part backordered for 3 days counts as 3 backorder-days, representing the total holding/delay cost).
- **Policies**: 
  - (A) Naive Point-Forecast: Forces drivers to clear backlogs instantly, causing massive unsafe overtime.
  - (B) Uncertainty-Aware (p90): Capped strictly; defers excess demand and recovers over time without overtime.

## Results: Base Cap (80 parts/day)
| Policy | Unsafe Assignments | Cumulative Backorder-Days | Shock Event Recovery Time |
|--------|--------------------|---------------------------|---------------------------|
| (A) Naive | 1835 | 2347 | 1 days |
| (B) p90 | **0** | 41445 | 53 days |

## Results: Sensitivity Test - Tight Cap (60 parts/day)
| Policy | Unsafe Assignments | Cumulative Backorder-Days | Shock Event Recovery Time |
|--------|--------------------|---------------------------|---------------------------|
| (A) Naive | 1944 | 2347 | 1 days |
| (B) p90 | **0** | 72153 | 56 days |

## Conclusion
During the massive shock event, Policy A blindly forces drivers into **thousands of unsafe overtime assignments** in a single day to clear the backlog instantly (recovering in 1 days). This is mathematically unfeasible in the real world.

Policy B, conversely, enforces the hard safety limit (0 unsafe assignments). As a result, it honestly absorbs a massive backlog during the shock event and slowly pays it down over **53 days** using spare capacity. Under the tighter cap (60/day), Policy B incurs significantly more backorders (72153) and takes longer to recover (56 days), proving it doesn't just magically solve capacity issues without cost—it makes an honest trade-off: **sacrificing service speed during black swan events to guarantee zero worker overloads.**

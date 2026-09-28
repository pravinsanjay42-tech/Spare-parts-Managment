# Demo Script (~3 Minutes)

## 1. Problem Statement (0:00 - 0:30)
"Welcome to UncertainSpares. In intermittent spare-parts supply chains, demand is exactly zero 85% of the time. When planners rely on traditional point-forecasts—like Croston's method—they are handed a single number that is almost always a flatline near zero. This creates an illusion of certainty. When a black swan event hits—like a heatwave causing sudden, massive part failure—that point forecast fails silently. Planners are caught completely off guard, leading to massive stockouts and warehouse chaos. Today, we're shifting from predicting a single number to predicting *uncertainty*."

## 2. Dashboard Walkthrough (0:30 - 1:30)
"Let's look at the Dashboard. First, notice the top KPI bar and the Red Alert Banner. The system has automatically triaged our entire network, identified that 23 SKUs are breaching critical safety stock limits in their right-tail forecast, and triggered a global alert. 

Scroll down to the **Network Risk Overview**. Instead of forcing a planner to look at thousands of flat lines, we sort every SKU by its uncertainty width (p90 minus p10). The rule-engine color-codes the action: Order Now, Monitor, or Normal. The planner knows exactly where to focus.

If we jump to the **Forecast Viewer**, we can look at one of these at-risk SKUs. The green expected line is flat, but look at the shaded uncertainty band—the p90 upper bound is tracking the real risk. Now, models can't predict unprecedented shocks. That requires human judgment. If a planner knows a heatwave is coming, they click the 'Heatwave' scenario toggle. Instantly, the UI applies an illustrative multiplier, visibly blowing up that uncertainty band so the planner can physically see the tail-risk and order early."

## 3. Evaluation & Honest Limitations (1:30 - 2:15)
"Let's look at the hard data in the **Calibration & Evaluation** tab. We built a LightGBM quantile regression model. Overall, we achieved a 39% improvement in Mean Absolute Error over the baseline. But we have to be honest about two limitations:

First, **Lower Quantile Collapse**. Because 85% of our data is exactly zero, the model's p10 and p50 predictions mathematically collapse to zero. The entire operational value of this model lives in the p90 right-tail bound.
Second, **Shock Event Miscalibration**. During shock events, our coverage drops to less than 5%. Machine learning cannot predict unprecedented structural breaks from historical features alone. That is exactly why the scenario toggle exists—to merge model bounds with human intuition."

## 4. Dispatch Safety Comparison (2:15 - 3:00)
"Finally, predicting the risk is only half the battle; we have to dispatch it safely. We built a discrete-event simulator capping the warehouse at 80 dispatches a day. 
If we use a legacy, naive policy that just tries to clear all backorders instantly based on the point-forecast, the black-swan shock forces the warehouse to attempt over 1,800 dispatches in a single day. That results in 1,700 unsafe workload violations—meaning mandatory overtime, chaos, and injury.
By switching to our Uncertainty-Aware Policy, we cap the daily dispatch strictly at 80. We accept 39,000 cumulative backorder-days as the shock organically clears over 51 days, but we incur **zero workload violations**. We trade unavoidable backorders for a sustainable, safe operation."

## 5. Wrap-Up (3:00 - 3:10)
"By quantifying uncertainty and respecting operational constraints, we stop fighting intermittent demand and start managing its risk. Thank you."

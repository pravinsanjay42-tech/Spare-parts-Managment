# Workflow Map: Legacy vs. Uncertainty-Aware

## Legacy Workflow: The Point-Forecast Trap

1. **Review Dashboard**: Planner looks at the daily forecast report.
2. **Read Single Number**: Planner sees a point forecast (e.g., "Expected demand: 0.8 units").
3. **Implicit Trust**: Planner assumes the model "knows best" and ignores external context (impending heatwave).
4. **Action**: Planner orders 1 unit (or 0 units) based on the flatline expected value.
5. **Shock Occurs**: Demand spikes to 200 units.
6. **Chaos / Firefighting**: Massive stockout. Warehouse is forced into severe overtime to clear the emergency backlog, violating physical dispatch constraints.

---

## New Workflow: The Uncertainty-Aware Planner

1. **Triage by Risk**: Planner opens the Network Risk Overview table, which automatically sorts all SKUs by uncertainty width (p90 - p10).
2. **Identify Exceptions**: The rule-engine has automatically color-coded the SKUs breaching safety-stock limits in the right-tail (🔴 Order Now). The planner ignores the 85% of SKUs that are Green.
3. **Investigate Context**: Planner clicks into a Red SKU on the Forecast Viewer. They see the p50 is near zero, but the p90 right-tail is ballooning due to an aging equipment fleet in the region.
4. **Inject Human Judgment (Scenario Check)**: The planner knows a severe heatwave is hitting that region next week (data the model hasn't seen yet). They toggle the **Heatwave Scenario**.
5. **Observe Tail Risk**: The dashboard applies the multiplier, explicitly showing the planner how wide the p90 tail risk could get during the heatwave.
6. **Workload-Safe Action**: Planner orders up to the p90 limit early. The automated dispatch system (Policy B) sequences these orders, strictly capping warehouse workload at 80 dispatches/day to ensure a safe operational environment.

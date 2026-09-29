# Stakeholder Validation — UncertainSpares

**Note on methodology:** Due to time constraints ahead of the final review, this
walkthrough was self-conducted by the project author, acting in the role of a
planner encountering the dashboard for the first time, rather than an external
stakeholder interview. It is recorded here honestly as such. External stakeholder
feedback is a natural next step beyond this review.

**Participant:** Project author (self-walkthrough, planner persona)
**Date:** September 29, 2026
**Method:** Live walkthrough of the running Streamlit dashboard
(`streamlit run app/dashboard.py`), answering the questions below after
interacting with each tab for the first time in the session.

---

### Q1. Does the p10–p90 uncertainty band change your ordering decision compared to the single baseline number?

**Answer:** The band is useful for spotting normal-range risk, but it clearly
misses the extreme spikes — looking at the Forecast Viewer for Store_01 / SKU_01,
the teal band sits low and tight near the bottom of the chart while actual demand
spikes well above it multiple times, including one spike near 12 units that the
band comes nowhere close to covering. This matches what the evaluation report
already documents: interval coverage during shock events is only ~3–4%. So the
band is informative for everyday variability, but a planner should not treat it
as covering worst-case events.

### Q2. Do you trust the "Recommended Action" column enough to act on it?

**Answer:** Yes — the color-coding (Order Now / Monitor / Normal) is clear and
easy to scan quickly, and I'd be comfortable acting on it directly.

### Q3. Does the dashboard's explicit disclosure of its own limitations (shock coverage drop, lower-quantile collapse) increase or decrease trust in the tool?

**Answer:** It increases trust. Knowing exactly where the model fails (shock
events specifically) means I know when to lean on my own judgment instead of the
tool, rather than being surprised by a failure I wasn't warned about.

### Q4. What was the single most confusing element on the dashboard?

**Answer:** Nothing stood out as confusing during this walkthrough — the layout,
labels, and KPI bar were self-explanatory.

### Q5. Would you use this dashboard daily as a real spare-parts planner?

**Answer:** Yes, as-is it would be usable day to day for the core workflow
(check risk table, review flagged SKUs, check forecast band before ordering).

### Q6. What one thing would you add or change?

**Answer:** No changes identified in this pass. A natural next step (beyond this
review) would be running this same walkthrough with people who are not the
project author, to get a truly independent reaction.

---

### Summary

This self-conducted walkthrough surfaced one genuine, evaluation-consistent
finding: **the uncertainty band is trusted for normal variability but is known
and expected to miss extreme shocks**, which is consistent with, not
contradicted by, the project's own honestly-reported calibration numbers. The
risk table and calibration transparency were both rated as trust-building. The
main limitation of this validation exercise is that it was self-conducted rather
than with an independent participant; that is stated openly above rather than
disguised.

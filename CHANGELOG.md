# Dashboard update — 24 October 2026

- Current nine-channel demonstration and labels, including Google Video, Shopping and Demand Gen; Smart excluded from the current fitted specification.
- All three original tabs retained: data summary/validation, model results, budget optimizer.
- Modeled-channel spend scope, OOS conditional evaluation, complete holdout periods and interval types made explicit.
- Signed component totals, media contributions, CPA plots, saturation extrapolation and draw-normalized adstock retained and checked.
- Headline optimization now compares with a feasible equal-budget reference. Historical-mix and historical-average comparisons remain separate.
- Total-order forecasts retain intercept, discounts, trend and seasonality. Configurable horizon, bounds, percentage moves, Search protection and budget cap retained.
- New notebook export cell and helper include current channels, OOS metrics/input quality and paired fitted-response checkpoints.
- CSV downloads contain full-precision allocation and reference results; budget bars display planning-period totals.

## Validation scope

Automated checks cover app rendering and optimizer widgets, constrained allocation, carryover math, chart calculations, data rejection and export/load round trips. Tests use synthetic data and a notebook-model adapter. The client's newly refitted nine-channel posterior must be exported from their notebook before deployment; it is not included in this source package.

Runtime: Python 3.12, dependencies pinned in requirements.txt. No application has been deployed by this package.

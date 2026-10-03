# Ecomignite MMM dashboard

A Streamlit dashboard for the current weekly Orders MMM, its saved validation results, and forward budget allocation. Python 3.12 is the tested runtime. The app entry point is `streamlit_app.py`.

## Included views

- **Data summary & validation:** model period and weekly granularity; orders, revenue, and channel-spend totals; KPI lines, channel lines, stacked weekly spending, ordered channel totals, and exported missingness/imputation information.
- **Model results:** within-sample and holdout fit plots and metrics; all-component and marketing-only contributions; spend/effect shares, modeled CPA, Hill saturation, and normalized geometric adstock.
- **Budget optimizer:** future horizon, historical reference window, total budget cap, percentage-change limits, weekly channel minimums/maximums, Google Search protection, future discount assumption, and channel activation. It produces allocation/CPA tables, CSV downloads, grouped budget bars, and future order forecasts.

## Model data

The source package contains a clearly labelled synthetic demo. It does **not** contain Oliver's live notebook variables, fitted posterior, customer records, or credentials. The dashboard must receive an export of the actual fitted model before displaying actual Ecomignite results.

`notebook/export_dashboard_bundle.py` and `notebook/export_cell.py` provide that export bridge. The editable notebook cell binds the full-period fitted model, the model dataframe, anchored control references, optional source-validation frames, and the existing holdout results. It exports a folder and ZIP with CSV/JSON/numeric NPZ files. No model refit or pickle is needed.

The app accepts this ZIP using **Upload model bundle**. A folder at `data/model_bundle/` is also supported. Uploads remain session inputs; they are not written into the source repository. Actual model bundles are excluded by `.gitignore`; the app runs the synthetic demo when no repository bundle is supplied. Source-data coverage is reported as unavailable when its audit fields were not exported. Missing holdout results are never reconstructed from full-model fitted values.

## Budget calculation

The planning period begins on the Monday after the last fitted week. One constant weekly allocation is repeated over the selected horizon. Finite-lag carryover starts from the actual fitted-period spending history. The solver maximizes expected modeled media orders across paired posterior draws, subject to the period budget ceiling and intersected channel constraints.

Default channel moves are ±50% of recent average weekly spending. Search protection prevents cuts below its recent average; it permits increases. The default reference window equals the planning horizon. A channel with zero reference spend stays at zero under percentage bounds; removing its percentage limit allows reopening it within other bounds.

The **same-budget historical-mix reference** preserves historical channel proportions at the optimized total budget and evaluates that reference over the same future weeks. It may lie outside newly imposed channel bounds. A separate feasible same-budget reference is included in the scenario details. Comparisons to continued historical spending may also contain a budget change and are labelled separately.

Total-order forecasts add the intercept, annual Fourier seasonality, anchored future trend, and selected discount assumption. Percentage KPI uplift uses the total-order reference, while modeled CPA divides channel spend by modeled channel orders. Intervals represent posterior uncertainty in the expected response, not future observation noise. Gains within the horizon exclude carryover after its end. Multistart SLSQP finds the best feasible allocation encountered; it does not guarantee a global optimum.

## Files

- `streamlit_app.py`: app entry point and three tabs.
- `ecomignite/data.py`: validated bundle loading and synthetic demo.
- `ecomignite/charts.py`: all Plotly views.
- `ecomignite/engine.py`: forward response and constrained allocation.
- `notebook/`: export helper and editable notebook cell.
- `tests/`: allocation, loader, export, and Streamlit integration checks.
- `requirements.txt`: tested app dependencies; no PyMC/JAX is needed on the server.
- `requirements-dev.txt`: additional test dependencies.
- `.streamlit/config.toml`: app theme and upload settings.

This package contains the application files only. It does not create a GitHub repository or deploy an app.

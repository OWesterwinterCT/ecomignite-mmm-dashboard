# Ecomignite MMM dashboard

A Streamlit dashboard for the current weekly Orders MMM, its saved validation results, and forward budget allocation. Python 3.12 is the tested runtime. The app entry point is `streamlit_app.py`.

## October 2026 update

The current notebook models nine channels: Meta Conversion, Meta Engagement, Meta Traffic, Google Search, Google Performance Max, Google Display, Google Video, Google Shopping, and Google Demand Gen. Google Smart remains outside the fitted model. Google Video can include YouTube and video partners; other Google campaign types can also deliver on YouTube.

All calculations follow the channels stored in the fitted export. Older bundles can still load, but the app displays a specification notice when they differ from the current nine-channel model. Updating application files alone does not create a new fitted model. See `UPDATE_AND_DEPLOY.md` for the export and deployment sequence.

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

The headline **feasible equal-budget reference** starts with historical proportions rescaled to the optimized budget and projects that allocation onto the same channel bounds. It uses the same total spend and future weeks as the optimized scenario. This agrees with the revised notebook's reference. The unconstrained historical-mix comparison remains available separately, along with continued historical-average spending. Comparisons to continued historical spending may also contain a budget change and are labelled separately.

Total-order forecasts add the intercept, annual Fourier seasonality, anchored future trend, and selected discount assumption. Percentage KPI uplift uses the total-order reference, while modeled CPA divides channel spend by modeled channel orders. Intervals represent posterior uncertainty in the expected response, not future observation noise. Gains within the horizon exclude carryover after its end. Multistart SLSQP finds the best feasible allocation encountered; it does not guarantee a global optimum.

The notebook's media-order uplift percentage and the dashboard's total-order percentage use different denominators. Absolute scenario differences agree when inputs and posterior draws match. For closer numerical agreement, match the historical reference window, planning horizon, caps and bounds, optimization draws, and exported posterior draw count. The app defaults to a historical reference window as long as the planning horizon; the notebook example uses eight reference weeks.

The default 12-week OOS evaluation uses a separate training-only model and all holdout weeks. It conditions on realized media spend and discounts. OOS predictive intervals include observation noise, while in-sample fitted-mean and optimizer intervals do not. Missing-spend imputation flags describe estimated inputs; they are not proof that the upstream source is complete. Unmodeled promotions, CRM/email activity and external events are not forecast explicitly.

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
- `UPDATE_AND_DEPLOY.md`: notebook export, local replacement and deployment steps.

Run checks with `python -m pip install -r requirements-dev.txt` followed by `python -m pytest -q` from this directory. The suite uses synthetic bundles and a synthetic notebook-model adapter; it does not refit or certify a client's actual MMM.

This package contains the application files only. It does not create a GitHub repository or deploy an app.


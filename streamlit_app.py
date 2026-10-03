"""Ecomignite: exported MMM results and forward budget planning."""
from pathlib import Path
from copy import copy
import hashlib
import json

import numpy as np
import pandas as pd
import streamlit as st

from ecomignite import charts
from ecomignite.data import load_bundle, make_demo_bundle, fit_metrics, validation_summary
from ecomignite.engine import optimize_budget

st.set_page_config(page_title="Ecomignite | Marketing Measurement", page_icon="◈", layout="wide")
st.markdown("""<style>
.stApp {background:#f7f9fc;color:#152b4f;}
.block-container {padding-top:2rem;padding-bottom:3rem;max-width:1550px;}
[data-testid="stMetric"] {background:white;border:1px solid #e4eaf3;border-radius:12px;padding:16px;}
[data-testid="stMetricLabel"] {color:#63718a;}
[data-testid="stMetricValue"] {font-size:1.8rem;}
[data-testid="stTabs"] button {font-weight:600;}
div[data-testid="stPlotlyChart"] {background:white;border-radius:12px;padding:6px;}
.eyebrow {font-size:12px;font-weight:700;letter-spacing:.13em;color:#2563eb;}
</style>""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent
REPOSITORY_BUNDLE = ROOT / "data" / "model_bundle"


@st.cache_data(show_spinner=False)
def demo_bundle():
    return make_demo_bundle()


def show_chart(figure, key):
    st.plotly_chart(figure, width="stretch", key=key,
                    config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}})


def csv_bytes(frame):
    return frame.to_csv(index=False).encode("utf-8-sig")


def number(value, decimals=0):
    return f"{value:,.{decimals}f}" if value is not None and np.isfinite(value) else "Not available"


def fit_explanation(metrics, oos=False):
    if not metrics:
        return "No validation results were supplied."
    r2, error = metrics["R²"], metrics["WAPE (%)"]
    pattern = ("follows most of the observed weekly variation" if r2 >= .8 else
               "captures part of the observed weekly variation" if r2 >= .5 else
               "has limited agreement with the observed weekly variation")
    scope = "held-out weeks" if oos else "weeks used for estimation"
    return (f"The model {pattern} across {metrics['Weeks']} {scope}. "
            f"Absolute error totals {number(error, 1)}% of observed orders (WAPE), "
            f"with an average absolute error of {number(metrics['MAE'], 1)} orders per week. "
            + ("These predictions assess performance on weeks excluded from that model's training."
               if oos else "This describes historical fit; the holdout results provide a separate forecasting check."))


def render_fit(frame, title, key, oos=False, interval_kind="expected_mean", interval_level=.90):
    st.subheader(title)
    if frame.empty:
        st.info("OOS results have not been exported. The notebook export cell accepts your existing holdout predictions.")
        return
    metrics = fit_metrics(frame)
    cards = st.columns(5)
    for col, label, decimals in zip(cards, ["R²", "MAPE (%)", "WAPE (%)", "RMSE", "MAE"], [3,1,1,1,1]):
        col.metric(label, number(metrics[label], decimals))
    if is_demo:
        st.write("These synthetic observations and prediction curves illustrate how fit metrics will be presented. "
                 "Actual model assessment appears after loading the notebook export.")
    else:
        st.write(fit_explanation(metrics, oos))
    show_chart(charts.fit_chart(frame, title), key + "_chart")
    with st.expander("Fit metrics and residuals"):
        st.dataframe(pd.DataFrame({"Metric": list(metrics), "Value": list(metrics.values())}), hide_index=True)
        show_chart(charts.residual_chart(frame), key + "_residual")
    interval_label = "posterior predictive" if interval_kind=="posterior_predictive" else "expected-mean credible"
    has_bands={"lower","upper"}.issubset(frame) and frame[["lower","upper"]].notna().any().any()
    band_text=(f"Exported bands are {100*interval_level:.0f}% {interval_label} intervals."
               if has_bands else "Interval bands were not supplied in this export.")
    st.caption("MAPE excludes weeks with zero observed orders. Bias is observed minus predicted. " + band_text)


# Data source selection. Uploaded bundles stay in the current app session.
sources = (["Repository model"] if (REPOSITORY_BUNDLE / "manifest.json").exists() else []) + ["Synthetic demo", "Upload model bundle"]
st.sidebar.title("Ecomignite")
source = st.sidebar.selectbox("Model data", sources)
try:
    if source == "Synthetic demo":
        bundle = demo_bundle()
    elif source == "Repository model":
        bundle = load_bundle(REPOSITORY_BUNDLE)
    else:
        uploaded = st.sidebar.file_uploader("Notebook model export (.zip)", type=["zip"])
        if uploaded is None:
            st.info("Upload the ZIP produced by the included notebook export cell.")
            st.stop()
        raw = uploaded.getvalue()
        signature = hashlib.sha256(raw).hexdigest()
        if st.session_state.get("uploaded_signature") != signature:
            st.session_state["uploaded_bundle"] = load_bundle(raw)
            st.session_state["uploaded_signature"] = signature
        bundle = st.session_state["uploaded_bundle"]
except (ValueError, KeyError, OSError, TypeError) as error:
    st.error(f"The model export could not be loaded: {error}")
    st.stop()

m = bundle.manifest
channels = m["channels"]
labels = m.get("channel_labels", {})
target = m.get("target_column", "eligible_orders")
target_label = m.get("target_label", "Orders")
is_demo = m.get("kind") == "demo"
weekly = bundle.weekly
st.markdown('<div class="eyebrow">MARKETING MEASUREMENT & PLANNING</div>', unsafe_allow_html=True)
st.title("Ecomignite")
st.write("Understand performance, inspect the model, and plan the next marketing budget.")
if is_demo:
    st.warning("SYNTHETIC DEMO — all numbers, curves, and validation examples are illustrative, not Ecomignite's fitted results.")
st.caption(f"{m.get('model_name', 'Orders MMM')}  ·  Weekly / Monday starts  ·  "
           f"{weekly.date.min():%d %b %Y} — {weekly.date.max():%d %b %Y}  ·  {len(weekly)} modeled weeks")

st.sidebar.caption("Display dates filter summaries and contributions. Optimization always starts after the full fitted period.")
period = st.sidebar.date_input("Display period", value=(weekly.date.min().date(), weekly.date.max().date()),
    min_value=weekly.date.min().date(), max_value=weekly.date.max().date())
if not isinstance(period, (tuple, list)) or len(period) != 2:
    st.info("Select both the start and end of the display period.")
    st.stop()
start, end = [pd.Timestamp(x) for x in period]
visible = weekly.loc[weekly.date.between(start,end)].copy()
if visible.empty:
    st.info("The selected display period contains no weekly records.")
    st.stop()
display_bundle = copy(bundle)
display_bundle.weekly = visible
display_bundle.contributions = bundle.contributions.loc[bundle.contributions.date.between(start,end)].copy()

summary_tab, results_tab, optimizer_tab = st.tabs(["Data summary & validation", "Model results", "Budget optimizer"])

with summary_tab:
    st.subheader("Data at a glance")
    cards = st.columns(4)
    cards[0].metric(target_label, number(visible[target].sum()))
    cards[1].metric("Net revenue", number(visible.net_revenue.sum(), 2) if "net_revenue" in visible else "Not supplied")
    cards[2].metric("Marketing spend", number(visible[channels].sum().sum(), 2))
    cards[3].metric("Displayed weeks", str(len(visible)))
    st.caption("Totals cover the selected display period. Spend and revenue use the currency of the exported data.")
    kpi_options = [target] + (["net_revenue"] if "net_revenue" in visible else [])
    kpis = st.multiselect("KPI timeline", kpi_options, default=[target],
        format_func=lambda c: target_label if c==target else "Net revenue")
    if kpis:
        show_chart(charts.kpi_chart(visible,target,target_label,kpis),"kpi")
    line_view, stacked_view = st.tabs(["Channel spend timeline", "Stacked weekly spend"])
    with line_view:
        show_chart(charts.spend_timeline(visible,channels,labels),"spend_line")
    with stacked_view:
        show_chart(charts.spend_stacked(visible,channels,labels),"spend_stack")
    show_chart(charts.spend_totals(visible,channels,labels),"spend_totals")
    st.subheader("Data validation")
    inputs, flags, reconstruction_gap = validation_summary(bundle)
    st.success(f"The exported model has {len(weekly)} consecutive Monday weeks, finite model inputs, "
               "and all fitted channels included.")
    st.dataframe(inputs,hide_index=True,width="stretch")
    st.caption("Zero spend is retained as zero. Complete modeled columns can include imputed observations.")
    if flags.empty:
        st.info("Raw spend coverage and imputation history were not included in this export. "
                "Finite modeled inputs alone do not establish complete source data.")
    else:
        st.write("Exported coverage and imputation fields")
        st.dataframe(flags,hide_index=True,width="stretch")
    st.caption(f"Maximum contribution reconstruction gap: {reconstruction_gap:,.6f} orders per week. "
               "Validation checks cover the full fitted period.")
    if not bundle.validation_daily.empty:
        with st.expander("Daily source-data validation"):
            st.dataframe(bundle.validation_daily,hide_index=True,width="stretch")
    with st.expander("Weekly model inputs"):
        st.dataframe(visible,hide_index=True,width="stretch")
        st.download_button("Download displayed weekly data",csv_bytes(visible),"ecomignite_weekly_data.csv","text/csv")

with results_tab:
    if is_demo:
        st.caption("The demo fit and holdout curves illustrate the dashboard; no model was trained or validated on this synthetic example.")
    render_fit(bundle.fit_in_sample,"Within-sample model fit","insample",
               interval_kind=m.get("fit_intervals_type","expected_mean"),interval_level=m.get("fit_interval_level",.90))
    render_fit(bundle.fit_oos,"Out-of-sample model fit","oos",oos=True,
               interval_kind=m.get("oos",{}).get("intervals_type",m.get("fit_intervals_type","expected_mean")),
               interval_level=m.get("fit_interval_level",.90))
    oos_metadata = m.get("oos", {})
    if oos_metadata.get("training_end"):
        st.caption(f"Holdout-model training ended {oos_metadata['training_end']}.")
    diagnostics=m.get("diagnostics",{})
    if diagnostics:
        with st.expander("Sampling diagnostics"):
            st.json(diagnostics)
    st.divider()
    st.subheader("All model components")
    c = display_bundle.contributions
    effects = c.groupby(["effect","group"],as_index=False)["mean"].sum().rename(columns={"mean":"Total modeled orders"})
    total_model = effects["Total modeled orders"].sum()
    effects["Share of modeled orders (%)"] = 100*effects["Total modeled orders"]/total_model if total_model>0 else np.nan
    nice = {**labels, **m.get("control_labels",{}), "Intercept":"Baseline / intercept", "Seasonality":"Annual seasonality"}
    effects["Component"] = effects.effect.map(nice).fillna(effects.effect)
    show_chart(charts.contribution_chart(c,m),"all_components")
    st.dataframe(effects[["Component","group","Total modeled orders","Share of modeled orders (%)"]].sort_values("Total modeled orders",ascending=False),hide_index=True,width="stretch")
    st.caption("Contributions cover the selected display period and are signed. Discounts and trend use their anchored model references.")
    st.subheader("Marketing contributions")
    marketing_total=effects.loc[effects.group.eq("Marketing"),"Total modeled orders"].sum()
    a,b=st.columns(2)
    a.metric("Modeled marketing orders",number(marketing_total))
    b.metric("Paid marketing share",f"{100*marketing_total/total_model:.1f}%" if total_model>0 else "Not available")
    show_chart(charts.contribution_chart(c,m,marketing_only=True),"marketing_components")
    st.subheader("Channel efficiency")
    efficiency = charts.efficiency_table(display_bundle)
    show_chart(charts.efficiency_share_chart(efficiency),"efficiency_share")
    show_chart(charts.cpa_chart(efficiency),"cpa")
    st.dataframe(efficiency.rename(columns={"channel_label":"Channel","total_spend":"Total spend",
        "modeled_orders":"Modeled orders","spend_share_pct":"Spend share (%)",
        "effect_share_pct":"Media-order share (%)","modeled_cpa":"Modeled CPA",
        "orders_per_spend":"Orders per spend unit"}).drop(columns=["channel"],errors="ignore"),
        hide_index=True,width="stretch")
    st.caption("Modeled CPA = channel spend / modeled channel orders for the same period. "
               "This differs from ad-platform attribution CPA. Channel estimates rely on the model and its available controls.")
    st.subheader("Response and carryover")
    selected_curve = st.selectbox("Curve detail", ["All channels"]+channels,
                                 format_func=lambda c:labels.get(c,c))
    curve_channels = None if selected_curve=="All channels" else [selected_curve]
    show_chart(charts.saturation_chart(bundle,curve_channels),"saturation")
    st.caption("Hill curves show sustained effective weekly spend with normalized adstock. "
               "The optimizer separately evaluates finite-horizon responses using the actual spend history.")
    show_chart(charts.adstock_chart(bundle,curve_channels),"adstock")
    st.caption("Lag 0 is the spending week. Normalized weights sum to one across the fitted lag window.")

with optimizer_tab:
    st.subheader("Plan the next marketing budget")
    st.write("Optimize a constant weekly allocation from the Monday after the fitted model ends. "
             "Historical spend initializes carryover; baseline, trend, discounts, and seasonality are shared across scenarios.")
    st.caption("The headline uplift compares the optimized allocation with the historical channel mix at "
               "the same total budget and over the same future weeks.")
    col1,col2,col3=st.columns(3)
    horizon=col1.number_input("Planning weeks",min_value=1,max_value=104,value=12,step=1)
    matched_reference=col2.checkbox("Use the same number of historical weeks",value=True)
    reference_weeks=col2.number_input("Historical reference weeks",min_value=1,max_value=len(weekly),
                                    value=min(int(horizon),len(weekly)),step=1,disabled=matched_reference)
    R=int(horizon) if matched_reference else int(reference_weeks)
    automatic_budget=col3.checkbox("Use the historical average budget",value=True)
    mean_budget=float(weekly[channels].tail(min(R,len(weekly))).mean().sum())
    budget_cap=col3.number_input("Total budget cap for the whole period",min_value=0.,
                                value=float(int(horizon)*mean_budget),step=1000.,disabled=automatic_budget)
    col1,col2,col3=st.columns(3)
    max_change=col1.number_input("Maximum spend change (%)",min_value=0.,max_value=1000.,value=50.,step=5.)
    protect_search=col2.checkbox("Protect Google Search from cuts",value=True)
    historical_cap=col3.checkbox("Cap channels at historical weekly maxima",value=True)
    discount_default=float(weekly.discount_rate.tail(min(R,len(weekly))).mean()*100) if "discount_rate" in weekly else 0.
    discount=col1.number_input("Future discount rate (%)",min_value=0.,max_value=100.,
                               value=min(max(discount_default,0),100.),step=.5)
    st.caption("Percentage limits are relative to each channel's average weekly spend in the reference window. "
               "Channel amounts below are weekly; the total cap applies to the whole planning horizon.")
    bounds_frame=pd.DataFrame({"channel":channels,"Channel":[labels.get(c,c) for c in channels],
        "Minimum weekly spend":[None]*len(channels),"Maximum weekly spend":[None]*len(channels),
        "Apply percentage limit":[True]*len(channels)})
    bounds=st.data_editor(bounds_frame,key=f"bounds_{bundle.fingerprint}",hide_index=True,
        disabled=["channel","Channel"],column_config={"channel":None,
          "Minimum weekly spend":st.column_config.NumberColumn(min_value=0.),
          "Maximum weekly spend":st.column_config.NumberColumn(min_value=0.),
          "Apply percentage limit":st.column_config.CheckboxColumn()},width="stretch")
    st.caption("Leave a bound blank for no additional limit. To reopen a channel with zero recent spend, "
               "uncheck its percentage limit and provide an appropriate weekly maximum.")
    with st.expander("Calculation settings"):
        draws=st.number_input("Posterior draws used during optimization",min_value=50,max_value=2000,value=500,step=50)
        starts=st.number_input("Optimization starting points",min_value=1,max_value=20,value=6,step=1)
    channel_bounds={}
    overrides={}
    for _,row in bounds.iterrows():
        channel_bounds[row.channel]={"min":None if pd.isna(row["Minimum weekly spend"]) else float(row["Minimum weekly spend"]),
                                     "max":None if pd.isna(row["Maximum weekly spend"]) else float(row["Maximum weekly spend"])}
        if not row["Apply percentage limit"]:
            overrides[row.channel]=None
    settings={"horizon_weeks":int(horizon),"reference_weeks":R,
        "total_budget_cap":None if automatic_budget else float(budget_cap),"max_spend_change":float(max_change)/100,
        "protect_search":protect_search,"search_channel":"google_search_spend", "cap_at_historical_max":historical_cap,
        "channel_weekly_bounds":channel_bounds,"channel_change_overrides":overrides,
        "future_discount_rate":float(discount)/100,"optimization_draws":int(draws),"n_starts":int(starts),"random_seed":42}
    settings_signature=hashlib.sha256((bundle.fingerprint+json.dumps(settings,sort_keys=True)).encode()).hexdigest()
    references=m.get("control_scaling",{}) or {}
    required_references=set()
    if "discount_rate_scaled" in m.get("controls",[]):
        required_references.add("discount_unit")
    if "linear_trend_scaled" in m.get("controls",[]):
        required_references.update(["trend_origin","trend_span_days"])
    can_optimize=required_references.issubset(references)
    if not can_optimize:
        st.info("The control reference values were not exported. Model results remain available; "
                "include CONTROL_SCALING in the notebook export to enable total-order budget forecasts.")
    if st.button("Run budget optimizer",type="primary",key="run_optimizer",disabled=not can_optimize):
        try:
            with st.spinner("Evaluating future allocations and posterior uncertainty…"):
                result=optimize_budget(bundle,settings)
            st.session_state["optimizer_result"]=(settings_signature,result)
        except (ValueError,RuntimeError,AssertionError,KeyError) as error:
            st.error(str(error))
    stored=st.session_state.get("optimizer_result")
    if stored and stored[0]==settings_signature:
        result=stored[1]
        s=result["summary"]
        gain=s["gain_vs_historical_same_budget"]
        for warning in s.get("warnings",[]):
            st.warning(warning)
        a,b,c=st.columns(3)
        a.metric("Recommended period spend",number(s["optimized_period_budget"],2))
        b.metric("Estimated additional orders",number(gain["orders"],1))
        c.metric("Estimated total-order uplift",f"{gain['pct_total']:+.1f}%" if gain.get("pct_total") is not None else "Not available")
        st.caption(f"Planning: {s['planning_start']} — {s['planning_end']}  ·  Reference: {s['reference_start']} — {s['reference_end']}  ·  "
                   f"90% order-gain interval [{gain['lower_90']:+,.1f}, {gain['upper_90']:+,.1f}]")
        st.write("Historical average and optimized allocation")
        allocation=result["allocation"]
        table_columns={"channel_label":"Channel", "historical_weekly_spend":"Historical avg weekly spend",
            "optimized_weekly_spend":"Optimized weekly spend", "spend_change_pct":"Change (%)",
            "historical_modeled_cpa":"Current modeled CPA", "optimized_modeled_cpa":"Optimized modeled CPA",
            "historical_period_spend":"Current period spend", "optimized_period_spend":"Optimized period spend"}
        allocation_display=allocation[list(table_columns)].rename(columns=table_columns)
        st.dataframe(allocation_display.round(2),hide_index=True,width="stretch")
        st.download_button("Download optimizer table (CSV)",csv_bytes(allocation_display),
            f"ecomignite_budget_{s['horizon_weeks']}weeks.csv","text/csv",key="optimizer_csv")
        show_chart(charts.budget_chart(allocation,m),"optimized_budget")
        show_chart(charts.forecast_chart(result["weekly_forecast"]),"optimized_forecast")
        st.caption("The same-budget reference preserves historical channel proportions and can lie outside new channel constraints. "
                   "Both scenarios use the same future weeks, carryover history, and non-marketing assumptions. "
                   "Both CPA estimates divide future-period spend by modeled future-period media orders; "
                   "the current scenario continues historical average spending. The uplift percentage uses total modeled orders.")
        with st.expander("Scenario details, weekly plan, and optimizer checks"):
            st.json(s)
            st.dataframe(allocation,hide_index=True,width="stretch")
            st.dataframe(result["weekly_plan"],hide_index=True,width="stretch")
            st.dataframe(result["solver_runs"],hide_index=True,width="stretch")
            st.download_button("Download weekly plan",csv_bytes(result["weekly_plan"]),"ecomignite_weekly_budget_plan.csv","text/csv")
            st.download_button("Download forecast",csv_bytes(result["weekly_forecast"]),"ecomignite_budget_forecast.csv","text/csv")
    elif stored:
        st.info("The inputs or model have changed. Run the optimizer again to update the results.")
    else:
        st.info("Set the budget and channel limits, then run the optimizer.")

st.caption("Ecomignite · Bayesian marketing measurement · Exported model results; no model fitting or database credentials required.")

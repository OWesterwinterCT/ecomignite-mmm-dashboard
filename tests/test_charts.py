"""Check chart mathematics and labels independently of the fitted-model demo."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ecomignite import charts


CHANNELS = [
    "meta_conversion_spend", "meta_engagement_spend", "meta_traffic_spend",
    "google_search_spend", "google_pmax_spend", "google_display_spend",
    "google_video_spend", "google_shopping_spend", "google_demand_gen_spend",
]


@pytest.fixture
def bundle():
    dates = pd.date_range("2026-01-05", periods=10, freq="7D")
    spend = np.arange(1, 11)[:, None] * np.arange(1, 10)[None, :]
    weekly = pd.DataFrame(spend, columns=CHANNELS)
    weekly.insert(0, "date", dates)
    weekly["eligible_orders"] = 200.0
    samples, channels = 3, len(CHANNELS)
    post = {
        "alpha": np.broadcast_to(np.array([.05, .4, .85])[:, None], (samples, channels)).copy(),
        "beta": np.broadcast_to(np.array([.1, .3, .9])[:, None], (samples, channels)).copy(),
        "kappa": np.broadcast_to(np.array([.2, .7, 1.5])[:, None], (samples, channels)).copy(),
        "slope": np.broadcast_to(np.array([.8, 1.2, 1.8])[:, None], (samples, channels)).copy(),
        "channel_scale": spend.max(axis=0).astype(float),
        "target_scale": np.asarray(700.0),
    }
    records = []
    for date in dates:
        for i, channel in enumerate(CHANNELS):
            records.append({"date": date, "effect": channel, "group": "Marketing", "mean": i + 1.0})
        records.extend([
            {"date": date, "effect": "Intercept", "group": "Intercept", "mean": 120.0},
            {"date": date, "effect": "linear_trend_scaled", "group": "Context", "mean": -3.0},
            {"date": date, "effect": "Seasonality", "group": "Seasonality", "mean": -1.0},
        ])
    return SimpleNamespace(
        manifest={"channels": CHANNELS, "adstock_l_max": 4, "adstock_normalize": True},
        weekly=weekly, posterior=post, contributions=pd.DataFrame(records),
    )


def test_saturation_uses_paired_draws_and_marks_extrapolation(bundle):
    fig = charts.saturation_chart(bundle)
    lines = [trace for trace in fig.data if trace.mode == "lines"]
    assert len(lines) == 2 * len(CHANNELS)
    assert len([trace for trace in lines if trace.showlegend]) == len(CHANNELS)
    for i, channel in enumerate(CHANNELS):
        solid, dashed = [trace for trace in lines if trace.legendgroup == channel]
        assert solid.line.dash == "solid" and dashed.line.dash == "dash"
        maximum = bundle.weekly[channel].max()
        assert np.max(solid.x) == maximum
        assert np.min(dashed.x) == maximum
        scaled = np.asarray(solid.x)[None, :] / bundle.posterior["channel_scale"][i]
        beta = bundle.posterior["beta"][:, i, None]
        kappa = bundle.posterior["kappa"][:, i, None]
        slope = bundle.posterior["slope"][:, i, None]
        draws = 700 * beta * scaled ** slope / (scaled ** slope + kappa ** slope)
        np.testing.assert_allclose(solid.y, draws.mean(axis=0))
        np.testing.assert_allclose(solid.customdata, np.quantile(draws, [.05, .95], axis=0).T)
        assert solid.y[0] == 0
    assert {trace.name for trace in lines} >= {"Google Video", "Google Shopping", "Google Demand Gen"}
    assert fig.layout.legend.y < 0
    assert "Sustained" in fig.layout.xaxis.title.text
    fig.to_json()


def test_adstock_normalizes_each_draw_before_averaging(bundle):
    fig = charts.adstock_chart(bundle)
    assert len(fig.data) == len(CHANNELS)
    alpha = bundle.posterior["alpha"][:, 0]
    weights = alpha[:, None] ** np.arange(4)
    weights /= weights.sum(axis=1, keepdims=True)
    expected = 100 * weights.mean(axis=0)
    wrong = alpha.mean() ** np.arange(4)
    wrong = 100 * wrong / wrong.sum()
    assert not np.allclose(expected, wrong)
    for trace in fig.data:
        np.testing.assert_allclose(trace.y, expected)
        np.testing.assert_allclose(trace.customdata, np.quantile(weights * 100, [.05, .95], axis=0).T)
        assert np.sum(trace.y) == pytest.approx(100)
    assert fig.layout.legend.y < 0


def test_fit_interval_kind_and_level_are_explicit():
    frame = pd.DataFrame({
        "date": pd.date_range("2026-08-03", periods=12, freq="7D"),
        "observed": np.r_[1000, np.full(11, 100)],
        "predicted": np.full(12, 120), "lower": np.full(12, 80), "upper": np.full(12, 180),
    })
    predictive = charts.fit_chart(frame, "Holdout", "posterior_predictive", .8)
    expected = charts.fit_chart(frame, "Fit")
    assert "80% predictive interval" in [trace.name for trace in predictive.data]
    assert "90% expected-mean credible interval" in [trace.name for trace in expected.data]
    assert len(predictive.data[-1].x) == 12
    assert max(predictive.data[-1].y) == 1000


def test_signed_components_and_marketing_totals(bundle):
    fig = charts.contribution_chart(bundle.contributions, bundle.manifest)
    plotted = {label: value for trace in fig.data for label, value in zip(trace.x, trace.y)}
    assert plotted["Trend"] == -30
    assert plotted["Annual seasonality"] == -10
    assert sum(plotted.values()) == pytest.approx(bundle.contributions["mean"].sum())
    media = charts.contribution_chart(bundle.contributions, bundle.manifest, marketing_only=True)
    assert sum(len(trace.x) for trace in media.data) == len(CHANNELS)
    assert "selected period" in fig.layout.yaxis.title.text


def test_efficiency_handles_zero_orders_and_mismatched_periods(bundle):
    bundle.contributions.loc[bundle.contributions.effect.eq(CHANNELS[0]), "mean"] = 0.0
    bundle.contributions.loc[bundle.contributions.effect.eq(CHANNELS[1]), "mean"] = 0.0
    bundle.weekly[CHANNELS[1]] = 0.0
    table = charts.efficiency_table(bundle).set_index("channel")
    assert np.isinf(table.loc[CHANNELS[0], "modeled_cpa"])
    assert np.isnan(table.loc[CHANNELS[1], "modeled_cpa"])
    fig = charts.cpa_chart(table.reset_index())
    assert {annotation.text for annotation in fig.layout.annotations} == {"Unbounded", "n/a"}
    assert len(fig.data[0].x) == len(CHANNELS)
    assert fig.layout.legend.y < 0
    bundle.weekly = bundle.weekly.iloc[:-1]
    with pytest.raises(ValueError, match="same period"):
        charts.efficiency_table(bundle)


def test_budget_period_totals_and_zero_reference_labels(bundle):
    allocation = pd.DataFrame({
        "channel": CHANNELS[:3], "historical_weekly_spend": [10., 0., 0.],
        "optimized_weekly_spend": [15., 5., 0.], "historical_period_spend": [40., 0., 0.],
        "optimized_period_spend": [60., 20., 0.], "historical_modeled_cpa": [2., 0., np.nan],
        "optimized_modeled_cpa": [1.5, 2., np.nan],
    })
    fig = charts.budget_chart(allocation, bundle.manifest, period_totals=True)
    assert sum(fig.data[0].y) == 40
    assert sum(fig.data[1].y) == 80
    assert {annotation.text for annotation in fig.layout.annotations} == {"+50.0%", "New", "No change"}
    assert "horizon" in fig.layout.yaxis.title.text
    assert "Projected media CPA" in fig.data[1].hovertemplate


def test_forecast_distinguishes_feasible_reference_from_historical_mix():
    frame = pd.DataFrame({
        "date": pd.date_range("2026-10-05", periods=3, freq="7D"),
        "current_total_orders": [100, 100, 100],
        "reference_total_orders": [110, 111, 112],
        "feasible_reference_total_orders": [120, 121, 122],
        "optimized_total_orders": [130, 131, 132],
        "optimized_total_orders_lower_90": [80, 81, 82],
        "optimized_total_orders_upper_90": [180, 181, 182],
    })
    fig = charts.forecast_chart(frame)
    traces = {trace.name: trace for trace in fig.data}
    np.testing.assert_array_equal(traces["Feasible equal-budget reference"].y, [120, 121, 122])
    assert "Historical mix, same budget" not in traces
    assert "Optimized: 90% expected-mean credible interval" in traces
    expanded = charts.forecast_chart(frame, include_historical_mix=True)
    assert "Historical mix, same budget" in [trace.name for trace in expanded.data]
    legacy = charts.forecast_chart(frame.drop(columns="feasible_reference_total_orders"))
    assert "Historical mix, same budget" in [trace.name for trace in legacy.data]


def test_spend_plots_include_all_nine_channels(bundle):
    for fn in [charts.spend_timeline, charts.spend_stacked, charts.spend_totals]:
        fig = fn(bundle.weekly, CHANNELS, {})
        assert len(fig.data) == len(CHANNELS)
        assert {trace.name for trace in fig.data} >= {"Google Video", "Google Shopping", "Google Demand Gen"}
        assert fig.layout.legend.y < 0
    totals = charts.spend_totals(bundle.weekly, CHANNELS, {})
    assert sum(float(trace.y[0]) for trace in totals.data) == pytest.approx(bundle.weekly[CHANNELS].sum().sum())


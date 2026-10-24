"""Plotly views for the exported Ecomignite MMM and forward optimizer.

All calculations preserve the bundle's channel order. Marketing effects are
model estimates; the figures do not imply experimental proof of incrementality.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.special import expit


NAVY = "#152B4F"
BLUE = "#2563EB"
TEAL = "#0D9488"
ORANGE = "#F97316"
COLORS = [
    "#2563EB", "#8B5CF6", "#06B6D4", "#DB2777",
    "#64748B", "#16A34A", "#F59E0B", "#EF4444",
    "#0D9488", "#6366F1", "#A855F7", "#D97706",
]
DEFAULT_LABELS = {
    "meta_spend": "Meta",
    "meta_conversion_spend": "Meta Conversion",
    "meta_engagement_spend": "Meta Engagement",
    "meta_traffic_spend": "Meta Traffic",
    "meta_thruplay_spend": "Meta ThruPlay",
    "meta_other_spend": "Meta Other",
    "google_search_spend": "Google Search",
    "google_pmax_spend": "Google Performance Max",
    "google_display_spend": "Google Display",
    "google_video_spend": "Google Video",
    "google_shopping_spend": "Google Shopping",
    "google_demand_gen_spend": "Google Demand Gen",
}

GROUP_COLORS = {
    "Marketing": BLUE,
    "Context": TEAL,
    "Seasonality": "#8B5CF6",
    "Intercept": NAVY,
}


def _get(bundle: Any, name: str) -> Any:
    return bundle[name] if isinstance(bundle, Mapping) else getattr(bundle, name)


def _labels(manifest: Mapping[str, Any]) -> dict[str, str]:
    labels = dict(DEFAULT_LABELS)
    labels.update(manifest.get("labels", {}))
    labels.update(manifest.get("channel_labels", {}))
    labels.update(manifest.get("control_labels", {}))
    labels.update({
        "intercept": "Baseline / intercept",
        "Intercept": "Baseline / intercept",
        "yearly_seasonality_contribution": "Annual seasonality",
        "Seasonality": "Annual seasonality",
        "seasonality": "Annual seasonality",
        "discount_rate_scaled": "Discounts",
        "linear_trend_scaled": "Trend",
    })
    return labels


def _rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    rgb = tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})"


def _layout(
    figure: go.Figure,
    title: str,
    y_title: str,
    height: int = 440,
    x_title: str = "Week starting Monday",
    legend: bool = True,
) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=19, color=NAVY)),
        font=dict(family="Arial, sans-serif", size=12, color=NAVY),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=height,
        margin=dict(l=65, r=30, t=65, b=145 if legend else 70),
        hovermode="x unified",
        showlegend=legend,
        legend=dict(
            orientation="h", yanchor="top", y=-0.23,
            xanchor="center", x=0.5, title_text="", groupclick="togglegroup",
        ),
        xaxis=dict(title=x_title, gridcolor="#EEF2F7", zeroline=False),
        yaxis=dict(title=y_title, gridcolor="#EEF2F7", zeroline=False, tickformat=",.0f"),
    )
    return figure


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    return result.sort_values("date").reset_index(drop=True)


def _band(
    fig: go.Figure, x: Sequence, lower: Sequence, upper: Sequence,
    color: str, name: str = "90% interval", legendgroup: str | None = None,
) -> None:
    fig.add_trace(go.Scatter(
        x=x, y=lower, mode="lines", line=dict(width=0),
        hoverinfo="skip", showlegend=False, legendgroup=legendgroup,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=upper, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor=_rgba(color, 0.13),
        hoverinfo="skip", name=name, legendgroup=legendgroup,
    ))


def kpi_chart(
    weekly: pd.DataFrame,
    target_col: str = "eligible_orders",
    target_label: str = "Orders",
    selected_kpis: Sequence[str] | None = None,
) -> go.Figure:
    """Show the selected KPI; two selected KPIs use separate labeled axes."""
    frame = _sorted(weekly)
    columns = list(selected_kpis or [target_col])
    if not columns or len(columns) > 2:
        raise ValueError("Select one or two KPIs for the KPI chart.")
    nice = {target_col: target_label, "net_revenue": "Net revenue", "eligible_orders": "Orders"}
    fig = make_subplots(specs=[[{"secondary_y": len(columns) == 2}]])
    for i, column in enumerate(columns):
        if column not in frame:
            raise ValueError(f"KPI column is not available: {column}")
        fig.add_trace(go.Scatter(
            x=frame["date"], y=frame[column], mode="lines",
            name=nice.get(column, column.replace("_", " ").title()),
            line=dict(color=[BLUE, TEAL][i], width=2.6), connectgaps=False,
            hovertemplate="%{x|%d %b %Y}<br>%{y:,.1f}<extra>%{fullData.name}</extra>",
        ), secondary_y=i == 1)
    _layout(fig, "Weekly KPI history", nice.get(columns[0], target_label))
    if len(columns) == 2:
        fig.update_yaxes(title_text=nice.get(columns[1], columns[1]), secondary_y=True, tickformat=",.0f")
    return fig


def spend_timeline(
    weekly: pd.DataFrame, channels: Sequence[str], labels: Mapping[str, str],
) -> go.Figure:
    frame = _sorted(weekly)
    fig = go.Figure()
    for i, channel in enumerate(channels):
        fig.add_trace(go.Scatter(
            x=frame["date"], y=frame[channel], mode="lines",
            name=labels.get(channel, DEFAULT_LABELS.get(channel, channel)), connectgaps=False,
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            hovertemplate="%{x|%d %b %Y}<br>Spend: %{y:,.2f}<extra>%{fullData.name}</extra>",
        ))
    return _layout(fig, "Marketing spend over time", "Weekly spend", height=500)


def spend_stacked(
    weekly: pd.DataFrame, channels: Sequence[str], labels: Mapping[str, str],
) -> go.Figure:
    """Weekly channel totals as stacked bars, with every fitted channel shown."""
    frame = _sorted(weekly)
    total = frame[list(channels)].sum(axis=1, min_count=len(channels))
    fig = go.Figure()
    for i, channel in enumerate(channels):
        fig.add_bar(
            x=frame["date"], y=frame[channel], name=labels.get(channel, DEFAULT_LABELS.get(channel, channel)),
            marker_color=COLORS[i % len(COLORS)],
            customdata=total.to_numpy()[:, None],
            hovertemplate=(
                "%{x|%d %b %Y}<br>Channel spend: %{y:,.2f}<br>"
                "All-channel spend: %{customdata[0]:,.2f}<extra>%{fullData.name}</extra>"
            ),
        )
    _layout(fig, "Weekly marketing spend composition", "Total weekly spend", height=500)
    fig.update_layout(barmode="stack", bargap=0.08)
    fig.update_yaxes(rangemode="tozero")
    return fig


def spend_totals(
    weekly: pd.DataFrame, channels: Sequence[str], labels: Mapping[str, str],
) -> go.Figure:
    totals = weekly[list(channels)].sum(min_count=1).sort_values(ascending=False)
    if totals.isna().any():
        raise ValueError("At least one channel has no observed spending.")
    fig = go.Figure()
    ordered_labels = [labels.get(channel, DEFAULT_LABELS.get(channel, channel)) for channel in totals.index]
    for channel, value in totals.items():
        color = COLORS[list(channels).index(channel) % len(COLORS)]
        fig.add_bar(
            x=[labels.get(channel, DEFAULT_LABELS.get(channel, channel))], y=[value],
            name=labels.get(channel, DEFAULT_LABELS.get(channel, channel)), marker_color=color,
            text=[value], texttemplate="%{text:,.0f}", textposition="outside",
            hovertemplate="<b>%{x}</b><br>Total spend: %{y:,.2f}<extra></extra>",
        )
    _layout(fig, "Total spend by channel", "Spend over the selected period", height=500, x_title="")
    fig.update_layout(bargap=0.32)
    fig.update_xaxes(categoryorder="array", categoryarray=ordered_labels, tickangle=-22)
    fig.update_yaxes(rangemode="tozero")
    return fig


def fit_chart(
    frame: pd.DataFrame, title: str, interval_kind: str = "expected_mean",
    interval_level: float = 0.90,
) -> go.Figure:
    if not 0 < interval_level < 1:
        raise ValueError("Interval level must be between zero and one.")
    interval_names = {
        "expected_mean": "expected-mean credible interval",
        "posterior_predictive": "predictive interval",
        "unavailable": "exported interval",
    }
    if interval_kind not in interval_names:
        raise ValueError("Unrecognized fit interval kind.")
    data = _sorted(frame)
    fig = go.Figure()
    if {"lower", "upper"}.issubset(data.columns):
        available = data[["lower", "upper"]].notna().all(axis=1)
        if available.any():
            _band(fig, data["date"], data["lower"], data["upper"], BLUE,
                  f"{100 * interval_level:g}% {interval_names[interval_kind]}")
    fig.add_trace(go.Scatter(
        x=data["date"], y=data["predicted"], name="Predicted orders", mode="lines",
        line=dict(color=ORANGE, width=2.7),
        hovertemplate="Predicted: %{y:,.1f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=data["date"], y=data["observed"], name="Observed orders", mode="lines+markers",
        line=dict(color=BLUE, width=2), marker=dict(size=4),
        hovertemplate="Observed: %{y:,.1f}<extra></extra>",
    ))
    return _layout(fig, title, "Orders")


def residual_chart(frame: pd.DataFrame) -> go.Figure:
    data = _sorted(frame)
    fig = go.Figure(go.Scatter(
        x=data["date"], y=data["observed"] - data["predicted"],
        name="Observed − predicted", mode="lines+markers",
        line=dict(color=BLUE, width=1.8), marker=dict(size=4),
        hovertemplate="%{x|%d %b %Y}<br>Residual: %{y:+,.1f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#94A3B8", line_dash="dash")
    return _layout(fig, "Residuals over time", "Orders: observed − predicted", height=320, legend=False)


def contribution_chart(
    contributions: pd.DataFrame,
    manifest: Mapping[str, Any],
    marketing_only: bool = False,
) -> go.Figure:
    """Sum posterior means over dates; never sum pointwise interval endpoints."""
    data = contributions.copy()
    if marketing_only:
        data = data.loc[data["group"].eq("Marketing")]
    totals = (
        data.groupby(["effect", "group"], as_index=False, sort=False)["mean"]
        .sum().sort_values("mean", ascending=False)
    )
    nice = _labels(manifest)
    totals["label"] = totals["effect"].map(nice).fillna(totals["effect"])
    fig = go.Figure()
    # One trace per group retains a clear explanatory legend.
    for group in ["Marketing", "Intercept", "Context", "Seasonality"]:
        subset = totals.loc[totals["group"].eq(group)]
        if subset.empty:
            continue
        fig.add_bar(
            x=subset["label"], y=subset["mean"], name=group,
            marker_color=GROUP_COLORS[group], text=subset["mean"],
            texttemplate="%{text:,.0f}", textposition="outside",
            hovertemplate="<b>%{x}</b><br>Total modeled orders: %{y:,.1f}<extra>%{fullData.name}</extra>",
        )
    fig.add_hline(y=0, line_color="#CBD5E1", line_width=1)
    title = "Marketing contributions" if marketing_only else "All model components"
    _layout(fig, title, "Total modeled orders over the selected period", height=500, x_title="")
    fig.update_xaxes(categoryorder="array", categoryarray=totals["label"].tolist(), tickangle=-28)
    fig.update_layout(bargap=0.28)
    return fig


def efficiency_table(bundle: Any) -> pd.DataFrame:
    manifest = _get(bundle, "manifest")
    channels = list(manifest["channels"])
    weekly = _get(bundle, "weekly")
    contributions = _get(bundle, "contributions")
    nice = _labels(manifest)
    if set(pd.to_datetime(weekly["date"])) != set(pd.to_datetime(contributions["date"])):
        raise ValueError("Spending and contribution dates must cover the same period.")
    media = contributions.loc[contributions["group"].eq("Marketing")]
    orders = media.groupby("effect")["mean"].sum().reindex(channels)
    if orders.isna().any():
        raise ValueError("Marketing contributions are missing for one or more fitted channels.")
    spend = weekly[channels].sum(min_count=1)
    table = pd.DataFrame({"channel": channels})
    table["channel_label"] = table["channel"].map(nice).fillna(table["channel"])
    table["total_spend"] = spend.reindex(channels).to_numpy()
    table["modeled_orders"] = orders.to_numpy()
    spend_total = table["total_spend"].sum()
    effects_total = table["modeled_orders"].sum()
    table["spend_share_pct"] = 100 * table["total_spend"] / spend_total if spend_total > 0 else np.nan
    table["effect_share_pct"] = 100 * table["modeled_orders"] / effects_total if effects_total > 0 else np.nan
    table["modeled_cpa"] = np.divide(
        table["total_spend"], table["modeled_orders"],
        out=np.full(len(table), np.nan), where=table["modeled_orders"].to_numpy() > 0,
    )
    table.loc[
        table["total_spend"].gt(0) & table["modeled_orders"].eq(0), "modeled_cpa"
    ] = np.inf
    table["orders_per_spend"] = np.divide(
        table["modeled_orders"], table["total_spend"],
        out=np.full(len(table), np.nan), where=table["total_spend"].to_numpy() > 0,
    )
    return table.sort_values("modeled_orders", ascending=False).reset_index(drop=True)


def efficiency_share_chart(table: pd.DataFrame) -> go.Figure:
    data = table.sort_values("effect_share_pct", ascending=False)
    fig = go.Figure()
    for column, label, color in [
        ("spend_share_pct", "Spend share", BLUE),
        ("effect_share_pct", "Modeled marketing-order share", ORANGE),
    ]:
        fig.add_bar(
            x=data["channel_label"], y=data[column], name=label, marker_color=color,
            hovertemplate="<b>%{x}</b><br>Share: %{y:.1f}%<extra>%{fullData.name}</extra>",
        )
    share_values = data[["spend_share_pct", "effect_share_pct"]].to_numpy(dtype=float)
    finite_shares = share_values[np.isfinite(share_values)]
    peak = max(float(finite_shares.max()), 1.0) if finite_shares.size else 1.0
    for row in data.itertuples():
        label = (f"CPA {row.modeled_cpa:,.1f}" if np.isfinite(row.modeled_cpa)
                 else "CPA unbounded" if np.isposinf(row.modeled_cpa) else "CPA n/a")
        row_shares = np.asarray([row.spend_share_pct, row.effect_share_pct], dtype=float)
        row_shares = row_shares[np.isfinite(row_shares)]
        if not row_shares.size:
            continue
        fig.add_annotation(
            x=row.channel_label, y=row_shares.max() + peak * 0.055,
            text=label, showarrow=False, font=dict(size=11, color=NAVY),
        )
    _layout(fig, "Spend share vs modeled marketing-order share", "Share (%)", height=490, x_title="")
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_xaxes(categoryorder="array", categoryarray=data["channel_label"].tolist(), tickangle=-25)
    fig.update_yaxes(range=[0, peak * 1.2], ticksuffix="%", tickformat=".0f")
    return fig


def cpa_chart(table: pd.DataFrame) -> go.Figure:
    data = table.sort_values("modeled_cpa", ascending=True, na_position="last").copy()
    finite = np.isfinite(data["modeled_cpa"])
    fig = go.Figure(go.Bar(
        x=data["channel_label"], y=data["modeled_cpa"].where(finite), name="Modeled CPA",
        marker_color=TEAL, text=data["modeled_cpa"].where(finite), texttemplate="%{text:,.1f}", textposition="outside",
        hovertemplate="<b>%{x}</b><br>Modeled CPA: %{y:,.2f}<extra></extra>",
    ))
    for row in data.loc[~finite].itertuples():
        fig.add_annotation(x=row.channel_label, y=0, text="Unbounded" if np.isposinf(row.modeled_cpa) else "n/a",
                           showarrow=False, yshift=12)
    _layout(fig, "Cost per modeled media order", "Spend per modeled order", height=460, x_title="")
    fig.update_xaxes(categoryorder="array", categoryarray=data["channel_label"].tolist(), tickangle=-25)
    fig.update_yaxes(rangemode="tozero", tickformat=",.1f")
    return fig


def _posterior(bundle: Any) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    posterior = _get(bundle, "posterior")
    manifest = _get(bundle, "manifest")
    n_channels = len(manifest["channels"])
    values = {name: np.asarray(posterior[name], dtype=float) for name in ("alpha", "beta", "kappa", "slope")}
    shape = values["alpha"].shape
    if len(shape) != 2 or shape[1] != n_channels or any(v.shape != shape for v in values.values()):
        raise ValueError("Posterior arrays must share paired (sample, channel) dimensions.")
    if any(not np.isfinite(v).all() for v in values.values()):
        raise ValueError("Posterior arrays contain non-finite values.")
    if ((values["alpha"] < 0) | (values["alpha"] > 1)).any():
        raise ValueError("Adstock alpha must lie in [0, 1].")
    if any((values[name] <= 0).any() for name in ("beta", "kappa", "slope")):
        raise ValueError("Hill parameters must be positive.")
    scale_source = posterior.get("channel_scale")
    target_source = posterior.get("target_scale")
    if scale_source is None:
        scale_source = _get(bundle, "channel_scale")
    if target_source is None:
        target_source = _get(bundle, "target_scale")
    scale = np.asarray(scale_source, dtype=float).reshape(-1)
    target_scale = float(np.asarray(target_source).item())
    if len(scale) != n_channels or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Each channel scale must be finite and positive.")
    if not np.isfinite(target_scale) or target_scale <= 0:
        raise ValueError("Target scale must be finite and positive.")
    return values, scale, target_scale


def saturation_chart(
    bundle: Any, selected_channels: Sequence[str] | None = None,
) -> go.Figure:
    """Paired-posterior Hill curves in orders; normalized steady-state input.

    These are descriptive response curves, not a finite-horizon forecast. With
    normalized geometric adstock, constant spend equals its steady-state input.
    """
    manifest = _get(bundle, "manifest")
    channels = list(manifest["channels"])
    chosen = list(channels if selected_channels is None else selected_channels)
    if not chosen or any(channel not in channels for channel in chosen):
        raise ValueError("Select at least one fitted channel for saturation curves.")
    post, scales, target_scale = _posterior(bundle)
    if not manifest.get("adstock_normalize", True):
        raise ValueError("Sustained-spend curves require normalized adstock.")
    nice = _labels(manifest)
    weekly = _get(bundle, "weekly")
    fig = go.Figure()
    for channel in chosen:
        i = channels.index(channel)
        color = COLORS[i % len(COLORS)]
        historical_max = float(weekly[channel].max())
        mean_spend = float(weekly[channel].mean())
        if not np.isfinite(historical_max) or historical_max <= 0:
            raise ValueError(f"Channel has no positive historical spend: {channel}")
        grid = np.unique(np.r_[np.linspace(0, historical_max * 1.2, 220), historical_max, mean_spend])
        x = grid / scales[i]
        response = np.zeros((post["beta"].shape[0], len(grid)))
        positive = x > 0
        response[:, positive] = (
            post["beta"][:, i, None] * expit(
                post["slope"][:, i, None]
                * (np.log(x[None, positive]) - np.log(post["kappa"][:, i, None]))
            ) * target_scale
        )
        low, high = np.quantile(response, [0.05, 0.95], axis=0)
        if len(chosen) == 1:
            _band(fig, grid, low, high, color, "90% credible interval", legendgroup=channel)
        for extrapolation in (False, True):
            mask = grid >= historical_max if extrapolation else grid <= historical_max
            fig.add_trace(go.Scatter(
                x=grid[mask], y=response.mean(axis=0)[mask], mode="lines", name=nice.get(channel, channel),
                legendgroup=channel, showlegend=not extrapolation,
                line=dict(color=color, width=2.7, dash="dash" if extrapolation else "solid"),
                customdata=np.column_stack([low[mask], high[mask]]),
                hovertemplate=("Sustained weekly spend: %{x:,.0f}<br>Modeled channel orders/week: %{y:,.1f}<br>"
                               "90% credible interval: %{customdata[0]:,.1f}–%{customdata[1]:,.1f}"
                               "<extra>%{fullData.name}</extra>"),
            ))
        mean_x = mean_spend / scales[i]
        mean_orders = 0.0 if mean_x <= 0 else float(np.mean(
            post["beta"][:, i] * expit(post["slope"][:, i] * (np.log(mean_x) - np.log(post["kappa"][:, i]))) * target_scale
        ))
        fig.add_trace(go.Scatter(
            x=[mean_spend], y=[mean_orders], mode="markers", name=nice.get(channel, channel),
            showlegend=False, legendgroup=channel,
            marker=dict(size=9, color=color, line=dict(color="white", width=1.5)),
            hovertemplate=("Historical-average weekly spend: %{x:,.0f}<br>"
                           "Response at sustained average spend: %{y:,.1f}<extra>%{fullData.name}</extra>"),
        ))
    _layout(fig, "Channel saturation response<br><sup>Dashed: above historical maximum spend</sup>",
            "Modeled channel orders per week", height=550, x_title="Sustained weekly spend")
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(tickformat=",.0f", rangemode="tozero")
    fig.update_yaxes(rangemode="tozero")
    return fig


def adstock_chart(
    bundle: Any, selected_channels: Sequence[str] | None = None,
) -> go.Figure:
    manifest = _get(bundle, "manifest")
    channels = list(manifest["channels"])
    chosen = list(channels if selected_channels is None else selected_channels)
    if not chosen or any(channel not in channels for channel in chosen):
        raise ValueError("Select at least one fitted channel for adstock curves.")
    post, _, _ = _posterior(bundle)
    length = int(manifest.get("adstock_l_max", 8))
    if length < 1:
        raise ValueError("Adstock lag window must be positive.")
    normalized = bool(manifest.get("adstock_normalize", True))
    lags = np.arange(length)
    nice = _labels(manifest)
    fig = go.Figure()
    for channel in chosen:
        i = channels.index(channel)
        color = COLORS[i % len(COLORS)]
        weights = post["alpha"][:, i, None] ** lags[None, :]
        if normalized:
            weights /= weights.sum(axis=1, keepdims=True)
        low, high = np.quantile(weights * 100, [0.05, 0.95], axis=0)
        if len(chosen) == 1:
            _band(fig, lags, low, high, color, "90% credible interval", legendgroup=channel)
        fig.add_trace(go.Scatter(
            x=lags, y=100 * weights.mean(axis=0), name=nice.get(channel, channel), mode="lines+markers",
            line=dict(color=color, width=2.4), marker=dict(size=5), legendgroup=channel,
            customdata=np.column_stack([low, high]),
            hovertemplate=("Lag: %{x} weeks<br>Adstock weight: %{y:.1f}%<br>"
                           "90% credible interval: %{customdata[0]:.1f}%–%{customdata[1]:.1f}%"
                           "<extra>%{fullData.name}</extra>"),
        ))
    _layout(fig, "Weekly adstock weights<br><sup>Spend memory before saturation</sup>",
            "Share of normalized adstock kernel (%)" if normalized else "Weight relative to week 0 (%)",
            height=520, x_title="Weeks since media exposure (0 = same week)")
    fig.update_xaxes(tickmode="linear", dtick=1)
    fig.update_yaxes(rangemode="tozero", tickformat=".0f", ticksuffix="%")
    return fig


def budget_chart(
    allocation: pd.DataFrame, manifest: Mapping[str, Any], period_totals: bool = False,
) -> go.Figure:
    """Compare continuation and optimized spend; CPA includes historical carry-in."""
    data = allocation.copy()
    aliases = {
        "current_spend": "historical_weekly_spend",
        "recommended_spend": "optimized_weekly_spend",
        "current_total_spend": "historical_period_spend",
        "recommended_total_spend": "optimized_period_spend",
        "current_CPA": "historical_modeled_cpa",
        "optimized_CPA": "optimized_modeled_cpa",
    }
    for old, new in aliases.items():
        if new not in data and old in data:
            data[new] = data[old]
    suffix = "period_spend" if period_totals else "weekly_spend"
    current, optimized = f"historical_{suffix}", f"optimized_{suffix}"
    required = {"channel", current, optimized}
    if not required.issubset(data.columns):
        raise ValueError(f"Budget table is missing: {sorted(required.difference(data.columns))}")
    if not np.isfinite(data[[current, optimized]].to_numpy(dtype=float)).all():
        raise ValueError("Budget amounts must be finite.")
    if data[[current, optimized]].lt(0).any().any():
        raise ValueError("Budget amounts must be nonnegative.")
    data = data.sort_values(optimized, ascending=False)
    nice = _labels(manifest)
    data["label"] = data["channel"].map(nice).fillna(data["channel"])
    change = data[optimized] - data[current]
    percent = np.divide(
        100 * change.to_numpy(), data[current].to_numpy(),
        out=np.full(len(data), np.nan), where=data[current].to_numpy() > 0,
    )
    change_labels = [
        "New" if old == 0 and new > 0 else "No change" if old == 0 else f"{pct:+.1f}%"
        for old, new, pct in zip(data[current], data[optimized], percent)
    ]
    fig = go.Figure()
    unit = "Planning-period spend" if period_totals else "Weekly spend"
    for column, cpa_column, label, color in [
        (current, "historical_modeled_cpa", "Historical-average continuation", BLUE),
        (optimized, "optimized_modeled_cpa", "Optimized allocation", ORANGE),
    ]:
        cpa = data[cpa_column] if cpa_column in data else pd.Series(np.nan, index=data.index)
        cpa_labels = [f"{value:,.2f}" if np.isfinite(value)
                      else "Unbounded" if np.isposinf(value) else "n/a" for value in cpa]
        fig.add_bar(
            x=data["label"], y=data[column], name=label, marker_color=color,
            customdata=np.column_stack([cpa_labels, change_labels]),
            hovertemplate=("<b>%{x}</b><br>" + unit + ": %{y:,.2f}<br>"
                           "Projected media CPA: %{customdata[0]}<br>"
                           "Recommended spend change: %{customdata[1]}<extra>%{fullData.name}</extra>"),
        )
    peak = max(float(data[[current, optimized]].max().max()), 1.0)
    for index, (_, row) in enumerate(data.iterrows()):
        fig.add_annotation(
            x=row["label"], y=max(row[current], row[optimized]) + peak * 0.04,
            text=change_labels[index], showarrow=False,
            font=dict(size=12, color=TEAL if change.iloc[index] > 0 else "#B91C1C" if change.iloc[index] < 0 else "#64748B"),
        )
    title = "Budget allocation over the planning period" if period_totals else "Weekly budget allocation"
    _layout(fig, title, "Total spend over planning horizon" if period_totals else "Weekly spend", height=530, x_title="")
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_xaxes(categoryorder="array", categoryarray=data["label"].tolist(), tickangle=-25)
    fig.update_yaxes(range=[0, peak * 1.2])
    return fig


def forecast_chart(
    weekly_forecast: pd.DataFrame, media_only: bool = False,
    include_historical_mix: bool = False,
) -> go.Figure:
    data = _sorted(weekly_forecast)
    fig = go.Figure()
    measure = "media_orders" if media_only else "total_orders"
    columns = [("current", "Continue historical spending", BLUE)]
    if f"feasible_reference_{measure}" in data:
        columns.append(("feasible_reference", "Feasible equal-budget reference", "#64748B"))
        if include_historical_mix:
            columns.append(("reference", "Historical mix, same budget", "#8B5CF6"))
    else:
        columns.append(("reference", "Historical mix, same budget", "#64748B"))
    columns.append(("optimized", "Optimized allocation", ORANGE))
    for scenario, name, color in columns:
        column = f"{scenario}_{measure}"
        if column not in data and f"{scenario}_expected_orders" in data:
            column = f"{scenario}_expected_orders"
        if column not in data:
            continue
        if scenario == "optimized":
            for lower, upper in [(f"{column}_lower_90", f"{column}_upper_90"), ("optimized_lower", "optimized_upper"), ("optimized_lower_90", "optimized_upper_90")]:
                if {lower, upper}.issubset(data.columns):
                    _band(fig, data["date"], data[lower], data[upper], color, "Optimized: 90% expected-mean credible interval")
                    break
        fig.add_trace(go.Scatter(
            x=data["date"], y=data[column], name=name, mode="lines+markers",
            line=dict(color=color, width=2.6, dash="dot" if "reference" in scenario else "solid"),
            marker=dict(size=5), hovertemplate="%{x|%d %b %Y}<br>Modeled orders: %{y:,.1f}<extra>%{fullData.name}</extra>",
        ))
    if not fig.data:
        raise ValueError("Forward forecasts contain no recognized order columns.")
    return _layout(fig, "Forward modeled marketing orders" if media_only else "Forward expected total orders", "Weekly modeled media orders" if media_only else "Weekly expected total orders", height=430)


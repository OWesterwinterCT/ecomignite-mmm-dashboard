"""Portable posterior-response calculations and finite-horizon budget allocation.

No fitting, PyMC, pickle, external credentials, or remote execution is required.
All uncertainty calculations preserve each exported posterior draw as a pair.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

CHANNEL_LABELS = {
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

DEFAULT_SETTINGS = {
    "horizon_weeks": 12,
    "reference_weeks": None,
    "total_budget_cap": None,
    "max_spend_change": 0.50,
    "channel_weekly_bounds": {},
    "channel_change_overrides": {},
    "protect_search": True,
    "search_channel": "google_search_spend",
    "cap_at_historical_max": True,
    "future_discount_rate": None,
    "optimization_draws": 500,
    "n_starts": 6,
    "random_seed": 42,
}


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _finite_amount(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return result


def hill_response(x: np.ndarray, beta: np.ndarray, kappa: np.ndarray,
                  slope: np.ndarray) -> np.ndarray:
    """Exact zero response at zero; the ordinary Hill curve for positive input."""
    x = np.asarray(x, dtype=float)
    positive = x > 0
    safe = np.where(positive, x, 1.0)
    return np.where(positive, beta * expit(slope * (np.log(safe) - np.log(kappa))), 0.0)


def adstock_weights(alpha: np.ndarray, l_max: int) -> np.ndarray:
    """Normalized geometric weights, shape (posterior sample, channel, lag)."""
    alpha = np.asarray(alpha, dtype=float)
    weights = alpha[..., None] ** np.arange(l_max)
    return weights / weights.sum(axis=-1, keepdims=True)


def _model_inputs(bundle):
    manifest, posterior = bundle.manifest, bundle.posterior
    channels = list(manifest["channels"])
    if not channels or len(channels) != len(set(channels)):
        raise ValueError("The bundle must have distinct fitted channel names.")
    if not manifest.get("adstock_normalize", manifest.get("adstock_normalized", manifest.get("normalized", True))):
        raise ValueError("This optimizer requires normalized geometric adstock.")
    if not manifest.get("adstock_first", True):
        raise ValueError("This optimizer requires adstock before Hill saturation.")
    if manifest.get("time_varying_media", False) or manifest.get("time_varying_intercept", False):
        raise ValueError("Time-varying coefficients require another export/forecast implementation.")
    l_max = _positive_integer(manifest["adstock_l_max"], "adstock_l_max")
    weekly = bundle.weekly.copy()
    weekly["date"] = pd.to_datetime(weekly["date"], errors="raise")
    weekly = weekly.sort_values("date").reset_index(drop=True)
    dates = pd.DatetimeIndex(weekly["date"])
    if (dates.hasnans or not dates.is_unique or not (dates.dayofweek == 0).all()
            or not (np.diff(dates.values) == np.timedelta64(7, "D")).all()):
        raise ValueError("Model dates must be unique, consecutive Monday week starts.")
    if len(weekly) < l_max:
        raise ValueError(f"At least {l_max} fitted weeks are needed to initialize carryover.")
    observed = weekly[channels].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(observed).all() or (observed < 0).any():
        raise ValueError("Fitted media spend must be finite and nonnegative.")
    arrays = [np.asarray(posterior[name], dtype=float) for name in ["alpha", "beta", "kappa", "slope"]]
    alpha, beta, kappa, slope = arrays
    if alpha.ndim != 2 or any(a.shape != alpha.shape for a in arrays) or alpha.shape[1] != len(channels):
        raise ValueError("Posterior media arrays must share shape (draw, channel).")
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("Posterior media arrays contain nonfinite values.")
    if ((alpha < 0) | (alpha > 1)).any() or (beta < 0).any() or (kappa <= 0).any() or (slope <= 0).any():
        raise ValueError("Posterior media parameters are outside their valid support.")
    channel_scale = np.asarray(posterior["channel_scale"], dtype=float)
    target_scale = float(np.asarray(posterior["target_scale"]).item())
    if (channel_scale.shape != (len(channels),) or not np.isfinite(channel_scale).all()
            or (channel_scale <= 0).any() or not np.isfinite(target_scale) or target_scale <= 0):
        raise ValueError("The exported channel and target scales must be finite and positive.")
    return channels, weekly, observed, arrays, channel_scale, target_scale, l_max


def _forward_parts(bundle, horizon: int):
    channels, weekly, observed, arrays, scale_x, scale_y, lag = _model_inputs(bundle)
    alpha, beta, kappa, slope = arrays
    weights = adstock_weights(alpha, lag)
    samples, count = alpha.shape
    history = np.zeros((samples, horizon, count))
    growth = np.ones((samples, horizon, count))
    for h in range(min(horizon, lag)):
        growth[:, h, :] = weights[:, :, :h + 1].sum(axis=2)
        for delay in range(h + 1, lag):
            history[:, h, :] += weights[:, :, delay] * observed[-(delay - h)] / scale_x
    return history, growth, beta[:, None, :], kappa[:, None, :], slope[:, None, :], scale_x, scale_y


def forward_media_draws(bundle, weekly_spend: np.ndarray) -> np.ndarray:
    """Predict future per-channel media orders: shape (draw, future week, channel).

    weekly_spend may vary by future week. The optimizer below intentionally uses
    one constant weekly allocation. Carryover after the supplied horizon is excluded.
    """
    channels, _, observed, arrays, scale_x, scale_y, lag = _model_inputs(bundle)
    spend = np.asarray(weekly_spend, dtype=float)
    if spend.ndim != 2 or spend.shape[1] != len(channels) or not len(spend):
        raise ValueError("Future spend must have shape (week, channel).")
    if not np.isfinite(spend).all() or (spend < 0).any():
        raise ValueError("Future spend must be finite and nonnegative.")
    alpha, beta, kappa, slope = arrays
    weights = adstock_weights(alpha, lag)
    sequence = np.vstack([observed[-lag:], spend]) / scale_x
    adstocked = np.zeros((len(alpha), len(spend), len(channels)))
    for h in range(len(spend)):
        for delay in range(lag):
            adstocked[:, h, :] += weights[:, :, delay] * sequence[lag + h - delay]
    return scale_y * hill_response(adstocked, beta[:, None, :], kappa[:, None, :], slope[:, None, :])


def nonmedia_draws(bundle, dates, future_discount_rate: float) -> tuple[np.ndarray, dict]:
    """Paired additive intercept, anchored controls and yearly Fourier contributions."""
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    manifest, posterior = bundle.manifest, bundle.posterior
    scale = float(np.asarray(posterior["target_scale"]).item())
    samples = np.asarray(posterior["alpha"]).shape[0]
    intercept = np.asarray(posterior["intercept"], dtype=float).reshape(-1)
    if intercept.shape != (samples,) or not np.isfinite(intercept).all():
        raise ValueError("Intercept posterior must have one finite value per paired draw.")
    controls = list(manifest.get("controls", []))
    unsupported = set(controls) - {"discount_rate_scaled", "linear_trend_scaled"}
    if unsupported:
        raise ValueError(f"Supply an implementation of future assumptions for controls: {sorted(unsupported)}")
    gamma = np.asarray(posterior.get("gamma_control", np.empty((samples, 0))), dtype=float)
    if gamma.shape != (samples, len(controls)) or not np.isfinite(gamma).all():
        raise ValueError("Control coefficients do not match the paired posterior/control order.")
    references = manifest.get("control_scaling", {})
    values = []
    for name in controls:
        if name == "discount_rate_scaled":
            unit = float(references["discount_unit"])
            if not np.isfinite(unit) or unit <= 0:
                raise ValueError("discount_unit must be positive.")
            values.append(np.full(len(dates), future_discount_rate / unit))
        else:
            origin = pd.Timestamp(references["trend_origin"])
            span = float(references["trend_span_days"])
            if not np.isfinite(span) or span <= 0:
                raise ValueError("trend_span_days must be positive.")
            values.append(np.asarray((dates - origin).days, dtype=float) / span)
    control_contribution = gamma @ np.asarray(values) if controls else np.zeros((samples, len(dates)))
    order = int(manifest.get("yearly_seasonality", 0))
    gamma_fourier = np.asarray(posterior.get("gamma_fourier", np.empty((samples, 0))), dtype=float)
    if gamma_fourier.shape != (samples, 2 * order) or not np.isfinite(gamma_fourier).all():
        raise ValueError("Fourier coefficients do not match the exported yearly order.")
    if order:
        period = float(manifest.get("fourier_period_days", 365.25))
        if not np.isfinite(period) or period <= 0:
            raise ValueError("Fourier period must be finite and positive.")
        phase = 2 * np.pi * dates.dayofyear.to_numpy()[:, None] / period * np.arange(1, order + 1)
        modes = np.concatenate([np.sin(phase), np.cos(phase)], axis=1)
        default_names = [f"sin_{i}" for i in range(1, order + 1)] + [f"cos_{i}" for i in range(1, order + 1)]
        names = manifest.get("fourier_names", manifest.get("fourier_columns", default_names))
        if sorted(names) != sorted(default_names):
            raise ValueError("Fourier nodes must match sin_1..sin_n and cos_1..cos_n.")
        modes = modes[:, [default_names.index(name) for name in names]]
        seasonal = gamma_fourier @ modes.T
    else:
        seasonal = np.zeros((samples, len(dates)))
    parts = {"intercept": scale * np.broadcast_to(intercept[:, None], seasonal.shape),
             "controls": scale * control_contribution, "seasonality": scale * seasonal}
    total = parts["intercept"] + parts["controls"] + parts["seasonality"]
    return total, parts


def _bounded_projection(values, lower, upper, target):
    """Euclidean projection on box bounds and exact sum, including fixed channels."""
    if target < lower.sum() - 1e-9 or target > upper.sum() + 1e-9:
        raise ValueError("Cannot project allocation onto an infeasible spending total.")
    if not len(values):
        return np.asarray(values).copy()
    left, right = float(np.min(values - upper)), float(np.max(values - lower))
    for _ in range(90):
        midpoint = (left + right) / 2
        trial = np.clip(values - midpoint, lower, upper)
        if trial.sum() > target:
            left = midpoint
        else:
            right = midpoint
    return np.clip(values - (left + right) / 2, lower, upper)


def validate_fitted_checkpoint(bundle):
    """Verify paired exported draws against the final fitted week, when supplied.

    Ordinary contribution CSV means may use more draws than posterior.npz, so
    they are not an exact checkpoint for a subsampled model bundle.
    """
    posterior = bundle.posterior
    if "fitted_last_channel_contribution" not in posterior:
        return {"status": "unavailable", "reason": "This bundle has no paired fitted-response checkpoint."}
    channels, weekly, observed, arrays, scale_x, scale_y, lag = _model_inputs(bundle)
    alpha, beta, kappa, slope = arrays
    expected = np.asarray(posterior["fitted_last_channel_contribution"], dtype=float)
    if expected.shape != alpha.shape or not np.isfinite(expected).all():
        raise ValueError("The fitted-response checkpoint must match the exported sample/channel axes.")
    weights = adstock_weights(alpha, lag)
    last_input = (weights * (observed[-lag:][::-1] / scale_x).T[None, :, :]).sum(axis=2)
    reconstructed = hill_response(last_input, beta, kappa, slope)
    if not np.allclose(reconstructed, expected, rtol=1e-5, atol=1e-7):
        raise ValueError("Exported parameters and history do not reproduce the fitted media checkpoint. Re-export the matching model and data.")
    result = {"status": "verified", "media_max_abs_model_scale_gap": float(np.max(np.abs(reconstructed - expected)))}
    if "fitted_last_mu" in posterior:
        expected_mu = np.asarray(posterior["fitted_last_mu"], dtype=float)
        if expected_mu.shape != (len(alpha),) or not np.isfinite(expected_mu).all():
            raise ValueError("The fitted mean checkpoint must have one value per exported draw.")
        if "discount_rate_scaled" in bundle.manifest.get("controls", []):
            if "discount_rate" not in weekly:
                raise ValueError("Raw discount_rate is required to verify the fitted mean checkpoint.")
            last_discount = float(weekly["discount_rate"].iloc[-1])
        else:
            last_discount = 0.0
        nonmedia, _ = nonmedia_draws(bundle, [weekly["date"].iloc[-1]], last_discount)
        reconstructed_mu = reconstructed.sum(axis=1) + nonmedia[:, 0] / scale_y
        if not np.allclose(reconstructed_mu, expected_mu, rtol=1e-5, atol=1e-7):
            raise ValueError("Exported baseline, controls or seasonality do not reproduce the fitted mean checkpoint. Re-export the matching model and control references.")
        result["mu_max_abs_model_scale_gap"] = float(np.max(np.abs(reconstructed_mu - expected_mu)))
    return result


def optimize_budget(bundle, settings: dict | None = None) -> dict:
    """Find the best feasible repeated weekly allocation across a future horizon.

    The headline compares with a feasible same-budget reference formed from the
    historical channel mix projected onto the active channel bounds. The original
    historical proportions at the same budget and unscaled historical-average
    continuation remain separate comparisons. Multistart SLSQP is not a guarantee
    of a global optimum.
    """
    options = deepcopy(DEFAULT_SETTINGS)
    options.update(settings or {})
    horizon = _positive_integer(options["horizon_weeks"], "horizon_weeks")
    reference = horizon if options["reference_weeks"] is None else _positive_integer(options["reference_weeks"], "reference_weeks")
    starts_count = _positive_integer(options["n_starts"], "n_starts")
    channels, weekly, observed, arrays, scale_x, scale_y, lag = _model_inputs(bundle)
    checkpoint = validate_fitted_checkpoint(bundle)
    if reference > len(weekly):
        raise ValueError("The historical reference window exceeds the available fitted weeks.")
    current = observed[-reference:].mean(axis=0)
    if current.sum() <= 0:
        raise ValueError("The reference window has no positive media spend; a historical-mix comparison is undefined.")
    future_dates = pd.date_range(weekly["date"].iloc[-1] + pd.Timedelta(weeks=1), periods=horizon, freq="7D")
    cap = horizon * current.sum() if options["total_budget_cap"] is None else _finite_amount(options["total_budget_cap"], "total_budget_cap")
    cap = float(cap)
    weekly_cap = cap / horizon
    money_scale = max(weekly_cap, current.sum(), 1.0)
    tolerance = max(1e-6, money_scale * 1e-8)
    lower, upper = np.zeros(len(channels)), np.full(len(channels), weekly_cap)
    absolute, overrides = options["channel_weekly_bounds"], options["channel_change_overrides"]
    unknown = (set(absolute) | set(overrides)) - set(channels)
    if unknown:
        raise ValueError(f"Unknown fitted channel constraints: {sorted(unknown)}")
    changes = []
    for i, channel in enumerate(channels):
        change = overrides.get(channel, options["max_spend_change"])
        if change is not None:
            change = _finite_amount(change, f"Percentage-change limit for {channel}")
            lower[i] = current[i] * max(0, 1 - change)
            upper[i] = min(upper[i], current[i] * (1 + change))
        changes.append(change)
        bounds = absolute.get(channel, {})
        if not isinstance(bounds, dict) or set(bounds) - {"min", "max"}:
            raise ValueError(f"Use only 'min'/'max' weekly bounds for {channel}.")
        if bounds.get("min") is not None:
            lower[i] = max(lower[i], _finite_amount(bounds["min"], f"Minimum for {channel}"))
        if bounds.get("max") is not None:
            upper[i] = min(upper[i], _finite_amount(bounds["max"], f"Maximum for {channel}"))
    protected = np.zeros(len(channels), dtype=bool)
    if options["protect_search"]:
        if options["search_channel"] not in channels:
            raise ValueError("The protected Search channel is not in the fitted model.")
        index = channels.index(options["search_channel"])
        lower[index] = max(lower[index], current[index])
        protected[index] = True
    if options["cap_at_historical_max"]:
        upper = np.minimum(upper, observed.max(axis=0))
    if (lower > upper).any():
        labels = {**CHANNEL_LABELS, **bundle.manifest.get("channel_labels", {})}
        details = "; ".join(f"{labels.get(channels[i], channels[i])}: minimum {lower[i]:,.2f}, "
                            f"maximum {upper[i]:,.2f}" for i in np.flatnonzero(lower > upper))
        raise ValueError(f"The budget and channel limits conflict. {details}. "
                         "Increase the budget or adjust channel bounds, percentage limits, or Search protection.")
    if lower.sum() > weekly_cap + tolerance:
        raise ValueError(f"The channel minimums require {horizon * lower.sum():,.2f}, above the budget cap {cap:,.2f}.")

    history, growth, beta, kappa, slope, _, _ = _forward_parts(bundle, horizon)
    samples = len(beta)
    rng = np.random.default_rng(options["random_seed"])
    requested = options["optimization_draws"]
    indices = np.arange(samples) if requested is None else np.sort(rng.choice(samples, min(samples, _positive_integer(requested, "optimization_draws")), replace=False))
    all_arrays = (history, growth, beta, np.log(kappa), slope)
    sampled_arrays = tuple(a[indices] for a in all_arrays)

    def evaluate(spend, subset=False, gradient=False):
        offset, g, b, log_k, sl = sampled_arrays if subset else all_arrays
        adstocked = offset + g * spend[None, None, :] / scale_x
        positive = adstocked > 0
        safe = np.where(positive, adstocked, 1.0)
        z = sl * (np.log(safe) - log_k)
        probability = expit(z)
        response = np.where(positive, scale_y * b * probability, 0.0)
        if not gradient:
            return response
        derivative = np.where(positive, scale_y * b * sl * probability * expit(-z) / safe * g / scale_x, 0.0)
        marginal = derivative.sum(axis=1).mean(axis=0)
        marginal[np.any(~positive, axis=(0, 1))] = np.nan
        return response, marginal

    def feasible(spend):
        return bool(np.isfinite(spend).all() and (spend >= lower - tolerance).all()
                    and (spend <= upper + tolerance).all() and spend.sum() <= weekly_cap + tolerance)

    free = np.flatnonzero(upper > lower)
    fixed_sum = lower[upper == lower].sum()
    lo, hi = lower[free] / money_scale, upper[free] / money_scale
    available = (weekly_cap - fixed_sum) / money_scale
    selected_total = min(weekly_cap, upper.sum())
    target = (selected_total - fixed_sum) / money_scale
    scaled_current = current * selected_total / current.sum()
    seed = _bounded_projection(scaled_current[free] / money_scale, lo, hi, target)

    def allocation(shares):
        result = lower.copy()
        result[free] = shares * money_scale
        return result

    best = allocation(seed)
    best_value = float(evaluate(best).sum(axis=(1, 2)).mean())
    solution_label = "Feasible historical-mix starting allocation"
    objective_scale = max(best_value, 1.0)
    analytic = bool(len(free) and np.all(lower[free] > 0))

    def objective(shares):
        spend = allocation(shares)
        if analytic:
            response, marginal = evaluate(spend, subset=True, gradient=True)
            jacobian = -money_scale * marginal[free] / objective_scale
            if not np.isfinite(jacobian).all():
                raise ValueError("Nonfinite optimizer gradient; review parameter support.")
            return -float(response.sum(axis=(1, 2)).mean()) / objective_scale, jacobian
        return -float(evaluate(spend, subset=True).sum(axis=(1, 2)).mean()) / objective_scale

    records = []
    if len(free) <= 1 or selected_total <= lower.sum() + tolerance or selected_total >= upper.sum() - tolerance:
        records.append({"start": 0, "success": True, "feasible": feasible(best), "expected_media_orders": best_value,
                        "message": "Direct monotone solution: allocation fixed by channel or total bounds"})
        solution_label = records[0]["message"]
    else:
        starting_points = [seed] + [_bounded_projection(rng.uniform(lo, hi), lo, hi, target) for _ in range(starts_count - 1)]
        constraint = {"type": "ineq", "fun": lambda q: available - q.sum(), "jac": lambda q: -np.ones_like(q)}
        for number, start in enumerate(starting_points, 1):
            result = minimize(objective, start, method="SLSQP", jac=True if analytic else None,
                              bounds=list(zip(lo, hi)), constraints=[constraint],
                              options={"maxiter": 1500, "ftol": 1e-10, "eps": 1e-7})
            candidate = allocation(result.x)
            is_feasible = feasible(candidate)
            value = float(evaluate(candidate).sum(axis=(1, 2)).mean()) if is_feasible else np.nan
            accepted = bool(result.success and is_feasible and np.isfinite(value))
            records.append({"start": number, "success": bool(result.success), "feasible": is_feasible,
                            "accepted": accepted, "expected_media_orders": value, "message": str(result.message)})
            if accepted and value > best_value:
                best, best_value, solution_label = candidate.copy(), value, f"Optimization start {number}"
        if not any(record.get("accepted", False) for record in records):
            raise RuntimeError("No optimization run converged to a feasible allocation. " + "; ".join(r["message"] for r in records))

    # Compare every candidate with a feasible same-budget reference using ALL draws.
    historical_mix = current * best.sum() / current.sum()
    feasible_mix = money_scale * _bounded_projection(historical_mix / money_scale, lower / money_scale,
                                                     upper / money_scale, best.sum() / money_scale)
    mix_value = float(evaluate(feasible_mix).sum(axis=(1, 2)).mean())
    if mix_value > best_value + 1e-10 * max(abs(best_value), 1.0):
        best, best_value = feasible_mix.copy(), mix_value
        solution_label = "Feasible same-budget reference retained on all posterior draws"
    if not feasible(best):
        raise RuntimeError("Selected optimizer solution is infeasible.")
    historical_mix = current * best.sum() / current.sum()
    np.testing.assert_allclose(historical_mix.sum(), best.sum(), atol=tolerance)
    np.testing.assert_allclose(feasible_mix.sum(), best.sum(), atol=tolerance)
    if not feasible(feasible_mix):
        raise RuntimeError("The equal-budget reference is not feasible under the selected constraints.")

    future_discount = options["future_discount_rate"]
    if future_discount is None:
        if "discount_rate" not in weekly:
            raise ValueError("Raw discount_rate is required for the default future discount assumption.")
        discount_values = pd.to_numeric(weekly["discount_rate"].iloc[-reference:], errors="raise")
        if not np.isfinite(discount_values.to_numpy(dtype=float)).all() or not discount_values.between(0, 1).all():
            raise ValueError("Historical discount rates must be finite fractions between 0 and 1.")
        future_discount = float(discount_values.mean())
    future_discount = _finite_amount(future_discount, "future_discount_rate")
    if future_discount > 1:
        raise ValueError("future_discount_rate must be a fraction from 0 to 1, not a percentage.")
    nonmedia, nonmedia_components = nonmedia_draws(bundle, future_dates, future_discount)
    scenario_spends = {"current": current, "reference": historical_mix, "optimized": best, "feasible_reference": feasible_mix}
    draws = {name: evaluate(spend) for name, spend in scenario_spends.items()}
    media_totals = {name: value.sum(axis=(1, 2)) for name, value in draws.items()}
    total_orders = {name: value + nonmedia.sum(axis=1) for name, value in media_totals.items()}
    total_forecasts = {name: value.sum(axis=2) + nonmedia for name, value in draws.items()}
    nonpositive_total = any(np.any(value.mean(axis=0) <= 0) for value in total_forecasts.values())
    warnings = []
    if nonpositive_total:
        warnings.append("At least one expected weekly total-order forecast is nonpositive. Total-order uplift percentages are unavailable; revise future controls/model assumptions.")

    def gains(comparison):
        gain = media_totals["optimized"] - media_totals[comparison]
        low, high = np.quantile(gain, [0.05, 0.95])
        denominator_media = float(media_totals[comparison].mean())
        denominator_total = float(total_orders[comparison].mean())
        return {"orders": float(gain.mean()), "lower_90": float(low), "upper_90": float(high),
                "probability_positive": float(np.mean(gain > 1e-10)),
                "pct_media": float(100 * gain.mean() / denominator_media) if denominator_media > 0 else None,
                "pct_total": float(100 * gain.mean() / denominator_total) if denominator_total > 0 and not nonpositive_total else None}

    labels = dict(CHANNEL_LABELS)
    labels.update(bundle.manifest.get("channel_labels", {}))
    current_channel = draws["current"].sum(axis=1).mean(axis=0)
    optimized_channel = draws["optimized"].sum(axis=1).mean(axis=0)
    reference_channel = draws["reference"].sum(axis=1).mean(axis=0)
    feasible_reference_channel = draws["feasible_reference"].sum(axis=1).mean(axis=0)
    safe_ratio = lambda spend, orders: np.divide(spend, orders, out=np.full(len(channels), np.nan), where=orders > 0)
    allocation_table = pd.DataFrame({
        "channel": channels, "channel_label": [labels.get(name, name) for name in channels],
        "historical_weekly_spend": current, "optimized_weekly_spend": best,
        "historical_period_spend": horizon * current, "optimized_period_spend": horizon * best,
        "spend_change_pct": np.divide(100 * (best - current), current, out=np.full(len(channels), np.nan), where=current > 0),
        "historical_media_orders": current_channel, "optimized_media_orders": optimized_channel,
        "historical_modeled_cpa": safe_ratio(horizon * current, current_channel),
        "optimized_modeled_cpa": safe_ratio(horizon * best, optimized_channel),
        "reference_weekly_spend": historical_mix, "reference_period_spend": horizon * historical_mix,
        "reference_media_orders": reference_channel,
        "reference_modeled_cpa": safe_ratio(horizon * historical_mix, reference_channel),
        "feasible_reference_weekly_spend": feasible_mix,
        "feasible_reference_period_spend": horizon * feasible_mix,
        "feasible_reference_media_orders": feasible_reference_channel,
        "feasible_reference_modeled_cpa": safe_ratio(horizon * feasible_mix, feasible_reference_channel),
        "min_weekly_spend": lower, "max_weekly_spend": upper, "search_protected": protected,
    }).sort_values("optimized_weekly_spend", ascending=False).reset_index(drop=True)
    weekly_plan = pd.DataFrame(np.tile(best, (horizon, 1)), columns=channels)
    weekly_plan.insert(0, "date", future_dates)
    weekly_plan["total_spend"] = best.sum()
    weekly_forecast = pd.DataFrame({"date": future_dates})
    for name, value in draws.items():
        media = value.sum(axis=2)
        total = total_forecasts[name]
        for measure, samples_array in [("media_orders", media), ("total_orders", total)]:
            weekly_forecast[f"{name}_{measure}"] = samples_array.mean(axis=0)
            q = np.quantile(samples_array, [0.05, 0.95], axis=0)
            weekly_forecast[f"{name}_{measure}_lower_90"] = q[0]
            weekly_forecast[f"{name}_{measure}_upper_90"] = q[1]
    summary = {
        "horizon_weeks": horizon, "reference_weeks": reference,
        "planning_start": str(future_dates[0].date()),
        "planning_end": str((future_dates[-1] + pd.Timedelta(days=6)).date()),
        "reference_start": str(weekly["date"].iloc[-reference].date()),
        "reference_end": str((weekly["date"].iloc[-1] + pd.Timedelta(days=6)).date()),
        "current_weekly_budget": float(current.sum()), "optimized_weekly_budget": float(best.sum()),
        "current_period_budget": float(horizon * current.sum()), "optimized_period_budget": float(horizon * best.sum()),
        "total_budget_cap": cap, "unused_budget_cap": float(max(0, cap - horizon * best.sum())),
        "current_total_media_orders": float(media_totals["current"].mean()),
        "reference_total_media_orders": float(media_totals["reference"].mean()),
        "feasible_reference_total_media_orders": float(media_totals["feasible_reference"].mean()),
        "optimized_total_media_orders": float(media_totals["optimized"].mean()),
        "current_total_orders": float(total_orders["current"].mean()),
        "reference_total_orders": float(total_orders["reference"].mean()),
        "feasible_reference_total_orders": float(total_orders["feasible_reference"].mean()),
        "optimized_total_orders": float(total_orders["optimized"].mean()),
        "gain_vs_current": gains("current"), "gain_vs_historical_same_budget": gains("reference"),
        "gain_vs_feasible_same_budget": gains("feasible_reference"),
        "gain_vs_same_budget_reference": gains("feasible_reference"),
        "headline_comparison": "feasible_reference",
        "headline_comparison_label": "Feasible equal-budget reference",
        "feasible_reference_adjusted_historical_proportions": not bool(np.allclose(feasible_mix, historical_mix, rtol=1e-8, atol=tolerance)),
        "same_budget_reference_feasible": feasible(historical_mix),
        "future_discount_rate": future_discount,
        "selected_solution": solution_label, "analytic_gradient": analytic,
        "optimization_draws": len(indices), "evaluation_draws": samples,
        "fitted_response_check": checkpoint,
        "warnings": warnings,
    }
    comparison = pd.DataFrame({"current_media_orders": media_totals["current"],
                               "reference_media_orders": media_totals["reference"],
                               "feasible_reference_media_orders": media_totals["feasible_reference"],
                               "optimized_media_orders": media_totals["optimized"],
                               "current_total_orders": total_orders["current"],
                               "reference_total_orders": total_orders["reference"],
                               "feasible_reference_total_orders": total_orders["feasible_reference"],
                               "optimized_total_orders": total_orders["optimized"]})
    comparison["gain_vs_historical_same_budget"] = media_totals["optimized"] - media_totals["reference"]
    comparison["gain_vs_feasible_same_budget"] = media_totals["optimized"] - media_totals["feasible_reference"]
    comparison["gain_vs_current"] = media_totals["optimized"] - media_totals["current"]
    basis = pd.DataFrame({"channel": channels, "historical_weekly_spend": current,
                          "historical_max_weekly_spend": observed.max(axis=0),
                          "percentage_change_limit": changes, "min_weekly_spend": lower,
                          "max_weekly_spend": upper, "search_protected": protected})
    return {"allocation": allocation_table, "weekly_plan": weekly_plan,
            "weekly_forecast": weekly_forecast, "summary": summary,
            "posterior_comparison": comparison, "spend_basis": basis,
            "solver_runs": pd.DataFrame(records), "settings": options}


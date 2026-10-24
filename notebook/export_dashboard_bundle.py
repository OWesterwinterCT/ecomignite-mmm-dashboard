"""Export fitted notebook results for the Streamlit app, without refitting.

Run inside the environment containing the fitted PyMC-Marketing model. The
portable bundle contains CSV, JSON and numeric NPZ files, never a pickle or a
serialized executable model. Exported posterior draws remain paired across
parameters. This exporter supports the current additive Ecomignite orders MMM:
normalized geometric adstock, Hill saturation, fixed intercept/media effects,
discount and trend controls, and annual Fourier seasonality.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile

import numpy as np
import pandas as pd


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
CONTROL_LABELS = {
    "discount_rate_scaled": "Discounts",
    "linear_trend_scaled": "Trend",
}


def _date_frame(frame, date_column="date"):
    out = frame.copy()
    if date_column not in out and "week_start" in out:
        out = out.rename(columns={"week_start": date_column})
    if date_column not in out:
        raise ValueError(f"The data needs a {date_column!r} column.")
    out[date_column] = pd.to_datetime(out[date_column], errors="raise")
    if out[date_column].isna().any() or out[date_column].duplicated().any():
        raise ValueError("Dates must be present and unique.")
    return out.sort_values(date_column).reset_index(drop=True)


def _dates_from_posterior(model):
    return pd.DatetimeIndex(pd.to_datetime(
        model.idata.posterior.coords["date"].values, errors="raise"
    ))


def _scale(model, name, channels=None):
    constant = getattr(model.idata, "constant_data", None)
    if constant is not None and name in constant:
        arr = constant[name]
        if channels is not None and "channel" in arr.dims:
            arr = arr.sel(channel=channels)
        value = np.asarray(arr.values, dtype=float)
    elif name in model.model.named_vars:
        value = np.asarray(model.model.named_vars[name].eval(), dtype=float)
    else:
        raise ValueError(f"Cannot find explicit {name}; no scale guessing is used.")
    if not np.isfinite(value).all() or not (value > 0).all():
        raise ValueError(f"{name} must be finite and positive.")
    return value


def _samples(arr, dimensions=(), selections=None):
    """Return samples first, preserving the same chain/draw stacking order."""
    if selections:
        arr = arr.sel(**selections)
    if "chain" in arr.dims and "draw" in arr.dims:
        arr = arr.stack(sample=("chain", "draw"))
    elif "sample" not in arr.dims:
        raise ValueError(f"{arr.name!r} has no posterior sampling dimensions.")
    extra = set(arr.dims) - {"sample", *dimensions}
    if extra:
        raise ValueError(f"Unexpected dimensions in {arr.name!r}: {sorted(extra)}")
    result = np.asarray(arr.transpose("sample", *dimensions).values, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"Nonfinite posterior samples in {arr.name!r}.")
    return result


def _summary_frame(dates, values, interval_level):
    tail = (1.0 - interval_level) / 2
    return pd.DataFrame({
        "date": dates,
        "predicted": values.mean(axis=0),
        "lower": np.quantile(values, tail, axis=0),
        "upper": np.quantile(values, 1 - tail, axis=0),
    })


def _contribution_samples(post, name, target_scale, dimensions, selections=None):
    original = f"{name}_original_scale"
    if original in post:
        return _samples(post[original], dimensions, selections)
    if name in post:
        return _samples(post[name], dimensions, selections) * target_scale
    raise ValueError(f"Required contribution {name!r} is absent from the posterior.")


def _validate_fit_table(frame, interval_level, expected_dates=None):
    out = _date_frame(frame)
    required = ["observed", "predicted"]
    if not set(required).issubset(out):
        raise ValueError("OOS table needs date, observed and predicted columns.")
    numeric = required + [c for c in ("lower", "upper") if c in out]
    if ("lower" in out) != ("upper" in out):
        raise ValueError("Supply both lower and upper OOS interval columns, or neither.")
    out[numeric] = out[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(out[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Fit table contains missing/nonfinite numerical values.")
    if (out["observed"] < 0).any():
        raise ValueError("Observed orders cannot be negative.")
    if "lower" in out and (out["lower"] > out["upper"]).any():
        raise ValueError("OOS interval lower bounds exceed upper bounds.")
    if expected_dates is not None and not pd.DatetimeIndex(out.date).equals(expected_dates):
        raise ValueError("OOS predictions must align exactly with the holdout dates.")
    if len(out) > 1 and not out.date.diff().dropna().eq(pd.Timedelta(days=7)).all():
        raise ValueError("The OOS holdout must use consecutive weekly dates.")
    return out[["date", *numeric]]


def _oos_from_model(model, X, y, interval_level):
    """Evaluate an already-fitted training-only model; never fit or tune it."""
    supplied_dates = pd.DatetimeIndex(pd.to_datetime(X["date"], errors="raise"))
    if not supplied_dates.is_monotonic_increasing:
        raise ValueError("Sort OOS X and y together by date before exporting.")
    frame = _date_frame(X)
    training_dates = _dates_from_posterior(model)
    if len(frame) == 0 or frame.date.iloc[0] != training_dates.max() + pd.Timedelta(days=7):
        raise ValueError("OOS must begin one week after the training-only model ends.")
    observed = np.asarray(y, dtype=float).reshape(-1)
    if len(observed) != len(frame):
        raise ValueError("OOS X and y have different lengths.")
    # A dated y Series must match X; positional y arrays retain their supplied order.
    if isinstance(y, pd.Series) and isinstance(y.index, pd.DatetimeIndex):
        if not y.index.equals(pd.DatetimeIndex(frame.date)):
            raise ValueError("The OOS y dates do not align with X.")
    arguments = {
        "extend_idata": False,
        "combined": True,
        "include_last_observations": True,
        "var_names": ["mu"],
        "random_seed": 42,
        "progressbar": False,
    }
    signature = inspect.signature(model.sample_posterior_predictive).parameters
    if "original_scale" in signature:
        arguments["original_scale"] = False
    pp = model.sample_posterior_predictive(frame, **arguments)
    if hasattr(pp, "posterior_predictive"):
        pp = pp.posterior_predictive
    if "mu" not in pp:
        raise ValueError("OOS sampling did not return mu. Supply an explicit OOS fit table instead.")
    mu = pp["mu"].sel(date=frame.date.to_numpy())
    mu = _samples(mu, ("date",)) * float(_scale(model, "target_scale").item())
    out = _summary_frame(frame.date.to_numpy(), mu, interval_level)
    out.insert(1, "observed", observed)
    return _validate_fit_table(out, interval_level, pd.DatetimeIndex(frame.date))


def _diagnostics(model):
    """Sampling checks are optional; a missing diagnostic is not a passing one."""
    result = {}
    sample_stats = getattr(model.idata, "sample_stats", None)
    if sample_stats is not None and "diverging" in sample_stats:
        result["divergences"] = int(sample_stats["diverging"].sum().item())
    try:
        import arviz as az
        names = [rv.name for rv in model.model.free_RVs if rv.name in model.idata.posterior]
        summary = az.summary(model.idata, var_names=names, kind="diagnostics", round_to="none")
        for key, column, operation in [
            ("max_rhat", "r_hat", "max"),
            ("min_bulk_ess", "ess_bulk", "min"),
            ("min_tail_ess", "ess_tail", "min"),
        ]:
            value = float(getattr(summary[column], operation)(skipna=False))
            result[key] = value if np.isfinite(value) else None
        bfmi = np.asarray(az.bfmi(model.idata), dtype=float)
        result["min_bfmi"] = float(bfmi.min()) if np.isfinite(bfmi).all() else None
    except (ImportError, AttributeError, KeyError, TypeError, ValueError):
        result["details"] = "Some sampling diagnostics could not be exported."
    return result


def _control_references(control_scaling):
    """Validate declared anchored references; never infer them from z-scores."""
    scaling = dict(control_scaling or {})
    if not scaling:
        return scaling
    required = {"discount_unit", "trend_origin", "trend_span_days"}
    missing = required - set(scaling)
    if missing:
        raise ValueError(f"CONTROL_SCALING is incomplete: {sorted(missing)}")
    origin = pd.Timestamp(scaling["trend_origin"])
    if pd.isna(origin):
        raise ValueError("CONTROL_SCALING trend_origin must be a valid date.")
    scaling["trend_origin"] = origin.date().isoformat()
    for name in ("discount_unit", "trend_span_days"):
        value = float(scaling[name])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"CONTROL_SCALING {name} must be finite and positive.")
        scaling[name] = value
    return scaling


def _audit_table(frame, index_name):
    """Keep optional aggregate OOS tables portable and JSON-safe."""
    if frame is None:
        return None, []
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Optional OOS metrics/quality must be pandas DataFrames.")
    out = frame.copy()
    if not isinstance(out.index, pd.RangeIndex) or out.index.name is not None:
        out = out.rename_axis(index_name).reset_index()
    if not out.columns.is_unique:
        raise ValueError("Optional OOS tables contain duplicate columns.")
    # Pandas serializes NaN/infinite diagnostic values to null, not invalid JSON.
    records = json.loads(out.to_json(orient="records", date_format="iso"))
    return out, records


def export_dashboard_bundle(
    mmm,
    fitted_data,
    output_dir,
    *,
    control_scaling=None,
    weekly_validation=None,
    daily_validation=None,
    oos_fit=None,
    oos_model=None,
    oos_X=None,
    oos_y=None,
    oos_training_end=None,
    oos_evaluation_type=None,
    oos_metrics=None,
    oos_input_quality=None,
    fit_interval_level=0.90,
    oos_intervals_type="expected_mean",
    currency="",
    max_posterior_draws=2000,
    random_seed=42,
):
    """Create a real-data bundle directory and ZIP; return their paths.

    `fitted_data` must be the exact weekly frame used by `mmm`, with target in
    original orders and media in original currency. Optional validation frames
    are joined by date without replacing any model inputs. Supply either an OOS
    table in original order units, or a separate training-only fitted model plus
    holdout X/y. For a direct OOS table, `oos_training_end` records the last date
    used to estimate its model. The supplied table must contain every trailing
    fitted-period week after that cutoff. Optional OOS metadata describes the
    evaluation and its aggregate metrics/input quality, without rerunning it.
    """
    if not 0 < float(fit_interval_level) < 1:
        raise ValueError("fit_interval_level must be between 0 and 1.")
    if max_posterior_draws is not None and (
        isinstance(max_posterior_draws, bool)
        or not isinstance(max_posterior_draws, (int, np.integer))
        or max_posterior_draws < 2
    ):
        raise ValueError("max_posterior_draws must be at least 2 or None.")
    if getattr(mmm, "time_varying_media", False) or getattr(mmm, "time_varying_intercept", False):
        raise ValueError("This dashboard contract requires fixed media parameters/intercept.")
    if type(mmm.adstock).__name__ != "GeometricAdstock" or not bool(mmm.adstock.normalize):
        raise ValueError("The dashboard requires normalized GeometricAdstock.")
    if not getattr(mmm, "adstock_first", True):
        raise ValueError("The dashboard requires adstock before Hill saturation.")
    if type(mmm.saturation).__name__ not in {"HillSaturation", "ZeroSafeHillSaturation"}:
        raise ValueError("The dashboard requires Hill or ZeroSafeHill saturation.")

    post = mmm.idata.posterior
    dates = _dates_from_posterior(mmm)
    if dates.has_duplicates or dates.isna().any() or not dates.is_monotonic_increasing:
        raise ValueError("Fitted posterior dates must be unique and increasing.")
    if not (dates.dayofweek == 0).all() or (
        len(dates) > 1 and not (np.diff(dates.values) == np.timedelta64(7, "D")).all()
    ):
        raise ValueError("Fitted data must contain consecutive Monday-starting weeks.")
    channels = [str(c) for c in post.coords["channel"].values]
    controls = [str(c) for c in post.coords["control"].values] if "control" in post.coords else []
    scaling = _control_references(control_scaling)
    data = _date_frame(fitted_data)
    if not pd.DatetimeIndex(data.date).equals(dates):
        raise ValueError("fitted_data must match every posterior fitting date, in order.")
    required = ["eligible_orders", *channels, *controls]
    missing = [c for c in required if c not in data]
    if missing:
        raise ValueError(f"Missing fitted-data columns: {missing}")
    data[required] = data[required].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(data[required].to_numpy(dtype=float)).all():
        raise ValueError("Fitted target, media and controls must be finite; no missing values are filled.")
    if (data[["eligible_orders", *channels]] < 0).any().any():
        raise ValueError("Orders and media spend must be nonnegative.")
    # Restrict exported fields to aggregate model inputs/quality checks.
    quality_names = {
        "days", "meta_imputed_days", "google_imputed_days", "google_backfilled_days",
        "meta_imputed_share", "google_imputed_share", "meta_imputed_spend",
        "google_imputed_spend", "any_imputation", "high_imputation_week",
        "meta_record_missing", "google_record_missing", "discount_amount",
        "gross_revenue", "refund_amount", "discount_rate", "refund_rate",
    }
    for prefix in ("meta", "google"):
        quality_names.update(f"{prefix}_{suffix}" for suffix in (
            "source_missing", "source_missing_days", "spend_imputed",
            "backfilled_days", "record_missing_days",
        ))
    for channel in channels:
        quality_names.update(f"{channel}_{suffix}" for suffix in (
            "original", "imputed", "imputed_days", "imputed_spend",
            "imputed_share", "backfilled_days", "record_missing",
        ))
    exported_columns = ["date", "eligible_orders", *channels, *controls]
    exported_columns += [c for c in data if c in quality_names or c == "net_revenue"]
    data = data[list(dict.fromkeys(exported_columns))]
    if weekly_validation is not None:
        quality = _date_frame(weekly_validation)
        extras = [c for c in quality if c in quality_names or c == "net_revenue"]
        for column in extras:
            joined = quality.set_index("date")[column].reindex(dates).to_numpy()
            if column not in data:
                data[column] = joined
            else:
                data[column] = data[column].where(data[column].notna(), joined)

    if scaling and "discount_rate_scaled" in controls:
        # The declared transform is anchored at no discount, not mean-centered.
        # Inverting its exact unit recovers raw rates even when X only kept controls.
        raw_discount = data["discount_rate_scaled"] * scaling["discount_unit"]
        if "discount_rate" not in data:
            data["discount_rate"] = raw_discount
        else:
            known_discount = pd.to_numeric(data["discount_rate"], errors="raise")
            data["discount_rate"] = known_discount.where(known_discount.notna(), raw_discount)
        np.testing.assert_allclose(data.discount_rate, raw_discount, rtol=1e-7, atol=1e-10,
            err_msg="Raw and scaled discounts disagree with CONTROL_SCALING; check anchored controls.")
        if not data.discount_rate.between(0, 1).all():
            raise ValueError("Discount rates must be fractions in [0, 1]; check control units/centering.")
    if scaling and "linear_trend_scaled" in controls:
        declared_trend = ((data.date - pd.Timestamp(scaling["trend_origin"])).dt.days
                          / scaling["trend_span_days"])
        np.testing.assert_allclose(data.linear_trend_scaled, declared_trend, rtol=1e-7, atol=1e-10,
            err_msg="Trend values disagree with CONTROL_SCALING; rerun anchored-control preparation.")

    scale_y = float(_scale(mmm, "target_scale").item())
    scale_x = _scale(mmm, "channel_scale", channels).reshape(-1)
    if scale_x.shape != (len(channels),):
        raise ValueError("channel_scale must have one value per fitted channel.")
    mu = _samples(post["mu"], ("date",), {"date": dates.values}) * scale_y
    fit = _summary_frame(dates, mu, fit_interval_level)
    fit.insert(1, "observed", data["eligible_orders"].to_numpy())

    effects = []
    component_sum = np.zeros_like(mu)

    def add_effect(name, group, values):
        nonlocal component_sum
        if values.shape != mu.shape:
            raise ValueError(f"Contribution {name} has a different shape from mu.")
        component_sum += values
        summary = _summary_frame(dates, values, fit_interval_level)
        summary = summary.rename(columns={"predicted": "mean"})
        summary.insert(1, "effect", name)
        summary.insert(2, "group", group)
        effects.append(summary)

    media = _contribution_samples(post, "channel_contribution", scale_y,
        ("date", "channel"), {"date": dates.values, "channel": channels})
    for i, channel in enumerate(channels):
        add_effect(channel, "Marketing", media[:, :, i])
    if controls:
        context = _contribution_samples(post, "control_contribution", scale_y,
            ("date", "control"), {"date": dates.values, "control": controls})
        for i, control in enumerate(controls):
            add_effect(control, "Context", context[:, :, i])
    yearly_order = int(getattr(mmm, "yearly_seasonality", 0) or 0)
    if yearly_order:
        seasonal = _contribution_samples(post, "yearly_seasonality_contribution", scale_y,
            ("date",), {"date": dates.values})
        add_effect("Seasonality", "Seasonality", seasonal)
    else:
        add_effect("Seasonality", "Seasonality", np.zeros_like(mu))
    intercept = _samples(post["intercept"]) * scale_y
    add_effect("Intercept", "Intercept", np.broadcast_to(intercept[:, None], mu.shape))
    np.testing.assert_allclose(component_sum, mu, rtol=1e-5, atol=1e-5,
        err_msg="Contributions do not reconstruct mu; check target scaling/model specification.")
    contributions = pd.concat(effects, ignore_index=True)

    names = {"alpha": "adstock_alpha", "beta": "saturation_beta",
             "kappa": "saturation_kappa", "slope": "saturation_slope"}
    samples = {key: _samples(post[name], ("channel",), {"channel": channels})
               for key, name in names.items()}
    sample_count = samples["alpha"].shape[0]
    if sample_count < 2:
        raise ValueError("At least two paired posterior draws are needed.")
    if any(value.shape != (sample_count, len(channels)) for value in samples.values()):
        raise ValueError("All media parameters must have paired sample/channel shapes.")
    if ((samples["alpha"] < 0) | (samples["alpha"] > 1)).any():
        raise ValueError("Adstock alpha must lie in [0, 1].")
    if any((samples[key] <= 0).any() for key in ("beta", "kappa", "slope")):
        raise ValueError("Hill beta, kappa and slope must be positive.")
    keep = np.arange(sample_count)
    if max_posterior_draws is not None and sample_count > max_posterior_draws:
        keep = np.sort(np.random.default_rng(random_seed).choice(
            sample_count, size=int(max_posterior_draws), replace=False))
    arrays = {key: values[keep] for key, values in samples.items()}
    arrays.update(channel_scale=scale_x, target_scale=np.asarray(scale_y), intercept=intercept[keep] / scale_y)
    # Model-scale references let the dashboard verify its forward-response
    # implementation against the fitted model using the same retained draws.
    arrays["fitted_last_channel_contribution"] = media[keep, -1, :] / scale_y
    arrays["fitted_last_mu"] = mu[keep, -1] / scale_y
    arrays["gamma_control"] = (
        _samples(post["gamma_control"], ("control",), {"control": controls})[keep]
        if controls else np.empty((len(keep), 0))
    )
    fourier_names = []
    if yearly_order:
        fourier = post["gamma_fourier"]
        fourier_dims = [d for d in fourier.dims if d not in {"chain", "draw"}]
        if len(fourier_dims) != 1:
            raise ValueError("gamma_fourier must have one Fourier coefficient dimension.")
        fourier_dimension = fourier_dims[0]
        fourier_names = [str(v) for v in fourier.coords[fourier_dimension].values]
        arrays["gamma_fourier"] = _samples(fourier, (fourier_dimension,))[keep]
        if len(fourier_names) != 2 * yearly_order:
            raise ValueError("Fourier coefficient count does not match yearly_seasonality.")
    else:
        arrays["gamma_fourier"] = np.empty((len(keep), 0))
    if "y_sigma" in post:
        arrays["noise_sigma"] = _samples(post["y_sigma"])[keep]

    if oos_fit is not None and oos_model is not None:
        raise ValueError("Supply an OOS table OR an OOS model, not both.")
    allowed_evaluations = {
        None, "conditional_on_realized_media_and_discounts",
        "forecast_with_planned_inputs",
    }
    if oos_evaluation_type not in allowed_evaluations:
        raise ValueError("Unsupported oos_evaluation_type.")
    metric_table, metric_records = _audit_table(oos_metrics, "model")
    quality_table, quality_records = _audit_table(oos_input_quality, "channel")
    oos = None
    oos_info = {"status": "unavailable", "reason": "No training-only OOS evaluation was supplied."}
    if oos_fit is not None:
        if oos_training_end is None:
            raise ValueError("A direct OOS table requires oos_training_end to document the training cutoff.")
        oos = _validate_fit_table(oos_fit, fit_interval_level)
        cutoff = pd.Timestamp(oos_training_end)
        if pd.isna(cutoff) or len(oos) == 0 or cutoff >= oos.date.min():
            raise ValueError("The OOS training cutoff must precede the holdout.")
        if oos_intervals_type not in {"expected_mean", "posterior_predictive", "unavailable"}:
            raise ValueError("Invalid oos_intervals_type.")
        oos_info = {"status": "available", "source": "supplied_holdout_table",
                    "training_end": cutoff.date().isoformat(),
                    "intervals_type": oos_intervals_type if "lower" in oos else "unavailable"}
    elif oos_model is not None:
        if oos_model is mmm:
            raise ValueError("OOS must use a separate training-only model, not the full-period fitted model.")
        if oos_X is None or oos_y is None:
            raise ValueError("An OOS model requires oos_X and oos_y.")
        oos = _oos_from_model(oos_model, oos_X, oos_y, fit_interval_level)
        oos_info = {"status": "available", "source": "training_only_model",
                    "training_end": _dates_from_posterior(oos_model).max().date().isoformat(),
                    "intervals_type": "expected_mean"}
    if oos is not None:
        cutoff = pd.Timestamp(oos_info["training_end"])
        if cutoff not in dates:
            raise ValueError("OOS training cutoff must be a fitted-period week.")
        if not pd.DatetimeIndex(oos.date).equals(dates[dates > cutoff]):
            raise ValueError("OOS must include every trailing week after the training cutoff.")
        if not set(oos.date).issubset(set(dates)):
            raise ValueError("OOS dates must be held-out weeks within the full-period dashboard dataset.")
        expected_y = data.set_index("date").loc[oos.date, "eligible_orders"].to_numpy()
        np.testing.assert_allclose(oos.observed, expected_y, rtol=1e-8, atol=1e-8,
            err_msg="OOS observed orders differ from the full-period data.")
        oos_info.update(weeks=len(oos), start=oos.date.min().date().isoformat(),
                        end=oos.date.max().date().isoformat(),
                        evaluation_type=oos_evaluation_type or "not_supplied",
                        metrics=metric_records, input_quality=quality_records)
    elif metric_table is not None or quality_table is not None or oos_evaluation_type is not None:
        raise ValueError("OOS metadata requires an OOS fit table or training-only model.")

    # Do not infer centering or pretend unknown references are known.
    scaling_available = all(k in scaling for k in ("discount_unit", "trend_origin", "trend_span_days"))
    manifest = {
        "schema_version": 1, "kind": "real", "model_name": "Ecomignite Orders MMM",
        "exporter_version": "2026.10.24", "spend_scope": "modeled_channels_only",
        "target_column": "eligible_orders", "target_label": "Orders", "date_column": "date",
        "granularity": "weekly", "week_starts": "Monday", "currency": str(currency),
        "channels": channels, "channel_labels": {c: CHANNEL_LABELS.get(c, c) for c in channels},
        "channel_notes": ({"google_video_spend": "Google Video can include YouTube and video partners; other Google campaign types may also deliver on YouTube."}
                          if "google_video_spend" in channels else {}),
        "controls": controls, "control_labels": {c: CONTROL_LABELS.get(c, c) for c in controls},
        "adstock_l_max": int(mmm.adstock.l_max), "adstock_normalize": True, "adstock_first": True,
        "saturation": "Hill", "yearly_seasonality": yearly_order, "fourier_names": fourier_names,
        "control_scaling": scaling, "control_scaling_available": scaling_available,
        "fit_interval_level": float(fit_interval_level), "fit_intervals_type": "expected_mean",
        "training_start": dates.min().date().isoformat(), "training_end": dates.max().date().isoformat(),
        "posterior_draws_available": sample_count, "posterior_draws_exported": len(keep),
        "diagnostics": _diagnostics(mmm), "oos": oos_info,
        "validation": {"weekly_fields": [c for c in data if c in quality_names],
                       "daily_available": daily_validation is not None,
                       "note": "Missing observations remain missing; absent quality flags do not imply complete data."},
    }

    output = Path(output_dir).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Complete and validate in a temporary directory before writing deliverables.
    with TemporaryDirectory(prefix="ecomignite_dashboard_export_") as temporary:
        stage = Path(temporary)
        data.to_csv(stage / "weekly_data.csv", index=False, date_format="%Y-%m-%d")
        fit.to_csv(stage / "fit_in_sample.csv", index=False, date_format="%Y-%m-%d")
        contributions.to_csv(stage / "contributions.csv", index=False, date_format="%Y-%m-%d")
        np.savez_compressed(stage / "posterior.npz", **arrays)
        if oos is not None:
            oos.to_csv(stage / "fit_oos.csv", index=False, date_format="%Y-%m-%d")
        if metric_table is not None:
            metric_table.to_csv(stage / "oos_metrics.csv", index=False)
        if quality_table is not None:
            quality_table.to_csv(stage / "oos_input_quality.csv", index=False)
        if daily_validation is not None:
            daily = _date_frame(daily_validation)
            columns = ["date"] + [c for c in daily if c in quality_names or c in channels]
            daily[columns].to_csv(stage / "validation_daily.csv", index=False, date_format="%Y-%m-%d")
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
        output.mkdir(parents=True, exist_ok=True)
        for filename in ("fit_oos.csv", "validation_daily.csv", "oos_metrics.csv", "oos_input_quality.csv"):
            old = output / filename
            if not (stage / filename).exists() and old.exists():
                old.unlink()  # Avoid displaying stale optional exports from a prior run.
        for path in stage.iterdir():
            (output / path.name).write_bytes(path.read_bytes())
        archive = output.with_suffix(".zip")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(stage.iterdir()):
                zf.write(path, arcname=path.name)
    print("Dashboard bundle:", output)
    print("Uploadable ZIP:", archive)
    print("OOS:", oos_info["status"])
    return {"bundle_dir": output, "zip_path": archive, "manifest": manifest}


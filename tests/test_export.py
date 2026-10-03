"""Notebook exporter checks using a numeric xarray mock, without fitting PyMC."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ecomignite.data import load_bundle, validation_summary
from notebook.export_dashboard_bundle import export_dashboard_bundle


@pytest.fixture
def fitted_mock():
    """Paired two-chain draws and an exactly additive orders decomposition."""
    rng = np.random.default_rng(19)
    T, C, K, chains, draws = 30, 8, 2, 2, 12
    dates = pd.date_range("2025-01-06", periods=T, freq="7D")
    channels = ["meta_conversion_spend", "meta_engagement_spend", "meta_traffic_spend",
        "meta_thruplay_spend", "meta_other_spend", "google_search_spend",
        "google_pmax_spend", "google_display_spend"]
    controls = ["discount_rate_scaled", "linear_trend_scaled"]
    # Distinct draw values make sample-order mistakes detectable.
    alpha = rng.uniform(.05, .5, (chains, draws, C))
    beta = alpha + .1
    kappa = alpha + 1.
    slope = alpha + 1.2
    intercept = rng.uniform(.1, .3, (chains, draws))
    media = rng.uniform(.01, .2, (chains, draws, T, C))
    context = rng.uniform(0, .03, (chains, draws, T, K))
    seasonal = rng.normal(0, .01, (chains, draws, T))
    mu = media.sum(axis=3) + context.sum(axis=3) + seasonal + intercept[:, :, None]
    posterior = xr.Dataset({
        "adstock_alpha": (("chain", "draw", "channel"), alpha),
        "saturation_beta": (("chain", "draw", "channel"), beta),
        "saturation_kappa": (("chain", "draw", "channel"), kappa),
        "saturation_slope": (("chain", "draw", "channel"), slope),
        "intercept": (("chain", "draw"), intercept),
        "gamma_control": (("chain", "draw", "control"), np.broadcast_to([.02, .03], (chains, draws, K))),
        "gamma_fourier": (("chain", "draw", "fourier_mode"), np.broadcast_to([.02, -.01, .03, .01], (chains, draws, 4))),
        "mu": (("chain", "draw", "date"), mu),
        "channel_contribution": (("chain", "draw", "date", "channel"), media),
        "control_contribution": (("chain", "draw", "date", "control"), context),
        "yearly_seasonality_contribution": (("chain", "draw", "date"), seasonal),
        "y_sigma": (("chain", "draw"), np.full((chains, draws), .05)),
    }, coords={"chain": np.arange(chains), "draw": np.arange(draws), "date": dates,
        "channel": channels, "control": controls, "fourier_mode": ["sin_1", "sin_2", "cos_1", "cos_2"]})
    constants = xr.Dataset({
        "channel_scale": ("channel", np.full(C, 1000.)),
        "target_scale": xr.DataArray(500.),
    }, coords={"channel": channels})
    GeometricAdstock = type("GeometricAdstock", (), {})
    HillSaturation = type("HillSaturation", (), {})
    adstock = GeometricAdstock()
    adstock.normalize = True
    adstock.l_max = 8
    model = SimpleNamespace(
        idata=SimpleNamespace(posterior=posterior, constant_data=constants),
        model=SimpleNamespace(named_vars={}, free_RVs=[]), adstock=adstock,
        saturation=HillSaturation(), yearly_seasonality=2, adstock_first=True,
        time_varying_media=False, time_varying_intercept=False,
    )
    data = pd.DataFrame({"date": dates, "eligible_orders": mu.mean((0, 1)) * 500,
        "net_revenue": np.full(T, 10000.), "discount_rate_scaled": np.full(T, .1),
        "linear_trend_scaled": np.arange(T) / (T - 1)})
    for channel in channels:
        data[channel] = rng.uniform(1, 1000, T)
    quality = pd.DataFrame({"week_start": dates, "google_imputed_days": 0})
    quality.loc[3, "google_imputed_days"] = 2
    scaling = {"discount_unit": .1, "trend_origin": dates.min(),
               "trend_span_days": (dates.max() - dates.min()).days}
    return model, data, quality, scaling


def test_paired_numeric_zip_and_exact_reconstruction(tmp_path, fitted_mock):
    model, data, quality, scaling = fitted_mock
    result = export_dashboard_bundle(model, data, tmp_path / "bundle",
        control_scaling=scaling, weekly_validation=quality, max_posterior_draws=10)
    bundle = load_bundle(result["zip_path"])
    assert bundle.posterior["alpha"].shape == (10, 8)
    np.testing.assert_allclose(bundle.posterior["beta"], bundle.posterior["alpha"] + .1)
    np.testing.assert_allclose(bundle.posterior["kappa"], bundle.posterior["alpha"] + 1.)
    np.testing.assert_allclose(bundle.posterior["slope"], bundle.posterior["alpha"] + 1.2)
    assert bundle.manifest["kind"] == "real"
    assert bundle.manifest["posterior_draws_available"] == 24
    assert bundle.manifest["posterior_draws_exported"] == 10
    assert bundle.weekly.google_imputed_days.sum() == 2
    np.testing.assert_allclose(bundle.weekly.discount_rate, .01)
    assert validation_summary(bundle)[2] < 1e-9
    assert bundle.fit_oos.empty
    assert bundle.manifest["oos"]["status"] == "unavailable"
    assert not (Path(result["bundle_dir"]) / "fit_oos.csv").exists()


def test_holdout_cutoff_and_stale_oos_file_cleanup(tmp_path, fitted_mock):
    model, data, _, scaling = fitted_mock
    holdout = data.tail(12)[["date", "eligible_orders"]].rename(columns={"eligible_orders": "observed"})
    holdout["predicted"] = holdout.observed + 1.
    result = export_dashboard_bundle(model, data, tmp_path / "bundle", control_scaling=scaling,
        oos_fit=holdout, oos_training_end=data.date.iloc[-13], max_posterior_draws=None)
    bundle = load_bundle(result["zip_path"])
    assert len(bundle.fit_oos) == 12
    assert bundle.manifest["oos"]["intervals_type"] == "unavailable"
    assert bundle.manifest["oos"]["training_end"] == str(data.date.iloc[-13].date())
    with pytest.raises(ValueError, match="training cutoff"):
        export_dashboard_bundle(model, data, tmp_path / "invalid", control_scaling=scaling,
            oos_fit=holdout, oos_training_end=data.date.iloc[-1])
    with pytest.raises(ValueError, match="requires oos_training_end"):
        export_dashboard_bundle(model, data, tmp_path / "invalid", control_scaling=scaling,
            oos_fit=holdout)
    export_dashboard_bundle(model, data, tmp_path / "bundle", control_scaling=scaling)
    assert not (tmp_path / "bundle" / "fit_oos.csv").exists()
    assert load_bundle(tmp_path / "bundle.zip").fit_oos.empty


def test_wrong_fit_dates_and_centered_controls_are_rejected(tmp_path, fitted_mock):
    model, data, _, scaling = fitted_mock
    with pytest.raises(ValueError, match="every posterior fitting date"):
        export_dashboard_bundle(model, data.iloc[:-1], tmp_path / "missing", control_scaling=scaling)
    centered = data.copy()
    centered["discount_rate"] = .01
    centered["discount_rate_scaled"] = 0.
    with pytest.raises(AssertionError, match="Raw and scaled discounts disagree"):
        export_dashboard_bundle(model, centered, tmp_path / "centered", control_scaling=scaling)
    with pytest.raises(ValueError, match="CONTROL_SCALING is incomplete"):
        export_dashboard_bundle(model, data, tmp_path / "incomplete", control_scaling={"discount_unit": .1})

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
    T, C, K, chains, draws = 30, 9, 2, 2, 12
    dates = pd.date_range("2025-01-06", periods=T, freq="7D")
    channels = ["meta_conversion_spend", "meta_engagement_spend", "meta_traffic_spend",
        "google_search_spend", "google_pmax_spend", "google_display_spend",
        "google_video_spend", "google_shopping_spend", "google_demand_gen_spend"]
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
    assert bundle.posterior["alpha"].shape == (10, 9)
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



def test_new_channels_quality_and_paired_reference_arrays(tmp_path, fitted_mock):
    model, data, quality, scaling = fitted_mock
    quality["google_video_spend_imputed_days"] = 0
    quality.loc[4, "google_video_spend_imputed_days"] = 3
    quality["google_smart_spend"] = 100.0
    daily = pd.DataFrame({
        "date": data.date,
        "google_video_spend": 100.0,
        "google_video_spend_original": 100.0,
        "google_video_spend_imputed": False,
        "customer_email": "not_exported@example.test",
    })
    result = export_dashboard_bundle(
        model, data, tmp_path / "bundle", control_scaling=scaling,
        weekly_validation=quality, daily_validation=daily,
        max_posterior_draws=10,
    )
    bundle = load_bundle(result["zip_path"])
    labels = bundle.manifest["channel_labels"]
    assert labels["google_video_spend"] == "Google Video"
    assert labels["google_shopping_spend"] == "Google Shopping"
    assert labels["google_demand_gen_spend"] == "Google Demand Gen"
    assert "google_smart_spend" not in bundle.weekly
    assert "customer_email" not in bundle.validation_daily
    assert "google_video_spend_original" in bundle.validation_daily
    assert bundle.weekly.google_video_spend_imputed_days.sum() == 3
    assert bundle.manifest["spend_scope"] == "modeled_channels_only"
    assert "video partners" in bundle.manifest["channel_notes"]["google_video_spend"]

    # Exported reference arrays must use the same sampled draw IDs as parameters.
    full_alpha = model.idata.posterior.adstock_alpha.values.reshape(24, 9)
    draw_ids = [np.flatnonzero(np.all(full_alpha == row, axis=1)).item()
                for row in bundle.posterior["alpha"]]
    last_channel = model.idata.posterior.channel_contribution.values[:, :, -1, :].reshape(24, 9)
    last_mu = model.idata.posterior.mu.values[:, :, -1].reshape(24)
    np.testing.assert_allclose(
        bundle.posterior["fitted_last_channel_contribution"], last_channel[draw_ids]
    )
    np.testing.assert_allclose(bundle.posterior["fitted_last_mu"], last_mu[draw_ids])


def test_oos_metadata_and_complete_holdout_are_preserved(tmp_path, fitted_mock):
    model, data, _, scaling = fitted_mock
    holdout = data.tail(12)[["date", "eligible_orders"]].rename(columns={"eligible_orders": "observed"})
    holdout["predicted"] = holdout.observed + 1
    holdout["lower"] = holdout.predicted - 50
    holdout["upper"] = holdout.predicted + 50
    metrics = pd.DataFrame({"MAPE (%)": [5.0], "Correlation": [np.nan]}, index=["Bayesian MMM"])
    quality = pd.DataFrame({"training_missing_days": [12], "holdout_missing_days": [0]},
                           index=["google_video_spend"])
    kwargs = dict(control_scaling=scaling, oos_fit=holdout,
                  oos_training_end=data.date.iloc[-13],
                  oos_intervals_type="posterior_predictive",
                  oos_evaluation_type="conditional_on_realized_media_and_discounts",
                  oos_metrics=metrics, oos_input_quality=quality)
    result = export_dashboard_bundle(model, data, tmp_path / "bundle", **kwargs)
    bundle = load_bundle(result["zip_path"])
    oos = bundle.manifest["oos"]
    assert oos["weeks"] == 12
    assert oos["intervals_type"] == "posterior_predictive"
    assert oos["evaluation_type"] == "conditional_on_realized_media_and_discounts"
    assert oos["metrics"][0]["model"] == "Bayesian MMM"
    assert oos["metrics"][0]["Correlation"] is None
    assert oos["input_quality"][0]["channel"] == "google_video_spend"
    assert (tmp_path / "bundle" / "oos_metrics.csv").is_file()
    assert (tmp_path / "bundle" / "oos_input_quality.csv").is_file()
    np.testing.assert_allclose(bundle.fit_oos.observed, holdout.observed)
    np.testing.assert_allclose(bundle.fit_oos.lower, holdout.lower)

    for shortened in [holdout.iloc[1:], holdout.iloc[:-1]]:
        with pytest.raises(ValueError, match="every trailing week"):
            export_dashboard_bundle(
                model, data, tmp_path / "invalid", control_scaling=scaling,
                oos_fit=shortened, oos_training_end=data.date.iloc[-13],
            )
    export_dashboard_bundle(model, data, tmp_path / "bundle", control_scaling=scaling)
    assert not (tmp_path / "bundle" / "oos_metrics.csv").exists()
    assert not (tmp_path / "bundle" / "oos_input_quality.csv").exists()


def test_original_scale_contributions_are_not_scaled_twice(tmp_path, fitted_mock):
    model, data, _, scaling = fitted_mock
    for name in ["channel_contribution", "control_contribution", "yearly_seasonality_contribution"]:
        model.idata.posterior[name + "_original_scale"] = model.idata.posterior[name] * 500.0
    result = export_dashboard_bundle(model, data, tmp_path / "bundle", control_scaling=scaling)
    bundle = load_bundle(result["zip_path"])
    assert validation_summary(bundle)[2] < 1e-9


def test_oos_metadata_without_holdout_is_rejected(tmp_path, fitted_mock):
    model, data, _, scaling = fitted_mock
    with pytest.raises(ValueError, match="OOS metadata requires"):
        export_dashboard_bundle(
            model, data, tmp_path / "invalid", control_scaling=scaling,
            oos_evaluation_type="conditional_on_realized_media_and_discounts",
        )


def test_copy_paste_export_cell_uses_current_notebook_variables(tmp_path, fitted_mock):
    """Exercise the user-facing bridge, changing only its local project path."""
    import ast
    import shutil

    model, data, quality, scaling = fitted_mock
    root = Path(__file__).resolve().parents[1]
    app = tmp_path / "Ecomignite_MMM_Streamlit"
    (app / "notebook").mkdir(parents=True)
    shutil.copy2(root / "notebook" / "export_dashboard_bundle.py", app / "notebook")

    tree = ast.parse((root / "notebook" / "export_cell.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "PROJECT_DIR"
            for target in node.targets
        ):
            node.value = ast.Call(
                func=ast.Name(id="Path", ctx=ast.Load()),
                args=[ast.Constant(str(tmp_path))], keywords=[],
            )
    ast.fix_missing_locations(tree)

    holdout = data.tail(12).copy()
    predictions = holdout[["date", "eligible_orders"]].rename(
        columns={"eligible_orders": "observed_orders"}
    )
    predictions["predicted_orders"] = predictions.observed_orders + 1
    predictions["predictive_lower_90"] = predictions.predicted_orders - 50
    predictions["predictive_upper_90"] = predictions.predicted_orders + 50
    state = {
        "mmm": model, "model_data": data, "CONTROL_SCALING": scaling,
        "df_weekly": quality, "train_oos": data.iloc[:-12],
        "holdout_oos": holdout, "oos_predictions": predictions,
    }
    exec(compile(tree, "export_cell.py", "exec"), state)
    exported = load_bundle(app / "data" / "model_bundle")
    assert len(exported.manifest["channels"]) == 9
    assert len(exported.fit_oos) == 12
    assert exported.manifest["oos"]["evaluation_type"] == "conditional_on_realized_media_and_discounts"

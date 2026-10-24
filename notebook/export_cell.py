"""Paste this complete cell after the full-model and 12-week OOS sections.

It exports existing fitted results without fitting or sampling. OOS predictions
must come from the separate training-only model, including all holdout weeks.
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

# ==========================================================
# SETTINGS
# ==========================================================
PROJECT_DIR = Path(
    r"C:\Users\owest\OneDrive\Desktop\Upwork\Project_Ecomignite"
)
APP_DIR = PROJECT_DIR / "Ecomignite_MMM_Streamlit"
EXPORTER_FILE = APP_DIR / "notebook" / "export_dashboard_bundle.py"
BUNDLE_DIR = PROJECT_DIR / "Outputs" / "dashboard_bundle"
CURRENCY = ""  # Set only when the currency is confirmed.
EXPORT_OOS = True
EXPECTED_HOLDOUT_WEEKS = 12
COPY_TO_LOCAL_APP = True

EXPECTED_CHANNELS = [
    "meta_conversion_spend",
    "meta_engagement_spend",
    "meta_traffic_spend",
    "google_search_spend",
    "google_pmax_spend",
    "google_display_spend",
    "google_video_spend",
    "google_shopping_spend",
    "google_demand_gen_spend",
]

# ==========================================================
# FULL-PERIOD MODEL
# ==========================================================
for variable in ["mmm", "model_data", "CONTROL_SCALING"]:
    if variable not in globals():
        raise NameError(f"Run the updated model preparation/fitting cells: {variable} is missing.")

FITTED_DATA = model_data.copy()
fitted_channels = list(mmm.idata.posterior.coords["channel"].values)
assert fitted_channels == EXPECTED_CHANNELS, (
    "The fitted model is not the current nine-channel model. "
    "Rerun preparation and training before exporting."
)
if "mmm_oos" in globals():
    assert mmm is not mmm_oos, "Export the full-period model as mmm."
assert EXPORTER_FILE.is_file(), f"Exporter not found: {EXPORTER_FILE}"

# ==========================================================
# EXISTING OOS RESULTS — NO REFIT OR RESAMPLING
# ==========================================================
OOS_FIT = None
OOS_TRAINING_END = None
OOS_METRICS = None
OOS_INPUT_QUALITY = None
OOS_EVALUATION_TYPE = None

if EXPORT_OOS:
    for variable in ["oos_predictions", "train_oos", "holdout_oos"]:
        if variable not in globals():
            raise NameError(
                f"{variable} is missing. Run the OOS cells first, "
                "or explicitly set EXPORT_OOS=False."
            )
    OOS_FIT = (
        oos_predictions.rename(columns={
            "observed_orders": "observed",
            "predicted_orders": "predicted",
            "predictive_lower_90": "lower",
            "predictive_upper_90": "upper",
        })[["date", "observed", "predicted", "lower", "upper"]]
        .copy().sort_values("date").reset_index(drop=True)
    )
    OOS_FIT["date"] = pd.to_datetime(OOS_FIT["date"], errors="raise")
    OOS_TRAINING_END = pd.Timestamp(train_oos["date"].max())
    expected_holdout = holdout_oos.sort_values("date").reset_index(drop=True)
    assert len(OOS_FIT) == EXPECTED_HOLDOUT_WEEKS
    assert pd.DatetimeIndex(OOS_FIT["date"]).equals(
        pd.DatetimeIndex(pd.to_datetime(expected_holdout["date"]))
    )
    assert OOS_FIT["date"].min() == OOS_TRAINING_END + pd.Timedelta(weeks=1)
    np.testing.assert_allclose(OOS_FIT["observed"], expected_holdout["eligible_orders"])
    assert np.isfinite(
        OOS_FIT[["observed", "predicted", "lower", "upper"]].to_numpy(float)
    ).all()
    assert OOS_FIT["lower"].le(OOS_FIT["upper"]).all()
    if "mmm_oos" in globals():
        assert pd.Timestamp(
            mmm_oos.idata.posterior.coords["date"].values.max()
        ) == OOS_TRAINING_END
    OOS_METRICS = globals().get("oos_metrics")
    OOS_INPUT_QUALITY = globals().get("oos_imputation_counts")
    OOS_EVALUATION_TYPE = "conditional_on_realized_media_and_discounts"

# ==========================================================
# BUILD AND VALIDATE THE PORTABLE MODEL BUNDLE
# ==========================================================
spec = spec_from_file_location("ecomignite_bundle_exporter", EXPORTER_FILE)
assert spec is not None and spec.loader is not None
exporter = module_from_spec(spec)
sys.modules[spec.name] = exporter
spec.loader.exec_module(exporter)

dashboard_export = exporter.export_dashboard_bundle(
    mmm=mmm,
    fitted_data=FITTED_DATA,
    output_dir=BUNDLE_DIR,
    control_scaling=CONTROL_SCALING,
    weekly_validation=globals().get("df_weekly"),
    daily_validation=globals().get("df_daily"),
    oos_fit=OOS_FIT,
    oos_training_end=OOS_TRAINING_END,
    oos_intervals_type="posterior_predictive",
    oos_evaluation_type=OOS_EVALUATION_TYPE,
    oos_metrics=OOS_METRICS,
    oos_input_quality=OOS_INPUT_QUALITY,
    fit_interval_level=0.90,
    currency=CURRENCY,
    max_posterior_draws=2000,
    random_seed=42,
)
manifest = dashboard_export["manifest"]
assert manifest["channels"] == EXPECTED_CHANNELS
if EXPORT_OOS:
    assert manifest["oos"]["status"] == "available"
    assert manifest["oos"]["weeks"] == EXPECTED_HOLDOUT_WEEKS
    assert manifest["oos"]["intervals_type"] == "posterior_predictive"

source_dir = Path(dashboard_export["bundle_dir"])
if COPY_TO_LOCAL_APP:
    destination = APP_DIR / "data" / "model_bundle"
    destination.mkdir(parents=True, exist_ok=True)
    # Remove only stale optional bundle files from an earlier export.
    for name in ["fit_oos.csv", "validation_daily.csv", "oos_metrics.csv", "oos_input_quality.csv"]:
        target_file = destination / name
        if not (source_dir / name).exists() and target_file.exists():
            target_file.unlink()
    for source_file in source_dir.iterdir():
        if source_file.is_file():
            shutil.copy2(source_file, destination / source_file.name)
    print("Updated local Streamlit bundle:", destination)

print("Model channels:", manifest["channels"])
print("OOS status:", manifest["oos"]["status"])
print("Uploadable ZIP:", dashboard_export["zip_path"])
print(
    "In-sample bands describe uncertainty in the expected mean. "
    "OOS predictive bands also include observation noise."
)

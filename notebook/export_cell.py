"""Paste this cell after fitting the full-period MMM.

Keep export_dashboard_bundle.py next to this notebook or edit EXPORTER_FILE.
This cell exports existing fitted results; it does not train a model.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


# ==========================================================
# EDITABLE EXPORT SETTINGS
# ==========================================================

EXPORTER_FILE = Path("notebook/export_dashboard_bundle.py")
if not EXPORTER_FILE.exists():
    EXPORTER_FILE = Path("export_dashboard_bundle.py")

BUNDLE_DIR = Path(
    r"C:\Users\owest\OneDrive\Desktop\Upwork"
    r"\Project_Ecomignite\Outputs\dashboard_bundle"
)

# Set currency only when confirmed, for example "INR".
CURRENCY = ""

# Quality data is optional. Existing df_weekly/df_daily are used if present.
WEEKLY_VALIDATION = globals().get("df_weekly")
DAILY_VALIDATION = globals().get("df_daily")

# Use ONE option for OOS. Missing OOS is displayed as unavailable, never invented.
# Option A: an existing original-scale holdout table with columns:
# date, observed, predicted; optionally lower and upper (90% interval).
OOS_FIT = None
OOS_TRAINING_END = None  # Required with OOS_FIT; e.g. "2026-06-29".
OOS_INTERVALS_TYPE = "expected_mean"  # Or "posterior_predictive" / "unavailable".

# Option B: a separately fitted training-only model and its untouched holdout.
# Replace None with your actual notebook variables when ready.
OOS_MODEL = None
OOS_X = None
OOS_Y = None


# ==========================================================
# EXPORT EXISTING RESULTS
# ==========================================================

if "mmm" not in globals():
    raise NameError("Run the full-period fitting cell first: mmm is missing.")
if "model_data" in globals():
    FITTED_DATA = model_data.copy()
elif "df_orders_model" in globals():
    FITTED_DATA = df_orders_model.copy()
else:
    raise NameError("The exact fitting frame model_data or df_orders_model is required.")

if not EXPORTER_FILE.is_file():
    raise FileNotFoundError(
        "Put export_dashboard_bundle.py next to the notebook, "
        "or edit EXPORTER_FILE to its full path."
    )

spec = spec_from_file_location("ecomignite_bundle_exporter", EXPORTER_FILE)
exporter = module_from_spec(spec)
spec.loader.exec_module(exporter)

dashboard_export = exporter.export_dashboard_bundle(
    mmm=mmm,
    fitted_data=FITTED_DATA,
    output_dir=BUNDLE_DIR,
    control_scaling=globals().get("CONTROL_SCALING"),
    weekly_validation=WEEKLY_VALIDATION,
    daily_validation=DAILY_VALIDATION,
    oos_fit=OOS_FIT,
    oos_training_end=OOS_TRAINING_END,
    oos_intervals_type=OOS_INTERVALS_TYPE,
    oos_model=OOS_MODEL,
    oos_X=OOS_X,
    oos_y=OOS_Y,
    fit_interval_level=0.90,
    currency=CURRENCY,
    max_posterior_draws=2000,
    random_seed=42,
)

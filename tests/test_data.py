from copy import deepcopy
from io import BytesIO
import json
import zipfile

import numpy as np
import pytest

from ecomignite.data import make_demo_bundle, load_bundle, validate_bundle, fit_metrics
from ecomignite.data import CURRENT_CHANNELS


def bundle_zip(bundle):
    arrays = BytesIO()
    np.savez_compressed(arrays, **bundle.posterior)
    raw = BytesIO()
    with zipfile.ZipFile(raw, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(bundle.manifest))
        z.writestr("posterior.npz", arrays.getvalue())
        for name, frame in [("weekly_data.csv",bundle.weekly),
                            ("fit_in_sample.csv",bundle.fit_in_sample),
                            ("fit_oos.csv",bundle.fit_oos),
                            ("contributions.csv",bundle.contributions)]:
            z.writestr(name,frame.to_csv(index=False))
    return raw.getvalue()


def test_numeric_bundle_roundtrip():
    original = make_demo_bundle()
    loaded = load_bundle(bundle_zip(original))
    np.testing.assert_allclose(loaded.posterior["alpha"], original.posterior["alpha"])
    assert len(loaded.weekly) == len(original.weekly)
    assert loaded.manifest["kind"] == "demo"


def test_nonfinite_scale_and_incomplete_components_rejected():
    bundle = make_demo_bundle()
    invalid = deepcopy(bundle)
    invalid.posterior["target_scale"] = np.asarray(np.nan)
    with pytest.raises(ValueError,match="target scale"):
        validate_bundle(invalid)
    invalid = deepcopy(bundle)
    invalid.contributions = invalid.contributions.iloc[1:].copy()
    with pytest.raises(ValueError,match="Each fitted week"):
        validate_bundle(invalid)


def test_missing_weeks_rejected():
    bundle = make_demo_bundle()
    bundle.weekly = bundle.weekly.drop(index=30)
    with pytest.raises(ValueError,match="missing weeks"):
        validate_bundle(bundle)


def test_mape_excludes_zero_observations():
    import pandas as pd
    frame = pd.DataFrame({"observed":[0.,10.,20.],"predicted":[3.,9.,22.]})
    metrics = fit_metrics(frame)
    assert metrics["MAPE eligible weeks"] == 2
    assert metrics["MAPE (%)"] == pytest.approx(10.)


def test_current_demo_and_oos_observations():
    bundle = make_demo_bundle()
    assert bundle.manifest["channels"] == CURRENT_CHANNELS
    assert len(CURRENT_CHANNELS) == 9
    invalid = deepcopy(bundle)
    invalid.fit_oos.loc[0, "observed"] += 1
    with pytest.raises(ValueError, match="observed orders differ"):
        validate_bundle(invalid)


def test_oos_cutoff_and_intervals_rejected_when_invalid():
    bundle = make_demo_bundle()
    invalid = deepcopy(bundle)
    invalid.manifest["oos"]["training_end"] = str(invalid.fit_oos.date.max().date())
    with pytest.raises(ValueError, match="cutoff"):
        validate_bundle(invalid)
    invalid = deepcopy(bundle)
    invalid.fit_oos.loc[0, "lower"] = invalid.fit_oos.loc[0, "upper"] + 1
    with pytest.raises(ValueError, match="interval bounds"):
        validate_bundle(invalid)


def test_bad_reconstruction_and_control_rejected():
    bundle = make_demo_bundle()
    invalid = deepcopy(bundle)
    invalid.contributions.loc[0, "mean"] += 5
    with pytest.raises(ValueError, match="reconstruct"):
        validate_bundle(invalid)
    invalid = deepcopy(bundle)
    invalid.weekly.loc[0, "discount_rate_scaled"] = np.nan
    with pytest.raises(ValueError, match="modeled control"):
        validate_bundle(invalid)


"""Load aggregate dashboard bundles; no database access or model execution."""
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import hashlib
import json
import zipfile

import numpy as np
import pandas as pd


@dataclass
class ModelBundle:
    manifest: dict
    weekly: pd.DataFrame
    posterior: dict
    fit_in_sample: pd.DataFrame
    fit_oos: pd.DataFrame
    contributions: pd.DataFrame
    validation_daily: pd.DataFrame
    fingerprint: str = ""


MAX_BUNDLE_BYTES = 64 * 1024 * 1024
MAX_EXPANDED_BYTES = 160 * 1024 * 1024


def _frame(raw, optional=False):
    if raw is None:
        if optional:
            return pd.DataFrame()
        raise ValueError("A required CSV file is absent from the bundle.")
    result = pd.read_csv(BytesIO(raw))
    if "date" in result:
        result["date"] = pd.to_datetime(result["date"], errors="raise")
    return result


def load_bundle(source):
    """Read a directory, zip path, or uploaded zip bytes without extracting."""
    required = ["manifest.json", "weekly_data.csv", "posterior.npz",
                "fit_in_sample.csv", "contributions.csv"]
    optional = ["fit_oos.csv", "validation_daily.csv"]
    if isinstance(source, (str, Path)) and Path(source).is_dir():
        directory = Path(source)
        content = {name: (directory / name).read_bytes()
                   for name in required + optional if (directory / name).is_file()}
    else:
        raw = source if isinstance(source, bytes) else Path(source).read_bytes()
        if len(raw) > MAX_BUNDLE_BYTES:
            raise ValueError("Model bundle exceeds the 64 MB upload limit.")
        with zipfile.ZipFile(BytesIO(raw)) as archive:
            if sum(i.file_size for i in archive.infolist()) > MAX_EXPANDED_BYTES:
                raise ValueError("Expanded model bundle is too large.")
            manifests = [n for n in archive.namelist() if n.endswith("manifest.json")]
            if len(manifests) != 1:
                raise ValueError("Bundle must contain one manifest.json.")
            prefix = manifests[0][:-len("manifest.json")]
            content = {name: archive.read(prefix + name) for name in required + optional
                       if prefix + name in archive.namelist()}
    absent = set(required) - set(content)
    if absent:
        raise ValueError(f"Missing bundle files: {sorted(absent)}")
    manifest = json.loads(content["manifest.json"])
    with zipfile.ZipFile(BytesIO(content["posterior.npz"])) as archive:
        if sum(i.file_size for i in archive.infolist()) > MAX_EXPANDED_BYTES:
            raise ValueError("Expanded posterior arrays are too large.")
    with np.load(BytesIO(content["posterior.npz"]), allow_pickle=False) as arrays:
        posterior = {name: arrays[name].astype(float) for name in arrays.files}
    fingerprint = hashlib.sha256(b"".join(content[k] for k in sorted(content))).hexdigest()
    bundle = ModelBundle(manifest, _frame(content["weekly_data.csv"]), posterior,
        _frame(content["fit_in_sample.csv"]), _frame(content.get("fit_oos.csv"), True),
        _frame(content["contributions.csv"]), _frame(content.get("validation_daily.csv"), True),
        fingerprint)
    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle):
    m, w, p = bundle.manifest, bundle.weekly, bundle.posterior
    if m.get("schema_version") != 1:
        raise ValueError("Unsupported bundle schema. Use the included notebook exporter.")
    channels = m.get("channels", [])
    controls = m.get("controls", [])
    if not channels or len(channels) != len(set(channels)):
        raise ValueError("Channels must be present and unique.")
    target = m.get("target_column", "eligible_orders")
    missing = set(["date", target] + channels) - set(w)
    if missing:
        raise ValueError(f"Missing weekly data columns: {sorted(missing)}")
    w.sort_values("date", inplace=True)
    w.reset_index(drop=True, inplace=True)
    dates = w["date"]
    if dates.isna().any() or dates.duplicated().any() or not dates.dt.dayofweek.eq(0).all():
        raise ValueError("Weekly dates must be unique Monday starts.")
    if not dates.diff().dropna().eq(pd.Timedelta(days=7)).all():
        raise ValueError("The fitted weekly sequence contains missing weeks.")
    for c in [target] + channels:
        w[c] = pd.to_numeric(w[c], errors="raise")
        if not np.isfinite(w[c]).all() or w[c].lt(0).any():
            raise ValueError(f"Invalid modeled input: {c}")
    C, K = len(channels), len(controls)
    if "alpha" not in p or p["alpha"].ndim != 2:
        raise ValueError("Posterior alpha must have sample and channel axes.")
    S = p["alpha"].shape[0]
    if S < 2:
        raise ValueError("At least two paired posterior samples are required.")
    for name in ["alpha", "beta", "kappa", "slope"]:
        if name not in p or p[name].shape != (S, C) or not np.isfinite(p[name]).all():
            raise ValueError(f"Invalid posterior shape/values: {name}")
    if ((p["alpha"] < 0) | (p["alpha"] > 1)).any():
        raise ValueError("Adstock alpha must lie in [0, 1].")
    if any((p[n] <= 0).any() for n in ["beta", "kappa", "slope"]):
        raise ValueError("Hill parameters must be positive.")
    if (p.get("channel_scale", np.array([])).shape != (C,)
            or not np.isfinite(p["channel_scale"]).all() or (p["channel_scale"] <= 0).any()):
        raise ValueError("Invalid channel scales.")
    if ("target_scale" not in p or p["target_scale"].size != 1
            or not np.isfinite(p["target_scale"]).all() or float(p["target_scale"].item()) <= 0):
        raise ValueError("Invalid target scale.")
    for name, shape in [("intercept", (S,)), ("gamma_control", (S, K)),
                        ("gamma_fourier", (S, 2 * int(m.get("yearly_seasonality", 0))))]:
        if name not in p or p[name].shape != shape or not np.isfinite(p[name]).all():
            raise ValueError(f"Invalid posterior shape/values: {name}")
    if not m.get("adstock_normalize", True) or not m.get("adstock_first", True):
        raise ValueError("This dashboard requires normalized geometric adstock before Hill saturation.")
    for name, frame in [("in-sample", bundle.fit_in_sample), ("OOS", bundle.fit_oos)]:
        if frame.empty:
            continue
        if not {"date", "observed", "predicted"}.issubset(frame):
            raise ValueError(f"Invalid {name} fit columns.")
        if frame["date"].isna().any() or frame["date"].duplicated().any():
            raise ValueError(f"Invalid {name} fit dates.")
        if not np.isfinite(frame[["observed", "predicted"]].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite {name} fit values.")
    if set(bundle.fit_in_sample["date"]) != set(dates):
        raise ValueError("In-sample fit dates do not match the full fitted data.")
    c = bundle.contributions
    if not {"date", "effect", "group", "mean"}.issubset(c):
        raise ValueError("Invalid contribution table.")
    if c.duplicated(["date", "effect"]).any() or not np.isfinite(c["mean"]).all():
        raise ValueError("Contribution rows must be unique and finite.")
    expected_effects = set(channels + controls + ["Intercept", "Seasonality"])
    if not expected_effects.issubset(set(c["effect"])):
        raise ValueError("Contributions are missing one or more model components.")
    if set(c["date"]) != set(dates) or any(
        not expected_effects.issubset(set(rows["effect"])) for _, rows in c.groupby("date")
    ):
        raise ValueError("Each fitted week must include all model components.")


def fit_metrics(frame):
    if frame.empty:
        return {}
    y, pred = frame["observed"].to_numpy(float), frame["predicted"].to_numpy(float)
    residual = y - pred
    positive = y > 0
    variance = np.square(y - y.mean()).sum()
    values = {"Weeks": len(y), "R²": 1 - np.square(residual).sum() / variance if variance > 0 else np.nan,
        "Correlation": np.corrcoef(y, pred)[0, 1] if y.std() > 0 and pred.std() > 0 else np.nan,
        "MAE": np.abs(residual).mean(), "RMSE": np.sqrt(np.square(residual).mean()),
        "MAPE (%)": 100 * np.mean(np.abs(residual[positive]) / y[positive]) if positive.any() else np.nan,
        "WAPE (%)": 100 * np.abs(residual).sum() / np.abs(y).sum() if np.abs(y).sum() > 0 else np.nan,
        "Bias": residual.mean(), "MAPE eligible weeks": int(positive.sum())}
    if {"lower", "upper"}.issubset(frame) and frame[["lower", "upper"]].notna().all().all():
        values["Interval coverage (%)"] = 100 * ((y >= frame["lower"]) & (y <= frame["upper"])).mean()
    return values


def validation_summary(bundle):
    w = bundle.weekly
    required = [bundle.manifest.get("target_column", "eligible_orders")] + bundle.manifest["channels"]
    summary = pd.DataFrame({"Input": required,
        "Missing modeled values": [int(w[c].isna().sum()) for c in required],
        "Zero modeled values": [int(w[c].eq(0).sum()) for c in required],
        "Positive modeled values": [int(w[c].gt(0).sum()) for c in required]})
    quality = [c for c in w if any(t in c for t in ["imputed", "backfilled", "record_missing", "high_imputation"])]
    records = []
    for c in quality:
        values = pd.to_numeric(w[c], errors="coerce")
        is_share = "share" in c
        records.append({"Field": c, "Statistic": "Mean (%)" if is_share else "Sum",
                        "Value": 100 * values.mean() if is_share else values.sum(min_count=1)})
    flags = pd.DataFrame(records)
    reconstructed = bundle.contributions.groupby("date")["mean"].sum()
    fitted = bundle.fit_in_sample.set_index("date")["predicted"]
    gap = (reconstructed.reindex(fitted.index) - fitted).abs().max()
    return summary, flags, float(gap)


def make_demo_bundle():
    """Deterministic synthetic example, visibly distinct from client results."""
    from scipy.special import expit
    rng = np.random.default_rng(42)
    channels = ["meta_conversion_spend", "meta_engagement_spend", "meta_traffic_spend",
        "meta_thruplay_spend", "meta_other_spend", "google_search_spend",
        "google_pmax_spend", "google_display_spend"]
    labels = dict(zip(channels, ["Meta Conversion", "Meta Engagement", "Meta Traffic",
        "Meta ThruPlay", "Meta Other", "Google Search", "Google PMax", "Google Display"]))
    controls = ["discount_rate_scaled", "linear_trend_scaled"]
    T, S, L = 156, 600, 8
    dates = pd.date_range("2023-10-02", periods=T, freq="7D")
    t = np.arange(T)
    multipliers = np.array([110000, 5500, 3000, 2300, 1000, 14000, 24000, 9000])
    spend = multipliers * np.maximum(.2, 1 + .28*np.sin(2*np.pi*t[:,None]/52)
        + rng.normal(0,.14,(T,8)) + .12*t[:,None]/T)
    spend[:35,2:5] = 0
    spend[-12:,4] = 0
    spend[45:58,3] = 0
    scales = spend.max(axis=0)
    target_scale = 700.
    alpha = np.clip(rng.normal([.35,.45,.25,.55,.30,.015,.30,.40], .025, (S,8)), .001,.9)
    beta = np.maximum(.001, rng.normal([.33,.028,.035,.025,.014,.11,.08,.045], .006,(S,8)))
    kappa = np.exp(rng.normal(np.log(.6),.06,(S,8)))
    slope = np.exp(rng.normal(np.log(1.2),.05,(S,8)))
    intercept = rng.normal(.18,.008,S)
    gamma_control = rng.normal([.06,.055],[.004,.004],(S,2))
    gamma_fourier = rng.normal([.018,-.005,.025,.007], .003,(S,4))
    noise = np.full(S,.04)
    discount = np.clip(.015 + .005*np.sin(t/4) + rng.normal(0,.004,T),0,.08)
    discount[[9,31,58,82,115,140]] += .05
    control_data = np.column_stack([discount/.1,t/(T-1)])
    weights = alpha[:,:,None] ** np.arange(L)
    weights /= weights.sum(axis=2,keepdims=True)
    media = np.zeros((S,T,8))
    for i in range(T):
        A = sum(weights[:,:,lag]*spend[i-lag]/scales for lag in range(min(L,i+1)))
        safe = np.where(A>0,A,1.)
        media[:,i,:] = np.where(A>0,target_scale*beta*expit(slope*(np.log(safe)-np.log(kappa))),0.)
    phase = dates.dayofyear.to_numpy()/365.25
    fourier = np.column_stack([np.sin(2*np.pi*phase),np.sin(4*np.pi*phase),
                              np.cos(2*np.pi*phase),np.cos(4*np.pi*phase)])
    context = target_scale*gamma_control[:,None,:]*control_data[None,:,:]
    seasonal = target_scale*gamma_fourier@fourier.T
    mu = media.sum(axis=2) + context.sum(axis=2) + seasonal + target_scale*intercept[:,None]
    observed = np.maximum(0,np.round(mu.mean(axis=0)+rng.normal(0,28,T)))
    weekly = pd.DataFrame(spend,columns=channels)
    weekly.insert(0,"date",dates)
    weekly["eligible_orders"] = observed
    weekly["net_revenue"] = observed*(1250+rng.normal(0,40,T))
    weekly["discount_rate"] = discount
    weekly["discount_rate_scaled"] = control_data[:,0]
    weekly["linear_trend_scaled"] = control_data[:,1]
    weekly["meta_imputed_days"] = 0
    weekly.loc[[50,60,70,100],"meta_imputed_days"] = [3,4,2,4]
    weekly["google_imputed_days"] = 0
    weekly.loc[[40,65,90,100,121],"google_imputed_days"] = [1,2,5,7,3]
    weekly["high_imputation_week"] = (weekly[["meta_imputed_days","google_imputed_days"]].max(axis=1)>3)
    def fit(draws, actual, fit_dates):
        low,high=np.quantile(draws,[.05,.95],axis=0)
        return pd.DataFrame({"date":fit_dates,"observed":actual,"predicted":draws.mean(axis=0),"lower":low,"upper":high})
    in_fit=fit(mu,observed,dates)
    # Illustrative holdout predictions; the demo is not a fitted validation study.
    oos_fit=fit(mu[:,-12:]+rng.normal(0,20,(S,12)),observed[-12:],dates[-12:])
    rows=[]
    for effect, group, draws in ([(c,"Marketing",media[:,:,i]) for i,c in enumerate(channels)]
        +[(c,"Context",context[:,:,i]) for i,c in enumerate(controls)]
        +[("Intercept","Intercept",np.repeat(target_scale*intercept[:,None],T,axis=1)),
          ("Seasonality","Seasonality",seasonal)]):
        lo,hi=np.quantile(draws,[.05,.95],axis=0)
        rows.extend({"date":dates[i],"effect":effect,"group":group,"mean":draws[:,i].mean(),
                     "lower":lo[i],"upper":hi[i]} for i in range(T))
    manifest={"schema_version":1,"kind":"demo","model_name":"Ecomignite Orders MMM",
        "target_column":"eligible_orders","target_label":"Orders","date_column":"date",
        "granularity":"weekly","channels":channels,"channel_labels":labels,"controls":controls,
        "control_labels":{"discount_rate_scaled":"Discounts","linear_trend_scaled":"Trend"},
        "adstock_l_max":L,"adstock_normalize":True,"adstock_first":True,"yearly_seasonality":2,
        "fourier_columns":["sin_1","sin_2","cos_1","cos_2"],"fourier_period_days":365.25,
        "control_scaling":{"discount_unit":.1,"trend_origin":str(dates[0].date()),"trend_span_days":(T-1)*7},
        "currency":"","fit_interval_level":.9,"fit_intervals_type":"expected_mean",
        "diagnostics":{},
        "training_start":str(dates[0].date()),"training_end":str(dates[-1].date())}
    posterior={"alpha":alpha,"beta":beta,"kappa":kappa,"slope":slope,"channel_scale":scales,
        "target_scale":np.asarray(target_scale),"intercept":intercept,"gamma_control":gamma_control,
        "gamma_fourier":gamma_fourier,"noise_sigma":noise}
    bundle=ModelBundle(manifest,weekly,posterior,in_fit,oos_fit,pd.DataFrame(rows),pd.DataFrame(),"synthetic-demo-v1")
    validate_bundle(bundle)
    return bundle

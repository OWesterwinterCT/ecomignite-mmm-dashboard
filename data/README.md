# Model export location

The app can load an actual notebook export from `data/model_bundle/`, or accept the exported ZIP through the sidebar file uploader. No real model data is included in this source package.

The synthetic demo is generated in code and is always labelled. An actual export must contain `manifest.json`, `weekly_data.csv`, `posterior.npz`, `fit_in_sample.csv`, and `contributions.csv`. Optional files are `fit_oos.csv` and `validation_daily.csv`.

Use the included notebook exporter to preserve channel order, units, posterior pairing, anchored control references, and independent OOS predictions. Do not substitute screenshots for the numeric exports.

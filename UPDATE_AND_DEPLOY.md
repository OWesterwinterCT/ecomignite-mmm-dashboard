# Update the notebook and deployed dashboard

## 1. Replace the application source

Extract `Ecomignite_MMM_Streamlit_Update.zip`. Copy the contents of its `Ecomignite_MMM_Streamlit` folder into your existing application folder, replacing matching source files. Keep your current `data/model_bundle` directory until the fresh export is ready. The archive contains source, tests and a synthetic demo; it does not contain the newly fitted client model.

Existing local app folder:

```text
C:\Users\owest\OneDrive\Desktop\Upwork\Project_Ecomignite\Ecomignite_MMM_Streamlit
```

## 2. Export the new fitted results

Run the revised notebook through the full-model diagnostics and all 12 OOS weeks. If those variables are already available in the kernel, no refit is needed for exporting.

Use the updated `notebook/export_cell.py` as your final notebook cell. It uses the full-period `mmm` and `model_data`, current anchored `CONTROL_SCALING`, `oos_predictions`, `train_oos`, and `holdout_oos`. It also includes OOS benchmarks and imputation counts when available. Check its `PROJECT_DIR` and `APP_DIR` values.

It creates:

```text
Project_Ecomignite\Outputs\dashboard_bundle\
Project_Ecomignite\Outputs\dashboard_bundle.zip
```

By default, the cell also copies the validated bundle into the local app's `data\model_bundle` folder. The alternative is uploading `dashboard_bundle.zip` using the app's model-source selector. A session upload does not replace the default model for other visitors; use a repository bundle for that.

## 3. Check locally

In PowerShell, activate your existing dashboard environment, enter the app folder and run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Choose **Repository model**. Check that the channel list includes Video, Shopping and Demand Gen, the OOS plot has 12 weeks, and an optimizer run returns all nine channels. The synthetic demo is explicitly labelled if no model bundle is present.

## 4. Copy into the GitHub checkout

For the client repository used previously, replace the application source under `mmm\dashboard` with this updated folder's contents, including the new `data\model_bundle` export. Preserve the rest of the client repository.

From the repository root, review and stage the application changes:

```powershell
git branch --show-current
git status --short
git add mmm/dashboard
git add -f mmm/dashboard/data/model_bundle
git diff --cached --stat
git commit -m "Update MMM dashboard for nine-channel model and revised validation"
git push origin oliver-ecomignite
```

These commands assume you are on `oliver-ecomignite` and `origin` points to the client repository. If your Streamlit app deploys from another repository or branch, update that deployment source instead. The model bundle is ignored by default, so the explicit `git add -f` is needed when choosing to publish it. Files bundled with a public repository or public app can expose the exported aggregate results.

## 5. Refresh Streamlit

Use the repository and branch actually connected to your Streamlit app. For the client-repository layout, the main file is `mmm/dashboard/streamlit_app.py`; for a standalone dashboard repository it is `streamlit_app.py`. Keep `requirements.txt` alongside the app and use Python 3.12.

After pushing, check the deployment logs and reload the app. If the process still serves the older source, reboot it from Streamlit's app management. Verify **Repository model**, all nine channels and the updated OOS results in the deployed app.

No PyMC, NumPyro, database credentials or model refitting are required on the Streamlit server.

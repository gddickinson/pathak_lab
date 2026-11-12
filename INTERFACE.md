# INTERFACE.md: navigation map

Python code for analysing TIRF recordings of PIEZO1 (FLIKA plugins + pipeline scripts) and its web interface.

## flika_plugins/ (FLIKA GUI plugins; need FLIKA + Qt)
- `locsAndTracksPlotter/`: point/track display and analysis (main: `locsAndTracksPlotter.py`; `joinTracks.py`,
  `diffusionPlot.py`, `trackPlotter.py`, `roiZoomPlotter.py`, ...). Also holds the SVM training-features CSV (lab data).
- `overlay/`, `overlayMultipleRecordings/`, `translateAndScale/`, `videoExporter/`: overlays, alignment, video export.

## flika_scripts/piezo1_analysis/
- `1_preprocessing/`: binByFrame, cropLength, convert64bitTiffto32bit, removeAutofluoresence.
- `2_analysis/`: `Step_1` ... `Step_11` scripts, run in order (each reads the previous step's CSV). Qt-free modules
  used by Step 2 and the web app:
  - `pointLinking.py`: `PointLinker` (link_pts, extend_track, getIntensities), `addID`, `load_points`,
    `savetracksCSV`.
  - `trackFeatures.py`: Rg/asymmetry/skewness/kurtosis, FractalDimension, NetDisplacementEfficiency,
    SummedSinesCosines, addLagDisplacementToDF, getNN, `addFeaturesToTracksDF`.
  - `svmClassification.py`: `predict_SPT_class` (SVM classes 1 mobile, 2 intermediate, 3 trapped).
  - `Step_2_...recursionErrorFix.py` imports these and adds FLIKA display, file batch loops and trackpy linking.
- `3_postprocessing/`: compile/filter/summarise results. `testing_misc/`: development scripts.
- `minimal_model/`: example recording, ThunderSTORM protocol, ROI, results table (lab data, not exported).

## web/ (FastAPI + plain ES modules)
- `app.py`: routes (`/api/info`, `/api/catalogue`, `/api/tool/{id}`, `/api/pipeline`, POST `/api/run`,
  `/api/runs/{id}/result.csv`), `STATIC_ENDPOINTS`.
- `catalogue.py` + `catalogue_parse.py`: tool catalogue read statically (ast) from every .py file and folder READMEs.
- `analysis.py`: runs the pipeline steps on a user's file by calling the repo's functions.
- `repo_scripts.py`: imports pipeline scripts by path (folders start with digits).
- `static/`: `index.html`, `core.js` (kit router), `main.js`, pages (`home`, `tools`, `pipeline`, `linking`,
  `features`, `mobility`, `run`, `about`), engines (`link_engine.js`, `features_engine.js`, `mobility_engine.js`:
  JS ports), `synth.js` (synthetic demo data), `state.js`, `plot.js`, `md.js`, `browser_run.js`, `static_api.js`,
  `engines.test.mjs` (node test vs `tests/fixtures/engine_cases.json`).

## scripts/, tests/
- `scripts/build_static.py` (Pages export), `scripts/screenshots.py` (smoke test), `scripts/make_engine_fixture.py`.
- `tests/test_analysis_fixes.py` (regression tests for fixed bugs), `tests/test_web.py`, `tests/test_engine_fixture.py`,
  `tests/script_loader.py`.
- `run_web.py` (port 8202), `requirements-web.txt`, `.github/workflows/pages.yml`.

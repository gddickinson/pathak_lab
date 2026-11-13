# Session log

## 2026-11-12: web interface and static site
- Factored Step 2's pure code into Qt-free modules (`pointLinking.py`, `trackFeatures.py`, `svmClassification.py`);
  moved verbatim, Step 2 imports them back (checked importing Step 2 in the flika_new env).
- Fixed (regression tests in `tests/test_analysis_fixes.py`):
  - numpy 2: `np.linalg.eig` returns complex arrays (radius_gyration written as complex); `np.cross` on 2D vectors
    removed (Straight feature crashed).
  - pandas 3: `float(Series)` in Step 5; `DataFrame.append` in compileIterationSVMClasses and Step 2's
    filterDFandLocs_SVM2; groupby-agg over a text column; SVM training labels stayed text after `replace`.
  - Step 6: `np.insert(..., -1, 0)` dropped the frame interval/velocity of each file's second-to-last row.
- Not fixed (need FLIKA to test, or a decision): the Step 6 dt pattern and `float(Series)` remain in Step 10 and in
  copies under flika_plugins/locsAndTracksPlotter/joinTracks.py and testing_misc/; the eig pattern in those copies.
- Needs a decision: in predict_SPT_class, `X.eval('col = @X_.col')` returns a copy, so the Box-Cox transform is never
  applied (the SVM trains and predicts on untransformed features); `savetracksCSV` writes x [nm] with 108 nm regardless
  of pixel size (Steps 3, 4, 7 also default to 108); skewness sign depends on LAPACK's eigenvector sign (not used by
  the SVM).
- Web: catalogue of all 98 .py files, pipeline page, three demos on synthetic data (JS ports node-tested against
  Python: 32k checks), Run page (local: repo Python incl. SVM; static: JS ports in browser).
- Lab data (minimal_model recording/results, SVM training CSV) kept out of the static export (test).

## 2026-11-12: decisions from George applied (SVM transform, pixel size, skewness, remaining numpy/pandas patterns)
- SVM Box-Cox now applied: `ShiftedBoxCox` pipeline step, lambdas fitted on the training tracks and reused for
  prediction. Option `feature_transform` = `boxcox` (default, `pipeline_defaults.py`) | `none` (legacy, as published):
  Step 2 `featureTransform`, JoinTracks `featureTransform` (plugin + testing_misc), web Run form. No desktop dialog runs
  the SVM. Committed training CSV: 10-fold CV accuracy none 89.5% vs boxcox 86.4% (held-out 20%: 89.5% vs 84.5%);
  CytoD rows: 106/1000 change class; minimal_model: `none` reproduces the committed SVM column for all 119 tracks,
  `boxcox` agrees on 55.5%. Shift margin / Yeo-Johnson variants give the same picture (the change is Box-Cox itself).
- Pixel size: one setting `PIXEL_SIZE_NM` (108 nm, the lab's camera) threaded through load_points, savetracksCSV, Step
  2 (pixelSize now in nm), Steps 3, 4, 7, removeAutofluoresence, the web app and the JS port (node test at 160 nm).
- Skewness: axis oriented along the net displacement (`orientedDominantAxis`), so mirror/rotation/time reversal keep
  it; |skewness| unchanged, sign flips for 47 of 111 minimal_model tracks; SVM does not use it.
- numpy 2 / pandas 3 patterns fixed in Step 10, JoinTracks (plugin + testing_misc), calcuateRGandFeatures; also
  JoinTracks.join kept only the last group, and Step 10 addPointsToUnbinnedTracks copied features by index alignment
  (NaN rows). Step 10 tested with flika_new (skipped without FLIKA).
- Tests: `pytest tests` 141 passed, 2 skipped (Step 10, no FLIKA in the venv); flika_new: 27 passed; node 36,049 checks.
- Follow-up decisions: SVM `feature_transform` default is now `none` (raw features, as published; CV 89.5 % vs 86.4 %
  for `boxcox`, which stays an option). Step 10 now adds the unlinked localisations back (they were taken after being
  removed); test case: 46 -> 49 rows, track neighbour counts within 3/5/10 px 12/12/12 -> 15/16/16.

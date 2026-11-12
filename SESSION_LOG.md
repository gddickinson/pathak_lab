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

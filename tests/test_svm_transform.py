"""SVM feature transform (Box-Cox) regression tests.

predict_SPT_class meant to Box-Cox transform the features, but `X.eval('col = @X_.col')` returns a copy, so the
classifier always trained and predicted on raw features. The fix fits the Box-Cox lambdas on the training data inside
the classifier pipeline and reuses them at prediction; feature_transform="none" keeps the legacy (published) behaviour.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from script_loader import load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRAINING = ROOT / "flika_plugins" / "locsAndTracksPlotter" / "tdTomato_37Degree_CytoD_training_feats.csv"
FEATS = ["NetDispl", "Straight", "Asymmetry", "radiusGyration", "Kurtosis", "fracDimension"]
PRED_NAMES = {"NetDispl": "netDispl", "Asymmetry": "asymmetry", "radiusGyration": "radius_gyration",
              "Kurtosis": "kurtosis"}


def _synthetic(rng, n, label):
    loc = {"mobile": 0.0, "confined": 0.6, "trapped": 1.2}[label]
    return {f: float(np.exp(rng.normal(-loc + 0.3 * k, 0.45))) for k, f in enumerate(FEATS)}


def _write(tmp_path, scale_rg=1.0):
    rng = np.random.default_rng(7)
    rows = [{"Experiment": "tdTomato_37Degree", "TrackID": i, **_synthetic(rng, i, lab), "Elected_Label": lab}
            for i, lab in enumerate(["mobile", "confined", "trapped"] * 40)]
    train = pd.DataFrame(rows)
    pred = pd.DataFrame([_synthetic(rng, i, lab) for i, lab in enumerate(["mobile", "confined", "trapped"] * 15)])
    lo, hi = train[FEATS].min(), train[FEATS].max()
    pred = pred.clip(lower=lo, upper=hi, axis=1)
    train["radiusGyration"] *= scale_rg
    pred["radiusGyration"] *= scale_rg
    pred = pred.rename(columns=PRED_NAMES)
    pred.insert(0, "track_number", range(len(pred)))
    d = tmp_path / f"s{scale_rg:g}"
    d.mkdir()
    train.to_csv(d / "train.csv", index=False)
    pred.to_csv(d / "x_tracksRG.csv", index=False)
    return d


def _classes(svm, d, **kw):
    svm.predict_SPT_class(str(d / "train.csv"), str(d / "x_tracksRG.csv"), "x", level="", **kw)
    return pd.read_csv(d / "x_tracksRG_SVMPredicted.csv")["SVM"].tolist()


def test_boxcox_is_applied(tmp_path):
    """Box-Cox (with fitted lambdas and standardisation) makes the classifier blind to the units of a strictly
    positive feature; raw features feed PCA unscaled, so multiplying one column by 1000 changes the classes."""
    svm = load("2_analysis/svmClassification.py")
    a = _classes(svm, _write(tmp_path, 1.0), feature_transform="boxcox")
    b = _classes(svm, _write(tmp_path, 1000.0), feature_transform="boxcox")
    assert a == b


def test_default_is_raw_features():
    """Default 'none': Box-Cox was less accurate on the lab's labelled tracks (CV 86.4 % vs 89.5 %)."""
    svm = load("2_analysis/svmClassification.py")
    assert svm.check_transform(None) == "none"
    assert [n for n, _ in svm.make_pipeline().steps] == ["pca", "scaler", "SVC"]


def test_boxcox_lambdas_fit_on_training_and_reused():
    """The transform is fitted once on the training features; a single track (or a whole file) is transformed with
    the stored lambdas, so a track's class does not depend on the other tracks in its file."""
    svm = load("2_analysis/svmClassification.py")
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.lognormal(size=(80, 6)), columns=FEATS)
    X["Straight"] -= 2.0  # has values <= 0: shifted as in prepare_box_cox_data
    t = svm.ShiftedBoxCox().fit(X)
    assert t.shift_[FEATS.index("Straight")] > 0 and t.shift_[0] == 0
    whole = t.transform(X.iloc[:10])
    one = t.transform(X.iloc[[3]])
    assert np.allclose(whole[3], one[0])
    assert np.allclose(t.transform(X).mean(0), 0, atol=1e-9)  # standardised on the training data
    below = X.iloc[[0]].copy()
    below["Straight"] = X["Straight"].min() - 5  # below the training minimum: clipped, still finite
    assert np.isfinite(t.transform(below)).all()


def test_feature_transform_option_validated(tmp_path):
    svm = load("2_analysis/svmClassification.py")
    d = _write(tmp_path)
    with pytest.raises(ValueError):
        _classes(svm, d, feature_transform="log")


@pytest.mark.skipif(os.environ.get("CI") == "true" or os.environ.get("CI") == "1",
                    reason="discrete SVM classes from float maths may differ across platforms")
def test_none_matches_legacy_classes(tmp_path):
    """feature_transform='none' reproduces the classes the old code gave (predicting the committed CytoD rows from the
    tdTomato_37Degree training rows; fixture made with the code before the fix)."""
    if not TRAINING.exists():
        pytest.skip("training CSV not present")
    svm = load("2_analysis/svmClassification.py")
    d = pd.read_csv(TRAINING)
    c = d[d.Experiment == "tdTomato_CytoD"].rename(columns={"TrackID": "track_number", **PRED_NAMES})
    c.to_csv(tmp_path / "cytod_tracksRG.csv", index=False)
    svm.predict_SPT_class(str(TRAINING), str(tmp_path / "cytod_tracksRG.csv"), "x", "", feature_transform="none")
    got = pd.read_csv(tmp_path / "cytod_tracksRG_SVMPredicted.csv")["SVM"].astype(int).tolist()
    want = json.loads((ROOT / "tests" / "fixtures" / "svm_legacy_cytod.json").read_text())
    assert got == want


def _load_path(path, name):
    import importlib.util
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PLUGIN = ROOT / "flika_plugins" / "locsAndTracksPlotter"
MISC = ROOT / "flika_scripts" / "piezo1_analysis" / "testing_misc"


def test_plugin_transform_copy_matches_pipeline():
    svm = load("2_analysis/svmClassification.py")
    plug = _load_path(PLUGIN / "svmTransform.py", "plugin_svmTransform")
    rng = np.random.default_rng(2)
    X = rng.lognormal(size=(50, 6)) - 0.5
    assert np.allclose(svm.ShiftedBoxCox().fit(X).transform(X), plug.ShiftedBoxCox().fit(X).transform(X))
    defaults = load("2_analysis/pipeline_defaults.py")
    assert plug.FEATURE_TRANSFORM == defaults.FEATURE_TRANSFORM
    assert tuple(plug.FEATURE_TRANSFORMS) == tuple(defaults.FEATURE_TRANSFORMS)


@pytest.mark.parametrize("which", ["plugin", "testing_misc"])
def test_join_tracks_copies_apply_boxcox(tmp_path, which):
    """The predict_SPT_class copies in JoinTracks (plugin and testing_misc) had the same no-op Box-Cox."""
    d1, d2 = _write(tmp_path, 1.0), _write(tmp_path, 1000.0)
    path = PLUGIN / "joinTracks.py" if which == "plugin" else MISC / "joinTracks.py"
    jt = _load_path(path, f"jt_{which}")
    grid = jt.GridSearchCV  # one grid point keeps the test fast; the transform is what is under test
    jt.GridSearchCV = lambda pipe, param_grid, **kw: grid(
        pipe, param_grid={"pca__n_components": [3], "SVC__C": [10], "SVC__gamma": [0.1]}, **kw)
    out = []
    for d in (d1, d2):
        pred = pd.read_csv(d / "x_tracksRG.csv")
        got = []
        for i in range(0, len(pred), 3):
            j = jt.JoinTracks()
            assert j.featureTransform == "none"  # default
            j.featureTransform = "boxcox"
            row = pred.iloc[[i]].reset_index(drop=True)
            if which == "plugin":
                jt.train_data_path = str(d / "train.csv")
                got.append(int(j.predict_SPT_class(row, "x", "")["SVM"].iloc[0]))
            else:
                got.append(int(j.predict_SPT_class(str(d / "train.csv"), row, "x", "")["SVM"].iloc[0]))
        out.append(got)
    assert out[0] == out[1]

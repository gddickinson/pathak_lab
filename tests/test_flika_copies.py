"""numpy 2 / pandas 3 patterns in the copies of the pipeline code: the JoinTracks class (FLIKA plugin
locsAndTracksPlotter and testing_misc), testing_misc/calcuateRGandFeatures.py and Step 10 (needs FLIKA: run with the
flika_new env, `/opt/anaconda3/envs/flika_new/bin/python -m pytest -q tests/test_flika_copies.py`).

Fixed patterns: np.linalg.eig returns complex arrays; np.cross no longer takes 2D vectors; float() of a one-row Series;
DataFrame.append (removed); np.insert(..., -1, 0) put the 0 before the last element (dt of the second-to-last row).
The copies also orient the skewness axis as trackFeatures does.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from script_loader import load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = ROOT / "flika_plugins" / "locsAndTracksPlotter" / "joinTracks.py"
MISC = ROOT / "flika_scripts" / "piezo1_analysis" / "testing_misc"


def _load_path(path, name):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tracks(n_tracks=3, n=12, seed=5, start=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_tracks):
        xy = np.cumsum(rng.normal(size=(n, 2)), axis=0) + 30
        for f in range(n):
            rows.append({"track_number": t, "frame": start + f + t * n, "id": len(rows), "x": xy[f, 0],
                         "y": xy[f, 1], "intensity": 100.0 + f, "Experiment": "e", "nnDist": 1.0,
                         "nnIndex_inFrame": 0, "nnDist_inFrame": 1.0})
    return pd.DataFrame(rows)


@pytest.fixture(params=["plugin", "testing_misc"])
def jt(request):
    path = PLUGIN if request.param == "plugin" else MISC / "joinTracks.py"
    return _load_path(path, f"jt_copy_{request.param}").JoinTracks()


def test_join_tracks_features_match_pipeline(jt):
    tf = load("2_analysis/trackFeatures.py")
    for t, g in _tracks(6).groupby("track_number"):
        got = jt.RadiusGyrationAsymmetrySkewnessKurtosis(g)
        want = tf.RadiusGyrationAsymmetrySkewnessKurtosis(g)
        assert not any(np.iscomplexobj(v) for v in got)
        assert np.allclose(got, want, rtol=1e-12, equal_nan=True)
        pts = g[["x", "y"]].to_numpy()
        assert np.isclose(jt.SummedSinesCosines(pts)[0], tf.SummedSinesCosines(pts)[0])


def test_join_tracks_velocity_and_diffusion(jt):
    df = pd.DataFrame({"track_number": [0] * 5, "frame": [0, 1, 3, 4, 6], "x": [0, 1, 2, 3, 4.0], "y": 0.0,
                       "lag": [1.0, 2.0, 3.0, 4.0, np.nan]})
    out = jt.addVelocitytoDF(jt.addDiffusiontoDF(df))
    assert out["dt"].tolist()[:4] == [1, 2, 1, 2] and np.isnan(out["dt"].iloc[4])


def test_join_tracks_join_keeps_every_group(jt):
    """join() replaced the previous group's joined track with each new one (newDF = df.append(...) in the loop)."""
    if hasattr(jt, "train_data_path"):  # the testing_misc copy classifies the joined track with the SVM
        jt.train_data_path = str(ROOT / "flika_plugins" / "locsAndTracksPlotter" / "tdTomato_37Degree_CytoD_training_feats.csv")
        g = jt.predict_SPT_class.__func__.__globals__
        grid = g["GridSearchCV"]  # one grid point keeps the test fast
        g["GridSearchCV"] = lambda pipe, param_grid, **kw: grid(
            pipe, param_grid={"pca__n_components": [3], "SVC__C": [10], "SVC__gamma": [0.1]}, **kw)
    df = _tracks(4)
    out = jt.join(df.copy(), [[0, 1], [2, 3]])
    assert sorted(out["track_number"].unique()) == [4, 5]
    assert len(out) == len(df)


def test_calculate_rg_features_copy():
    m = _load_path(MISC / "calcuateRGandFeatures.py", "misc_calcRG")
    tf = load("2_analysis/trackFeatures.py")
    g = _tracks(1)
    assert np.allclose(m.RadiusGyrationAsymmetrySkewnessKurtosis(g), tf.RadiusGyrationAsymmetrySkewnessKurtosis(g))
    pts = g[["x", "y"]].to_numpy()
    assert np.isclose(m.SummedSinesCosines(pts)[2], tf.SummedSinesCosines(pts)[2])


def _step10():
    pytest.importorskip("flika", reason="Step 10 imports FLIKA: run with the flika_new env")
    return load("2_analysis/Step_10_addInterpolatedPointstoBinnedRecording.py")


def _binned(tmp_path):
    import tifffile
    df = _tracks(2, n=6)
    df["frame"] = df.groupby("track_number").cumcount() * 2  # gaps, to be interpolated
    for c, v in {"radius_gyration": 1.0, "asymmetry": 0.1, "skewness": 0.2, "kurtosis": 0.3, "fracDimension": 1.5,
                 "netDispl": 2.0, "Straight": 0.1, "SVM": 3, "n_segments": 6, "roi_1": 5.0,
                 "camera black estimate": 1.0}.items():
        df[c] = v
    df["SVM"] = [3] * 6 + [1] * 6
    # columns the earlier steps (5, 6, 9) add to the *_BGsubtract.csv input
    for c in ("zeroed_X", "zeroed_Y", "lagNumber", "distanceFromOrigin", "dy-dt: distance", "lag", "meanLag",
              "track_length", "radius_gyration_scaled", "radius_gyration_scaled_nSegments",
              "radius_gyration_scaled_trackLength", "d_squared", "lag_squared", "dt", "velocity",
              "direction_Relative_To_Origin", "meanVelocity", "intensity - mean roi1",
              "intensity - mean roi1 and black"):
        df[c] = 0.0
    tif = tmp_path / "rec.tif"
    tifffile.imwrite(tif, np.random.default_rng(0).integers(0, 50, (40, 64, 64)).astype(np.uint16))
    return df, str(tif)


def test_step10_unbinned_tracks(tmp_path):
    s10 = _step10()
    df, tif = _binned(tmp_path)
    out = s10.addPointsToUnbinnedTracks(df, tif, np.ones(40), np.zeros(40), bin_size=2)
    t0 = out[out["track_number"] == 0]
    assert len(t0) == 21 and t0["SVM"].eq(3).all() and t0["radius_gyration"].eq(1.0).all()
    dt = out["dt"].to_numpy()
    assert np.isnan(dt[20]) and np.isnan(dt[-1]) and dt[-2] == 1  # the second-to-last row keeps its interval


def test_step10_missing_points_on_trapped_sites(tmp_path):
    s10 = _step10()
    df, tif = _binned(tmp_path)
    out = s10.addMissingPoints(df, tif, np.ones(40), np.zeros(40), tracksToKeep="all", grouping="none")
    trapped = out[out["SVM"] == 3]
    assert len(trapped) == 40 and sorted(trapped["frame"].tolist()) == list(range(40))
    assert (out["SVM"] == 1).sum() == 6


def _with_unlinked(df):
    """Unlinked localisations (track_number NaN), as Step 7 adds them: near track 0's first point, in frames 0 and 2."""
    t0 = df[df["track_number"] == 0].iloc[0]
    extra = pd.DataFrame({"frame": [0, 2, 2], "x": t0["x"] + np.array([1.0, 1.5, 40.0]), "y": t0["y"],
                          "track_number": np.nan, "id": [900, 901, 902], "intensity": 50.0})
    return pd.concat([df, extra], ignore_index=True)


def test_step10_adds_unlinked_points_back(tmp_path):
    """Step 10 took the unlinked rows after removing them, so 'add unlinked points back' added nothing (and Step 10's
    neighbour counts, run on its output, missed every unlinked localisation)."""
    s10 = _step10()
    df, tif = _binned(tmp_path)
    df = _with_unlinked(df)
    out = s10.addMissingPoints(df.copy(), tif, np.ones(40), np.zeros(40), tracksToKeep="all", grouping="none")
    un = out[out["track_number"].isna()]
    assert len(un) == 3 and un["intensity"].eq(50.0).all()
    nn = s10.getNNinFrame(out, radiusList=[3])
    near = nn[(nn["track_number"] == 0) & (nn["frame"].isin([0, 2]))]["nnCountInFrame_within_3_pixels"]
    assert (near >= 1).all()  # track 0 now has the unlinked localisation within 3 px in frames 0 and 2
    out = s10.addPointsToUnbinnedTracks(df.copy(), tif, np.ones(40), np.zeros(40), bin_size=2)
    un = out[out["track_number"].isna()]
    assert len(un) == 3 and un["intensity"].eq(50.0).all()
    assert sorted(un["frame"].tolist()) == [0, 4, 4]  # on the unbinned time axis, like the tracks

"""Regression tests for bugs fixed in the pipeline scripts (each failed before its fix).

Run: python -m pytest -q tests/test_analysis_fixes.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from script_loader import load  # noqa: E402


def _track():
    rng = np.random.default_rng(3)
    xy = np.cumsum(rng.normal(size=(12, 2)), axis=0) + 20
    return pd.DataFrame({"track_number": 0, "frame": np.arange(12), "id": np.arange(12),
                         "x": xy[:, 0], "y": xy[:, 1], "intensity": 100.0})


def test_radius_of_gyration_is_real():
    """np.linalg.eig returns complex arrays in recent numpy; Rg, skewness and kurtosis must stay real floats."""
    tf = load("2_analysis/trackFeatures.py")
    rg, asym, skew, kurt = tf.RadiusGyrationAsymmetrySkewnessKurtosis(_track())
    for v in (rg, asym, skew, kurt):
        assert not np.iscomplexobj(v) and np.isfinite(v)
    out = tf.addFeaturesToTracksDF(_track(), minNumberSegments=6)
    assert out["radius_gyration"].dtype.kind == "f"
    pts = _track()[["x", "y"]].to_numpy()
    expected = np.sqrt(((pts - pts.mean(0)) ** 2).sum(1).mean())
    assert abs(rg - expected) < 1e-12


def test_diffusion_origin_with_current_pandas():
    """Step 5 took float() of a one-row Series, which raises TypeError in pandas 3."""
    s5 = load("2_analysis/Step_5_addDiffusion.py")
    df = _track()
    df["lag"] = 1.0
    out = s5.addDiffusiontoDF(df)
    assert out["zeroed_X"].iloc[0] == 0 and out["lagNumber"].tolist() == list(range(12))
    d = np.hypot(df["x"] - df["x"].iloc[0], df["y"] - df["y"].iloc[0])
    assert np.allclose(out["distanceFromOrigin"], d)


def test_velocity_dt_second_last_row():
    """Step 6 built the next-frame array with np.insert(..., -1, 0), which put the 0 before the last element: the
    second-to-last row lost its frame interval (and velocity)."""
    s6 = load("2_analysis/Step_6_addVelocity.py")
    df = pd.DataFrame({"track_number": [0] * 5, "frame": [0, 1, 3, 4, 6], "lag": [1.0, 2.0, 3.0, 4.0, np.nan],
                       "distanceFromOrigin": [0, 1, 2, 3, 4.0], "lagNumber": [0, 1, 3, 4, 6],
                       "zeroed_X": [0, 1, 2, 3, 4.0], "zeroed_Y": [0, 0, 0, 0, 0.0]})
    out = s6.addVelocitytoDF(df)
    assert out["dt"].tolist()[:4] == [1, 2, 1, 2]
    assert np.isnan(out["dt"].iloc[4])
    assert np.allclose(out["velocity"].to_numpy()[:4], [1.0, 1.0, 3.0, 2.0])


def test_compile_svm_results_without_dataframe_append(tmp_path):
    """compileIterationSVMClasses used DataFrame.append, removed in pandas 2."""
    cs = load("3_postprocessing/compileIterationSVMClasses.py")
    for cutoff in (5, 10):
        for rep in range(2):
            d = tmp_path / f"run{rep}"
            d.mkdir(exist_ok=True)
            pd.DataFrame({"Experiment": "e", "track_number": [0, 0, 1, 2], "SVM": [1, 1, 3, 2]}).to_csv(
                d / f"x_locsID_tracksRG_SVMPredicted_cutoff_{cutoff}.csv", index=False)
    results, stats = cs.compileSVMresults(str(tmp_path))
    assert len(results) == 4 and results["n_SVM_1"].tolist() == [1, 1, 1, 1]
    assert results["cutoff"].tolist() == [5, 5, 10, 10]
    assert stats.loc[5, ("n_SVM_3", "mean")] == 1

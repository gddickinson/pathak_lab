"""Pixel size is one setting (pipeline_defaults.PIXEL_SIZE_NM, 108 nm) threaded through every step that converts
between pixels and nm. savetracksCSV used to write x [nm] = x * 108 whatever pixel size the points were loaded with,
and Step 3 divided by a hard-coded 108.

Run: python -m pytest -q tests/test_pixel_size.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from script_loader import load  # noqa: E402


def _locs(path):
    rng = np.random.default_rng(4)
    rows = []
    for t in range(4):
        x, y = rng.uniform(5, 40, 2)
        for f in range(1, 9):
            rows.append({"frame": f, "x [nm]": (x + 0.2 * f) * 160, "y [nm]": (y - 0.1 * f) * 160})
    df = pd.DataFrame(rows).sort_values("frame", kind="stable")  # ThunderSTORM tables are frame-ordered
    df["id"] = np.arange(1, len(df) + 1)
    df.to_csv(path, index=False)
    return df


def test_saved_tracks_keep_the_pixel_size(tmp_path):
    pl = load("2_analysis/pointLinking.py")
    locs = _locs(tmp_path / "a_locs.csv")
    pl.addID(str(tmp_path / "a_locs.csv"))
    txy = pl.load_points(str(tmp_path / "a_locsID.csv"), pixelsize=160)
    p = pl.PointLinker(txy)
    p.link_pts(1, 3)
    p.intensities = [1.0] * len(txy)
    pl.savetracksCSV(p, str(tmp_path / "t.csv"), str(tmp_path / "a_locsID.csv"), pixelSize=160)
    t = pd.read_csv(tmp_path / "t.csv").merge(locs, on="id", suffixes=("", "_orig"))
    assert len(t) == len(locs)
    assert np.allclose(t["x [nm]"], t["x [nm]_orig"]) and np.allclose(t["y [nm]"], t["y [nm]_orig"])
    assert np.allclose(t["x"] * 160, t["x [nm]"])


def test_defaults_come_from_one_module(tmp_path, monkeypatch):
    pl = load("2_analysis/pointLinking.py")
    import pipeline_defaults
    assert pipeline_defaults.PIXEL_SIZE_NM == 108
    monkeypatch.setattr(pipeline_defaults, "PIXEL_SIZE_NM", 160.0)
    locs = _locs(tmp_path / "a_locs.csv")
    pl.addID(str(tmp_path / "a_locs.csv"))
    txy = pl.load_points(str(tmp_path / "a_locsID.csv"))
    assert np.allclose(txy[:, 1], locs["x [nm]"] / 160)
    nn = pd.DataFrame({"id": [1, 2], "nnDist_inFrame": [320.0, 480.0]})
    s4 = load("2_analysis/Step_4_joinNNtoResultDF.py")
    out = s4.addNNtoDF(pd.DataFrame({"id": [1, 2]}), nn)
    assert out["nnDist_inFrame"].tolist() == [2.0, 3.0]
    s7 = load("2_analysis/Step_7_addMissingPoints.py")
    out = s7.addMissingLocsToDF(pd.DataFrame({"id": [1], "frame": [0], "x": [0.0], "y": [0.0]}),
                                pd.DataFrame({"id": [1, 2], "frame": [0, 1], "x [nm]": [0, 320.0], "y [nm]": [0, 160.0]}))
    assert out["x"].tolist() == [0.0, 2.0] and out["y"].tolist() == [0.0, 1.0]


def test_step3_nn_distance_uses_pixel_size(tmp_path):
    s3 = load("2_analysis/Step_3_nearestNeighbour.py")
    pd.DataFrame({"id": [1, 2], "SVM": [1, 2]}).to_csv(tmp_path / "x_SVM-ALL.csv", index=False)
    pd.DataFrame({"id": [1, 2], "nnDist_inFrame": [320.0, 160.0]}).to_csv(tmp_path / "x_locsID_NN.csv", index=False)
    s3.addNNtoSVMFiles([str(tmp_path / "x_SVM-ALL.csv")], pixelSize=160)
    out = pd.read_csv(tmp_path / "x_SVM-ALL_NN.csv")
    assert out["nnDist_inFrame"].tolist() == [2.0, 1.0]

"""Write tests/fixtures/engine_cases.json: synthetic localisations and what the repo's Python makes of them.

The JS ports in web/static/*_engine.js are checked against this file by web/static/engines.test.mjs, and
tests/test_engine_fixture.py checks that the file is still what the Python code produces.
The data are SYNTHETIC (seeded random walks), not lab recordings.

Run: python scripts/make_engine_fixture.py
"""
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from web.repo_scripts import load  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "engine_cases.json"
LINK_CASES = [(1, 3.0), (18, 3.0), (2, 1.5)]
MOB_COLS = ["track_number", "frame", "zeroed_X", "zeroed_Y", "lagNumber", "distanceFromOrigin", "d_squared",
            "lag_squared", "dy-dt: distance", "dt", "velocity", "direction_Relative_To_Origin", "meanVelocity",
            "nnCountInFrame_within_3_pixels", "nnCountInFrame_within_10_pixels", "mean_X", "mean_Y",
            "distanceFromMean", "meanLocDistanceFromCenter"]


def synthetic_locs(seed=7, n_frames=60, size=48.0):
    """Seeded synthetic localisations (pixels): mobile, confined and immobile particles with missed detections."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(24):
        kind = ("mobile", "confined", "immobile")[k % 3]
        x0, y0 = rng.uniform(4, size - 4, 2)
        start, life = int(rng.integers(1, 20)), int(rng.integers(12, 40))
        x, y = x0, y0
        for f in range(start, min(n_frames, start + life) + 1):
            if kind == "mobile":
                x, y = x + rng.normal(0, 0.45), y + rng.normal(0, 0.45)
            elif kind == "confined":
                x, y = x0 + 0.6 * (x - x0) + rng.normal(0, 0.3), y0 + 0.6 * (y - y0) + rng.normal(0, 0.3)
            else:
                x, y = x0 + rng.normal(0, 0.06), y0 + rng.normal(0, 0.06)
            if rng.random() < 0.88:
                rows.append((f, x, y, rng.normal(400 if kind == "immobile" else 250, 30)))
    for _ in range(30):
        rows.append((int(rng.integers(1, n_frames + 1)), *rng.uniform(0, size, 2), rng.normal(150, 20)))
    rows.sort(key=lambda r: r[0])
    return [[int(f), i + 1, round(x * 108, 3), round(y * 108, 3), round(inten, 2)]
            for i, (f, x, y, inten) in enumerate(rows)]


OTHER_PIXEL_NM = 160.0  # a non-default camera, for the pixel-size case


def edge_tracks():
    """Hand-made tracks for corner cases: diagonal-free (b=0) tensors, y-dominant spread, repeated positions."""
    return {
        "axis_x": [[0, 0], [2, 0.5], [4, 0], [6, 0.5], [8, 0], [10, 0.5]],
        "axis_y": [[0.5, 0], [0, 2], [0.5, 4], [0, 6], [0.5, 8], [0, 10]],
        "repeats": [[1, 1], [1, 1], [2, 1.5], [2.5, 3], [2.5, 3], [4, 2], [3, 0.5], [1.5, 0]],
        "negative_corr": [[0, 5], [1, 4.2], [2.2, 3.1], [2.9, 2], [4.1, 1.2], [5, 0.3], [4.4, 1.0]],
        "positive_corr": [[0, 0], [1, 0.8], [2.1, 2.2], [3, 2.7], [4.2, 4.1], [3.6, 3.0], [5, 5.2]],
        "closed_loop": [[0, 0], [2, 0], [4, 0.6], [3, 1.4], [1.5, 0.9], [0, 0]],  # net displacement 0: fallback sign
        "mirrored": [[0, 0], [-1, 0.8], [-2.1, 2.2], [-3, 2.7], [-4.2, 4.1], [-3.6, 3.0], [-5, 5.2]],
    }


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **k)


def records(df, cols):
    out = []
    for row in df[cols].itertuples(index=False):
        out.append([None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v)
                    for v in row])
    return out


def build():
    pl, tf = load("2_analysis/pointLinking.py"), load("2_analysis/trackFeatures.py")
    s5, s6 = load("2_analysis/Step_5_addDiffusion.py"), load("2_analysis/Step_6_addVelocity.py")
    s8, s11 = load("2_analysis/Step_8_addNNcounts.py"), load("2_analysis/Step_11_addLocalizationError.py")
    locs = synthetic_locs()
    res = {"about": "SYNTHETIC data (seeded random walks) and the outputs of the repo's Python functions",
           "locs_columns": ["frame", "id", "x [nm]", "y [nm]", "intensity"], "locs": locs,
           "pixel_size_nm": load("2_analysis/pipeline_defaults.py").PIXEL_SIZE_NM,
           "link": [], "edge_tracks": {}}
    with tempfile.TemporaryDirectory() as tmp:
        lp = Path(tmp) / "syn_locs.csv"
        pd.DataFrame(locs, columns=res["locs_columns"]).to_csv(lp, index=False)
        quiet(pl.addID, str(lp))
        txy = quiet(pl.load_points, str(Path(tmp) / "syn_locsID.csv"), pixelsize=res["pixel_size_nm"])
        for skip, dist in LINK_CASES:
            p = pl.PointLinker(txy)
            quiet(p.link_pts, skip, dist)
            p.intensities = [r[4] for r in locs]
            quiet(pl.savetracksCSV, p, str(Path(tmp) / "t.csv"), str(Path(tmp) / "syn_locsID.csv"))
            table = pd.read_csv(Path(tmp) / "t.csv")
            res["link"].append({"skip": skip, "distance": dist, "recursiveFailure": p.recursiveFailure,
                                "tracks": [[int(i) for i in t] for t in p.tracks],
                                "table_columns": ["track_number", "frame", "x", "y", "intensity", "id"],
                                "table": records(table, ["track_number", "frame", "x", "y", "intensity", "id"])})
        # another pixel size: load_points and savetracksCSV must use the same one (x [nm] round-trips)
        skip, dist = LINK_CASES[1]
        txy = quiet(pl.load_points, str(Path(tmp) / "syn_locsID.csv"), pixelsize=OTHER_PIXEL_NM)
        p = pl.PointLinker(txy)
        quiet(p.link_pts, skip, dist)
        p.intensities = [r[4] for r in locs]
        quiet(pl.savetracksCSV, p, str(Path(tmp) / "t.csv"), str(Path(tmp) / "syn_locsID.csv"), pixelSize=OTHER_PIXEL_NM)
        cols = ["track_number", "frame", "x", "y", "id", "x [nm]", "y [nm]"]
        res["pixel_case"] = {"pixel_size_nm": OTHER_PIXEL_NM, "skip": skip, "distance": dist, "table_columns": cols,
                             "table": records(pd.read_csv(Path(tmp) / "t.csv"), cols)}
    tracks = pd.DataFrame(res["link"][1]["table"], columns=res["link"][1]["table_columns"])
    per = []
    for t, g in tracks.groupby("track_number", sort=False):
        if len(g) > 6:
            pts = g[["x", "y"]].to_numpy()
            rg, asym, skew, kurt = tf.RadiusGyrationAsymmetrySkewnessKurtosis(g)
            nd, eff = tf.NetDisplacementEfficiency(pts)
            per.append({"track_number": int(t), "rg": rg, "asymmetry": asym, "skewness": skew, "kurtosis": kurt,
                        "fracDimension": tf.FractalDimension(pts), "netDispl": nd, "efficiency": eff,
                        "straight": tf.SummedSinesCosines(pts)[2]})
    res["per_track"] = per
    feat = quiet(tf.addFeaturesToTracksDF, tracks.copy(), 6)
    fcols = [c for c in feat.columns if c != "nnIndex_inFrame"]
    res["features"] = {"min_segments": 6, "columns": fcols, "rows": records(feat, fcols)}
    mob = quiet(s5.addDiffusiontoDF, feat.copy())
    mob = quiet(s6.addVelocitytoDF, mob)
    mob = quiet(s8.getNNinFrame, mob, radiusList=[3, 10])
    mob = s11.addLocalizationError(mob)
    res["mobility"] = {"radii": [3, 10], "columns": MOB_COLS, "rows": records(mob, MOB_COLS)}
    for name, pts in edge_tracks().items():
        a = np.array(pts, float)
        df = pd.DataFrame({"x": a[:, 0], "y": a[:, 1]})
        rg, asym, skew, kurt = tf.RadiusGyrationAsymmetrySkewnessKurtosis(df)
        res["edge_tracks"][name] = {"points": pts, "rg": rg, "asymmetry": asym, "skewness": skew, "kurtosis": kurt,
                                    "straight": tf.SummedSinesCosines(a)[2], "fracDimension": tf.FractalDimension(a)}
    return res


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = build()
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {OUT.relative_to(ROOT)}: {len(data['locs'])} localisations, "
          f"{len(data['features']['rows'])} feature rows, {OUT.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()

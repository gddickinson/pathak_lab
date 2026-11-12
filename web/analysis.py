"""Run the repo's pipeline steps on a user's file (local app only). Every step calls the repo's own functions.

Input: a ThunderSTORM localisation table (frame, id, x [nm], y [nm]; frames start at 1) or a linked tracks table
(track_number, frame, x, y in pixels). Optional: the recording (.tif) for per-point intensities, and a training-features
file for the SVM classes. Steps, in the pipeline's order:
  link      Step 2   pointLinking: addID, load_points, PointLinker.link_pts, getIntensities, savetracksCSV
  features  Step 2   trackFeatures.addFeaturesToTracksDF (as calcFeaturesforFiles)
  svm       Step 2   svmClassification.predict_SPT_class
  nn        Steps 3+4 getNN on all localisations, addNNtoDF
  diffusion Step 5   addDiffusiontoDF        velocity Step 6 addVelocitytoDF
  missing   Step 7   addMissingLocsToDF      nncount  Step 8 getNNinFrame        locerr Step 11 addLocalizationError
Steps that need FLIKA or an ROI file (9, 10) are not offered.
"""
import io
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .repo_scripts import ROOT, load

STEPS = ["link", "features", "svm", "nn", "diffusion", "velocity", "missing", "nncount", "locerr"]
_D = load("2_analysis/pipeline_defaults.py")  # the pipeline's single defaults module
DEFAULTS = {"pixel_size_nm": float(_D.PIXEL_SIZE_NM), "skip_frames": _D.SKIP_FRAMES,
            "link_distance": float(_D.LINK_DISTANCE), "min_segments": _D.MIN_SEGMENTS,
            "radii": ",".join(str(r) for r in _D.NN_RADII), "feature_transform": _D.FEATURE_TRANSFORM}
FEATURE_TRANSFORMS = list(_D.FEATURE_TRANSFORMS)
TRAINING = ROOT / "flika_plugins" / "locsAndTracksPlotter" / "tdTomato_37Degree_CytoD_training_feats.csv"
EXAMPLE = ROOT / "minimal_model" / "Endothelial_NonBapta_bin10_crop_RESULTS-FILE.csv"
TRACK_COLS = ["n_segments", "radius_gyration", "asymmetry", "skewness", "kurtosis", "fracDimension", "netDispl",
              "Straight", "track_length", "meanLag", "meanVelocity", "meanLocDistanceFromCenter", "SVM",
              "track_intensity_mean"]


class AnalysisError(ValueError):
    """A problem with the input or parameters, shown to the user."""


def mods():
    return {"link": load("2_analysis/pointLinking.py"), "feat": load("2_analysis/trackFeatures.py"),
            "svm": load("2_analysis/svmClassification.py"), "s3": load("2_analysis/Step_3_nearestNeighbour.py"),
            "s4": load("2_analysis/Step_4_joinNNtoResultDF.py"), "s5": load("2_analysis/Step_5_addDiffusion.py"),
            "s6": load("2_analysis/Step_6_addVelocity.py"), "s7": load("2_analysis/Step_7_addMissingPoints.py"),
            "s8": load("2_analysis/Step_8_addNNcounts.py"), "s11": load("2_analysis/Step_11_addLocalizationError.py")}


def parse_params(form):
    p = {}
    try:
        p["pixel_size_nm"] = float(form.get("pixel_size_nm") or DEFAULTS["pixel_size_nm"])
        p["skip_frames"] = int(form.get("skip_frames") or DEFAULTS["skip_frames"])
        p["link_distance"] = float(form.get("link_distance") or DEFAULTS["link_distance"])
        p["min_segments"] = int(form.get("min_segments") or DEFAULTS["min_segments"])
        p["radii"] = [float(r) if float(r) % 1 else int(float(r))
                      for r in str(form.get("radii") or DEFAULTS["radii"]).split(",") if r.strip()]
    except ValueError as e:
        raise AnalysisError(f"Parameter is not a number: {e}")
    p["feature_transform"] = str(form.get("feature_transform") or DEFAULTS["feature_transform"]).lower()
    if p["feature_transform"] not in FEATURE_TRANSFORMS:
        raise AnalysisError(f"SVM feature transform must be one of: {', '.join(FEATURE_TRANSFORMS)}.")
    if p["pixel_size_nm"] <= 0 or p["link_distance"] <= 0 or p["skip_frames"] < 0 or p["min_segments"] < 0:
        raise AnalysisError("Pixel size and linking distance must be positive; gap and minimum length not negative.")
    steps = [s for s in STEPS if str(form.get("step_" + s, "")).lower() in ("1", "true", "on")]
    if not steps:
        raise AnalysisError("Choose at least one step.")
    return p, steps


def read_table(data):
    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception as e:
        raise AnalysisError(f"Could not read the CSV file: {e}")
    if {"track_number", "frame", "x", "y"} <= set(df.columns):
        return "tracks", df
    if {"frame", "x [nm]", "y [nm]"} <= set(df.columns):
        return "locs", df
    raise AnalysisError("Expected a ThunderSTORM localisation file (frame, x [nm], y [nm]) or a tracks table "
                        "(track_number, frame, x, y).")


def read_tiff(data):
    """The recording, oriented as Step 2's linkFilesNoFlika does (rot90 over axes 1,2 then fliplr)."""
    import tifffile
    try:
        A = tifffile.imread(io.BytesIO(data))
    except Exception as e:
        raise AnalysisError(f"Could not read the TIFF recording: {e}")
    if A.ndim != 3:
        raise AnalysisError(f"Expected a 3D (frames, y, x) recording, got shape {A.shape}.")
    return np.fliplr(np.rot90(A, axes=(1, 2)))


def _check(kind, steps):
    if kind == "locs" and "link" not in steps:
        raise AnalysisError("A localisation file must be linked first: tick 'Link' (Step 2).")
    if kind == "tracks" and "link" in steps:
        raise AnalysisError("This file is already linked (it has track_number): untick 'Link'.")
    for s in ("nn", "missing"):
        if s in steps and kind != "locs":
            raise AnalysisError("Steps 3+4 and 7 need the localisation file (they use every localisation).")
    need = {"svm": "features", "diffusion": "features", "velocity": "diffusion"}
    for s, req in need.items():
        if s in steps and req not in steps:
            raise AnalysisError(f"Step '{s}' needs '{req}' too.")


def run(kind, df, params, steps, tiff=None, training=None, name="upload.csv", log=None):
    """Run the chosen steps; returns (result dict, output DataFrame)."""
    _check(kind, steps)
    m, warn = mods(), []
    say = log or (lambda s: None)
    locsID = None
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if kind == "locs":
            if "id" not in df.columns:
                df = df.assign(id=np.arange(1, len(df) + 1))
                warn.append("No 'id' column: localisations were numbered 1..n in file order.")
            locs_path = tmp / "upload_locs.csv"
            df.to_csv(locs_path, index=False)
            m["link"].addID(str(locs_path))
            locsID = pd.read_csv(tmp / "upload_locsID.csv")
        if "link" in steps:
            say("linking")
            txy = m["link"].load_points(str(tmp / "upload_locsID.csv"), pixelsize=params["pixel_size_nm"])
            p = m["link"].PointLinker(txy)
            p.link_pts(params["skip_frames"], params["link_distance"])
            if p.recursiveFailure:
                warn.append("Recursion limit reached while linking (a track longer than 10000 points): Step 2 "
                            "skips such files unless includeRecursionProblem is set; the result is kept here.")
            if tiff is not None:
                p.getIntensities(tiff)
            else:
                p.intensities = [math.nan] * len(txy)
                warn.append("No recording given: intensity columns are empty (Step 2 reads them from the .tif).")
            if not any(len(t) > 1 for t in p.tracks):
                raise AnalysisError("No localisations could be linked with these settings.")
            m["link"].savetracksCSV(p, str(tmp / "tracks.csv"), str(tmp / "upload_locsID.csv"))
            df = pd.read_csv(tmp / "tracks.csv")
            if params["pixel_size_nm"] != 108:
                warn.append("savetracksCSV writes x [nm] = x × 108 whatever the pixel size; Steps 3, 4 and 7 use "
                            "the pixel size you set.")
        n_tracks_linked = int(df["track_number"].nunique())
        if "features" in steps:
            say("features")
            if "id" not in df.columns:
                df = df.assign(id=np.arange(len(df)))
            if "intensity" not in df.columns:
                df = df.assign(intensity=math.nan)
            try:
                df = m["feat"].addFeaturesToTracksDF(df, params["min_segments"])
            except Exception as e:
                raise AnalysisError(f"Track features failed ({e}). Step 2 skips the whole file in this case.")
            if df.empty:
                raise AnalysisError("No track is longer than the minimum number of segments.")
        if "svm" in steps:
            say("svm")
            rg = tmp / "upload_tracksRG.csv"
            df.to_csv(rg)
            train = training if training is not None else TRAINING
            m["svm"].predict_SPT_class(str(train), str(rg), Path(name).stem, level="",
                                       feature_transform=params["feature_transform"])
            df = pd.read_csv(tmp / "upload_tracksRG_SVMPredicted.csv")
        if "nn" in steps:
            say("nearest neighbours")
            nn = m["s3"].getNN(locsID.copy())[["id", "frame", "x [nm]", "y [nm]", "nnDist_inFrame"]]
            df = m["s4"].addNNtoDF(df, nn, pixelSize=params["pixel_size_nm"])
        if "diffusion" in steps:
            df = m["s5"].addDiffusiontoDF(df)
        if "velocity" in steps:
            df = m["s6"].addVelocitytoDF(df)
        if "missing" in steps:
            df = m["s7"].addMissingLocsToDF(df, locsID.copy(), pixelSize=params["pixel_size_nm"])
        if "nncount" in steps:
            df = m["s8"].getNNinFrame(df, radiusList=params["radii"])
        if "locerr" in steps:
            df = m["s11"].addLocalizationError(df)
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    return summarise(df, kind, params, steps, warn, name, n_tracks_linked, locsID), df


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else round(f, 6)


def summarise(df, kind, params, steps, warn, name, n_tracks_linked, locsID, max_tracks=1500):
    linked = df[df["track_number"].notna()]
    per = linked.groupby("track_number", sort=True)
    cols = [c for c in TRACK_COLS if c in df.columns]
    first = per.first()
    tracks = [{"track_number": int(t), "n_points": int(n), **{c: _num(first.at[t, c]) for c in cols}}
              for t, n in per.size().items()]
    coords = {}
    for t, g in list(per)[:max_tracks]:
        g = g.sort_values("frame")
        coords[str(int(t))] = [g["frame"].astype(int).tolist(), [_num(v) for v in g["x"]], [_num(v) for v in g["y"]]]
    unlinked = df[df["track_number"].isna()]
    res = {"name": name, "kind": kind, "params": params, "steps": steps, "warnings": warn,
           "n_rows": int(len(df)), "n_locs": None if locsID is None else int(len(locsID)),
           "n_tracks_linked": n_tracks_linked, "n_tracks": int(linked["track_number"].nunique()),
           "n_unlinked": int(len(unlinked)), "columns": [str(c) for c in df.columns], "track_columns": cols,
           "tracks": tracks[:2000], "coords": coords, "coords_truncated": len(per) > max_tracks,
           "unlinked_xy": [[_num(x), _num(y)] for x, y in unlinked[["x", "y"]].head(5000).to_numpy()]}
    if "SVM" in df.columns:
        res["svm_counts"] = {str(int(k)): int(v) for k, v in per["SVM"].first().value_counts().sort_index().items()}
    return res


def example_frame():
    """The repo's minimal_model results table, used as a tracks input in the local app only (not exported)."""
    df = pd.read_csv(EXAMPLE)
    return df[["track_number", "frame", "x", "y", "intensity"]].dropna(subset=["track_number"])

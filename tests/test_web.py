"""Web interface tests: every exported endpoint answers, the static build is complete, and the JS engine tests pass.

Run: python -m pytest -q tests/test_web.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from web.app import STATIC, STATIC_ENDPOINTS, app  # noqa: E402

client = TestClient(app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "static/main.js" in r.text


@pytest.mark.parametrize("path", STATIC_ENDPOINTS)
def test_endpoint(path):
    r = client.get("/api" + path)
    assert r.status_code == 200, r.text
    assert r.json() is not None


def test_static_build(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_static
    build_static.build(tmp_path / "site")
    assert (tmp_path / "site" / "index.html").read_text().count("APP_STATIC") == 1
    for path in STATIC_ENDPOINTS:
        assert (tmp_path / "site" / "api" / (path.strip("/") + ".json")).exists()


def test_static_assets_referenced_exist():
    html = (STATIC / "index.html").read_text()
    for name in ("style.css", "main.js"):
        assert f"static/{name}" in html and (STATIC / name).exists()


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_engine():
    tests = sorted(STATIC.glob("*.test.mjs"))
    for t in tests:
        r = subprocess.run(["node", str(t)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


# --- project-specific -----------------------------------------------------------------------------------------
import io  # noqa: E402
import json  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "engine_cases.json"
LAB_FILES = [ROOT / "minimal_model" / "Endothelial_NonBapta_bin10_crop_RESULTS-FILE.csv",
             ROOT / "flika_plugins" / "locsAndTracksPlotter" / "tdTomato_37Degree_CytoD_training_feats.csv"]


def _locs_csv():
    fx = json.loads(FIXTURE.read_text())
    lines = ["frame,id,x [nm],y [nm]"] + [f"{f},{i},{x},{y}" for f, i, x, y, _ in fx["locs"]]
    return "\n".join(lines).encode()


def test_catalogue_covers_every_python_file():
    tools = client.get("/api/catalogue").json()["tools"]
    on_disk = [p for base in ("flika_scripts", "flika_plugins") for p in (ROOT / base).rglob("*.py") if "__pycache__" not in p.parts]
    assert len(tools) == len(on_disk) and all(t["id"] for t in tools)
    step5 = client.get("/api/tool/2_analysis__Step_5_addDiffusion.py").json()
    assert "*_SVMPredicted_NN.csv" in step5["inputs"] and "_diffusion.csv" in step5["outputs"]
    assert step5["demos"] == ["mobility"] and step5["local_run"]


def test_catalogue_has_no_authorship_or_private_paths():
    blob = json.dumps([client.get("/api" + p).json() for p in STATIC_ENDPOINTS])
    for bad in ("@author", "/Users/", "gmail.com", "Created on"):
        assert bad not in blob, bad


def test_run_links_and_measures_uploaded_localisations():
    form = {f"step_{s}": "1" for s in ("link", "features", "diffusion", "velocity", "nncount", "locerr", "nn", "missing")}
    r = client.post("/api/run", data=form, files={"file": ("syn_locs.csv", _locs_csv(), "text/csv")})
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["kind"] == "locs" and res["n_tracks"] > 5 and "meanLocDistanceFromCenter" in res["columns"]
    assert res["n_unlinked"] > 0  # Step 7 added the unlinked localisations
    csv = client.get(f"/api/runs/{res['run_id']}/result.csv")
    assert csv.status_code == 200 and csv.text.startswith("track_number")


def test_run_reports_input_problems():
    r = client.post("/api/run", data={"step_features": "1"}, files={"file": ("a.csv", b"a,b\n1,2\n", "text/csv")})
    assert r.status_code == 400 and "Expected" in r.json()["detail"]
    r = client.post("/api/run", data={"step_velocity": "1"}, files={"file": ("l.csv", _locs_csv(), "text/csv")})
    assert r.status_code == 400


def test_run_example_tracks_locally():
    r = client.post("/api/run", data={"example": "1", "step_features": "1", "step_locerr": "1"})
    assert r.status_code == 200, r.text
    assert r.json()["n_tracks"] > 50


def test_static_export_holds_no_lab_data(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_static
    build_static.build(tmp_path / "site")
    exported = [p for p in (tmp_path / "site").rglob("*") if p.is_file()]
    assert not [p for p in exported if p.suffix.lower() in (".tif", ".tiff", ".csv", ".mp4")]
    text = "".join(p.read_text(errors="ignore") for p in exported)
    for lab in LAB_FILES:
        rows = lab.read_text().splitlines()[1:4]
        assert not any(row[:60] in text for row in rows), lab.name


def test_svm_feature_transform_option():
    info = client.get("/api/info").json()
    assert info["defaults"]["feature_transform"] == "boxcox" and info["feature_transforms"] == ["boxcox", "none"]
    r = client.post("/api/run", data={"step_link": "1", "feature_transform": "log"},
                    files={"file": ("l.csv", _locs_csv(), "text/csv")})
    assert r.status_code == 400 and "feature transform" in r.json()["detail"]

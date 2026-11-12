"""FastAPI web app for the Pathak lab TIRF analysis tools: serves the front end in static/ and the JSON API under /api.

Run with `python run_web.py` (http://127.0.0.1:8202, local only). `scripts/build_static.py` exports every path in
STATIC_ENDPOINTS to JSON for the GitHub Pages site, so keep that list in step with the GET routes below.

The catalogue is read from the repo's own files (web/catalogue.py); /api/run calls the repo's pipeline functions on
the user's file (web/analysis.py). The repo's committed recording, results table and SVM training file are used by
the local app only and are never exported to the static site.
"""
import contextlib
import io
import re
import tempfile
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi.concurrency import run_in_threadpool
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from . import analysis as an
from .catalogue import GROUPS, ROOT, by_id, catalogue, groups, summary

STATIC = Path(__file__).resolve().parent / "static"
TITLE = "Pathak lab TIRF tools"
MAX_UPLOAD = 500 * 1024 * 1024
DEMO_ROUTES = ["linking", "features", "mobility"]
INSTRUCTIONS = {n: f"flika_scripts/piezo1_analysis/2_analysis/Step_{n}_instruction{'' if n == '1' else 's'}.md"
                for n in "1234"}  # the step instruction notes next to Steps 1-4

app = FastAPI(title=TITLE)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
RUNS: "OrderedDict[str, tuple]" = OrderedDict()
MAX_RUNS = 5


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    """Counts for the home page, the groups, the plugins and the local-run defaults."""
    entries, plugins = catalogue()
    scripts = [e for e in entries if e["group"] != "plugin"]
    return {
        "title": TITLE, "version": "1",
        "counts": {"files": len(entries), "scripts": len(scripts), "plugins": len(plugins),
                   "plugin_files": len(entries) - len(scripts),
                   "functions": sum(e["n_functions"] for e in entries),
                   "pipeline_steps": sum(1 for e in scripts if e["name"].startswith("Step_")),
                   "demos": len(DEMO_ROUTES), "local_run": sum(1 for e in entries if e["local_run"])},
        "groups": groups(), "plugins": plugins, "defaults": an.DEFAULTS, "feature_transforms": an.FEATURE_TRANSFORMS, "steps": an.STEPS,
        "paper": {"title": "Visualizing PIEZO1 localization and activity in hiPSC-derived single cells and organoids "
                           "with HaloTag technology", "journal": "Nature Communications 16, 5556 (2025)",
                  "doi": "10.1038/s41467-025-59150-1"},
    }


@app.get("/api/catalogue")
def catalogue_list():
    """Every Python file, summarised, with the group and plugin descriptions."""
    entries, plugins = catalogue()
    return {"groups": groups(), "plugins": plugins, "tools": [summary(e) for e in entries]}


@app.get("/api/tool/{tool}")
def tool(tool: str):
    """Everything read from one file: purpose, functions, parameters, inputs/outputs, instructions."""
    e = by_id(tool)
    if e is None:
        raise HTTPException(404, "unknown tool")
    e = dict(e)
    m = re.match(r"Step_(\d+)_", e["name"]) if e["group"] == "2_analysis" else None
    md = INSTRUCTIONS.get(m.group(1)) if m else None
    e["instructions"] = (ROOT / md).read_text(encoding="utf-8") if md else None
    e["group_title"] = GROUPS[e["group"]][0]
    return e


@app.get("/api/pipeline")
def pipeline():
    """The analysis scripts in pipeline order with the file patterns each reads and writes."""
    entries, _ = catalogue()
    def key(e):
        n = e["name"].split("_")[1] if e["name"].startswith("Step_") else "99"
        return (list(GROUPS).index(e["group"]), int(n) if n.isdigit() else 99, e["name"])
    keep = [e for e in entries if e["group"] in ("1_preprocessing", "2_analysis", "3_postprocessing")]
    return {"stages": [{k: e[k] for k in ("id", "name", "group", "purpose", "inputs", "outputs", "demos",
                                          "local_run", "needs_gui")} for e in sorted(keep, key=key)]}


@app.post("/api/run")
async def run(request: Request, file: UploadFile | None = File(None), tiff: UploadFile | None = File(None),
              training: UploadFile | None = File(None), example: str = Form("")):
    """Run the chosen pipeline steps on an uploaded CSV (or the repo's example) with the repo's functions."""
    form = dict((await request.form()).items())
    try:
        params, steps = an.parse_params(form)
        if example == "1":
            kind, df, name = "tracks", an.example_frame(), an.EXAMPLE.name
        elif file is not None and file.filename:
            data = await file.read(MAX_UPLOAD + 1)
            if len(data) > MAX_UPLOAD:
                raise an.AnalysisError("File too large (limit 500 MB).")
            name = Path(file.filename).name
            kind, df = an.read_table(data)
        else:
            raise an.AnalysisError("Upload a CSV file (or tick the repo example).")
        rec = an.read_tiff(await tiff.read()) if tiff is not None and tiff.filename else None
        train_path = None
        with contextlib.ExitStack() as stack:
            if training is not None and training.filename:
                tmpdir = stack.enter_context(tempfile.TemporaryDirectory())
                train_path = Path(tmpdir) / "training.csv"
                train_path.write_bytes(await training.read())
            result, out = await run_in_threadpool(_quiet_run, kind, df, params, steps, rec, train_path, name)
    except an.AnalysisError as e:
        raise HTTPException(400, str(e))
    run_id = uuid.uuid4().hex[:12]
    RUNS[run_id] = (Path(name).stem, out)
    while len(RUNS) > MAX_RUNS:
        RUNS.popitem(last=False)
    result["run_id"] = run_id
    return result


def _quiet_run(kind, df, params, steps, rec, train_path, name):
    """an.run with the scripts' progress printing (print, tqdm) kept out of the server log."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return an.run(kind, df, params, steps, tiff=rec, training=train_path, name=name)


@app.get("/api/runs/{run_id}/result.csv")
def run_csv(run_id: str):
    """The full output table of a run."""
    if run_id not in RUNS:
        raise HTTPException(404, "run not found (the server keeps the last few runs only)")
    stem, df = RUNS[run_id]
    return Response(df.to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{stem}_analysed.csv"'})


# GET paths (without the /api prefix) written to the static site as api/<path>.json.
STATIC_ENDPOINTS = ["/info", "/catalogue", "/pipeline"] + [f"/tool/{e['id']}" for e in catalogue()[0]]

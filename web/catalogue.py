"""The tool catalogue: one entry per Python file in flika_scripts/ and flika_plugins/, built from the repo itself.

Purpose text is the file's own module docstring, else the README of its folder (quoted, with the source named).
Inputs/outputs, parameters and functions come from catalogue_parse (static reading, nothing is executed).
DEMOS and LOCAL_RUN say which tools have an in-browser demo (JS port tested against the Python) or run in the local
app on the user's own files.
"""
import ast
import hashlib
import re
from functools import lru_cache
from pathlib import Path

from . import catalogue_parse as cp

ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/gddickinson/pathak_lab"
GROUPS = {
    "1_preprocessing": ("Preprocessing", "Prepare recordings before localisation: binning, cropping, bit depth, autofluorescence."),
    "2_analysis": ("Analysis pipeline", "Steps 1-11: localise, link, classify and measure PIEZO1 puncta and tracks."),
    "3_postprocessing": ("Postprocessing", "Compile, filter and summarise analysed track tables."),
    "testing_misc": ("Testing and miscellaneous", "Development and test scripts kept for reference (the folder README says they may be incomplete)."),
    "plugin": ("FLIKA plugins", "Interactive plugins for the FLIKA image-analysis program."),
}
A = "2_analysis/"
DEMOS = {  # repo-relative path under flika_scripts/piezo1_analysis -> demo routes
    A + "pointLinking.py": ["linking"],
    A + "Step_2_linkingAndClassification_trackpyOption_recursionErrorFix.py": ["linking", "features"],
    A + "trackFeatures.py": ["features"],
    A + "Step_5_addDiffusion.py": ["mobility"], A + "Step_6_addVelocity.py": ["mobility"],
    A + "Step_8_addNNcounts.py": ["mobility"], A + "Step_11_addLocalizationError.py": ["mobility"],
}
LOCAL_RUN = {A + p for p in ("pointLinking.py", "trackFeatures.py", "svmClassification.py",
                             "Step_2_linkingAndClassification_trackpyOption_recursionErrorFix.py",
                             "Step_3_nearestNeighbour.py", "Step_4_joinNNtoResultDF.py", "Step_5_addDiffusion.py",
                             "Step_6_addVelocity.py", "Step_7_addMissingPoints.py", "Step_8_addNNcounts.py",
                             "Step_11_addLocalizationError.py")}


def _files():
    out = []
    for base in ("flika_scripts", "flika_plugins"):
        for p in sorted((ROOT / base).rglob("*.py")):
            if "__pycache__" not in p.parts:
                out.append(p)
    return out


def tool_id(rel):
    """URL- and file-safe id: path below flika_scripts/piezo1_analysis or flika_plugins, '/' -> '__'."""
    rel = rel.replace("flika_scripts/piezo1_analysis/", "").replace("flika_plugins/", "plugin/")
    return re.sub(r"[^\w.-]", "_", rel.replace("/", "__"))


def _group(rel):
    parts = rel.split("/")
    if parts[0] == "flika_plugins":
        return "plugin", parts[1]
    return parts[2], None


def _readme(folder):
    p = folder / "README.md"
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def _first_paragraph(md):
    paras = [p.strip() for p in re.split(r"\n\s*\n", md) if p.strip() and not p.lstrip().startswith(("#", "```", "-"))]
    return paras[0] if paras else ""


def _plugin_info(folder):
    info = {"name": folder.name, "summary": _first_paragraph(_readme(folder)), "version": None, "menu": None}
    xml = folder / "info.xml"
    if xml.exists():
        t = xml.read_text(errors="replace")
        m = re.search(r"<version>\s*([^<]+?)\s*</version>", t)
        info["version"] = m.group(1) if m else None
        m = re.search(r'function="([^"]+)"', t)
        info["menu"] = m.group(1) if m else None
    return info


def _entry(path):
    rel = path.relative_to(ROOT).as_posix()
    group, plugin = _group(rel)
    src = path.read_text(encoding="utf-8", errors="replace")
    tree, magic, err = cp.parse_source(src)
    sub = rel.replace("flika_scripts/piezo1_analysis/", "")
    e = {"id": tool_id(rel), "path": rel, "name": path.name, "group": group, "plugin": plugin,
         "lines": src.count("\n") + 1, "source_url": f"{REPO_URL}/blob/main/{rel}",
         "sha1": hashlib.sha1(src.encode()).hexdigest(), "ipython_magic": magic, "parse_error": err,
         "demos": DEMOS.get(sub, []), "local_run": sub in LOCAL_RUN}
    doc = cp.clean_doc(ast.get_docstring(tree)) if tree else ""
    note = cp.readme_note(_readme(path.parent), path.name) if group != "plugin" else None
    e["docstring"] = doc
    if group == "plugin":
        e["purpose"], e["purpose_source"] = _plugin_info(path.parent)["summary"], f"flika_plugins/{plugin}/README.md"
    elif doc:
        e["purpose"], e["purpose_source"] = doc, "module docstring"
    elif note:
        e["purpose"], e["purpose_source"] = note, f"{path.parent.relative_to(ROOT).as_posix()}/README.md"
    else:
        e["purpose"], e["purpose_source"] = "", None
    e["readme_note"] = note if doc and note else None
    if tree:
        e["functions"], e["classes"] = cp.definitions(tree)
        e["inputs"], e["outputs"] = cp.files_used(tree)
        e["settings"], e["has_main"] = cp.main_settings(tree), cp.has_main(tree)
        mods = cp.imports(tree)
    else:
        e.update(functions=[], classes=[], inputs=[], outputs=[], settings=[], has_main=False)
        mods = []
    e["imports"] = [m for m in mods if m in cp.HEAVY]
    e["needs"] = sorted({cp.HEAVY[m] for m in mods if m in cp.HEAVY})
    e["needs_gui"] = any(m in mods for m in ("flika", "qtpy", "PyQt5", "pyqtgraph"))
    e["n_functions"] = len(e["functions"]) + sum(len(c["methods"]) for c in e["classes"])
    return e


@lru_cache(maxsize=1)
def catalogue():
    """All entries plus duplicate detection (identical files) and plugin summaries."""
    entries = [_entry(p) for p in _files()]
    by_hash = {}
    for e in entries:
        by_hash.setdefault(e["sha1"], []).append(e["id"])
    for e in entries:
        e["identical_to"] = [i for i in by_hash[e["sha1"]] if i != e["id"]]
    plugins = [_plugin_info(d) for d in sorted((ROOT / "flika_plugins").iterdir()) if d.is_dir()]
    return entries, plugins


def summary(e):
    """The fields the list view needs."""
    keys = ("id", "path", "name", "group", "plugin", "lines", "purpose", "purpose_source", "demos", "local_run",
            "needs_gui", "n_functions", "identical_to", "ipython_magic")
    s = {k: e[k] for k in keys}
    s["purpose"] = (s["purpose"].split("\n\n")[0])[:320]
    return s


def by_id(tool):
    entries, _ = catalogue()
    return next((e for e in entries if e["id"] == tool), None)


def groups():
    return [{"id": k, "title": v[0], "about": v[1]} for k, v in GROUPS.items()]

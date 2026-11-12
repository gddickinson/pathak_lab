"""Find and import the repo's analysis scripts by path.

The pipeline folders (1_preprocessing, 2_analysis, ...) start with digits, so they are not Python packages: scripts
are loaded with importlib, with their own folder on sys.path so sibling imports (pointLinking, trackFeatures) work.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "flika_scripts" / "piezo1_analysis"
ANALYSIS = SCRIPTS / "2_analysis"
_cache = {}


def load(rel):
    """Import flika_scripts/piezo1_analysis/<rel> as a module (cached)."""
    if rel in _cache:
        return _cache[rel]
    path = SCRIPTS / rel
    folder = str(path.parent)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    spec = importlib.util.spec_from_file_location("pl_" + path.stem.replace("-", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _cache[rel] = mod
    return mod

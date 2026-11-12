"""Load a pipeline script from flika_scripts/ by its path (the folders start with digits, so they are not packages)."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "flika_scripts" / "piezo1_analysis"


def load(rel):
    """Import flika_scripts/piezo1_analysis/<rel> as a module (its folder goes on sys.path for sibling imports)."""
    path = SCRIPTS / rel
    folder = str(path.parent)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    name = "pl_" + path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

"""Load a pipeline script from flika_scripts/ by its path (re-exports web.repo_scripts.load for the tests)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from web.repo_scripts import load  # noqa: E402,F401

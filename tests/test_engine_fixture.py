"""The JS-vs-Python fixture (tests/fixtures/engine_cases.json) is still what the repo's Python produces.

If this fails after a change to the analysis code, run `python scripts/make_engine_fixture.py` and
`node web/static/engines.test.mjs` (the JS ports must keep matching). Numbers compare with a tolerance (libm differs
in the last digit between platforms).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import make_engine_fixture as mk  # noqa: E402


def _same(a, b, where="root"):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), where
        for k in a:
            _same(a[k], b[k], f"{where}.{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{where}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert (a is None and b is None) or np.isclose(a, b, rtol=1e-9, atol=1e-12), f"{where}: {a} vs {b}"
    else:
        assert a == b, f"{where}: {a} vs {b}"


def test_fixture_is_current():
    stored = json.loads(mk.OUT.read_text())
    fresh = json.loads(json.dumps(mk.build()))
    _same(fresh, stored)

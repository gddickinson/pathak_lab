# pathak_lab

@INTERFACE.md

- Tests: `python -m pytest -q tests`; JS ports: `node web/static/engines.test.mjs`.
- After changing pointLinking.py / trackFeatures.py / Steps 5, 6, 8, 11: run `python scripts/make_engine_fixture.py`
  and the node test; the JS ports in web/static/*_engine.js must keep matching the Python.
- Web app: `python run_web.py` (port 8202). Static site: `python scripts/build_static.py --out site`.
- Lab data (minimal_model/, the SVM training CSV) must never be exported to the static site (tested).
- Keep new files under 500 lines; update INTERFACE.md and SESSION_LOG.md.

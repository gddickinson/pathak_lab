"""Web interface for the Pathak lab TIRF tools (FastAPI + plain ES modules; see INTERFACE.md).

Contents: app.py (routes, STATIC_ENDPOINTS), catalogue.py + catalogue_parse.py (tool catalogue read from the repo's
files), analysis.py (runs the pipeline steps on a user's file), repo_scripts.py (imports the pipeline scripts by
path), static/ (front end, JS ports of the analysis and their node test).
"""

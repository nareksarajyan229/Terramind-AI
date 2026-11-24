#!/bin/bash
cd "$(dirname "$0")"
command -v python3 >/dev/null || { echo "Install Python from python.org first"; open https://www.python.org/downloads/; exit 1; }
[ -x .venv/bin/python ] || { echo "First start: installing TerraMind AI..."; python3 -m venv .venv; }
.venv/bin/python -m pip install -q -r requirements.txt
.venv/bin/python -m streamlit run app.py

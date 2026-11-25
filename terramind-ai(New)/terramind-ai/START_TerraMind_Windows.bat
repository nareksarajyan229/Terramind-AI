@echo off
title TerraMind AI
cd /d "%~dp0"
set PY=py
where py >nul 2>nul || set PY=python
where %PY% >nul 2>nul || (
  echo Python is not installed. Install it from python.org and tick "Add python.exe to PATH".
  start https://www.python.org/downloads/
  pause
  exit /b
)
if not exist ".venv\Scripts\python.exe" (
  echo First start: installing TerraMind AI, this takes a few minutes...
  %PY% -m venv .venv
)
".venv\Scripts\python.exe" -m pip install -q --disable-pip-version-check -r requirements.txt
".venv\Scripts\python.exe" -m streamlit run app.py
pause

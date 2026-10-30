@echo off
REM One-command launcher for Windows: double-click this file.
cd /d "%~dp0"
if not exist .venv ( py -3 -m venv .venv || python -m venv .venv )
call .venv\Scripts\activate.bat
if not exist .venv\deps_ok.txt (
  python -m pip install --upgrade pip
  pip install -r requirements.txt && echo ok> .venv\deps_ok.txt
)
streamlit run app.py
pause

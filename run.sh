#!/usr/bin/env bash
# One-command launcher: creates .venv, installs deps (first run), starts the app.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
[ -d .venv ] || $PY -m venv .venv
. .venv/bin/activate
if [ ! -f .venv/.deps_ok ] || [ requirements.txt -nt .venv/.deps_ok ]; then
  pip install -q --upgrade pip && pip install -q -r requirements.txt && touch .venv/.deps_ok
fi
exec streamlit run app.py "$@"

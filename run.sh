#!/bin/sh
# Taste MVP dev server. Creates the venv on first run.
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
exec .venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000

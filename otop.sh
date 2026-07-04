#!/bin/sh
# Launch the otop dashboard, creating the venv on first run.
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
    echo "first run: creating venv and installing dependencies..."
    python3 -m venv .venv && .venv/bin/pip -q install -r requirements.txt || exit 1
fi
exec .venv/bin/python otop.py "$@"

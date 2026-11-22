#!/bin/sh
# Invoke the Python launcher explicitly: macOS shebangs cannot contain spaces.
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_dir"
exec "$project_dir/.venv/bin/python" "$project_dir/.venv/bin/mjpython" \
  "$project_dir/scripts/watch_insertion.py" "$@"

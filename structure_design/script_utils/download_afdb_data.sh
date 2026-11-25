#!/usr/bin/env bash
# Preserve the index-file/output-directory interface; see --help for mirror options.
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(dirname -- "$script_dir")

if [[ -n "${PYTHON_EXEC:-}" ]]; then
    python_exec="$PYTHON_EXEC"
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
    python_exec="$repo_root/.venv/bin/python"
else
    python_exec=python3
fi

exec "$python_exec" "$script_dir/download_afdb_data.py" "$@"

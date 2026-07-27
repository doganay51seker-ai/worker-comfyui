#!/usr/bin/env bash
set -euo pipefail

if [ "${RUNTIME_MODEL_FETCH:-true}" = "true" ]; then
    /opt/venv/bin/python /usr/local/bin/fetch-runtime-models.py
else
    echo "runtime-models: fetch disabled"
fi

exec /start.sh "$@"

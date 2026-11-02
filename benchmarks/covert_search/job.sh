#!/usr/bin/env bash
# job.sh ISLAND START CLASS SEED [meter.py args...]: one candidate on the node, as launch.py runs it (research run --gpus 1).
set -u
P=${COVERT_PYTHON:-/workspace/jobs/venv312/bin/python}
here=$(cd "$(dirname "$0")" && pwd)
nvidia-smi --query-gpu=index,name --format=csv,noheader
exec "$P" "$here/meter.py" --island "$here/islands/$1" --start "$2" --class "$3" --seed "$4" --out "$OUT" "${@:5}"

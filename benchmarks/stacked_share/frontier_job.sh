#!/usr/bin/env bash
# The frontier census: atoms.py counts every atom frontier.py composes (property/v1 on vLLM's registry Definitions; CPU only,
# no torch), holds them to the int8 census's measured Llama-3.1-8B, and frontier.py gives the table. Run from the source
# tree's root; keeps atoms.json, atoms.log, frontier.json and frontier.md in $OUT. PY=PATH names a Python with numpy, else a
# venv with numpy is built in a scratch directory the job removes.
#
#   bash benchmarks/stacked_share/frontier_job.sh      # on a CPU node, from the source tree's root
set -euo pipefail
SRC=$(pwd)
export PYTHONPATH=$SRC:$SRC/integrations/vllm:$SRC/tools/research/src
if [ -z "${PY:-}" ]; then
  W=$(mktemp -d)
  trap 'rm -rf "$W"' EXIT
  uv venv -q --python 3.12 "$W/venv" && uv pip install -q --python "$W/venv/bin/python" numpy==2.5.3
  PY=$W/venv/bin/python
fi
HERE=$SRC/benchmarks/stacked_share
/usr/bin/time -v "$PY" "$HERE/atoms.py" --out "$OUT/atoms.json" 2> >(tee "$OUT/atoms.log" >&2)
"$PY" "$HERE/frontier.py" --atoms "$OUT/atoms.json" --out "$OUT/frontier.json"

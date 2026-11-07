#!/usr/bin/env bash
# OVERHEADDEF's spot_job.sh on vy-spot-glm53: the gpu-lease and TP8 shims (sent beside it) go on PATH and PYTHONPATH, and the
# job refuses to start unless vllm.EngineArgs is TP8 under them.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run this under research run}; IN=$D/inputs
S=$(mktemp -d); cp "$IN/gpu-lease" "$IN/sitecustomize.py" "$S/" && chmod +x "$S/gpu-lease" || exit 2
export PATH="$S:$PATH" PYTHONPATH="$S${PYTHONPATH:+:$PYTHONPATH}" SPOT_TP=8
"${VENV:?}/bin/python" -c 'import functools, vllm; e = vllm.EngineArgs
assert isinstance(e, functools.partial) and e.keywords == {"tensor_parallel_size": 8}, e; print("spot shim: EngineArgs TP8")' || exit 2
[ "$(command -v gpu-lease)" = "$S/gpu-lease" ] || { echo "spot shim: gpu-lease is $(command -v gpu-lease)"; exit 2; }
bash "$IN/spot_job.sh"

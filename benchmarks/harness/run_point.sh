#!/usr/bin/env bash
# `bash benchmarks/harness/run_point.sh ARGS...`: one sweep point (`python -m verity_bench _vllm-point ARGS...`) as the
# queued research run `verity-bench run SPEC` dispatches, from the source tree (--cwd source), in env.sh's environment.
set -euo pipefail
. "$(dirname "$0")/env.sh"
exec "$VENV/bin/python" -m verity_bench _vllm-point "$@"

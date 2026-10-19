#!/usr/bin/env bash
# `bash benchmarks/harness/run_point.sh ARGS...`: one sweep point (`python -m verity_bench _vllm-point ARGS...`) as the
# queued research run `verity-bench run SPEC` dispatches, from the source tree (--cwd source), in env.sh's environment.
# The dispatcher also forwards the spec's env, and its setup script as VERITY_BENCH_SETUP, which is sourced here (cwd the
# source tree, with env.sh's VENV, CUDA_HOME, PATH and PYTHONPATH set) so what it exports reaches the point.
set -euo pipefail
. "$(dirname "$0")/env.sh"
if [ -n "${VERITY_BENCH_SETUP:-}" ]; then
  # shellcheck source=/dev/null
  . "$ROOT/$VERITY_BENCH_SETUP"
fi
exec "$VENV/bin/python" -m verity_bench _vllm-point "$@"

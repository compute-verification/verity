#!/usr/bin/env bash
# `bash benchmarks/harness/run_point.sh ARGS...`: one sweep point (`python -m verity_bench _vllm-point ARGS...`) as the
# queued research run `verity-bench run SPEC` dispatches, from the source tree (--cwd source). The node's Python is the
# pinned-vLLM GPU environment integrations/vllm/verity_vllm/ops/pod_bootstrap.sh makes ($WORK/venv312), or
# VERITY_BENCH_VENV, which the dispatcher forwards when it is set; this tree's packages go on PYTHONPATH.
set -euo pipefail
WORK=${WORK:-/workspace}
VENV=${VERITY_BENCH_VENV:-$WORK/venv312}
if [ ! -x "$VENV/bin/python" ]; then
  echo "run_point: no Python at $VENV: run integrations/vllm/verity_vllm/ops/pod_bootstrap.sh on this node, or dispatch" \
       "with VERITY_BENCH_VENV set to the node's pinned-vLLM environment" >&2
  exit 2
fi
export HF_HOME=${HF_HOME:-$WORK/hf}
if [ -z "${CUDA_HOME:-}" ]; then
  for c in /usr/local/cuda-13.0 /usr/local/cuda; do [ -d "$c" ] && { export CUDA_HOME=$c; break; }; done
fi
export PATH=${CUDA_HOME:+$CUDA_HOME/bin:}$VENV/bin:$PATH
ROOT=$PWD
export PYTHONPATH=$ROOT/benchmarks/harness:$ROOT/integrations/vllm:$ROOT/tools/config:$ROOT/tools/research/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}
exec "$VENV/bin/python" -m verity_bench _vllm-point "$@"

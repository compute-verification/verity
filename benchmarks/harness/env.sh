# Sourced by the harness's launchers (run_point.sh, benchmarks/<name>/run.sh) from the source tree: the node's pinned-vLLM
# Python as $VENV, the checkpoint store, CUDA, and this tree's packages on PYTHONPATH. $VENV is VERITY_BENCH_VENV, which
# the dispatcher forwards when it is set, else the first of the nodes' pinned-vLLM environments that exists: the one
# integrations/vllm/verity_vllm/ops/pod_bootstrap.sh makes ($WORK/venv312), a jobs node's environment, the PoUW
# pod's and the single-GPU VMs'. `benchmarks` is on the path so a spec names an experimental arm module
# from there (`served_commit.prof_arm:SERVED_COMMIT_ROWS`).
WORK=${WORK:-/workspace}
VENVS="$WORK/venv312 $WORK/jobs/venv312 $WORK/pouw/mvp-e2e/venv312 $WORK/cp/venv312"
if [ -n "${VERITY_BENCH_VENV:-}" ]; then
  VENV=$VERITY_BENCH_VENV
else
  VENV=$WORK/venv312
  for v in $VENVS; do [ -x "$v/bin/python" ] && { VENV=$v; break; }; done
fi
if [ ! -x "$VENV/bin/python" ]; then
  echo "verity-bench: no Python at ${VERITY_BENCH_VENV:-any of $VENVS}: run" \
       "integrations/vllm/verity_vllm/ops/pod_bootstrap.sh on this node, or dispatch with VERITY_BENCH_VENV set to the" \
       "node's pinned-vLLM environment" >&2
  exit 2
fi
export HF_HOME=${HF_HOME:-$WORK/hf}
if [ -z "${CUDA_HOME:-}" ]; then
  for c in /usr/local/cuda-13.0 /usr/local/cuda; do [ -d "$c" ] && { export CUDA_HOME=$c; break; }; done
fi
export PATH=${CUDA_HOME:+$CUDA_HOME/bin:}$VENV/bin:$PATH
ROOT=$PWD
export PYTHONPATH=$ROOT/benchmarks/harness:$ROOT/integrations/vllm:$ROOT/tools/config:$ROOT/tools/research/src:$ROOT:$ROOT/benchmarks${PYTHONPATH:+:$PYTHONPATH}

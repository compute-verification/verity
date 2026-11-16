# Sourced by the harness's launchers (run_point.sh, benchmarks/<name>/run.sh) from the source tree: the node's pinned-vLLM
# Python as $VENV, the checkpoint store, CUDA, and this tree's packages on PYTHONPATH. $VENV is VERITY_BENCH_VENV, which
# the dispatcher forwards when it is set, else the first of the nodes' pinned-vLLM environments that exists: the one
# adapters/vllm/verity_vllm/ops/pod_bootstrap.sh makes ($WORK/venv312), a jobs node's environment, the PoUW
# pod's and the single-GPU VMs'. `benchmarks` is on the path so a spec names an experimental arm module
# from there (`served_commit.prof_arm:SERVED_COMMIT_ROWS`), and so is every workspace member's import root.
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
       "adapters/vllm/verity_vllm/ops/pod_bootstrap.sh on this node, or dispatch with VERITY_BENCH_VENV set to the" \
       "node's pinned-vLLM environment" >&2
  exit 2
fi
export HF_HOME=${HF_HOME:-$WORK/hf}
if [ -z "${CUDA_HOME:-}" ]; then
  for c in /usr/local/cuda-13.0 /usr/local/cuda; do [ -d "$c" ] && { export CUDA_HOME=$c; break; }; done
fi
export PATH=${CUDA_HOME:+$CUDA_HOME/bin:}$VENV/bin:$PATH
# The in-process challenger builds its Lean coins with `lake`, which elan puts in ~/.elan/bin (tools/verity/lean/setup.sh).
if ! command -v lake >/dev/null && [ -x "${HOME:-}/.elan/bin/lake" ]; then export PATH=$HOME/.elan/bin:$PATH; fi
ROOT=$PWD
# Every workspace member's import root, the parent of each package its wheel names (`members` under the root
# pyproject.toml's [tool.uv.workspace], `packages` under a member's [tool.hatch.build.targets.wheel]), since a pinned-vLLM
# environment made without the workspace has none of them installed.
toml_list() {  # FILE SECTION KEY: the strings of a list of strings, on one line or several
  awk -v s="[$2]" -v k="$3" '/^\[/ { in_s = ($0 == s) } in_s && $0 ~ "^" k " *=" { on = 1; sub(/^[^=]*=/, "") }
    on { print; if (/\]/) on = 0 }' "$1" | tr -d '[]",'
}
ROOTS=
for m in $(toml_list "$ROOT/pyproject.toml" tool.uv.workspace members); do
  for p in $(toml_list "$ROOT/$m/pyproject.toml" tool.hatch.build.targets.wheel packages); do
    r=$ROOT/$m/$p
    r=${r%/*}
    case ":$ROOTS:" in *":$r:"*) ;; *) ROOTS=${ROOTS:+$ROOTS:}$r ;; esac
  done
done
if [ -z "$ROOTS" ]; then
  echo "verity-bench: no workspace member names a package in $ROOT/pyproject.toml's members (run from the source tree's root)" >&2
  exit 2
fi
export PYTHONPATH=$ROOTS:$ROOT:$ROOT/benchmarks${PYTHONPATH:+:$PYTHONPATH}

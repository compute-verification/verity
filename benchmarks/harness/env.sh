# Sourced by the harness's launchers (run_point.sh, benchmarks/<name>/run.sh) from the source tree: the node's pinned-vLLM
# Python as $VENV, CUDA, and this tree's packages on PYTHONPATH. $VENV is VERITY_BENCH_VENV, which the launch gives
# (`research run --env VERITY_BENCH_VENV=...`; the environment adapters/vllm/verity_vllm/ops/pod_bootstrap.sh makes), and
# the checkpoint store HF_HOME, where it gives one. CUDA_HOME, where the launch gives none, is the toolkit the run's
# machine's [properties] list first (benchmarks/kernels/machine.py toolkit). `benchmarks` is on the path so a spec
# names an experimental arm module from there (`served_commit.prof_arm:SERVED_COMMIT_ROWS`), and so is every workspace
# member's import root.
VENV=${VERITY_BENCH_VENV:-}
if [ -z "$VENV" ] || [ ! -x "$VENV/bin/python" ]; then
  echo "verity-bench: no Python at VERITY_BENCH_VENV (${VENV:-unset}): launch with VERITY_BENCH_VENV set to the node's" \
       "pinned-vLLM environment (adapters/vllm/verity_vllm/ops/pod_bootstrap.sh makes one)" >&2
  exit 2
fi
if [ -z "${CUDA_HOME:-}" ] && [ -n "${RESEARCH_MACHINE:-}" ]; then
  if machine_cuda=$("$VENV/bin/python" benchmarks/kernels/machine.py toolkit); then
    export CUDA_HOME=$machine_cuda
  else
    echo "verity-bench: no CUDA_HOME, so no toolkit on the PATH" >&2
  fi
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

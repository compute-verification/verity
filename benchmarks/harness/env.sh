# Sourced by the harness's launchers (run_point.sh, benchmarks/<name>/run.sh) from the source tree: the node's pinned-vLLM
# Python as $VENV, CUDA, and this tree's packages on PYTHONPATH. $VENV is VERITY_BENCH_VENV, which the launch gives
# (`research run --env VERITY_BENCH_VENV=...`), else the environment the launch already runs in, $VIRTUAL_ENV (`uv run`'s,
# or an activated one), else the one this user's bootstrap recorded (adapters/vllm/verity_vllm/ops/pod_bootstrap.sh writes
# its path in ${XDG_STATE_HOME:-$HOME/.local/state}/verity-bench/venv); and the checkpoint store HF_HOME, where it gives one. CUDA_HOME, where the launch gives none, is the toolkit the run's
# machine's [properties] list first (benchmarks/kernels/machine.py toolkit). `benchmarks` is on the path so a spec
# names an experimental arm module from there (`served_commit.prof_arm:SERVED_COMMIT_ROWS`), and so is every workspace
# member's import root.
VENV=${VERITY_BENCH_VENV:-${VIRTUAL_ENV:-}}
venv_record=${XDG_STATE_HOME:-${HOME:+$HOME/.local/state}}
venv_record=${venv_record:+$venv_record/verity-bench/venv}
if [ -z "$VENV" ] && [ -n "$venv_record" ] && [ -f "$venv_record" ]; then
  IFS= read -r VENV < "$venv_record" || VENV=
fi
if [ -z "$VENV" ] || [ ! -x "$VENV/bin/python" ]; then
  echo "verity-bench: no Python at VERITY_BENCH_VENV, else VIRTUAL_ENV, else the path recorded in" \
       "${venv_record:-\$XDG_STATE_HOME/verity-bench/venv} (${VENV:-none of them set}): bootstrap this machine" \
       "(adapters/vllm/verity_vllm/ops/pod_bootstrap.sh, which records its environment), or record one it already has" \
       "with \`bash adapters/vllm/verity_vllm/ops/pod_bootstrap.sh --record-venv PATH\`, or launch with VERITY_BENCH_VENV" >&2
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
# Every workspace member's import root, by research's reader of the workspace declaration (research.pythonpath, which
# `research run --source` sets PYTHONPATH from), since a pinned-vLLM environment made without the workspace has none of
# them installed.
ROOTS=$("$VENV/bin/python" -c 'import sys; root = sys.argv[1]; sys.path[:0] = [f"{root}/tools/research/src"]
from research.pythonpath import import_roots_dir
print(":".join(f"{root}/{r}" for r in import_roots_dir(root)["roots"]))' "$ROOT")
if [ -z "$ROOTS" ]; then
  echo "verity-bench: no workspace member names a package in $ROOT/pyproject.toml's members (run from the source tree's root)" >&2
  exit 2
fi
export PYTHONPATH=$ROOTS:$ROOT:$ROOT/benchmarks${PYTHONPATH:+:$PYTHONPATH}

# Sourced by steady_point.sh (a steady-state spec's `setup`), cwd the source tree: builds this tree's Flock prover for the
# node's cards and names it, with the Python that drives it, for a steady-state point whose rows name their statements
# (a row staged in the run needs what the stager needs besides).
#
# - The prover: verity/protocols/flock/pod/70-class-sweep.sh BUILD_ONLY=1 GPU=1, which keeps the binary under FLOCK_WORK by
#   a hash of the prover's sources and build scripts, so a later point of the same sources finds it built
#   (VERITY_BENCH_STEADY_BIN). Its build tree stays under FLOCK_WORK, as the build scripts leave it.
# - The Python: 3.12 with numpy and blake3, kept under FLOCK_WORK too (VERITY_BENCH_STEADY_PY), with the PYTHONPATH
#   benchmarks/one_stage/server_rate.py imports from.
# - The CUDA 13.3 toolkit the prover is built with, first on PATH, where the node side reads its version, and its lib64
#   on LD_LIBRARY_PATH: the installed one, else the one verity/protocols/flock/pod/cuda-13.3.sh unpacks. Its compat
#   library, which a driver older than the toolkit needs, is left out: a spec names its driver.
# - The scratch (VERITY_BENCH_STEADY_SCRATCH): a directory on the node's disk beside the builds, never the node's memory,
#   so a statement a row names in the store can be fetched into it. A point removes its own directory there at its end; one
#   a killed point left is removed here once it is a day old, far past any point's bound.
#
# FLOCK_WORK is /workspace/jobs/verity-bench-steady unless the run's environment names another. The logs go to
# $RESEARCH_RUN_DIR/out/setup. A step that fails ends the run with 4, as a refusal does: nothing was timed.
export FLOCK_WORK=${FLOCK_WORK:-/workspace/jobs/verity-bench-steady}
steady_out=${RESEARCH_RUN_DIR:-$PWD}/out/setup
mkdir -p "$steady_out" "$FLOCK_WORK/flock-circuit" "$FLOCK_WORK/scratch" \
  || { echo "setup: cannot make $FLOCK_WORK or $steady_out" >&2; exit 4; }
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH PYTHONDONTWRITEBYTECODE=1
unset CARGO_TARGET_DIR

# the unpacking runs in a subshell, which keeps this shell's arguments (the point's) from its `set --`; sourced again,
# the script only names the toolkit it finds
( . verity/protocols/flock/pod/cuda-13.3.sh ) > "$steady_out/cuda.log" 2>&1 \
  || { tail -20 "$steady_out/cuda.log" >&2; echo "setup: the node has no CUDA 13.3 toolkit and none could be unpacked (out/setup/cuda.log)" >&2; exit 4; }
. verity/protocols/flock/pod/cuda-13.3.sh >> "$steady_out/cuda.log" 2>&1
steady_cuda=${CUDA_TK:-/usr/local/cuda-13.3}
export PATH=$steady_cuda/bin:$PATH LD_LIBRARY_PATH=$steady_cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# the build scripts write their own out/ files: they get a run directory of their own under the setup's
RESEARCH_RUN_DIR=$steady_out/build OUT=$steady_out/build/sweep bash verity/protocols/flock/pod/70-class-sweep.sh BUILD_ONLY=1 GPU=1 \
  > "$steady_out/build.txt" 2>&1 \
  || { tail -30 "$steady_out/build.txt" >&2; echo "setup: the prover's build failed (out/setup/build.txt)" >&2; exit 4; }
VERITY_BENCH_STEADY_BIN=$(sed -n 's/^BINARY=//p' "$steady_out/build.txt" | tail -1)
[ -x "$VERITY_BENCH_STEADY_BIN" ] \
  || { echo "setup: the prover's build named no binary (out/setup/build.txt)" >&2; exit 4; }

VERITY_BENCH_STEADY_PY=$FLOCK_WORK/flock-circuit/py/bin/python3
if [ ! -x "$VERITY_BENCH_STEADY_PY" ]; then
  steady_uv=$(command -v uv || echo "$HOME/.local/bin/uv")
  # one venv at a time per FLOCK_WORK; the second point finds the first's
  ( exec 8> "$FLOCK_WORK/flock-circuit/venv.lock"; flock 8
    [ -x "$VERITY_BENCH_STEADY_PY" ] || { "$steady_uv" venv -q --python 3.12 "$FLOCK_WORK/flock-circuit/py" \
      && "$steady_uv" pip install -q --python "$VERITY_BENCH_STEADY_PY" numpy blake3; } ) > "$steady_out/venv.log" 2>&1 \
    || { tail -20 "$steady_out/venv.log" >&2; echo "setup: no Python 3.12 could be made under $FLOCK_WORK (out/setup/venv.log)" >&2; exit 4; }
fi
"$VERITY_BENCH_STEADY_PY" -c 'import sys, numpy, blake3; print("python", sys.version.split()[0], "numpy", numpy.__version__, "blake3", blake3.__version__)' \
  > "$steady_out/python.txt" 2>&1 \
  || { cat "$steady_out/python.txt" >&2; echo "setup: $VERITY_BENCH_STEADY_PY lacks numpy or blake3 (out/setup/python.txt)" >&2; exit 4; }

find "$FLOCK_WORK/scratch" -mindepth 1 -maxdepth 1 -name 'verity-bench-steady-*' -mmin +1440 -exec rm -rf {} + 2> /dev/null || true
export VERITY_BENCH_STEADY_BIN VERITY_BENCH_STEADY_PY VERITY_BENCH_STEADY_SCRATCH=$FLOCK_WORK/scratch
export PYTHONPATH=$PWD:$PWD/verity/protocols/flock/prover/python:$PWD/adapters/vllm:$PWD/tools/verity/circuit_check/src:$PWD/benchmarks/numerical/python

# Sourced by a harness run on a machine with an NVIDIA driver but no CUDA 12.8+ toolkit (a RunPod pod, a node-2 job):
#
#   . benchmarks/pouw/harness/jobs/env.sh BUILD && "$PY" benchmarks/pouw/harness/bench.py ...
#
# BUILD is a build.sh output directory (libpouw_harness.so and build.json), or art:<id> of one in the evidence store. The library
# is built elsewhere: build.sh needs nvcc, and NVIDIA's nvcc wheel ships only ptxas. This makes (once, under a lock) a venv with
# numpy and the cuBLASLt and CUDA runtime wheels of the versions build.sh links against, puts their libraries on
# LD_LIBRARY_PATH, and exports PY and POUW_HARNESS_LIB. It puts this tree's `verity` and `verity.core.protocols.pouw` sources (numpy and
# stdlib only) on PYTHONPATH, so an arm imports its scheme from the tree it ships in. env: POUW_HARNESS_ENV (default
# /workspace/cache/pouw-harness-env).

_ph_build="${1:?usage: . env.sh BUILD_DIR|art:<id>}"
_ph_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POUW_HARNESS_ENV="${POUW_HARNESS_ENV:-/workspace/cache/pouw-harness-env}"
_ph_wheels="numpy nvidia-cublas-cu12==12.9.2.10 nvidia-cuda-runtime-cu12==12.9.79"
mkdir -p "$POUW_HARNESS_ENV" || return 1
(
  flock 9
  [[ "$(cat "$POUW_HARNESS_ENV/ready" 2>/dev/null)" == "$_ph_wheels" ]] && exit 0
  rm -rf "$POUW_HARNESS_ENV/venv"
  if command -v uv >/dev/null; then
    uv venv -q "$POUW_HARNESS_ENV/venv" && VIRTUAL_ENV="$POUW_HARNESS_ENV/venv" uv pip install -q $_ph_wheels
  else
    python3 -m venv "$POUW_HARNESS_ENV/venv" && "$POUW_HARNESS_ENV/venv/bin/pip" install -q $_ph_wheels
  fi && echo "$_ph_wheels" > "$POUW_HARNESS_ENV/ready"
) 9>"$POUW_HARNESS_ENV/.lock" || return 1
PY="$POUW_HARNESS_ENV/venv/bin/python"
_ph_site="$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")" || return 1
export LD_LIBRARY_PATH="$_ph_site/nvidia/cublas/lib:$_ph_site/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
_ph_root="$(cd "$_ph_here/../../../.." && pwd)"
export PYTHONPATH="$_ph_root:$_ph_root/tools/config:$_ph_root/verity/core/protocols/pouw${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$_ph_build" == art:* ]]; then
  _ph_build="$(PYTHONPATH="$_ph_here/../../../../tools/research/src" python3 -m research data fetch "$_ph_build" \
    --to "$POUW_HARNESS_ENV/builds/${_ph_build#art:}" | tail -1)" || return 1
fi
[[ -f "$_ph_build/libpouw_harness.so" && -f "$_ph_build/build.json" ]] || { echo "no build in $_ph_build" >&2; return 1; }
export PY POUW_HARNESS_LIB="$_ph_build/libpouw_harness.so"

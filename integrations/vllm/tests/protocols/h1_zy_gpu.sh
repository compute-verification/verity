#!/usr/bin/env bash
# H1's tests and h1_zy_gpu.py on one leased GPU (cwd = the source tree's root):
#
#   research run --on MACHINE --project verity --campaign hash --source . --cwd source --timeout 60m \
#     --env GPU_LEASE_WHO=kernels -- bash integrations/vllm/tests/protocols/h1_zy_gpu.sh
#
# Outside the lease: the workspace synced with the members' CUDA torch (`uv sync --all-packages --extra torch`). Under one
# lease of at most LEASE_MIN minutes (default 20): the H1 tests (TESTS, default tests/protocols/test_pouw_h1*.py) with
# the GPU, into $OUT/tests.log and $OUT/tests.xml, then h1_zy_gpu.py (with H1_ZY_ARGS) into $OUT/h1_zy_gpu.json. Each step's exit code is in
# $OUT/rc.txt, and the script exits non-zero if any failed. CUDA: the toolkit (default the first of /usr/local/cuda-13.0
# and /usr/local/cuda).
set -uo pipefail
export O=${OUT:?} V=integrations/vllm
for c in "${CUDA:-}" /usr/local/cuda-13.0 /usr/local/cuda; do [ -n "$c" ] && [ -x "$c/bin/nvcc" ] && { CUDA=$c; break; }; done
export CUDA PATH=$CUDA/bin:$HOME/.local/bin:$PATH LD_LIBRARY_PATH=$CUDA/lib64:${LD_LIBRARY_PATH:-}
: > "$O/rc.txt"
step() { echo "$1 rc $2 at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"; }
git rev-parse HEAD > "$O/commit.txt" 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}" > "$O/commit.txt"
nvcc --version > "$O/nvcc.txt" 2>&1
uv sync -q --all-packages --extra torch > "$O/sync.log" 2>&1
step sync $?
export PY=$PWD/.venv/bin/python TESTS=${TESTS:-tests/protocols/test_pouw_h1*.py} H1_ZY_ARGS=${H1_ZY_ARGS:-}
"$PY" -c 'import torch; print(torch.__version__, torch.version.cuda)' > "$O/python.txt" 2>&1
grep -qE "^sync rc [1-9]" "$O/rc.txt" && { cat "$O/rc.txt"; tail -20 "$O/sync.log"; exit 2; }

echo "h1_zy_gpu: asking for the GPU at $(date -u +%H:%M:%SZ)"
# shellcheck disable=SC2016
env -u GPU_LEASE_TIMED gpu-lease 1 --wait --max-min "${LEASE_MIN:-20}" -- bash -c '
  echo "leased at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=name,driver_version,clocks.max.sm --format=csv > "$O/gpu.txt" 2>&1
  # shellcheck disable=SC2086
  (cd "$V" && "$PY" -m pytest -rs --junitxml "$O/tests.xml" $TESTS) > "$O/tests.log" 2>&1
  echo "tests rc $? at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"
  # shellcheck disable=SC2086
  (cd "$V" && "$PY" tests/protocols/h1_zy_gpu.py --out "$O" $H1_ZY_ARGS) > "$O/h1_zy_gpu.log" 2>&1
  echo "time rc $? at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"'
step lease $?
cat "$O/rc.txt"; tail -3 "$O/tests.log" 2>/dev/null; tail -8 "$O/h1_zy_gpu.log" 2>/dev/null
! grep -qE " rc [1-9]" "$O/rc.txt"

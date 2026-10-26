#!/usr/bin/env bash
# check.py on one leased GPU of vy-dc-h200 (cwd = the source tree's root): the weights on the CPU before the lease, the
# linear and its timings inside one lease of at most 20 minutes, the twin and the Definition after it. The weights and the
# sampled rows stay in scratch (their sha256 are in $OUT/gpu.json and $OUT/weights.json; check.py rebuilds them).
#
#   research run --on vy-dc-h200 --project verity --campaign e2e-demo --cwd source --env GPU_LEASE_WHO=kernels-ks8simple \
#     [--send weights.py] -- bash benchmarks/torch_gpu/s8_simple/check.sh [--sets ...] [--shapes ...] [--ms ...]
#
# `--send` ships compute-accounting's `verity_toy/weights.py` for the served set while the tree lacks it. Exit: 0
# every checked word is the Definition's, 1 a mismatch, 4 the weights failed, else the GPU phase's code.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run under research run}; OUT=$D/out; mkdir -p "$OUT"
HERE=benchmarks/torch_gpu/s8_simple
TMP=/workspace/pouw; mkdir -p "$TMP" 2>/dev/null && [ -w "$TMP" ] || TMP=${TMPDIR:-/tmp}
S=$(mktemp -d "$TMP/tmp-s8simple.XXXXXX"); trap 'rm -rf "$S"' EXIT
for v in /workspace/jobs/venv312 /workspace/cp/venv312; do [ -z "${PY:-}" ] && [ -x "$v/bin/python3" ] && PY=$v/bin/python3; done
export PYTHONPATH=$PWD/examples/toy:$PWD PYTHONDONTWRITEBYTECODE=1
WM=(); [ -f "$D/inputs/weights.py" ] && WM=(--weights-module "$D/inputs/weights.py")
git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}" > "$OUT/commit.txt"
"$PY" "$HERE/check.py" weights --scratch "$S" "${WM[@]}" "$@" || { echo "weights 4" | tee "$OUT/status.txt"; exit 4; }
cp "$S/weights.json" "$OUT/"
echo "check.sh: asking for one GPU at $(date -u +%H:%M:%SZ)"
export S OUT HERE PY
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 20 -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm --format=csv -i "$GPU_LEASE_UUID" > "$OUT/gpu.txt" 2>&1
  exec "$PY" "$HERE/check.py" gpu --scratch "$S" --out "$OUT" "$@"' _ "$@"
rc=$?
echo "check.sh: lease ended ($rc) at $(date -u +%H:%M:%SZ)"
[ $rc = 0 ] || { echo "gpu $rc" | tee "$OUT/status.txt"; exit $rc; }
"$PY" "$HERE/check.py" verify --scratch "$S" --out "$OUT"
rc=$?
echo "verify $rc" | tee "$OUT/status.txt"
exit $rc

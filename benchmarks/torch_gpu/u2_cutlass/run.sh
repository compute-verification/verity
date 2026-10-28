#!/usr/bin/env bash
# bench.py on one leased GPU from build.sh's ship, sent with the run as u2-ship.tar (nothing is compiled here). The ship is
# unpacked into a scratch directory inside the run directory, held to its MANIFEST.sha256 and deleted at exit, so the run
# writes nothing on the node outside its own directory. Arguments go to bench.py; U2_MAX_MIN (default 20) bounds the lease.
# VERITY_BENCH_VENV names a Python with torch (default the node's venv312).
set -uo pipefail
D=${RESEARCH_RUN_DIR:?}; OUT=$D/out; mkdir -p "$OUT"
SHIP=${U2_SHIP:-$D/inputs/u2-ship.tar}
S=$D/scratch.$$; mkdir -p "$S"; trap 'rm -rf "$S"' EXIT
ROOT=$PWD
PY=${VERITY_BENCH_VENV:+$VERITY_BENCH_VENV/bin/python3}
for v in /workspace/cp/venv312 /workspace/cache/kernels-j10r/venv312 /workspace/jobs/venv312; do
  [ -z "${PY:-}" ] && [ -x "$v/bin/python3" ] && PY=$v/bin/python3
done
[ -x "${PY:-}" ] || { echo "u2_cutlass: no Python with torch (VERITY_BENCH_VENV)" >&2; exit 2; }
git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}" > "$OUT/commit.txt"
mkdir -p "$S/ship" && tar -C "$S/ship" -xf "$SHIP" && (cd "$S/ship" && sha256sum -c --quiet MANIFEST.sha256) \
  || { echo "u2_cutlass: the ship $SHIP is missing or fails its MANIFEST" >&2; exit 2; }
cp "$S/ship/build.json" "$S/ship/MANIFEST.sha256" "$S/ship/cutlass.patch" "$OUT/"
"$PY" -c 'import torch; print(torch.__version__, torch.version.cuda)' > "$OUT/python.txt" 2>&1
nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.max.sm,power.limit,enforced.power.limit --format=csv \
  > "$OUT/gpus.txt" 2>&1
echo "u2_cutlass: asking for one GPU at $(date -u +%H:%M:%SZ)"
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${U2_MAX_MIN:-20}" -- bash -c '
  sel=${GPU_LEASE_UUID:+-i $GPU_LEASE_UUID}
  nvidia-smi $sel -q -d CLOCK,PERFORMANCE,POWER > "$0/clocks-before.txt" 2>&1
  timeout $((60 * ${U2_MAX_MIN:-20} - 45)) "$1" "$2" --lib "$3" --out "$0/bench" --gpu "${GPU_LEASE_UUID:-}" "${@:4}"
  rc=$?
  nvidia-smi $sel -q -d CLOCK,PERFORMANCE,POWER > "$0/clocks-after.txt" 2>&1
  exit $rc' "$OUT" "$PY" "$ROOT/benchmarks/torch_gpu/u2_cutlass/bench.py" "$S/ship/libu2gemm.so" "$@" 2>&1 \
  | tee "$OUT/bench.log"
rc=${PIPESTATUS[0]}
echo "u2_cutlass: bench exited $rc at $(date -u +%H:%M:%SZ)" | tee -a "$OUT/rc.txt"
exit "$rc"

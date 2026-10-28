#!/usr/bin/env bash
# bench.py on one leased GPU from build.sh's ship, sent with the run as u2-ship.tar (nothing is compiled here), or read
# from U2_SHIP_DIR, a ship directory a build left on this node, which is read once the lease is held, so a run can queue
# for the GPU while its build finishes. The ship is copied into a scratch directory inside the run directory, held to its
# MANIFEST.sha256 and deleted at exit, so the run writes nothing on the node outside its own directory. Arguments go to
# bench.py; U2_MAX_MIN (default 20) bounds the lease. VERITY_BENCH_VENV names a Python with torch (default the node's
# venv312).
set -uo pipefail
D=${RESEARCH_RUN_DIR:?}; OUT=$D/out; mkdir -p "$OUT"
SHIP=${U2_SHIP:-$D/inputs/u2-ship.tar}
S=$D/scratch.$$; mkdir -p "$S"; trap 'rm -rf "$S"' EXIT
ROOT=$PWD
PY=${VERITY_BENCH_VENV:+$VERITY_BENCH_VENV/bin/python3}
for v in /workspace/cp/venv312 /workspace/cache/kernels-j10r/venv312 /workspace/jobs/venv312; do
  [ -z "${PY:-}" ] && [ -x "$v/bin/python3" ] && PY=$v/bin/python3
done
if [ -z "${PY:-}" ] && command -v uv > /dev/null; then
  # no venv on the node: torch through uv (U2_TORCH, U2_TORCH_INDEX), fetched here, before the lease
  PY=$S/python
  printf '#!/usr/bin/env bash\nexec uv run -q --no-project --with "torch==%s" --with numpy --index-url %s python "$@"\n' \
    "${U2_TORCH:-2.9.*}" "${U2_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}" > "$PY"
  chmod +x "$PY"
fi
[ -x "${PY:-}" ] || { echo "u2_cutlass: no Python with torch (VERITY_BENCH_VENV)" >&2; exit 2; }
git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}" > "$OUT/commit.txt"
ship() {
  rm -rf "$S/ship" && mkdir -p "$S/ship" || return 1
  if [ -n "${U2_SHIP_DIR:-}" ]; then
    [ -f "$U2_SHIP_DIR/MANIFEST.sha256" ] && cp -r "$U2_SHIP_DIR/." "$S/ship/" || return 1
  else
    tar -C "$S/ship" -xf "$SHIP" || return 1
  fi
  (cd "$S/ship" && sha256sum -c --quiet MANIFEST.sha256) && cp "$S/ship/build.json" "$S/ship/MANIFEST.sha256" \
    "$S/ship/cutlass.patch" "$OUT/"
}
export -f ship; export S SHIP OUT U2_SHIP_DIR
if [ -z "${U2_SHIP_DIR:-}" ]; then
  ship || { echo "u2_cutlass: the ship $SHIP is missing or fails its MANIFEST" >&2; exit 2; }
fi
"$PY" -c 'import torch; print(torch.__version__, torch.version.cuda)' > "$OUT/python.txt" 2>&1 \
  || { cat "$OUT/python.txt" >&2; echo "u2_cutlass: $PY has no torch" >&2; exit 2; }
nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.max.sm,power.limit,enforced.power.limit --format=csv \
  > "$OUT/gpus.txt" 2>&1
echo "u2_cutlass: asking for one GPU at $(date -u +%H:%M:%SZ)"
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${U2_MAX_MIN:-20}" -- bash -c '
  [ -f "$S/ship/libu2gemm.so" ] || ship || { echo "u2_cutlass: no ship at ${U2_SHIP_DIR:-$SHIP}" >&2; exit 2; }
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

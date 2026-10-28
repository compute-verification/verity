#!/usr/bin/env bash
# s8_mma on one datacenter GPU (cwd = the source tree's root): the build, SASS dump and case files outside the lease, the
# forms inside one short lease, the check against verity.ml.circuits.silicon.integer after it. The bulk words stay in
# scratch (their sha256 is kept; s8_mma.py gen rebuilds the cases from the seed).
#
#   research run --on vy-dc-h200 --project verity --campaign pouw --source <tree> --cwd source --custody-ttl 12h \
#     --env GPU_LEASE_WHO=LANE -- bash benchmarks/pouw/dc/s8_mma.sh sm90 [FAMILY ...]      (sm100 on vy-dc-b200)
#
# sm90 runs families mma (mma.sync m16n8k32), wg (wgmma m64n8k32) and wg64, wg128, wg256 (wgmma m64nNk32); sm100 runs mma
# and tc (tcgen05.mma kind::i8); FAMILY ... runs those alone. `s8_mma.sh build sm90|sm100` on a CPU node only compiles:
# out/s8_mma_<arch>, its .nvcc sidecar (nvcc --version) and its SASS; S8_MMA_BIN=<that binary, shipped with its sidecar>
# then runs it instead of compiling here. Exit: 0 every form of every family matches the sm_120 model with one tensor
# instruction a kernel, 1 a mismatch, 2 wrong device or no nvcc, 4 build or case generation failed, else the GPU part's code.
set -uo pipefail
MODE=run
[ "${1:-}" = build ] && { MODE=build; shift; }
FAM=${1:?usage: s8_mma.sh [build] sm90|sm100 [FAMILY ...]}
shift
export O=${RESEARCH_RUN_DIR:?run under research run}/out
mkdir -p "$O"
HERE=benchmarks/pouw/dc
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || { echo "s8_mma.sh: no nvcc at $NVCC (cuobjdump is beside it)" >&2; exit 2; }
export PATH="$(dirname "$NVCC"):$PATH" PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
PY=${PY:-python3}
case "$FAM" in
  sm90) ARCH=90a; CC=9; FAMS=${*:-mma wg wg64 wg128 wg256} ;;
  sm100) ARCH=100a; CC=10; FAMS=${*:-mma tc} ;;
  *) echo "s8_mma.sh: sm90 or sm100, not $FAM" >&2; exit 2 ;;
esac
build() {  # $1: the binary; writes $1.nvcc beside it
  "$NVCC" --version | tail -2 > "$1.nvcc"
  "$NVCC" -O2 -std=c++17 -gencode "arch=compute_$ARCH,code=sm_$ARCH" -Xptxas -v -o "$1" "$HERE/probes/s8_mma.cu" \
    > "$O/ptxas.txt" 2>&1 || { tail -40 "$O/ptxas.txt"; echo "build 4" | tee "$O/status.txt"; exit 4; }
}
sass() {
  cuobjdump -sass "$1" > "$O/sass.txt"
  grep -E 'Function|MMA|IDP' "$O/sass.txt" > "$O/sass_mma.txt"
}
if [ "$MODE" = build ]; then
  build "$O/s8_mma_$ARCH"
  sass "$O/s8_mma_$ARCH"
  echo "build 0" | tee "$O/status.txt"
  exit 0
fi
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
[ "${cap%%.*}" = "$CC" ] || { echo "s8_mma.sh: $FAM needs compute capability $CC.x, the device reports '$cap'" >&2; exit 2; }
export B; B=$(mktemp -d /tmp/s8-mma.XXXXXX); trap 'rm -rf "$B"' EXIT
git rev-parse HEAD > "$O/tree.txt" 2>/dev/null || true
if [ -n "${S8_MMA_BIN:-}" ]; then
  case "$S8_MMA_BIN" in /*) ;; *) S8_MMA_BIN=$RESEARCH_RUN_DIR/$S8_MMA_BIN ;; esac  # --send puts it in the run dir's inputs/
  cp "$S8_MMA_BIN" "$B/s8_mma" && cp "$S8_MMA_BIN.nvcc" "$B/s8_mma.nvcc" \
    || { echo "s8_mma.sh: no $S8_MMA_BIN or its .nvcc sidecar" | tee "$O/ptxas.txt"; echo "build 4" | tee "$O/status.txt"; exit 4; }
  echo "prebuilt $S8_MMA_BIN" > "$O/ptxas.txt"
else
  build "$B/s8_mma"
fi
cp "$B/s8_mma.nvcc" "$O/nvcc.txt"
sass "$B/s8_mma"
for f in $FAMS; do
  mkdir -p "$B/$f"
  "$PY" "$HERE/s8_mma.py" gen --family "$f" --out "$B/$f/cases.bin" --meta "$O/meta-$f.json" > "$O/gen-$f.txt" 2>&1 \
    || { cat "$O/gen-$f.txt"; echo "gen 4" | tee "$O/status.txt"; exit 4; }
done
LEASE=(gpu-lease 1 --wait --max-min 15 --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
export FAMS
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  for f in $FAMS; do "$B/s8_mma" "$f" "$B/$f/cases.bin" "$B/$f" > "$O/gpu-$f.txt" 2>&1 || exit $?; done'
rc=$?
[ $rc = 0 ] || { echo "gpu part failed ($rc)"; cat "$O"/gpu-*.txt; echo "gpu $rc" | tee "$O/status.txt"; exit $rc; }
rc=0
for f in $FAMS; do
  cp "$B/$f/device.txt" "$O/device-$f.txt"
  (cd "$B" && sha256sum s8_mma "$f"/cases.bin "$f"/d_*.bin) >> "$O/inputs.sha256"
  "$PY" "$HERE/s8_mma.py" check --family "$f" --cases "$B/$f/cases.bin" --meta "$O/meta-$f.json" --words "$B/$f" \
    --sass "$O/sass.txt" --json "$O/check-$f.json" --structured "$O/structured-$f.json" > "$O/check-$f.txt" 2>&1 || rc=1
  tail -20 "$O/check-$f.txt"
done
sha256sum "$HERE/probes/s8_mma.cu" "$HERE/s8_mma.py" "$HERE/dc.cuh" "$HERE/wgmma.cuh" >> "$O/inputs.sha256"
echo "check $rc" | tee "$O/status.txt"
exit $rc

#!/usr/bin/env bash
# s8_mma on one datacenter GPU (cwd = the source tree's root): the build, SASS dump and case files outside the lease, the
# forms inside one short lease, the check against verity.circuits.silicon.integer after it. The bulk words stay in
# scratch (their sha256 is kept; s8_mma.py gen rebuilds the cases from the seed).
#
#   research run --on H200 --project verity --campaign pouw --source <tree> --cwd source --custody-ttl 12h \
#     --env GPU_LEASE_WHO=LANE -- bash calibration/pouw/dc/s8_mma.sh [FAMILY ...]      (or on a B200)
#
# The card is the one the run opened: its card record's `arch` is the target (probe.py arch) and its architecture file
# nvcc's -gencode (probe.py gencode). sm_90a runs families mma (mma.sync m16n8k32), wg (wgmma m64n8k32) and wg64, wg128,
# wg256 (wgmma m64nNk32); sm_100a runs mma and tc (tcgen05.mma kind::i8); another target is refused before anything is
# built. FAMILY ... runs those alone. `s8_mma.sh build SKU` on a CPU node only compiles, for the card file
# calibration/hardware/cards/SKU.toml: out/s8_mma_<arch>, its .nvcc sidecar (nvcc --version) and its SASS;
# S8_MMA_BIN=<that binary, shipped with its sidecar> then runs it instead of compiling here. Exit: 0 every form of every
# family matches the sm_120 model with one tensor instruction a kernel, 1 a mismatch, 2 wrong device or no nvcc, 4 build
# or case generation failed, else the GPU part's code.
set -uo pipefail
MODE=run
CARD=()
if [ "${1:-}" = build ]; then
  MODE=build
  CARD=(--sku "${2:?usage: s8_mma.sh build SKU, the card file to build for (calibration/hardware/cards/SKU.toml)}")
  shift 2
fi
export O=${RESEARCH_RUN_DIR:?run under research run}/out
mkdir -p "$O"
HERE=calibration/pouw/dc
. "$HERE/../nvcc.sh"
export PATH="$(dirname "$NVCC"):$PATH" PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
PY=${PY:-python3}
ARCH=$("$PY" "$HERE/../opcode_probe/probe.py" arch "${CARD[@]}") || exit 2
FORMS=$("$PY" "$HERE/archs.py" "$ARCH" s8_forms) || exit 2
FAMS=${*:-$(echo $FORMS)}
GENCODE=$("$PY" "$HERE/../opcode_probe/probe.py" gencode "${CARD[@]}") || exit 2
build() {  # $1: the binary; writes $1.nvcc beside it
  "$NVCC" --version | tail -2 > "$1.nvcc"
  "$NVCC" -O2 -std=c++17 -gencode "$GENCODE" -Xptxas -v -o "$1" "$HERE/probes/s8_mma.cu" \
    > "$O/ptxas.txt" 2>&1 || { tail -40 "$O/ptxas.txt"; echo "build 4" | tee "$O/status.txt"; exit 4; }
}
sass() {
  cuobjdump -sass "$1" > "$O/sass.txt"
  grep -E 'Function|MMA|IDP' "$O/sass.txt" > "$O/sass_mma.txt"
}
if [ "$MODE" = build ]; then
  build "$O/s8_mma_${ARCH#sm_}"
  sass "$O/s8_mma_${ARCH#sm_}"
  echo "build 0" | tee "$O/status.txt"
  exit 0
fi
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

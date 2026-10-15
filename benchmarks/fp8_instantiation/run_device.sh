#!/usr/bin/env bash
# chain_probe on one leased sm_120 GPU (cwd = the clone's root): build and the SASS check outside the lease, the chains
# inside it, the integer families' check after it.  The operand files (exactness.py ops) arrive with --send.
#
#   research run --on vy-nebius-2 --project verity --campaign pouw --source . --cwd clone --env GPU_LEASE_WHO=... \
#     --send benchmarks/fp8_instantiation/run_device.sh --send OPS/ops0_pearlc.bin --send OPS/ops1_grids.bin \
#     --send OPS/ops_meta.json -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/run_device.sh"'
set -uo pipefail
export O=${RESEARCH_RUN_DIR:?}/out IN=$RESEARCH_RUN_DIR/inputs
mkdir -p "$O/words"
export PATH=/usr/local/cuda-13.0/bin:$PATH
SRC=benchmarks/fp8_instantiation
export B; B=$(mktemp -d /tmp/fp8-chain.XXXXXX); trap 'rm -rf "$B"' EXIT
nvcc --version | tail -2 > "$O/nvcc.txt"
nvcc -O2 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$B/chain_probe" $SRC/chain_probe.cu \
  > "$O/ptxas.txt" 2>&1 || { tail -40 "$O/ptxas.txt"; exit 4; }
cuobjdump -sass "$B/chain_probe" > "$B/sass.txt"
{
  echo "QMMA.16832.F32.E4M3.E4M3: $(grep -c 'QMMA.16832.F32.E4M3.E4M3' "$B/sass.txt")"
  echo "FADD: $(grep -c ' FADD' "$B/sass.txt")"
  echo "FFMA: $(grep -c 'FFMA' "$B/sass.txt")"
  echo "FTZ: $(grep -c 'FTZ' "$B/sass.txt")"
  grep -m2 'QMMA' "$B/sass.txt"
} > "$O/sass_gate.txt"
grep -q 'QMMA.16832.F32.E4M3.E4M3' "$B/sass.txt" || { echo "no QMMA in the binary"; exit 5; }
! grep -q 'FTZ' "$B/sass.txt" || { echo "FTZ in the binary"; exit 5; }
sha256sum "$B/chain_probe" $SRC/chain_probe.cu "$IN"/ops*.bin > "$O/inputs.sha256"
cp "$IN"/ops_meta.json "$O/"
LEASE=(gpu-lease 1 --wait --max-min 30 --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  "$B/chain_probe" gpu "$O/words" --ops "$IN/ops0_pearlc.bin" --ops "$IN/ops1_grids.bin" > "$O/gpu_stdout.txt" 2>&1'
rc=$?
[ $rc = 0 ] || { echo "gpu part failed ($rc)"; cat "$O/gpu_stdout.txt"; exit $rc; }
"$B/chain_probe" check "$O/words" > "$O/check.json"
echo "check exit $?" > "$O/check_status.txt"
cat "$O/check_status.txt" "$O/sass_gate.txt"

#!/usr/bin/env bash
# imma_probe on one leased sm_120 GPU of vy-nebius-2 (cwd = the clone's root): the build, SASS dump and case file outside
# the lease, the eight integer mma.sync forms inside it, the check against verity.ml.circuits.silicon.integer after it.
# The bulk words stay in scratch (their sha256 is kept; imma_probe.py gen rebuilds the cases from the seed).
#
#   research run --on vy-nebius-2 --project verity --campaign pouw --source . --cwd clone --custody-ttl 12h \
#     --env GPU_LEASE_WHO=... --send benchmarks/fp8_instantiation/run_imma_probe.sh -- \
#     bash -c 'bash "$RESEARCH_RUN_DIR/inputs/run_imma_probe.sh"'
set -uo pipefail
export O=${RESEARCH_RUN_DIR:?}/out
mkdir -p "$O"
export PATH=/usr/local/cuda-13.0/bin:$PATH PYTHONPATH=$PWD OMP_NUM_THREADS=8
PY=/workspace/jobs/venv312/bin/python
SRC=benchmarks/fp8_instantiation
export B; B=$(mktemp -d /tmp/imma-probe.XXXXXX); trap 'rm -rf "$B"' EXIT
nvcc --version | tail -2 > "$O/nvcc.txt"
nvcc -O2 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$B/imma_probe" $SRC/imma_probe.cu \
  > "$O/ptxas.txt" 2>&1 || { tail -40 "$O/ptxas.txt"; echo "build 4" | tee "$O/status.txt"; exit 4; }
cuobjdump -sass "$B/imma_probe" > "$O/sass.txt"
grep -E 'Function|MMA|IDP' "$O/sass.txt" > "$O/sass_mma.txt"
"$PY" $SRC/imma_probe.py gen --out "$B/cases.bin" --meta "$O/meta.json" > "$O/gen.txt" 2>&1 || { cat "$O/gen.txt"; exit 4; }
mkdir -p "$B/words"
LEASE=(gpu-lease 1 --wait --max-min 30 --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  "$B/imma_probe" gpu "$B/cases.bin" "$B/words" > "$O/gpu_stdout.txt" 2>&1'
rc=$?
[ $rc = 0 ] || { echo "gpu part failed ($rc)"; cat "$O/gpu_stdout.txt"; echo "gpu $rc" | tee "$O/status.txt"; exit $rc; }
cp "$B/words/device.txt" "$O/"
(cd "$B" && sha256sum imma_probe cases.bin words/d_*.bin) > "$O/inputs.sha256"
sha256sum $SRC/imma_probe.cu $SRC/imma_probe.py >> "$O/inputs.sha256"
"$PY" $SRC/imma_probe.py check --cases "$B/cases.bin" --meta "$O/meta.json" --words "$B/words" --sass "$O/sass.txt" \
  --json "$O/check.json" --structured "$O/structured.json" > "$O/check.txt" 2>&1
rc=$?
echo "check $rc" | tee "$O/status.txt"
cat "$O/check.txt" | tail -20
exit $rc

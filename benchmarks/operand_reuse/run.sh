#!/usr/bin/env bash
# Operand reuse on sm_120 (cwd = a clone of the source tree): builds probe.cu for sm_120a, writes its SASS, PTX and
# sass_check.py's per-instruction producer table, compiles tcgen05_tmem_a.cu for sm_100a (SASS only, never run), then
# runs the probe under one gpu-lease 1 --wait --max-min 15.  Everything kept goes to $OUT.
#
#   research run --on vy-nebius-2 --project verity --campaign pouw --source . --cwd clone \
#     --env GPU_LEASE_WHO=... -- bash benchmarks/operand_reuse/run.sh
set -uxo pipefail
: "${OUT:?}"
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/operand_reuse
B=$(mktemp -d /tmp/operand-reuse.XXXXXX); trap 'rm -rf "$B"' EXIT
nvcc --version > "$OUT/nvcc_version.txt" 2>&1
nvcc -O2 -std=c++17 -gencode=arch=compute_120a,code=sm_120a -Xptxas -v -o "$B/probe" $D/probe.cu > "$OUT/probe_build.txt" 2>&1 \
  || { tail -40 "$OUT/probe_build.txt"; exit 4; }
cuobjdump -sass "$B/probe" > "$OUT/probe_sm120a.sass"
nvcc -O2 -std=c++17 -arch=compute_120a -ptx -o "$OUT/probe_sm120a.ptx" $D/probe.cu
python3 $D/sass_check.py "$OUT/probe_sm120a.sass" > "$OUT/sass_check.json"
nvcc -O2 -std=c++17 -arch=sm_100a -cubin -o "$B/tcgen05.cubin" $D/tcgen05_tmem_a.cu > "$OUT/tcgen05_build.txt" 2>&1 \
  && cuobjdump -sass "$B/tcgen05.cubin" > "$OUT/tcgen05_sm100a.sass" \
  && python3 $D/sass_check.py "$OUT/tcgen05_sm100a.sass" > "$OUT/tcgen05_sass_check.json"
echo "tcgen05 build rc=$?" >> "$OUT/tcgen05_build.txt"
sha256sum "$B/probe" > "$OUT/probe.sha256"
gpu-lease 1 --wait --max-min 15 -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm --format=csv,noheader -i "$GPU_LEASE_UUID" > "$OUT/gpu.txt" 2>&1
  "$0"/probe > "$OUT/probe_run.json" 2> "$OUT/probe_run.err"' "$B"

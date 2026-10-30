#!/usr/bin/env bash
# P2's device rates and the INT8 linear over the P2 store on one H200, from build.sh's ship (cwd = the source tree's
# root):
#
#   tar -C SHIP -cf IN/ship.tar .
#   OUT=DIR IN=IN [STEPS=bench,scale] bash benchmarks/torch_gpu/s8_dec_linear/h200.sh [RUN_ARGS...]
#
# Needs CUDA torch with Triton and `gpu-lease` on PATH; IN defaults to a run directory's inputs/ (RESEARCH_RUN_DIR).
# Outside the lease: the ship unpacked and held to its MANIFEST.sha256, the card, the toolkit and the Python recorded. Under one
# timed lease of at most LEASE_MIN minutes (default 25), the STEPS (default bench,scale,run) in order, memory-accounting's
# tools unchanged on the shipped libp2v1.so (v4 keys, sm_90) first:
#   bench  benchmarks/pous/p2_v1/decode_bench.py: P2's decode rate, the keys alone, the kernel alone and the full pass, gated
#          against reference.py (decode-bench.json)
#   scale  benchmarks/pous/p2_v1/scale.py gpu: the device encode rate at SCALE_BATCHES blocks (bench.py's encode is the 2,048
#          one), each batch decoded back and sampled blocks held to reference.decode, and the decode rate (scale/gpu.json)
#   run    run.py with RUN_ARGS: the staging gates, the linear against its twin, and the overlap's co-run timing (run.jsonl)
# VERITY_BENCH_VENV names the Python with torch and Triton (default /workspace/cp/venv312), CUDA the toolkit. Every step's exit
# code is in $OUT/rc.txt; exits non-zero if any step failed.
set -uo pipefail
O=${OUT:?}; IN=${IN:-${RESEARCH_RUN_DIR:?set IN}/inputs}
D=benchmarks/torch_gpu/s8_dec_linear
PY=${VERITY_BENCH_VENV:-/workspace/cp/venv312}/bin/python
CUDA=${CUDA:-/usr/local/cuda-13.0}
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
STEPS=${STEPS:-bench,scale,run}; SCALE_BATCHES=${SCALE_BATCHES:-1,1024,2048,4096,16384,65536}
export O D PY S STEPS SCALE_BATCHES PATH=$CUDA/bin:$PATH TRITON_CACHE_DIR=$S/triton \
  PYTHONPATH=$PWD:$PWD/integrations/vllm${PYTHONPATH:+:$PYTHONPATH}
: > "$O/rc.txt"
mkdir -p "$S/ship" && tar -C "$S/ship" -xf "$IN/ship.tar" && (cd "$S/ship" && sha256sum -c --quiet MANIFEST.sha256)
echo "ship rc $?" >> "$O/rc.txt"
grep -q "ship rc 0" "$O/rc.txt" || { cat "$O/rc.txt"; exit 2; }
cp "$S/ship/source.txt" "$S/ship/toolchain.txt" "$S/ship/MANIFEST.sha256" "$O/"
for d in "$S"/ship/lib/*/; do cp "$d/sass.json" "$O/sass-$(basename "$d").json"; done
git rev-parse HEAD > "$O/commit.txt" 2>&1
nvcc --version > "$O/nvcc.txt" 2>&1
nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.max.sm,power.limit --format=csv > "$O/gpus.txt" 2>&1
"$PY" -c 'import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__)' > "$O/python.txt" 2>&1
echo "python rc $?" >> "$O/rc.txt"
nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv -l 1 \
  > "$O/smi.csv" 2>/dev/null &
SMI=$!
echo "s8_dec_linear: steps $STEPS, asking for the GPU at $(date -u +%H:%M:%SZ)"
# shellcheck disable=SC2016
GPU_LEASE_TIMED=1 gpu-lease 1 --wait --max-min "${LEASE_MIN:-25}" -- bash -c '
  echo "leased at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"
  nvidia-smi -i "$GPU_LEASE_UUID" -q -d CLOCK,PERFORMANCE,POWER > "$O/clocks.txt" 2>&1
  for step in ${STEPS//,/ }; do
    case $step in
      bench) "$PY" benchmarks/pous/p2_v1/decode_bench.py --lib "$S/ship/p2v1" --build-dir "$S/jit" --blocks $((8 << 16)) \
               --reps 10 --out "$O/decode-bench.json" > "$O/decode-bench.log" 2>&1 ;;
      scale) "$PY" benchmarks/pous/p2_v1/scale.py gpu --out "$O/scale" --p2-lib "$S/ship/p2v1" --build-dir "$S/jit" \
               --batches "$SCALE_BATCHES" > "$O/scale.log" 2>&1 ;;
      run) "$PY" "$D/run.py" --ship "$S/ship" --out "$O" "$@" > "$O/run.log" 2> "$O/run.err" ;;
      *) echo "unknown step $step"; false ;;
    esac
    echo "$step rc $? at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"
  done
  nvidia-smi -i "$GPU_LEASE_UUID" -q -d CLOCK,PERFORMANCE,POWER > "$O/clocks-end.txt" 2>&1
  echo "released at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"' _ "$@"
echo "lease rc $?" >> "$O/rc.txt"
kill "$SMI" 2>/dev/null
cat "$O/rc.txt"
tail -4 "$O/decode-bench.log" "$O/scale.log" "$O/run.err" 2>/dev/null
grep -E '"arm": "summary"|"passed": false' "$O/run.jsonl" 2>/dev/null | cut -c1-800 | head -20
! grep -E " rc [0-9]+" "$O/rc.txt" | grep -qv " rc 0"

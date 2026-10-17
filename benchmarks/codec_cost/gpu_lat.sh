#!/usr/bin/env bash
# One word's squaring latency on one GPU thread (gpu_lat.cu: sq.cu's chain kernel, one block of one thread) at 512,
# 768 and 1,024 bits, every form sq.cu builds at that width, each gated against Python's pow (32 words at K = 1,
# 777 and 1,000) before it is timed (cwd = the source tree). JSON lines in $OUT.
#
#   research run --on vy-nebius-2 --queue --gpus 1 --cwd source -- bash benchmarks/codec_cost/gpu_lat.sh
set -uxo pipefail
O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/codec_cost; R=benchmarks/rate_squaring
B=$(mktemp -d /tmp/codec-cost-gpu.XXXXXX); trap 'rm -rf "$B"' EXIT
WIDTHS=${WIDTHS:-"512:569:0,1,2,3,4,5,6,7,11,12 768:825:1,5,6,7,11,12 1024:105:1,5,6,7,11,12"}
for wcf in $WIDTHS; do
  IFS=: read -r w c f <<< "$wcf"
  nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -DSQ_BITS="$w" -DSQ_C="$c" -o "$B/sq$w" $R/sq.cu \
    >> "$O/build.log" 2>&1 || echo "sq $w build rc $?" >> "$O/errors.txt"
  nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -DSQ_BITS="$w" -DSQ_C="$c" -o "$B/lat$w" $D/gpu_lat.cu \
    >> "$O/build.log" 2>&1 || echo "lat $w build rc $?" >> "$O/errors.txt"
done
export O B R WIDTHS
LEASE=(gpu-lease 1 --wait --max-min "${MAXMIN:-25}" --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  for wcf in $WIDTHS; do
    IFS=: read -r w c f <<< "$wcf"
    "$B/sq$w" clock >> "$O/clock.jsonl" 2>> "$O/errors.txt"
    for fold in ${f//,/ }; do
      ok=1
      for k in 1 777 1000; do
        "$B/sq$w" gate $fold 1 $k 32 > "$B/gate.txt" 2>> "$O/errors.txt" || ok=0
        python3 $R/sq_gate.py < "$B/gate.txt" >> "$O/gate.jsonl" 2>&1 || ok=0
      done
      if [ $ok = 1 ]; then
        "$B/lat$w" $fold $(( 10000000 / w )) 7 >> "$O/lat.jsonl" 2>> "$O/errors.txt" || echo "lat $w $fold rc $?" >> "$O/errors.txt"
      else
        echo "bits $w fold $fold failed its gate: not timed" >> "$O/errors.txt"
      fi
    done
  done
  nvidia-smi --query-gpu=clocks.sm --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1'

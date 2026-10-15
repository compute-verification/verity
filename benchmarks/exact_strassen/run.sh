#!/usr/bin/env bash
# strassen.cu on one GPU (cwd = the source tree): builds outside the lease, then under `gpu-lease 1` runs every shape
# in SHAPES at both alphabets ({-5..5} and [-127, 127]); each invocation gates before it times. JSON lines in
# $OUT/results.jsonl, one invocation's stderr per file under $OUT/err/.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source \
#     --env GPU_LEASE_WHO=... -- bash benchmarks/exact_strassen/run.sh
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
export B; B=$(mktemp -d /tmp/exact-strassen.XXXXXX); trap 'rm -rf "$B"' EXIT
D=benchmarks/exact_strassen
mkdir -p "$O/err"
git rev-parse HEAD > "$O/commit.txt" 2>/dev/null || true
nvcc --version > "$O/nvcc.txt"
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xcompiler -fopenmp -Xptxas -v -o "$B/strassen" \
  $D/strassen.cu -lcublasLt -ldl > "$O/build.log" 2>&1 || { tail -40 "$O/build.log"; exit 4; }
export SHAPES=${SHAPES:-"1024,1024,1024,1,full 4096,4096,4096,1,s 8192,8192,8192,1,s 16384,16384,16384,1,s 8192,8192,128,32,s"}
export AMAXES=${AMAXES:-"5 127"} REPS=${REPS:-9}
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-20}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,clocks.mem,power.draw,power.limit,driver_version \
    --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  rc=0
  for s in $SHAPES; do
    IFS=, read -r m n k b mode <<< "$s"
    for a in $AMAXES; do
      f="$O/err/m$m-n$n-k$k-b$b-a$a.txt"
      STR_TAG="m$m-n$n-k$k-b$b-a$a" timeout 280 "$B/strassen" "$m" "$n" "$k" "$b" "$a" 0x5EED "$REPS" "$mode" \
        >> "$O/results.jsonl" 2> "$f"
      r=$?; echo "{\"kind\":\"exit\",\"tag\":\"m$m-n$n-k$k-b$b-a$a\",\"rc\":$r}" >> "$O/results.jsonl"
      [ "$r" = 0 ] || rc=1
    done
  done
  nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  exit $rc
'

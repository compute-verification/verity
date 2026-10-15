#!/usr/bin/env bash
# merge.cu on one GPU (cwd = the source tree): the exact tensor-core merge D = Z1 + Z2 checked bit for bit against
# Z1 + Z2 on at least 2^20 words for each digit count, a perturbed-weight control that must mismatch, then the
# instruction's rate on merge operands. Build and verdicts outside the lease; only the device runs under it.
# Results in $OUT.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-12 --source . --cwd clone \
#     --env GPU_LEASE_WHO=... --send benchmarks/tc_merge/merge.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/merge.sh"'
set -uxo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/tc_merge
export B; B=$(mktemp -d /tmp/tc-merge.XXXXXX); trap 'rm -rf "$B"' EXIT
nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xptxas -v -o "$B/merge" $D/merge.cu > "$O/build.log" 2>&1 \
  || { tail -40 "$O/build.log"; exit 4; }
nvcc --version > "$O/nvcc.txt"
git rev-parse HEAD > "$O/commit.txt" 2>&1
cuobjdump -sass "$B/merge" > "$B/merge.sass" 2>&1
grep -c 'IMMA' "$B/merge.sass" > "$O/imma-count.txt" 2>&1
grep -E 'IMMA\.' "$B/merge.sass" | sed 's/^ *//' | sort | uniq -c > "$O/imma-forms.txt" 2>&1
gzip -c "$B/merge.sass" > "$O/merge.sass.gz"
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-20}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.draw \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  ok=1
  # n s : tiles for >= 2^20 merged words (16*s words a tile); unsigned domain [0, Vmax(n)] unless signed
  for seed in 0x5EED1234 0xC0FFEE11; do
    "$B/merge" check 4  8192  $seed 0 195329 0           >> "$O/check.jsonl" 2>> "$O/errors.txt" || ok=0
    "$B/merge" check 8  16384 $seed 0 455429 0           >> "$O/check.jsonl" 2>> "$O/errors.txt" || ok=0
    "$B/merge" check 9  21846 $seed 0 516128 0           >> "$O/check.jsonl" 2>> "$O/errors.txt" || ok=0
    "$B/merge" check 17 65536 $seed -516128 516128 516128 >> "$O/check.jsonl" 2>> "$O/errors.txt" || ok=0
  done
  # control: one radix weight 254 instead of 255 must break the merge (rc 1 expected)
  "$B/merge" check 17 4096 7 -516128 516128 516128 1 >> "$O/control.jsonl" 2>> "$O/control-errors.txt"
  echo "control rc $?" >> "$O/control.jsonl"
  [ $ok = 1 ] || { echo "check failed: not timed" >> "$O/errors.txt"; exit 6; }
  for n in 4 8 9 17; do
    "$B/merge" time $n 200000 2 >> "$O/time.jsonl" 2>> "$O/errors.txt" || echo "time n=$n rc $?" >> "$O/errors.txt"
  done
  nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
'

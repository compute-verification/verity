#!/usr/bin/env bash
# attacks.cu on one GPU (cwd = the source tree): for each noise alphabet, the exact routes to A·Bᵀ checked bit for bit
# (every entry against a CPU reference at n = 1024; against int8's output at n = 4096 and 8192, whose int8 output is
# itself checked on sampled entries), then timed interleaved against int8. Build outside the lease. Clocks read only.
# SIZES and ALPHABETS override the defaults; ROUNDS is the number of interleaved rounds. Results in $OUT.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source \
#     --env GPU_LEASE_WHO=D4-ATT -- bash benchmarks/d4_attacks/run.sh
set -uxo pipefail
export O=${OUT:?}
export CUDA=/usr/local/cuda-13.0
export PATH=$CUDA/bin:$PATH
export SIZES=${SIZES:-1024 4096 8192} ALPHABETS=${ALPHABETS:-pm1 pm1pm3 pm5pm7 u2 u4 u8} ROUNDS=${ROUNDS:-12}
export B; B=$(mktemp -d /tmp/d4-attacks.XXXXXX); trap 'rm -rf "$B"' EXIT
D=benchmarks/d4_attacks
nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xptxas -v -o "$B/attacks" $D/attacks.cu \
  -L$CUDA/lib64 -L$CUDA/lib64/stubs -lcublasLt -lnvidia-ml > "$O/build.log" 2>&1 || { tail -40 "$O/build.log"; exit 4; }
nvcc --version > "$O/nvcc.txt"
git rev-parse HEAD > "$O/commit.txt" 2>&1
cuobjdump -sass "$B/attacks" > "$B/attacks.sass" 2>&1
for k in k_bin k_4r; do
  echo "== $k" >> "$O/opcodes.txt"
  awk -v pat="$k" '$0 ~ "Function : " {f = index($0, pat) > 0; next} f' "$B/attacks.sass" \
    | grep -oE '^\s+/\*[0-9a-f]+\*/\s+[A-Z][A-Z0-9_.]*' | awk '{print $2}' | sort | uniq -c | sort -rn >> "$O/opcodes.txt"
done
gzip -c "$B/attacks.sass" > "$O/attacks.sass.gz"
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-25}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.draw,temperature.gpu \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  rc=0
  for n in $SIZES; do
    for a in $ALPHABETS; do
      "$B/attacks" "$n" "$a" 0x5EED1234 "$ROUNDS" >> "$O/attacks.jsonl" 2>> "$O/errors.txt" || { echo "rc $? n=$n a=$a" >> "$O/errors.txt"; rc=1; }
    done
  done
  nvidia-smi --query-gpu=index,uuid,clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  exit $rc
'
rc=$?
python3 $D/table.py "$O/attacks.jsonl" > "$O/table.md" 2>> "$O/errors.txt"
cat "$O/table.md"
exit $rc

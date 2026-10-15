#!/usr/bin/env bash
# prefix.cu on one GPU (cwd = the source tree): the honest m16n8k32 int8 step against the CUDA-core routes of the
# low-rank attack, each attack checked bit for bit against the host's replay, then timed in SM clocks a committed word
# a block. Build and opcode counts outside the lease; only the device runs under it. Clocks are read, never set.
# BLOCKS sets the timed blocks (default 16384). Results in $OUT.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd clone \
#     --env GPU_LEASE_WHO=RS-D2A --send benchmarks/pouw/rs_attacks/prefix.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/prefix.sh"'
set -uxo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/pouw/rs_attacks
export BLOCKS=${BLOCKS:-16384}
export B; B=$(mktemp -d /tmp/rs-attacks.XXXXXX); trap 'rm -rf "$B"' EXIT
nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xptxas -v -o "$B/prefix" $D/prefix.cu > "$O/build.log" 2>&1 \
  || { tail -40 "$O/build.log"; exit 4; }
nvcc --version > "$O/nvcc.txt"
git rev-parse HEAD > "$O/commit.txt" 2>&1
cuobjdump -sass "$B/prefix" > "$B/prefix.sass" 2>&1
# each kernel's opcodes between its two clock reads: the timed block loop
awk '/Function : / {f = $3; on = 0; next}
     /SR_CLOCKLO/ {on = !on; next}
     on && match($0, /\/\*[0-9a-f]+\*\/ +(@!?U?P[0-9T] +)?[A-Z][A-Z0-9_.]*/) {
       n = split(substr($0, RSTART, RLENGTH), p, / +/); op = p[n]; sub(/\..*/, "", op); c[f " " op]++ }
     END {for (k in c) print k, c[k]}' "$B/prefix.sass" | sort > "$O/loop-opcodes.txt" 2>&1
gzip -c "$B/prefix.sass" > "$O/prefix.sass.gz"
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-15}" -- bash -c '
  q() { nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.draw \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1; }
  run() { "$B/prefix" "$@" >> "$O/rates.jsonl" 2>> "$O/errors.txt" || echo "rc $? for $*" >> "$O/errors.txt"; }
  q
  for w in 8 16; do
    run honest 0 $w $BLOCKS
    for r in 1 4 8; do run dp4a $r $w $BLOCKS; done
    for r in 1 2 4; do run imad $r $w $BLOCKS; done
    for r in 1 2 3 4 8; do run ffma $r $w $BLOCKS; done
    for r in 1 4; do run mix $r $w $BLOCKS; done
  done
  q
'
echo "lease rc $?" >> "$O/errors.txt"

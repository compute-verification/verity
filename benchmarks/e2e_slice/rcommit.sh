#!/usr/bin/env bash
# The receipts' commitment cost a call for each checkpoint spacing s (cwd = the source tree's root): COMMIT-GPU's committer
# (benchmarks/commit_gpu: hm96-sha512/row/v2 rows in frame-v3-sha512 trees) on o_proj's receipts as 2 KiB rows (512 words a
# leaf, 512 MiB / s a call), alone and beside the o_proj int8 GEMM with the commitment's stream at the highest priority; for
# reference round 2's layout (a 512-byte leaf a thread, s = 1) and the served operand's tile leaves (decision 46: R rows x
# 32 values of the 8192 x 4096 operand). Under one GPU lease; clocks are read, never set. Under $OUT: rcommit-*.jsonl.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source --env GPU_LEASE_WHO=E2E \
#     -- bash benchmarks/e2e_slice/rcommit.sh
# SS (default "1 4 8 16 32"), REPS (default 9), TILE_R (default 64).
set -uo pipefail
export O=${OUT:?} D=benchmarks/commit_gpu PY=python3 PYTHONPATH=. REPS=${REPS:-9} SS=${SS:-1 4 8 16 32} R=${TILE_R:-64}
export PATH=/usr/local/cuda-13.0/bin:$PATH
W=$(mktemp -d /tmp/e2e-rcommit.XXXXXX)
export W
trap 'rm -rf "$W"' EXIT
git rev-parse HEAD > "$O/commit.txt" 2>&1
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -o "$W/cg" $D/commit_gpu.cu -lcublasLt -lnvidia-ml \
  > "$O/build.log" 2>&1 || { tail -20 "$O/build.log"; exit 2; }
for s in $SS; do $PY $D/reference.py params "$W/p-r$s.bin" $((262144 / s)) 16384 || exit 3; done
$PY $D/reference.py params "$W/p-thread.bin" 1048576 4096 || exit 3
$PY $D/reference.py params "$W/p-tile.bin" $((8192 / R * 128)) $((256 * R)) || exit 3
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 20 -- bash -c '
  nvidia-smi --query-gpu=index,name,clocks.sm,clocks.max.sm,power.limit --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  "$W/cg" selftest > "$O/selftest.json" 2>&1; echo "selftest rc $?" >> "$O/rc.txt"
  for s in $SS; do
    "$W/cg" bench "$W/p-r$s.bin" rows $((262144 / s)) 16384 "$REPS" 8192 4096 4096 > "$O/rcommit-s$s.jsonl" 2>> "$O/bench.err"
    echo "bench s=$s rc $?" >> "$O/rc.txt"
  done
  "$W/cg" bench "$W/p-thread.bin" rows 1048576 4096 "$REPS" 8192 4096 4096 > "$O/rcommit-thread.jsonl" 2>> "$O/bench.err"
  echo "bench thread rc $?" >> "$O/rc.txt"
  "$W/cg" bench "$W/p-tile.bin" tile 8192 4096 "$R" "$REPS" 8192 4096 4096 > "$O/rcommit-tile.jsonl" 2>> "$O/bench.err"
  echo "bench tile R=$R rc $?" >> "$O/rc.txt"'
echo "lease rc $?" >> "$O/rc.txt"
cat "$O/rc.txt"

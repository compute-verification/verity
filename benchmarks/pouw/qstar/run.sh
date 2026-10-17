#!/usr/bin/env bash
# Build qstar, gate its SASS (every timed loop's whole opcode histogram, expect.json: clock reads, one loop, no spill, a
# live sink), then on one leased GPU the poison gate of every arm and the timed items at each shape; all into $OUT.
# From a pushed commit:
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source <tree> --cwd source --custody-ttl 12h \
#     --env GPU_LEASE_WHO=compute-accounting -- bash benchmarks/pouw/qstar/run.sh [M,K,N ...]
#
# The default shapes are 4096,14336,4096 and 4096,4096,4096. The arms' receipts are checked on the leased GPU against
# the CPU reference before any timing; a failed gate times nothing.
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc)
export PATH="$(dirname "$NVCC"):$PATH"
shapes=("$@")
[ ${#shapes[@]} -gt 0 ] || shapes=(4096,14336,4096 4096,4096,4096)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

"$NVCC" -std=c++20 -O3 -gencode arch=compute_120a,code=sm_120a -ldl -Xptxas -v -o "$WORK/qstar" "$HERE/qstar.cu" \
  >"$OUT/build.log" 2>&1 || { cat "$OUT/build.log" >&2; exit 1; }
cuobjdump -sass "$WORK/qstar" >"$OUT/qstar.sass" 2>&1
python3 "$HERE/count.py" "$WORK/qstar" --pin "$HERE/expect.json" --json "$OUT/count.json" >"$OUT/count.txt" 2>&1
rc=$?
cat "$OUT/count.txt"
[ $rc -eq 0 ] || exit $rc

cat >"$WORK/leased.sh" <<EOF
set -uo pipefail
nvidia-smi --query-gpu=uuid,name,clocks.sm,clocks.max.sm,power.limit --format=csv >"$OUT/gpu.csv" 2>&1
for s in ${shapes[*]}; do
  IFS=, read -r m k n <<<"\$s"
  "$WORK/qstar" --mode gate --m \$m --k \$k --n \$n --dump "$WORK/dump-\$m-\$k-\$n" >"$OUT/gate-\$m-\$k-\$n.txt" 2>&1
  rc=\$?
  cat "$OUT/gate-\$m-\$k-\$n.txt"
  [ \$rc -eq 0 ] || exit \$rc
  "$WORK/qstar" --mode time --m \$m --k \$k --n \$n --reps 10 --items "$OUT/items-\$m-\$k-\$n.jsonl" \
    >"$OUT/time-\$m-\$k-\$n.txt" 2>&1 || exit 4
  cat "$OUT/time-\$m-\$k-\$n.txt"
done
EOF
gpu-lease 1 --wait --max-min 45 -- bash "$WORK/leased.sh"

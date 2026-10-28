#!/usr/bin/env bash
# Build qstar, gate its SASS (every timed loop's whole opcode histogram, expect.json: clock reads, one loop, no spill, a
# live sink), then on one leased GPU the poison gate of every arm and the timed items at each shape; all into $OUT.
# From a pushed commit:
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source <tree> --cwd source --custody-ttl 12h \
#     --env GPU_LEASE_WHO=LANE -- bash benchmarks/pouw/qstar/run.sh [M,K,N ...]
#
# The default shapes are 4096,14336,4096 and 4096,4096,4096; at the first, the two-stream arms (IMMA on n - co2 columns
# beside fr16, or beside design A's bb, bx and bx_mix, on co2) again at co2 = 512 and 1024. QS_ARMS (comma-separated)
# limits the arms of the shape loop (default all); QS_SWEEPS picks the co2 sweeps (default "fr16 bit"). The arms'
# receipts are checked on the leased GPU against the CPU reference before any timing; a failed gate times nothing.
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc)
export PATH="$(dirname "$NVCC"):$PATH"
shapes=("$@")
arms=${QS_ARMS:+--arms ${QS_ARMS//,/ }}
sweeps=${QS_SWEEPS:-fr16 bit}
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
  "$WORK/qstar" --mode gate --m \$m --k \$k --n \$n $arms --dump "$WORK/dump-\$m-\$k-\$n" >"$OUT/gate-\$m-\$k-\$n.txt" 2>&1
  rc=\$?
  cat "$OUT/gate-\$m-\$k-\$n.txt"
  [ \$rc -eq 0 ] || exit \$rc
  "$WORK/qstar" --mode time --m \$m --k \$k --n \$n $arms --reps 10 --items "$OUT/items-\$m-\$k-\$n.jsonl" \
    >"$OUT/time-\$m-\$k-\$n.txt" 2>&1 || exit 4
  cat "$OUT/time-\$m-\$k-\$n.txt"
done
IFS=, read -r m k n <<<"${shapes[0]}"
for w in $sweeps; do
  case \$w in
    fr16) g="co2_none co2_fr16 co2_c16 co3_none co3_c32"; a="imma_p co2_none co2_fr16 co2_c16 imma_p3 co3_none co3_c32"; p=co2 ;;
    bit) g="co3_none co3_bb co3_bx co3_mix"; a="imma_p3 co3_none co3_bb co3_bx co3_mix"; p=bit-co2 ;;
    *) echo "unknown sweep \$w" >&2; exit 2 ;;
  esac
  for c in 512 1024; do
    t="\$p-\$c-\$m-\$k-\$n"
    "$WORK/qstar" --mode gate --m \$m --k \$k --n \$n --co2 \$c --arms \$g --dump "$WORK/dump-\$t" \
      >"$OUT/gate-\$t.txt" 2>&1 || { cat "$OUT/gate-\$t.txt"; exit 3; }
    "$WORK/qstar" --mode time --m \$m --k \$k --n \$n --co2 \$c --arms \$a --reps 10 \
      --items "$OUT/items-\$t.jsonl" >"$OUT/time-\$t.txt" 2>&1 || exit 4
    cat "$OUT/gate-\$t.txt" "$OUT/time-\$t.txt"
  done
done
EOF
gpu-lease 1 --wait --max-min 45 -- bash "$WORK/leased.sh"

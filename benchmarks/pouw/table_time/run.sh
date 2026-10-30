#!/usr/bin/env bash
# Build table_time, gate its SASS (no spills, no .FTZ / bare FP32 MUFU; every timed loop's histogram against expect.json
# when present), then on one leased GPU the poison gate of every arm and the timed items at each shape; all into $OUT.
# From a pushed commit:
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source <tree> --cwd source --custody-ttl 12h \
#     --env GPU_LEASE_WHO=LANE -- bash benchmarks/pouw/table_time/run.sh [M,K,N ...]
#
# The default shapes are 2048,14336,4096 4096,14336,4096 8192,14336,4096 (the M sweep for H1's build charge and A's
# per-instance builds). TT_ARMS (comma-separated) limits the arms of the shape loop; TT_SWEEPS picks the co2 sweeps
# (default "af"). The arms' receipts are checked on the leased GPU against the CPU reference before any timing.
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc)
export PATH="$(dirname "$NVCC"):$PATH"
shapes=("$@")
[ ${#shapes[@]} -gt 0 ] || shapes=(2048,14336,4096 4096,14336,4096 8192,14336,4096)
arms=${TT_ARMS:-imma_zero imma_bb bb a_fr8 fr16_c16 fr16_c32}
sweeps=${TT_SWEEPS:-af}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

KERNELS="k_imma_zero k_imma_gen k_dp4a k_bits k_fr8 k_fr16_c16 k_fr16_c32 k_bb k_bx k_bx_hs k_bx_mix k_a_fr8"
"$NVCC" -std=c++20 -O3 -gencode arch=compute_120a,code=sm_120a -ldl -Xptxas -v -o "$WORK/tt" "$HERE/table_time.cu" \
  >"$OUT/build.log" 2>&1 || { cat "$OUT/build.log" >&2; exit 1; }
cuobjdump -sass "$WORK/tt" >"$OUT/tt.sass" 2>&1
# The numeric SASS gate: no spills (LDL/STL), no .FTZ, no bare FP32 MUFU anywhere in a timed kernel. Fail closed.
python3 "$HERE/../gpu_kit/sass_gate.py" "$WORK/tt" --kernels $KERNELS --numeric --json "$OUT/sass.json" \
  >"$OUT/sass.txt" 2>&1
rc=$?
cat "$OUT/sass.txt"
[ $rc -eq 0 ] || exit $rc
pin=""
[ -f "$HERE/expect.json" ] && pin="--pin $HERE/expect.json"
python3 "$HERE/count.py" "$WORK/tt" $pin --json "$OUT/count.json" >"$OUT/count.txt" 2>&1
rc=$?
cat "$OUT/count.txt"
[ $rc -eq 0 ] || exit $rc

cat >"$WORK/leased.sh" <<EOF
set -uo pipefail
nvidia-smi --query-gpu=uuid,name,clocks.sm,clocks.max.sm,power.limit --format=csv >"$OUT/gpu.csv" 2>&1
for s in ${shapes[*]}; do
  IFS=, read -r m k n <<<"\$s"
  "$WORK/tt" --mode gate --m \$m --k \$k --n \$n --arms $arms --dump "$WORK/dump-\$m-\$k-\$n" >"$OUT/gate-\$m-\$k-\$n.txt" 2>&1
  rc=\$?
  cat "$OUT/gate-\$m-\$k-\$n.txt"
  [ \$rc -eq 0 ] || exit \$rc
  "$WORK/tt" --mode time --m \$m --k \$k --n \$n --arms $arms --reps 10 --items "$OUT/items-\$m-\$k-\$n.jsonl" \
    >"$OUT/time-\$m-\$k-\$n.txt" 2>&1 || exit 4
  cat "$OUT/time-\$m-\$k-\$n.txt"
done
IFS=, read -r m k n <<<"${shapes[1]}"
for w in $sweeps; do
  case \$w in
    af) g="co3_none co3_af"; a="imma_p3 co3_none co3_af"; p=af-co2 ;;
    *) echo "unknown sweep \$w" >&2; exit 2 ;;
  esac
  for c in 512 1024; do
    t="\$p-\$c-\$m-\$k-\$n"
    "$WORK/tt" --mode gate --m \$m --k \$k --n \$n --co2 \$c --arms \$g --dump "$WORK/dump-\$t" \
      >"$OUT/gate-\$t.txt" 2>&1 || { cat "$OUT/gate-\$t.txt"; exit 3; }
    "$WORK/tt" --mode time --m \$m --k \$k --n \$n --co2 \$c --arms \$a --reps 10 \
      --items "$OUT/items-\$t.jsonl" >"$OUT/time-\$t.txt" 2>&1 || exit 4
    cat "$OUT/gate-\$t.txt" "$OUT/time-\$t.txt"
  done
done
EOF
gpu-lease 1 --wait --max-min 45 -- bash "$WORK/leased.sh"

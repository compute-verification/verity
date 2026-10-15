#!/usr/bin/env bash
# run.sh's leased half (env: O, B, FC, PY, ARGS; L2_SWEEP=1 runs bench.py --l2-sweep's probe alone, L2_ONLY=1 the gate alone): the L2 write gate under ncu, then the bench, which reports it.
set -uxo pipefail
NCU=$(command -v ncu || ls "$CUDA_HOME"/nsight-compute*/ncu 2>/dev/null | head -1)
if [ -n "$NCU" ]; then
  "$NCU" --csv --print-units base --kernel-name regex:^fill_ \
    --metrics lts__t_sectors_op_write.sum,lts__t_sectors_op_atom.sum,lts__t_sectors_op_red.sum,dram__bytes_write.sum,lts__t_sectors_op_read.sum,dram__bytes_read.sum \
    "$PY" $FC/bench.py --lib "$B/libfill.so" --l2-probe "$O/l2-declared.json" ${L2_SWEEP:+--l2-sweep} > "$O/l2-ncu.csv" 2> "$O/l2-ncu.err"
fi
python3 $FC/l2_gate.py "$O/l2-declared.json" "$O/l2-ncu.csv" "$O/l2.json" || echo "L2 write gate exit $? (bench.py reports it per store)"
[ -n "${L2_SWEEP:-}${L2_ONLY:-}" ] && exit 0
# shellcheck disable=SC2086
"$PY" $FC/bench.py --lib "$B/libfill.so" --fixture "$B/check" --out "$O" --l2 "$O/l2.json" ${ARGS:-}

#!/usr/bin/env bash
# run.sh's leased half (env: O, B, FC, PY, ARGS): the L2 write gate under ncu, then the bench, which reports it.
set -uxo pipefail
NCU=$(command -v ncu || ls "$CUDA_HOME"/nsight-compute*/ncu 2>/dev/null | head -1)
if [ -n "$NCU" ]; then
  "$NCU" --csv --print-units base --kernel-name regex:^fill_ \
    --metrics lts__t_sectors_op_write.sum,lts__t_sectors_op_atom.sum,lts__t_sectors_op_red.sum,dram__bytes_write.sum \
    "$PY" $FC/bench.py --lib "$B/libfill.so" --l2-probe "$O/l2-declared.json" > "$O/l2-ncu.csv" 2> "$O/l2-ncu.err"
fi
python3 $FC/l2_gate.py "$O/l2-declared.json" "$O/l2-ncu.csv" "$O/l2.json"
# shellcheck disable=SC2086
"$PY" $FC/bench.py --lib "$B/libfill.so" --fixture "$B/check" --out "$O" --l2 "$O/l2.json" ${ARGS:-}

#!/usr/bin/env bash
# attacks.py at full resolution on a CPU pod (cwd = the source tree), one process per attack and word distribution,
# merged into $OUT/attacks.json.
#
#   research run --on vy-nebius-cpu-1 --project verity --campaign pouw --source . --cwd clone \
#     --send benchmarks/receipt_phi/attacks.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/attacks.sh"'
set -uo pipefail
O=${OUT:?}
D=benchmarks/receipt_phi
P=$(mktemp -d /tmp/receipt-attacks.XXXXXX); trap 'rm -rf "$P"' EXIT
pids=()
for dist in u32 int26 fp32; do
  python3 $D/attacks.py --only differential --dist $dist --samples-log2 "${DIFF_LOG2:-20}" > "$P/diff-$dist.json" 2>> "$O/errors.txt" & pids+=($!)
  python3 $D/attacks.py --only three_word --dist $dist --samples-log2 "${THREE_LOG2:-18}" > "$P/three-$dist.json" 2>> "$O/errors.txt" & pids+=($!)
  python3 $D/attacks.py --only linearity,carries --dist $dist > "$P/lin-$dist.json" 2>> "$O/errors.txt" & pids+=($!)
done
python3 $D/attacks.py --only small,mum_lattice --dist u32 > "$P/small.json" 2>> "$O/errors.txt" & pids+=($!)
python3 $D/attacks.py --only mad_sketch --dist u32 > "$P/sketch.json" 2>> "$O/errors.txt" & pids+=($!)
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
python3 - "$P" > "$O/attacks.json" <<'EOF'
import json, pathlib, sys

def merge(a, b):
    for k, v in b.items():
        a[k] = merge(a.get(k, {}), v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return a

res = {}
for f in sorted(pathlib.Path(sys.argv[1]).glob("*.json")):
    merge(res, json.loads(f.read_text()))
print(json.dumps(res, indent=1))
EOF
exit $rc

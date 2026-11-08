#!/usr/bin/env bash
# diff.sh OLD_ART NEW_ROW_DIR: circuits' compare.py (sent beside this script) between the pinned Build's Program descriptors,
# fetched from the store with this run's custody key, and a re-pin's Build left on this node. Run with --cwd clone.
set -uo pipefail
ART=$1; NEW=$2
OUT="$RESEARCH_RUN_DIR/out"; mkdir -p "$OUT"
T=$(mktemp -d /workspace/tmp/kstruct-diff.XXXXXX)
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/summary.txt"; }
say "old $ART new $NEW head $(git rev-parse HEAD) node $(hostname)"
export RESEARCH_SRC=$(for d in $(ls -td /workspace/research/tool/*/); do [ -d "$d/research/store" ] && { echo "$d"; break; }; done)
( cd integrations/vllm/tests/regression && python3 -c "
import sys, store_io
p = store_io._cli('fetch', sys.argv[1], '--path', '*descriptor.json.gz', '--to', sys.argv[2])
print(p.stdout[-2000:])" "$ART" "$T" ) > "$OUT/fetch.log" 2>&1
say "fetch rc=$? ($(find "$T" -name descriptor.json.gz | wc -l) descriptors)"
f=$(find "$T" -path '*/build/rank0/*/descriptor.json.gz' | head -1); OLD=${f%%/build/rank0/*}
g=$(find "$NEW" -path '*/build/rank0/*/descriptor.json.gz' | head -1); NROOT=${g%%/build/rank0/*}
say "old root $OLD new root $NROOT"
/usr/bin/time -v python3 "$RESEARCH_RUN_DIR/inputs/compare.py" "$OLD" "$NROOT" "$OUT/compare.json" > "$OUT/compare.log" 2> "$OUT/compare.time"
say "compare rc=$?"
grep -E "OK|DIFF|BAD" "$OUT/compare.log" | tee -a "$OUT/summary.txt"
rm -rf "$T"

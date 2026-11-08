#!/usr/bin/env bash
# diffp.sh OLD_ART NEW_ROW_DIR P: diff.sh with compare.py run per descriptor, P at a time (compare.py's path filter), then one
# verdict over all of them. Run with --cwd clone.
set -uo pipefail
ART=$1; NEW=$2; P=${3:-12}
OUT="$RESEARCH_RUN_DIR/out"; mkdir -p "$OUT/c"
T=$(mktemp -d /workspace/tmp/kstruct-diff.XXXXXX)
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/summary.txt"; }
say "old $ART new $NEW head $(git rev-parse HEAD) node $(hostname) P=$P"
export RESEARCH_SRC=$(for d in $(ls -td /workspace/research/tool/*/); do [ -d "$d/research/store" ] && { echo "$d"; break; }; done)
( cd integrations/vllm/tests/regression && python3 -c "
import sys, store_io
p = store_io._cli('fetch', sys.argv[1], '--path', '*descriptor.json.gz', '--to', sys.argv[2])
print(p.stdout[-2000:])" "$ART" "$T" ) > "$OUT/fetch.log" 2>&1
say "fetch rc=$? ($(find "$T" -name descriptor.json.gz | wc -l) descriptors)"
f=$(find "$T" -path '*/build/rank0/*/descriptor.json.gz' | head -1); OLD=${f%%/build/rank0/*}
g=$(find "$NEW" -path '*/build/rank0/*/descriptor.json.gz' | head -1); NROOT=${g%%/build/rank0/*}
say "old root $OLD new root $NROOT"
export OLD NROOT OUT
( cd "$NROOT" && find build -name descriptor.json.gz | sort ) > "$OUT/files.txt"
say "$(wc -l < "$OUT/files.txt") new descriptors"
xargs -P "$P" -I{} bash -c 'n=$(echo "{}" | tr / _); /usr/bin/time -v python3 "$RESEARCH_RUN_DIR/inputs/compare.py" "$OLD" "$NROOT" "$OUT/c/$n.json" "/{}" > "$OUT/c/$n.log" 2> "$OUT/c/$n.time"; echo "{} rc=$?"' < "$OUT/files.txt" > "$OUT/rcs.txt"
say "compares done: $(grep -c 'rc=0' "$OUT/rcs.txt") rc=0 of $(wc -l < "$OUT/rcs.txt")"
cat "$OUT"/c/*.log | grep -E " (OK|DIFF) " > "$OUT/compare.log"
say "OK $(grep -c ' OK ' "$OUT/compare.log") DIFF $(grep -c ' DIFF ' "$OUT/compare.log")"
grep -h "Maximum resident" "$OUT"/c/*.time | sort -k6 -n | tail -1 | tee -a "$OUT/summary.txt"
rm -rf "$T"

#!/usr/bin/env bash
# main's Facts.lean against FactsNew.lean on soundness: identical facts.json and canon.jsonl, and each one's wall time.
set -eu
src=$(pwd)
pkg=${PROBE_PKG:-/workspace/research/cache/verity-check/lean-records/c/backends/flock/verifier/lean/soundness}
out=$src/out
mkdir -p "$out"
scratch=$(mktemp -d)
for v in old new; do
  mkdir -p "$scratch/$v"
  python3 - "$pkg" "$out/config-$v.json" "$scratch/$v" <<'PY'
import json, sys
pkg, cfg, scratch = sys.argv[1:]
p = json.load(open(f"{pkg}/lean-audit.json"))
roots = p["roots"]
json.dump({"roots": roots, "meaning": p.get("meaning", roots), "allowed": p.get("axioms", ["propext", "Classical.choice", "Quot.sound"]),
           "pins": sorted(p.get("guarantees") or p.get("pins") or {}), "moved": [],
           "out": f"{scratch}/facts.json", "canon": f"{scratch}/canon.jsonl"}, open(cfg, "w"))
PY
done
cd "$pkg"
/usr/bin/time -v lake env lean --run "$src/tools/lean/Facts.lean" "$out/config-old.json" > "$out/stdout-old.txt" 2> "$out/stderr-old.txt" &
a=$!
/usr/bin/time -v lake env lean --run "$src/tools/lean/probe/FactsNew.lean" "$out/config-new.json" > "$out/stdout-new.txt" 2> "$out/stderr-new.txt" &
b=$!
wait $a; wait $b
cd "$scratch"
sha256sum old/* new/* | tee "$out/sha256.txt"
ls -l old new > "$out/sizes.txt"
for f in facts.json canon.jsonl; do
  if cmp -s "old/$f" "new/$f"; then echo "$f identical"; else echo "$f DIFFERS"; fi
done | tee "$out/compare.txt"
rm -rf "$scratch"
grep -H -E 'Maximum resident|Elapsed' "$out"/stderr-*.txt
! grep -q DIFFERS "$out/compare.txt"

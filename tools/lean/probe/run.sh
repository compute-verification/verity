#!/usr/bin/env bash
# Phase timings of Facts.lean on soundness: FactsTimed.lean (Facts.lean with timers) on check's kept fast-path tree on node 1.
set -eu
src=$(pwd)
pkg=${PROBE_PKG:-/workspace/research/cache/verity-check/lean-records/c/backends/flock/verifier/lean/soundness}
out=$src/out
mkdir -p "$out"
scratch=$(mktemp -d)
python3 - "$pkg" "$out/config.json" "$scratch" <<'EOF'
import json, sys
pkg, cfg, scratch = sys.argv[1:]
p = json.load(open(f"{pkg}/lean-audit.json"))
roots = p["roots"]
json.dump({"roots": roots, "meaning": p.get("meaning", roots), "allowed": p.get("axioms", ["propext", "Classical.choice", "Quot.sound"]),
           "pins": sorted(p.get("guarantees") or p.get("pins") or {}), "moved": [],
           "out": f"{scratch}/facts.json", "canon": f"{scratch}/canon.jsonl"}, open(cfg, "w"))
EOF
git -C "$pkg/../../../../.." log -1 --format=%H > "$out/tree-commit.txt" 2>/dev/null || true
cd "$pkg"
/usr/bin/time -v lake env lean --run "$src/tools/lean/probe/FactsTimed.lean" "$out/config.json" > "$out/stdout.txt" 2> "$out/stderr.txt"
ls -l "$scratch" > "$out/scratch-sizes.txt"
rm -rf "$scratch"
grep -E '^probe|Maximum resident|Elapsed' "$out/stderr.txt"

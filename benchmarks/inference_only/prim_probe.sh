#!/usr/bin/env bash
# Scratch, never for landing: the primitive-unit probe (PrimProbe.lean) on the benchmark's two Programs and the small
# served phase's, read where an earlier run of own_probe.sh left them on this node, else built here; then on the
# record check's difftest cases, generated here.
#   bash benchmarks/inference_only/prim_probe.sh        (from the repository's root, in a clone; $OUT set)
set -euo pipefail
root=$(pwd)
pkg=verity/protocols/inference_only/verifier/nci/lean
scratch=$(mktemp -d)
prior=${PRIOR:-/workspace/research/runs/r20261111-110708-253e/out/probe}
bash tools/verity/lean/setup.sh "$pkg" > "$scratch/setup.log" 2>&1 || { tail -40 "$scratch/setup.log"; exit 1; }
export PATH="$HOME/.elan/bin:$PATH"
mkdir -p "$OUT/probe"
src=$prior
if [ -s "$prior/small-program.json" ] && [ -s "$prior/count-instance.json" ] && [ -s "$prior/class-program.json" ]; then
  echo "== 1. the three Programs: read in place at $prior"
else
  echo "== 1. the three Programs: built here"
  uv sync -q --all-packages --extra torch-cpu
  uv run --no-sync python benchmarks/inference_only/own_probe.py "$OUT/probe" > "$OUT/probe/report.txt"
  (cd adapters/vllm && uv run --no-sync python own_small.py "$OUT/probe")
  src=$OUT/probe
fi
sha256sum "$src/small-program.json" "$src/count-instance.json" "$src/class-program.json"
echo "== 2. the probe: the root's nodes by kind, the test of before against the test of now"
cd "$root/$pkg"
lake build NciCheck.Record > "$scratch/build.log" 2>&1 || { tail -60 "$scratch/build.log"; exit 1; }
tail -2 "$scratch/build.log"
bad=0
for f in small-program count-instance class-program; do
  /usr/bin/time -v timeout 3000 lake env lean --run "$root/benchmarks/inference_only/PrimProbe.lean" instance "$src/$f.json" \
    2> "$scratch/$f.time" || { echo "PROBE $f: exit $?"; bad=1; }
  grep -E "Elapsed|Maximum resident" "$scratch/$f.time" || tail -5 "$scratch/$f.time"
done
echo "== 3. the probe on the record check's difftest cases, generated here (pinned nowhere by this run)"
uv sync -q --all-packages --extra torch-cpu
uv run --no-sync python scripts/record_difftest_vectors.py "$root" --out "$scratch/gen" > "$scratch/gen.log" 2>&1 \
  || { tail -30 "$scratch/gen.log"; exit 1; }
sha256sum "$scratch"/gen/record_difftest/*.jsonl
timeout 3000 lake env lean --run "$root/benchmarks/inference_only/PrimProbe.lean" cases "$scratch/gen/record_difftest/cases.jsonl" \
  || echo "PROBE cases: exit $? (1 is what the one case of a unit of a primitive gives)"
echo "== done: of the three Programs, moved or not read: $bad"
exit $bad

# scratch (lean/WIRED): a copy of a built private tree, its `wires` stage printing Spot.Wires.check's refusal, then the cases
set -eo pipefail
SRC=$1; shift
T=$RESEARCH_RUN_DIR/tree; mkdir -p "$T"; cp -a "$SRC/." "$T/"
cd "$T/verity/protocols/workload_classification/training_detection/inference_only/verifier/lean"
python3 - <<'PY'
f = "InferenceOnlyVerifier/Record.lean"; s = open(f).read()
a = '  let W ← recode "wires" (wiring f.P f.doc)\n'
b = '    | .error _ => throw "wires"\n'
assert s.count(a) == 1 and s.count(b) == 1
s = s.replace(a, '  let W ← match wiring f.P f.doc with\n    | .ok w => pure w\n    | .error e => throw s!"wires (the wiring): {e}"\n')
s = s.replace(b, '    | .error r => throw s!"wires (Spot.Wires.check): {r.code.name} at wire {r.wire} of {W.wires.length}"\n')
open(f, "w").write(s)
PY
S=$(date +%s); lake build inference-only-record-check 2>&1 | grep -E "error|Build completed|build failed" | head -20
echo "LEAN_BUILT in $(( $(date +%s) - S )) s"
for G in "$@"; do
  C=$(basename "$G" .case.json.gz); gunzip -c "$G" > "$OUT/$C.case.json"
  echo "== $C"
  .lake/build/bin/inference-only-record-check "$OUT/$C.case.json" 2>&1 | grep --line-buffered -E "check:|error" | cut -c1-900 || true
  rm -f "$OUT/$C.case.json"
done

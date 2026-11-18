# scratch (lean/WIRED): the timed stage probe, built in a private copy of the shipped tree, on one case
set -eo pipefail
CASE=$1
T=$RESEARCH_RUN_DIR/tree; mkdir -p "$T"
tar --exclude=./.venv --exclude='*/.lake' --exclude=./.lake -cf - . | tar -xf - -C "$T"
cd "$T/verity/protocols/workload_classification/training_detection/inference_only/verifier/lean"
cp "$RESEARCH_RUN_DIR/inputs/TimeProbe.lean" WiredTime.lean
printf '\n[[lean_exe]]\nname = "wired-time"\nroot = "WiredTime"\n' >> lakefile.toml
S=$(date +%s)
lake build wired-time inference-only-record-check 2>&1 | grep -E "error|Build completed|build failed" | head -30
echo "LEAN_BUILT in $(( $(date +%s) - S )) s"
gunzip -c "$CASE" > "$OUT/case.json"
/usr/bin/time -f "wired-time: %e s, %M KB" timeout 30m .lake/build/bin/wired-time "$OUT/case.json" > "$OUT/probe.txt" 2>&1 || echo "wired-time rc=$?"; grep -v "^ " "$OUT/probe.txt" | cut -c1-1500
rm -f "$OUT/case.json"
gunzip -c "$CASE" > "$OUT/case.json"
echo "== record check"
/usr/bin/time -f "record-check: %e s, %M KB" timeout 40m .lake/build/bin/inference-only-record-check "$OUT/case.json" > "$OUT/rc.txt" 2>&1 || echo "record-check rc=$?"
cut -c1-1500 "$OUT/rc.txt"; rm -f "$OUT/case.json"

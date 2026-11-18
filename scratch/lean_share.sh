# scratch (lean/WIRED): the Lean record check and the timed stage probe, built in a private copy of the shipped tree, on one case
set -o pipefail
CASE=$1
T=$RESEARCH_RUN_DIR/tree; mkdir -p "$T"
tar --exclude=./.venv --exclude='*/.lake' --exclude=./.lake -cf - . | tar -xf - -C "$T"
cd "$T/verity/protocols/workload_classification/training_detection/inference_only/verifier/lean"
S=$(date +%s)
lake build inference-only-record-check 2>&1 | grep -E "error|Build completed|build failed" | head -30
echo "LEAN_BUILT record check in $(( $(date +%s) - S )) s"
cp "$RESEARCH_RUN_DIR/inputs/ShareProbe.lean" WiredShare.lean
printf '\n[[lean_exe]]\nname = "wired-share"\nroot = "WiredShare"\n' >> lakefile.toml
lake build wired-share 2>&1 | grep -E "error|Build completed|build failed" | head -30
echo "LEAN_BUILT probe in $(( $(date +%s) - S )) s"
gunzip -c "$CASE" > "$OUT/case.json"

if [ -x .lake/build/bin/wired-share ]; then
  /usr/bin/time -f "wired-share: %e s, %M KB" timeout 40m .lake/build/bin/wired-share "$OUT/case.json" > "$OUT/share.txt" 2>&1 || echo "wired-share rc=$?"
fi

echo "== probe"; cut -c1-1500 "$OUT/share.txt"; echo "== record check"; cut -c1-1500 "$OUT/rc.txt"
rm -f "$OUT/case.json"; cd /; rm -rf "$T"

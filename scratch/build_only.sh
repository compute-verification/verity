# scratch (lean/WIRED): the record check's build alone, in a private copy of the tree
set -o pipefail
T=$RESEARCH_RUN_DIR/tree; mkdir -p "$T"
tar --exclude=./.venv --exclude='*/.lake' --exclude=./.lake -cf - . | tar -xf - -C "$T"
cd "$T/verity/protocols/workload_classification/training_detection/inference_only/verifier/lean"
S=$(date +%s)
lake build inference-only-record-check 2>&1 | grep -E "error|Build completed|build failed" | head -30
echo "LEAN_BUILT record check in $(( $(date +%s) - S )) s"
cd /; rm -rf "$T"

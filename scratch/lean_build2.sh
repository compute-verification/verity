set -eo pipefail
# builds in a private copy of the shipped tree (never the runner's shared per-commit cache)
T=$RESEARCH_RUN_DIR/tree; mkdir -p "$T"
tar --exclude=./.venv --exclude='*/.lake' --exclude=./.lake -cf - . | tar -xf - -C "$T"
cd "$T/verity/protocols/workload_classification/training_detection/inference_only/verifier/lean"
cp "$RESEARCH_RUN_DIR/inputs/Probe.lean" WiredProbe.lean
printf '\n[[lean_exe]]\nname = "wired-probe"\nroot = "WiredProbe"\n' >> lakefile.toml
S=$(date +%s)
lake build inference-only-record-check wired-probe 2>&1 | tail -4
echo "LEAN_BUILT in $(( $(date +%s) - S )) s at $PWD"

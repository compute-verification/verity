set -eo pipefail
L=verity/protocols/workload_classification/training_detection/inference_only/verifier/lean
cd "$L"
cp "$RESEARCH_RUN_DIR/inputs/Probe.lean" WiredProbe.lean
printf '\n[[lean_exe]]\nname = "wired-probe"\nroot = "WiredProbe"\n' >> lakefile.toml
S=$(date +%s)
lake build inference-only-record-check wired-probe
echo "LEAN_BUILT in $(( $(date +%s) - S )) s at $PWD"

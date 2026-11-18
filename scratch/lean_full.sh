set -eo pipefail
bash "$RESEARCH_RUN_DIR/inputs/lean_build2.sh"
P=$RESEARCH_RUN_DIR/tree/verity/protocols/workload_classification/training_detection/inference_only/verifier/lean
bash "$RESEARCH_RUN_DIR/inputs/lean_probe.sh" "$P" "$@"

# Sourced by run_point.sh (a spec's `setup`): the int8 Llama-3.1-8B pin (checkpoints-int8.json's LLAMA31_8B_INT8, a recipe
# pin made on the node, never downloaded) in the node's HF cache, made by benchmarks/hash/dither/pin.sh when its snapshot is
# missing, and lake on the PATH, since PoUW's in-process coins are the Lean verifier of record's (`pouw.verifier.coins`
# builds pouw-check-difftest once a process). pin.sh's output goes to $RESEARCH_RUN_DIR/out/setup.
INT8_SNAPSHOT=$("$VENV/bin/python" -c 'import json; print(next(c["local_path"] for c in json.load(open(
  "integrations/vllm/manifests/checkpoints-int8.json"))["checkpoints"] if c["role"] == "LLAMA31_8B_INT8"))')
if [ ! -d "$INT8_SNAPSHOT" ]; then
  mkdir -p "${RESEARCH_RUN_DIR:-$PWD}/out/setup"
  OUT=${RESEARCH_RUN_DIR:-$PWD}/out/setup bash benchmarks/hash/dither/pin.sh \
    || { echo "setup: benchmarks/hash/dither/pin.sh failed" >&2; exit 4; }
fi
command -v lake >/dev/null || export PATH="$HOME/.elan/bin:$PATH"
command -v lake >/dev/null || { echo "setup: no lake on the PATH for the Lean verifier's coins" >&2; exit 4; }

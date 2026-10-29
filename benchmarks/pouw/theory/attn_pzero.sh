#!/usr/bin/env bash
# attn_pzero.py on a CPU node (cwd = the source's root): its own venv with CPU torch, the model's pinned snapshot pulled
# into a scratch HF home, benchmarks/ppl's dataset sent with the run as inputs/.
#
#   research run --on vy-nebius-cpu-9 --project verity --source . --cwd source --mem-gb 48 --timeout 150m \
#     --send /path/to/wikitext-2-raw-v1-test.json --env PZ_ARGS="--ctx 8192 --windows 4" \
#     -- bash benchmarks/pouw/theory/attn_pzero.sh
set -uo pipefail
O=${RESEARCH_RUN_DIR:?}/out
mkdir -p "$O"
S=$(mktemp -d "${TMPDIR:-/tmp}/pzero.XXXX") || exit 1
trap 'rm -rf "$S"' EXIT
trap 'exit 143' TERM INT
PY=$S/venv/bin/python
{ uv venv -q "$S/venv" --python 3.12 &&
  uv pip install -q --python "$PY" --index-url https://download.pytorch.org/whl/cpu torch &&
  uv pip install -q --python "$PY" transformers safetensors numpy huggingface_hub; } > "$O/setup.log" 2>&1 ||
  { echo "setup failed" | tee "$O/status.txt"; tail -20 "$O/setup.log"; exit 3; }
export HF_HOME=$S/hf
"$PY" -c 'import sys; sys.path.insert(0, "benchmarks/ppl"); import models; from huggingface_hub import snapshot_download
m = models.get("llama-3.2-1b"); print(snapshot_download(m["repo"], revision=m["revision"]))' > "$O/pull.log" 2>&1 ||
  { echo "pull failed" | tee "$O/status.txt"; tail -20 "$O/pull.log"; exit 3; }
# shellcheck disable=SC2086
"$PY" benchmarks/pouw/theory/attn_pzero.py --data "$RESEARCH_RUN_DIR/inputs" --out "$O" ${PZ_ARGS:-} \
  > "$O/stdout.txt" 2> "$O/stderr.txt"
rc=$?
echo "pzero $rc" | tee "$O/status.txt"
tail -5 "$O/stderr.txt"
exit $rc

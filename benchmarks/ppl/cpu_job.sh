#!/usr/bin/env bash
# `run.py eval --device cpu` on a CPU node, for a model small enough (Llama-3.2-1B): its own venv and HF cache in a scratch
# directory, the model's pinned snapshot pulled from the hub, the datasets sent with the run (as run.py submit sends them).
#
#   research run --on auto --cpus 16 --mem-gb 48 --max-min 150 --project verity --source . --cwd source \
#     --send ~/.cache/verity-ppl/wikitext-2-raw-v1-test.json \
#     --env PPL_ARGS="--model llama-3.2-1b --set wikitext2 --ctx 2048 SPEC..." -- bash benchmarks/ppl/cpu_job.sh
set -uo pipefail
O=${RESEARCH_RUN_DIR:?}/out
mkdir -p "$O"
S=$(mktemp -d "${TMPDIR:-/tmp}/ppl-cpu.XXXX") || exit 1
trap 'rm -rf "$S"' EXIT
trap 'exit 143' TERM INT
PY=$S/venv/bin/python
{ uv venv -q "$S/venv" --python 3.12 &&
  uv pip install -q --python "$PY" --index-url https://download.pytorch.org/whl/cpu torch &&
  uv pip install -q --python "$PY" transformers safetensors numpy huggingface_hub; } > "$O/setup.log" 2>&1 ||
  { echo "setup failed" | tee "$O/status.txt"; tail -20 "$O/setup.log"; exit 3; }
export HF_HOME=$S/hf
model=$(sed -n 's/.*--model \([^ ]*\).*/\1/p' <<< "${PPL_ARGS:?}")
"$PY" -c 'import sys; sys.path.insert(0, "benchmarks/ppl"); import models; from huggingface_hub import snapshot_download
m = models.get(sys.argv[1]); print(snapshot_download(m["repo"], revision=m["revision"]))' "$model" > "$O/pull.log" 2>&1 ||
  { echo "pull failed" | tee "$O/status.txt"; tail -20 "$O/pull.log"; exit 3; }
{ nproc; lscpu | grep -E "^Model name|^CPU\(s\)"; "$PY" -c "import torch; print('torch', torch.__version__, torch.get_num_threads())"; } \
  > "$O/cpus.txt" 2>&1
# shellcheck disable=SC2086
"$PY" benchmarks/ppl/run.py eval --device cpu --out "$O" --data "$RESEARCH_RUN_DIR/inputs" $PPL_ARGS > "$O/stdout.txt" 2> "$O/stderr.txt"
rc=$?
echo "eval $rc" | tee "$O/status.txt"
tail -5 "$O/stderr.txt"
exit $rc

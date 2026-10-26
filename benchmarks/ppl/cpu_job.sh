#!/usr/bin/env bash
# `run.py eval --device cpu` on a CPU node (cwd = the source's root), in a venv of its own built in scratch from uv's
# cache: CPU torch, numpy and transformers (its tokenizer and the HF reference layer). `run.py submit --device cpu`
# writes the `research run` that calls this, and sends the two dataset files (datasets.json) with it. The model's
# snapshot is read from the node's HF cache, offline.
#
#   PPL_ARGS   eval's flags but --out and --data (`--model M --set S --ctx T --seeds 0,1,2 --device cpu --threads N SPEC...`)
set -uo pipefail
export O=${RESEARCH_RUN_DIR:?}/out
mkdir -p "$O"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 PPL_DATA=$RESEARCH_RUN_DIR/inputs
export PPL_ARGS=${PPL_ARGS:?}
S=$(mktemp -d "${TMPDIR:-/tmp}/ppl-$(basename "$RESEARCH_RUN_DIR").XXXX") || exit 1
export TMPDIR=$S
trap 'rm -rf "$S"' EXIT
trap 'exit 143' TERM INT
uv venv -q --python 3.12 "$S/venv" || exit 1
VIRTUAL_ENV="$S/venv" uv pip install -q --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 || exit 1
VIRTUAL_ENV="$S/venv" uv pip install -q numpy transformers==5.21.0 || exit 1
"$S/venv/bin/python" -c 'import torch, transformers; print(torch.__version__, transformers.__version__)' > "$O/env.txt"
lscpu > "$O/lscpu.txt"
# shellcheck disable=SC2086
"$S/venv/bin/python" benchmarks/ppl/run.py eval --out "$O" --data "$PPL_DATA" $PPL_ARGS > "$O/stdout.txt" 2> "$O/stderr.txt"
rc=$?
echo "eval $rc" | tee "$O/status.txt"
tail -5 "$O/stderr.txt" 2>/dev/null
exit $rc

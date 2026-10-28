#!/usr/bin/env bash
# The served table's measurements of one retained pass, CPU only, after the pass (served_table.py reads their inputs.json):
#   bash benchmarks/pouw/pearl_c_vllm/served_inputs.sh <pass run id> <label> [J] [SOURCE]
# Without SOURCE: each input's census or draw bound.  With SOURCE: the same inputs by N = 384 fresh trials each
# (HonestCap's fourth case).  Fixed here before any fresh run, as the completeness citation's conditions require:
# - the inputs: served_debit --per-input 2 at seed 20261002 (two of each (k, n) in decode and in prefill, 16 a model), census
#   at most 128 tiles, else 64 work-weighted draws;
# - N = 384 trials an input; m counts every bound the table states: Llama-3.1-8B and Qwen3-8B, 16 inputs each, a census or draw
#   bound and a fresh one each, so m = 64, δ = 0.05/64, and at k = 0 the fresh limit is 1 − δ^(1/384) = 1.846%;
# - SOURCE: 32 bytes of OS randomness (os.urandom) drawn by the launcher after the pass has ended with rc 0, so after its
#   activations are fixed, and written into the command, so the run's Attempt records it; every trial's draw and salt derive
#   from it (served_debit.FRESH).  One run per pass, and the cited limit uses every trial it ran.
set -euo pipefail
WORK=${WORK:-/workspace/pouw/mvp-e2e}; PY=$WORK/venv312/bin/python
PASS=$WORK/passes/${1:?the pass run id}; LABEL=${2:?a label}; JOBS=${3:-12}; SOURCE=${4:-}; OUT=${RESEARCH_RUN_DIR:-.}/out
export PYTHONPATH=$PWD:$PWD/tools/config:$PWD/verity/core/protocols/pouw:$PWD/integrations/vllm:$PWD/benchmarks/pouw
FRESH=(--cross-check 128); [ -n "$SOURCE" ] && FRESH=(--fresh-draws 384 --fresh-source "$SOURCE" --cross-check 384)
"$PY" -u benchmarks/pouw/pearl_c_vllm/served_debit.py "$PASS/decode/manifest.json" "$PASS/prefill/manifest.json" \
  --per-input 2 --seed 20261002 --census-max 128 --draws 64 --jobs "$JOBS" "${FRESH[@]}" \
  --out "$OUT/$LABEL-${1}${SOURCE:+-fresh}"

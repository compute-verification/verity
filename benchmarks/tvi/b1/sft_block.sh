#!/usr/bin/env bash
# beta_e on SmolLM2-135M: sft.py trains at the measured step's recipe, then block.py's exact blocks on each saved theta'_s
# (theta the registered SmolLM2-135M) over the batch step s + 1 reads, then beta_e.py. One leased GPU:
#   research run --on NODE --project verity --source CHECKOUT --cwd source -- bash benchmarks/tvi/b1/sft_block.sh
# Env: STEPS (1000), SAVE ("1,10,100,1000"), BOUNDARIES ("1,14,26"), LBS ("1,2,3"), MAXMIN (the lease, 120), and CONFIGS,
# block.py's FMT:RESID:checkpoints ("bf16:bf16:all fp8t:fp32:all bf16:fp32:last"): the step itself (bf16 products and a bf16
# residual stream), the format the Qwen2.5-1.5B measurement used (fp8t, fp32 residual), and B1's own model at bf16.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p /workspace/tmp 2>/dev/null
S=$(mktemp -d /workspace/tmp/betae.XXXXXX 2>/dev/null || mktemp -d)
trap 'rm -rf "$S"' EXIT
trap 'exit 143' TERM INT
REV=93efa2f097d58c2a74874c7e644dbc9b0cee75a2          # HuggingFaceTB/SmolLM2-135M
DREV=f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc         # HuggingFaceTB/smol-smoltalk
export STEPS="${STEPS:-1000}" SAVE="${SAVE:-1,10,100,1000}"
export BOUNDARIES="${BOUNDARIES:-1,14,26}" LBS="${LBS:-1,2,3}" FILLS=exact K=1e15
export CONFIGS="${CONFIGS:-bf16:bf16:all fp8t:fp32:all bf16:fp32:last}" LAST="${SAVE##*,}"
uv venv -q --python 3.12 "$S/venv" > "$OUT/venv.txt" 2>&1 || { cat "$OUT/venv.txt"; exit 1; }
uv pip install -q --python "$S/venv/bin/python" torch==2.14.0 transformers==4.57.1 numpy safetensors pyarrow \
  constriction==0.5.0 huggingface_hub >> "$OUT/venv.txt" 2>&1 || { tail -30 "$OUT/venv.txt"; exit 1; }
uv pip freeze --python "$S/venv/bin/python" > "$OUT/freeze.txt"
PY=$S/venv/bin/python
if [ -d /workspace/hf/hub ] && [ -w /workspace/hf/hub ]; then export HF_HUB_CACHE=/workspace/hf/hub; else export HF_HOME="$S/hf"; fi
BASE=$("$PY" -c "from huggingface_hub import snapshot_download as s; print(s('HuggingFaceTB/SmolLM2-135M', revision='$REV'))") || exit 1
SHARD=$("$PY" -c "from huggingface_hub import hf_hub_download as h; print(h('HuggingFaceTB/smol-smoltalk', \
'data/train-00000-of-00004.parquet', repo_type='dataset', revision='$DREV'))") || exit 1
cp "$HERE"/block.py "$HERE"/counted.py "$HERE"/quant.py "$HERE"/sft.py "$HERE"/beta_e.py "$S"/
cd "$S"
export PYTHONPATH="$S" CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false PY BASE SHARD
echo "base $BASE ($REV) data $SHARD ($DREV) steps $STEPS save $SAVE boundaries $BOUNDARIES lbs $LBS configs $CONFIGS"
gpu-lease 1 --wait --max-min "${MAXMIN:-120}" --mem-gb 60 -- bash -c '
  set -uo pipefail
  nvidia-smi --query-gpu=index,uuid,name,driver_version --format=csv
  "$PY" sft.py "$BASE" "$SHARD" "$PWD/run" 2>&1 | grep --line-buffered -v -E "Warning|warn\(" | tee "$OUT/log-sft.txt"
  [ "${PIPESTATUS[0]}" -eq 0 ] || exit 1
  cp run/losses.json "$OUT/"
  fail=0
  for s in ${SAVE//,/ }; do
    mkdir -p "$OUT/step$s"
    for c in $CONFIGS; do
      IFS=: read -r f r at <<< "$c"
      [ "$at" = all ] || [ "$s" = "$LAST" ] || continue
      OUT="$OUT/step$s" FMT=$f RESID=$r TOKENS="$PWD/run/batch$s.pt" "$PY" block.py "$BASE" "$PWD/run/step$s" 2>&1 \
        | grep --line-buffered -v -E "Warning|warn\(" | tee "$OUT/step$s/log-block-$f-resid-$r.txt"
      [ "${PIPESTATUS[0]}" -eq 0 ] || fail=1
    done
    sha256sum run/step$s/model.safetensors run/batch$s.pt >> "$OUT/digests.txt"
  done
  exit $fail'
rc=$?
"$PY" beta_e.py "$OUT" > "$OUT/beta_e.md" 2>&1
cat "$OUT/beta_e.md"
exit $rc

#!/usr/bin/env bash
# B1's fan-out block (block.py, or SCRIPT=cheap.py for the cheap-input block) on one leased GPU, at FMT (fp8t or nvfp4), on
# CKPT (default: mb.sh's full-arm seed-1 step 150).
#   research run --on NODE --project verity --source CHECKOUT --cwd source --send gsm8k_test.jsonl --send constriction.tgz \
#     --env FMT=fp8t -- env CHECK_SLOTS=... python3 tools/verity/check/slot.py --by training -- gpu-lease 1 --wait -- \
#     bash benchmarks/tvi/b1/fan.sh
# constriction.tgz: constriction 0.5.0's cp312 linux x86_64 package directory (the rANS coder), unpacked onto PYTHONPATH.
set -uo pipefail
IN="$RESEARCH_RUN_DIR/inputs"
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p /workspace/jobs/tvi/scratch
S=$(mktemp -d /workspace/jobs/tvi/scratch/b1fan.XXXXXX)
trap 'rm -rf "$S"' EXIT
trap 'exit 143' TERM INT
cp "$HERE"/block.py "$HERE"/cheap.py "$HERE"/counted.py "$HERE"/quant.py "$IN"/gsm8k_test.jsonl "$S"/
mkdir -p "$S/extra" && tar -C "$S/extra" -xzf "$IN/constriction.tgz"
cd "$S"
PYC=/workspace/jobs/venv312/bin/python
CKPT="${CKPT:-/workspace/jobs/tvi/ckpt/B1/s1/A2/gsm8k-full-seed1/step150}"
if [ -d /workspace/hf/hub ] && [ -w /workspace/hf/hub ]; then export HF_HUB_CACHE=/workspace/hf/hub; else export HF_HOME="$S/hf"; fi
export PYTHONPATH="$S/extra:$S" OUT FMT="${FMT:-fp8t}" DATA="$S/gsm8k_test.jsonl"
echo "fmt $FMT ckpt $CKPT boundaries ${BOUNDARIES:-1,13} lbs ${LBS:-1,2,3} k ${K:-2e7}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
"$PYC" -c 'import constriction, torch; print("constriction ok", torch.__version__, torch.cuda.get_device_name(0))'
SCRIPT="${SCRIPT:-block.py}"
"$PYC" "$SCRIPT" "${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}" "$CKPT" 2>&1 | grep --line-buffered -v -E 'Warning|warn\(' \
  | tee "$OUT/log-${SCRIPT%.py}-$FMT${EMB:+-emb-$EMB}.txt"
ls -la "$OUT"

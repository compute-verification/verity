#!/usr/bin/env bash
# The PoUS option's GPU smoke (pous_gpu.py) on one H200; cwd = the source tree's root:
#
#   research run --on vy-dc-h200-3 --project verity --campaign e2e-24h --source . --cwd source --timeout 90m \
#     --env GPU_LEASE_WHO=memory-accounting -- bash integrations/vllm/tests/protocols/pous_gpu.sh
#
# Outside the lease: libp2v1.so built by benchmarks/torch_gpu/p2_stage/build.sh into $OUT/p2lib (sm_90a); torch, numpy,
# safetensors and huggingface_hub from PyPI in a uv venv in the scratch directory; the base pin unsloth/Llama-3.2-1B at its
# pinned revision into HF (default /workspace/tmp/memory-accounting/hf, kept with `research keep`), and the int8 pin
# LLAMA32_1B_INT8 re-made beside it by `verity-vllm int8-checkpoint make`, which checks every file's sha256 against
# checkpoints-int8.json. Under one untimed lease of at most LEASE_MIN minutes (default 20): pous_gpu.py, writing
# $OUT/pous_gpu.json. Each step's exit code is in $OUT/rc.txt; exits non-zero if any failed.
set -uo pipefail
O=${OUT:?}; ROOT=$PWD; V=integrations/vllm
HF=${HF:-/workspace/tmp/memory-accounting/hf}
BASE_REPO=unsloth/Llama-3.2-1B; BASE_REV=9535bd9b1d1dea6acafbdc4813b728796aeb28da
PIN=$HF/hub/models--verity-int8--Llama-3.2-1B-INT8/snapshots/15b0976d400d796ee589e7737d03a919526bf800
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
: > "$O/rc.txt"
step() { echo "$1 rc $2 at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"; }
git rev-parse HEAD > "$O/commit.txt" 2>&1

ARCH=sm_90a bash benchmarks/torch_gpu/p2_stage/build.sh "$O/p2lib" > "$O/build.log" 2>&1
step build $?
uv venv -q -p 3.12 "$S/venv" && uv pip install -q -p "$S/venv" torch numpy safetensors huggingface_hub > "$O/venv.log" 2>&1
step venv $?
PY=$S/venv/bin/python
export PYTHONPATH=$ROOT:$ROOT/$V:$ROOT/tools/config:$ROOT/tools/research/src PYTHONDONTWRITEBYTECODE=1
"$PY" -c 'import torch; print(torch.__version__, torch.version.cuda)' > "$O/python.txt" 2>&1
nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.max.sm,power.limit --format=csv > "$O/gpus.txt" 2>&1
grep -qE "^(build|venv) rc [1-9]" "$O/rc.txt" && { cat "$O/rc.txt"; exit 2; }

mkdir -p "$HF/hub"
research keep "$HF" --owner @memory-accounting --expires 2026-11-02 --priority normal \
  --why "unsloth/Llama-3.2-1B's pin and its int8 pin LLAMA32_1B_INT8, for the PoUS option's GPU smoke" > /dev/null 2>&1
if [ ! -f "$PIN/model.safetensors" ]; then
  "$PY" -c 'import sys; from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2], cache_dir=sys.argv[3]))' "$BASE_REPO" "$BASE_REV" "$HF/hub" \
    > "$O/base.log" 2>&1
  step base $?
  (cd "$V" && "$PY" -m verity_vllm.pipeline.cli int8-checkpoint make --role LLAMA32_1B_INT8 --hf-home "$HF" --jobs 4) \
    > "$O/pin.log" 2>&1
  step pin $?
else
  step pin 0
fi
grep -qE "^(base|pin) rc [1-9]" "$O/rc.txt" && { cat "$O/rc.txt"; tail -20 "$O/base.log" "$O/pin.log" 2>/dev/null; exit 2; }
echo "pin $PIN" >> "$O/rc.txt"

echo "pous_gpu: asking for the GPU at $(date -u +%H:%M:%SZ)"
export O PY PIN V
# shellcheck disable=SC2016
env -u GPU_LEASE_TIMED gpu-lease 1 --wait --max-min "${LEASE_MIN:-20}" -- bash -c '
  echo "leased at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"
  nvidia-smi -i "$GPU_LEASE_UUID" -q -d CLOCK,PERFORMANCE,POWER > "$O/clocks.txt" 2>&1
  (cd "$V" && "$PY" tests/protocols/pous_gpu.py --snapshot "$PIN" --lib "$O/p2lib/lib" --out "$O" ${POUS_GPU_ARGS:-}) \
    > "$O/pous_gpu.log" 2> "$O/pous_gpu.err"
  echo "smoke rc $? at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"'
step lease $?
cat "$O/rc.txt"; tail -3 "$O/pous_gpu.log" 2>/dev/null; tail -30 "$O/pous_gpu.err" 2>/dev/null
grep -qE "rc [1-9]" "$O/rc.txt" && exit 1
exit 0

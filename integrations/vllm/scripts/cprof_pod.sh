#!/usr/bin/env bash
# CPROF pod setup + B0 store build (RTX 4090, driver 550 -> cu129 wheel variant, env/README.md).
# Run inside tmux on the pod: bash /workspace/veritor-cprof/scripts/cprof_pod.sh
# Ends with CPROF-STORE-DONE.
set -uxo pipefail
export DEBIAN_FRONTEND=noninteractive

WORK=/workspace
SRC=$WORK/veritor-cprof
POC=$SRC/vllm-poc
VENV=$WORK/venv-cu129
PY=$VENV/bin/python
export HF_HOME=$WORK/hf

cd $WORK

# --- venv + pinned vLLM (cu129 build of the frozen commit d9105ea80; env/README.md deviation table)
if [ ! -x "$PY" ]; then
  python3 -m venv $VENV
  $VENV/bin/pip install -q -U pip uv
  mkdir -p $WORK/wheels && cd $WORK/wheels
  W="vllm-0.28.1rc1.dev472+gd9105ea80.cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
  [ -f "$W" ] || curl -fLO "https://wheels.vllm.ai/d9105ea8001e0a6d77a96327d17515bb5791fb36/vllm-0.28.1rc1.dev472%2Bgd9105ea80.cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
  echo "7aa52ac71500caf74643135a340eb2d03ae0386d4282fc0b23846940029fbbb3  $W" | sha256sum -c
  $VENV/bin/uv pip install --python $PY \
      --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match \
      "$WORK/wheels/$W" numpy safetensors huggingface_hub zstandard pytest
fi
cd $WORK

# --- checkpoint at the exact pinned revision (checkpoints.json local_path layout)
$PY - <<'EOF'
from huggingface_hub import snapshot_download
p = snapshot_download("HuggingFaceTB/SmolLM2-135M",
                      revision="93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
                      allow_patterns=["*.json", "*.safetensors"])
print("checkpoint at", p)
EOF

# --- B0 store: production capture, then replay --mode commit (the TA1/INT path), then weights v2
ADAPTER_REV=$(cat $SRC/.cprof-adapter-rev 2>/dev/null || echo cprof-pod)
OUT=$WORK/results/B0/cprof
mkdir -p $OUT
export PYTHONPATH=$POC:$SRC/src
export VLLM_USE_V2_MODEL_RUNNER=1

if [ ! -f $OUT/production/production.jsonl ]; then
  rm -rf $OUT/production
  $PY -m verity_vllm_adapter.production --case B0 \
      --checkpoints $POC/manifests/checkpoints.json --workload $POC/e2e/workload-B0-small.json \
      --out $OUT/production --adapter-revision "$ADAPTER_REV" --enforce-eager \
      2>&1 | tee $OUT/production.log
fi

if [ ! -f $OUT/commit/commitments.json ]; then
  rm -rf $OUT/commit
  $PY -m verity_vllm_adapter.replay --case B0 \
      --checkpoints $POC/manifests/checkpoints.json --production $OUT/production \
      --out $OUT/commit --mode commit --weights-values store \
      --adapter-revision "$ADAPTER_REV" --arena-gb 24 --replay-exec eager \
      2>&1 | tee $OUT/replay-commit.log
fi

if [ ! -f $OUT/commit-v2/commitments.json ]; then
  rm -rf $OUT/commit-v2
  PYTHONPATH=$POC:$SRC/src $PY -m verity_vllm.weights_v2 $OUT/commit $OUT/commit-v2 --rows-per-leaf 1 \
      2>&1 | tee $OUT/weights-v2.log
fi

echo CPROF-STORE-DONE

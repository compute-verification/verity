#!/usr/bin/env bash
# The cuBLAS device pin of one GPU under Slurm, with no .git, gpu-lease, root, profiler or nvcc: the job `run.sh device-pin` runs,
# from a `git archive` tarball of this tree, in the served vLLM's pinned environment (ops/pod_bootstrap.sh's wheel and torch).
#   make:   git archive --format=tar.gz -o "$C/verity-pin.tar.gz" COMMIT
#   submit: sbatch --gres=gpu:1 --time=25 --chdir="$C" --output="$C/device-pin-%j.log" device-pin.slurm.sh
# C (default $HOME/verity-vcf) holds the tarball, the venv and the caches; out/ under the job's working directory gets the pin.
#SBATCH --job-name=h100-route-pin
#SBATCH --gres=gpu:1
#SBATCH --time=25
set -euo pipefail

C=${C:-$HOME/verity-vcf}
TGZ=${TGZ:-$C/verity-pin.tar.gz}
VENV=${VENV:-$C/venv312-pin}
OUT=${OUT:-$PWD/device-pin-out-${SLURM_JOB_ID:-local}}
DRY=${DRY:-0}                       # 1: every step but the probe, on a machine with no GPU
BUDGET_S=${BUDGET_S:-1200}
TORCH_PIN="2.13.0+cu129"
W="vllm-0.28.1rc1.dev472+gd9105ea80.cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
WSHA=7aa52ac71500caf74643135a340eb2d03ae0386d4282fc0b23846940029fbbb3
WURL="https://wheels.vllm.ai/d9105ea8001e0a6d77a96327d17515bb5791fb36/vllm-0.28.1rc1.dev472%2Bgd9105ea80.cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
TORCHCODEC="torchcodec==0.16.0+cu129"

mkdir -p "$OUT" "$C/wheels" "$C/cache"
exec > >(tee -a "$OUT/device-pin.log") 2>&1
STEP=start
trap 'rc=$?; [ $rc = 0 ] || echo "DEVICE_PIN_FAIL step=$STEP rc=$rc after ${SECONDS}s (budget ${BUDGET_S}s)"' EXIT
step() { STEP=$1; echo "[device-pin] $(date -u +%FT%TZ) +${SECONDS}s $1"; }
left() { local l=$(( BUDGET_S - SECONDS )); [ $l -gt 30 ] || { echo "DEVICE_PIN_FAIL step=$STEP: out of time (${SECONDS}s of ${BUDGET_S}s)" >&2; exit 124; }; echo $l; }
export XDG_CACHE_HOME=$C/cache UV_CACHE_DIR=$C/cache/uv UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-600}

step tree
[ -f "$TGZ" ] || { echo "no tree tarball at $TGZ: make it with \`git archive --format=tar.gz -o $TGZ COMMIT\`"; exit 2; }
COMMIT=$(gzip -dc "$TGZ" | git get-tar-commit-id 2>/dev/null || true)
[ -n "$COMMIT" ] || { echo "$TGZ carries no commit id: make it with \`git archive\` of a commit, not of a tree or a working copy"; exit 2; }
TREE=$C/tree-$COMMIT
[ -d "$TREE/adapters/vllm/verity_vllm" ] || { rm -rf "$TREE.part"; mkdir -p "$TREE.part"; tar -xzf "$TGZ" -C "$TREE.part"; mv "$TREE.part" "$TREE"; }
echo "commit $COMMIT"

step venv
command -v uv > /dev/null || { python3 -m pip install -q --user uv; export PATH=$HOME/.local/bin:$PATH; }
PY=$VENV/bin/python
[ -x "$PY" ] || uv venv -q "$VENV" --python 3.12
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

step install
if ! "$PY" -c "import torch, vllm; assert torch.__version__ == '$TORCH_PIN', torch.__version__" 2> /dev/null; then
  [ -f "$C/wheels/$W" ] || timeout "$(left)" curl -fsSL --retry 5 -o "$C/wheels/$W" "$WURL"
  echo "$WSHA  $C/wheels/$W" | sha256sum -c --quiet
  timeout "$(left)" uv pip install -q --python "$PY" --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match \
    "$C/wheels/$W" "torch==$TORCH_PIN" "$TORCHCODEC" "numpy>=1.26" safetensors huggingface_hub zstandard
fi

step import
cd "$TREE/adapters/vllm"
export PYTHONPATH=$TREE:$TREE/adapters/vllm:$TREE/adapters/torch:$TREE/tools/config:$TREE/tools/research/src
timeout "$(left)" "$PY" - << 'EOF'
import torch
from verity_vllm.pipeline import cli, device_pin  # noqa: F401
from verity_vllm.acquire.served import probe_routes  # noqa: F401
from verity_vllm.check.replay.driver import engine_facts_of_record  # noqa: F401
print(f"torch {torch.__version__} cuda {torch.version.cuda} cudnn {torch.backends.cudnn.version()} gpu {torch.cuda.is_available()}")
EOF

if [ "$DRY" = 1 ]; then
  step "dry run: stopped before the probe"
  exit 0
fi

step device
nvidia-smi --query-gpu=index,name,driver_version,memory.used --format=csv,noheader
DEVICE=$(timeout "$(left)" "$PY" -c 'import torch; print(torch.cuda.get_device_name())')
PIN=$OUT/device_pins/${DEVICE// /_}.json
mkdir -p "$(dirname "$PIN")"

step probe
timeout "$(left)" "$PY" -m verity_vllm.pipeline.cli device-pin probe --out "$PIN" \
  --evidence "slurm job ${SLURM_JOB_ID:-local} at commit $COMMIT"

step report
"$PY" - "$PIN" << 'EOF'
import json, sys
import torch
from verity_vllm.check.replay.driver import engine_facts_of_record
pin = json.load(open(sys.argv[1]))
facts = engine_facts_of_record()["cublas"]
print(f"device {torch.cuda.get_device_name()!r} torch {torch.__version__} cuda {torch.version.cuda} "
      f"cudnn {torch.backends.cudnn.version()} cublas {facts.get('cublas_version')} routes {len(pin['cublas'])}")
EOF
echo "pin $(sha256sum "$PIN" | cut -d' ' -f1)  $PIN"
step done

#!/usr/bin/env bash
# The e2e's environment on node 2, CPU only (no GPU lease), idempotent: vLLM d9105ea80 cu129 and torch in a Python 3.12 venv
# (pod_bootstrap.sh --cpu, no checkpoint cases), Llama-3.1-8B-Instruct at its pinned revision, and FlashInfer's sampling
# kernels, which vLLM's top-p/top-k sampler would otherwise JIT-compile on its first call inside a window.  They are built
# for window.sh's FLASHINFER_CUDA_ARCH_LIST (12.0f, what the card's (12, 0) maps to) with the venv's ninja first on PATH, so
# the window's build flags and cache path are these and nothing rebuilds there; the cache is the venv's FlashInfer build's own
# (flashinfer_env.sh).  pyarrow reads the quality run's WikiText-2.
#   research run --on vy-nebius-2 --project verity --campaign pouw --source <run tree> --cwd source \
#     --env GPU_LEASE_WHO=<you> -- bash benchmarks/pouw/pearl_c_vllm/setup.sh
# Env: WORK (/workspace/pouw/mvp-e2e), HF_HOME (/workspace/hf).  Writes $WORK/ready.json.  LINK_VENV=DIR links a venv already
# on the node (node 1's /workspace/jobs/venv312) as $WORK/venv312 instead of bootstrapping one, once its vLLM and torch are
# pod_bootstrap.sh's pins; nothing is installed into it, so pyarrow (the quality run's) is left out.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
export WORK=${WORK:-/workspace/pouw/mvp-e2e} HF_HOME=${HF_HOME:-/workspace/hf}
REPO=unsloth/Meta-Llama-3.1-8B-Instruct REV=a2856192dd7c25b842431f39c179a6c2c2f627d1
mkdir -p "$WORK"
if [ -n "${LINK_VENV:-}" ]; then
  "$LINK_VENV/bin/python" -c "import sys, torch, vllm
sys.exit(vllm.__version__ != '0.28.1rc1.dev472+gd9105ea80' or torch.__version__ != '2.13.0+cu129')" \
    || { echo "LINK_VENV $LINK_VENV is not pod_bootstrap.sh's vLLM d9105ea80 and torch 2.13.0+cu129" >&2; exit 3; }
  [ -e "$WORK/venv312" ] || ln -s "$LINK_VENV" "$WORK/venv312"
else
  (cd "$ROOT/integrations/vllm" && VENV=$WORK/venv312 bash verity_vllm/ops/pod_bootstrap.sh --cpu --cases "" --out "$WORK/bootstrap")
fi
"$WORK/venv312/bin/python" - "$REPO" "$REV" "$WORK/models/llama-3.1-8b-instruct" <<'EOF'
import json, os, sys
from pathlib import Path
from huggingface_hub import snapshot_download
repo, rev, link = sys.argv[1:4]
p = Path(snapshot_download(repo, revision=rev, allow_patterns=["*.json", "*.safetensors"], cache_dir=os.path.join(os.environ["HF_HOME"], "hub")))
Path(link).parent.mkdir(parents=True, exist_ok=True)
if not Path(link).exists():
    os.symlink(p, link)
import torch, vllm
ready = {"model": repo, "revision": rev, "path": str(p), "vllm": vllm.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
         "shards": sorted(f.name for f in p.glob("*.safetensors"))}
(Path(os.environ["WORK"]) / "ready.json").write_text(json.dumps(ready, indent=1) + "\n")
print(json.dumps(ready))
EOF
[ -n "${LINK_VENV:-}" ] || "$WORK/venv312/bin/python" -c "import pyarrow" 2>/dev/null || uv pip install --python "$WORK/venv312/bin/python" -q pyarrow
. "$ROOT/benchmarks/pouw/pearl_c_vllm/flashinfer_env.sh"
PATH=$WORK/venv312/bin:$PATH FLASHINFER_CUDA_ARCH_LIST=12.0f CUDA_VISIBLE_DEVICES= "$WORK/venv312/bin/python" - <<'EOF'
import json, os
from pathlib import Path
import flashinfer
from flashinfer.sampling import gen_sampling_module
spec = gen_sampling_module()
spec.build()
if not spec.is_compiled:
    raise SystemExit("FlashInfer's sampling module did not build")
ready = Path(os.environ["WORK"]) / "ready.json"
r = json.loads(ready.read_text())
r["flashinfer"] = {"version": flashinfer.__version__, "arch_list": os.environ["FLASHINFER_CUDA_ARCH_LIST"],
                   "workspace_base": os.environ["FLASHINFER_WORKSPACE_BASE"],
                   "prebuilt": {"sampling": str(spec.get_library_path())}}
ready.write_text(json.dumps(r, indent=1) + "\n")
print(json.dumps(r["flashinfer"]))
EOF

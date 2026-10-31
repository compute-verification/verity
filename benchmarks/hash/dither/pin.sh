#!/usr/bin/env bash
# The int8 Llama-3.1-8B pin (checkpoints-int8.json's LLAMA31_8B_INT8, recipe verity-int8-channel127/v1) in a node's HF cache:
# the BF16 base pin (LLAMA31_8B, unsloth/Llama-3.1-8B) downloaded if missing, then `verity-vllm int8-checkpoint make`, which
# checks every base shard and every made file against the manifests' sha256s. CPU only; cwd = the source tree.
#
#   bash benchmarks/hash/dither/pin.sh      (on the host that serves it, e.g. an H200; HF_HOME, default /workspace/hf)
set -euo pipefail
PY=${VERITY_BENCH_VENV:-/workspace/cp/venv312}/bin/python
export HF_HOME=${HF_HOME:-/workspace/hf} PYTHONPATH=$PWD/integrations/vllm:$PWD${PYTHONPATH:+:$PYTHONPATH}
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
mkdir -p "$OUT"
"$PY" - <<'EOF'
import json, os
from huggingface_hub import snapshot_download
base = next(c for c in json.load(open("integrations/vllm/manifests/checkpoints.json"))["checkpoints"] if c["role"] == "LLAMA31_8B")
print(snapshot_download(base["repo"], revision=base["revision"], allow_patterns=["*.json", "*.safetensors"],
                        cache_dir=os.environ["HF_HOME"] + "/hub"))
EOF
"$PY" -c 'import sys; from verity_vllm.pipeline.cli import main; sys.exit(main())' \
  int8-checkpoint make --role LLAMA31_8B_INT8 --hf-home "$HF_HOME" | tee "$OUT/pin.json"

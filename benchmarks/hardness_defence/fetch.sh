#!/usr/bin/env bash
# A Hugging Face model's weights, config and tokenizer into node RAM (/dev/shm), for a model too large for the GPU
# node's nearly full /workspace. HF_HOME points into the same directory, so no cache lands on disk.
#   bash benchmarks/hardness_defence/fetch.sh REPO DIR
# The caller removes DIR when its evals are done (`rm -rf DIR`: it is this lane's own scratch, not a retained path).
set -euo pipefail
repo=$1 dir=$2
here=$(cd "$(dirname "$0")" && pwd)
source "$here/../hardness_useful/env.sh"
mkdir -p "$dir"
export HF_HOME=$dir/.hf
python - "$repo" "$dir" <<'EOF'
import sys
import time

from huggingface_hub import snapshot_download

t0 = time.time()
p = snapshot_download(sys.argv[1], local_dir=sys.argv[2], max_workers=16,
                      allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
print(p, f"{time.time() - t0:.0f}s", flush=True)
EOF
du -sh "$dir"

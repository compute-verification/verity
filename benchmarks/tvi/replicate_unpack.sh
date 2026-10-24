# On a node without the venv: fetch replicate_pack.sh's artifact ($TVI_ART) and unpack it at the same absolute paths.
set -euo pipefail
: "${TVI_ART:?set TVI_ART=art:...}"
test ! -e /workspace/jobs/tvi/venv/READY || { echo "READY already here; refusing to overwrite"; exit 3; }
test ! -e /workspace/jobs/tvi/venv/bin/python || { echo "a venv is already here; refusing to overwrite"; exit 3; }
mkdir -p /workspace/jobs/tvi
S=$(mktemp -d -p /workspace/jobs/tvi .unpack.XXXXXX)
trap 'rm -rf "$S"' EXIT
time research data fetch "$TVI_ART" --to "$S"
F=$(find "$S" -name '*.tar.zst' | head -1)
sha256sum "$F" | tee "$OUT/sha256.txt"
time tar -I zstd -xf "$F" -C /
/workspace/jobs/tvi/venv/bin/python -c 'import torch, trl, peft, vllm, transformers, math_verify; print(torch.__version__, trl.__version__, peft.__version__, vllm.__version__, transformers.__version__)' | tee "$OUT/imports.txt"
cat /workspace/jobs/tvi/venv/READY
ls -l /workspace/jobs/tvi/templates

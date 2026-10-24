# R2's own venv on a node where A1's shared one isn't READY yet: TRL 1.16 with the vLLM it pins, PEFT, math-verify.
# Writes /workspace/jobs/tvi/r2-venv (with READY) and $OUT/versions.txt.
set -euo pipefail
V=/workspace/jobs/tvi/r2-venv
mkdir -p /workspace/jobs/tvi
test ! -e "$V/READY" || { echo "READY already"; cat "$V/READY"; exit 0; }
S=$(mktemp -d -p /workspace/jobs/tvi .r2-install.XXXXXX)
trap 'rm -rf "$S"' EXIT
export UV_CACHE_DIR="$S/uv" UV_LINK_MODE=copy
uv venv --python 3.12 "$V"
research keep "$V" --owner @training --expires 2026-10-27 --priority normal --why 'training-vs-inference program (R2 venv)' || true
time uv pip install --python "$V/bin/python" "trl[vllm]==1.16.0" peft math-verify datasets accelerate 2>&1 | tail -40
"$V/bin/python" -c 'import torch, trl, peft, vllm, transformers, accelerate, datasets, math_verify; print("torch", torch.__version__, torch.version.cuda, "trl", trl.__version__, "peft", peft.__version__, "vllm", vllm.__version__, "transformers", transformers.__version__, "accelerate", accelerate.__version__, "datasets", datasets.__version__)' | tee "$OUT/versions.txt"
uv pip freeze --python "$V/bin/python" > "$OUT/freeze.txt"
date -u +%FT%TZ > "$V/READY"
cat "$OUT/versions.txt" >> "$V/READY"

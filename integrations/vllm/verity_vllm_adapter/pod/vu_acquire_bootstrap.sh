#!/bin/bash
# VU-ACQUIRE pod bootstrap (vu-acquire-a100; A100-SXM4-80GB, driver 580.x -> pinned +cu130 wheel OK).
# Two venvs:
#   /workspace/venv-derived : uv py3.12, torch + numpy + pytest + editable veritor — the derived-boundary
#                             evaluators (verity_vllm.registry.derived_rows[_torch]) and the cost bench.
#   /workspace/venv         : the pinned vLLM env of bootstrap_w3c.sh (wheel d9105ea80 + requirements.lock),
#                             checkpoints restricted to role B0 — the plain-replay baseline
#                             (verity_vllm_adapter.replay --mode reference).
# usage: nohup bash vu_acquire_bootstrap.sh > /workspace/logs/bootstrap.log 2>&1 &
set -u
mkdir -p /workspace/logs /workspace/wheels /workspace/hf /workspace/results
export HF_HOME=/workspace/hf
echo "[$(date -u +%H:%M:%SZ)] vu-acquire bootstrap start on $(hostname)"

# --- derived venv (fast path, no vLLM) --------------------------------------------------------------
if [ ! -x /workspace/venv-derived/bin/python ]; then
  pip install -q uv || exit 1
  uv venv /workspace/venv-derived --python 3.12 || exit 1
  uv pip install --python /workspace/venv-derived/bin/python -q -e /workspace/veritor torch numpy pytest || exit 1
fi
echo "[$(date -u +%H:%M:%SZ)] venv-derived ok: $(/workspace/venv-derived/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"

# --- pinned vLLM venv (bootstrap_w3c.sh recipe, wheel step verbatim) --------------------------------
if [ ! -x /workspace/venv/bin/python ]; then
  python3 -m venv /workspace/venv && /workspace/venv/bin/pip install -q -U pip uv || exit 1
fi
cd /workspace/wheels
W=vllm-0.28.1rc1.dev472+gd9105ea80-cp38-abi3-manylinux_2_28_x86_64.whl
if [ ! -f $W ]; then
  curl -fL -o "$W" "https://wheels.vllm.ai/d9105ea8001e0a6d77a96327d17515bb5791fb36/vllm-0.28.1rc1.dev472%2Bgd9105ea80-cp38-abi3-manylinux_2_28_x86_64.whl" || exit 1
fi
echo "0198513ebd5561f154d9a5a241cdaacd1a7a211bff66aa8997e9e62481eb9120  $W" | sha256sum -c || exit 1
sed "s#^vllm @ .*#vllm @ file:///workspace/wheels/$W#" /workspace/veritor/vllm-poc/env/requirements.lock > /workspace/wheels/requirements.lock
/workspace/venv/bin/uv pip install --python /workspace/venv/bin/python -r /workspace/wheels/requirements.lock || exit 1
echo "[$(date -u +%H:%M:%SZ)] venv ok: $(/workspace/venv/bin/python -c 'import vllm,torch; print(vllm.__version__, torch.__version__)')"

# --- checkpoint: role B0 only (SmolLM2-135M at the frozen revision, sha-checked) --------------------
/workspace/venv/bin/python - <<'EOF'
import hashlib, json, sys
from pathlib import Path
from huggingface_hub import snapshot_download
ck = json.load(open('/workspace/veritor/manifests/checkpoints.json'))
bad = 0
for e in ck['checkpoints']:
    if not e['downloaded'] or e['role'] != 'B0':
        continue
    local = Path(snapshot_download(e['repo'], revision=e['revision'],
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "merges.txt", "vocab*"]))
    assert str(local) == e['local_path'], (str(local), e['local_path'])
    for f, want in e['safetensors_sha256'].items():
        h = hashlib.sha256()
        with open(local / f, 'rb') as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b''):
                h.update(chunk)
        ok = h.hexdigest() == want
        bad += (not ok)
        print(f"{e['repo']} {f} sha256={'OK' if ok else 'MISMATCH ' + h.hexdigest()}", flush=True)
sys.exit(1 if bad else 0)
EOF
[ $? = 0 ] || { echo "CHECKPOINT SHA MISMATCH"; exit 1; }
echo "[$(date -u +%H:%M:%SZ)] BOOTSTRAP_DONE"

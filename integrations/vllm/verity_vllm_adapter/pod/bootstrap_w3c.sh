#!/bin/bash
# W3c pod bootstrap: pinned venv (env/README.md), checkpoints B0+B1 at the frozen revisions, sha check.
# usage: nohup bash bootstrap_w3c.sh > /workspace/logs/bootstrap.log 2>&1 &
set -u
mkdir -p /workspace/logs /workspace/wheels /workspace/hf /workspace/results
export HF_HOME=/workspace/hf
echo "[$(date -u +%H:%M:%SZ)] bootstrap start on $(hostname)"
if [ ! -x /workspace/venv/bin/python ]; then
  python3 -m venv /workspace/venv && /workspace/venv/bin/pip install -q -U pip uv || exit 1
fi
cd /workspace/wheels
W=vllm-0.28.1rc1.dev472+gd9105ea80-cp38-abi3-manylinux_2_28_x86_64.whl
[ -f "vllm-0.28.1rc1.dev472%2Bgd9105ea80-cp38-abi3-manylinux_2_28_x86_64.whl" ] && mv "vllm-0.28.1rc1.dev472%2Bgd9105ea80-cp38-abi3-manylinux_2_28_x86_64.whl" "$W"
if [ ! -f $W ]; then
  curl -fL -o "$W" "https://wheels.vllm.ai/d9105ea8001e0a6d77a96327d17515bb5791fb36/vllm-0.28.1rc1.dev472%2Bgd9105ea80-cp38-abi3-manylinux_2_28_x86_64.whl" || exit 1
fi
echo "0198513ebd5561f154d9a5a241cdaacd1a7a211bff66aa8997e9e62481eb9120  $W" | sha256sum -c || exit 1
echo "[$(date -u +%H:%M:%SZ)] wheel ok"
# requirements.lock reproduces the resolved set exactly; its vllm URL line has an unencoded '+' that the
# server 404s, so the line is pointed at the sha-verified local wheel (same file).
sed "s#^vllm @ .*#vllm @ file:///workspace/wheels/$W#" /workspace/veritor/vllm-poc/env/requirements.lock > /workspace/wheels/requirements.lock
/workspace/venv/bin/uv pip install --python /workspace/venv/bin/python -r /workspace/wheels/requirements.lock || exit 1
echo "[$(date -u +%H:%M:%SZ)] venv ok: vllm=$(/workspace/venv/bin/python -c 'import vllm,torch,triton; print(vllm.__version__, torch.__version__, triton.__version__)')"
# checkpoints at the frozen revisions
/workspace/venv/bin/python - <<'EOF'
import hashlib, json, sys
from pathlib import Path
from huggingface_hub import snapshot_download
ck = json.load(open('/workspace/veritor/manifests/checkpoints.json'))
bad = 0
for e in ck['checkpoints']:
    if not e['downloaded']:
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

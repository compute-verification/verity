#!/bin/bash
# RC pod bootstrap (vllm-rc-a100, z04dtaigc3y9fi; A100-SXM4-80GB, driver 570.195.03 = CUDA 12.8).
# Same recipe as W3c's A100 pod: the pinned +cu130 torch needs driver >= 580, so the SAME vLLM commit
# (d9105ea80) is installed from its cu129 wheel variant (env/README.md "Deviation: cu129 variant") via
# bootstrap_w3c_cu129.sh, then the checkpoint step of bootstrap_w3c.sh restricted to roles B0 and B1
# (the other `downloaded: true` manifest entries are W4d's 7B/14B/32B rows, ~130 GB, not needed here).
# usage: nohup bash bootstrap_rc.sh > /workspace/logs/bootstrap.log 2>&1 &
set -u
mkdir -p /workspace/logs /workspace/wheels /workspace/hf /workspace/rc/results
export HF_HOME=/workspace/hf
echo "[$(date -u +%H:%M:%SZ)] RC bootstrap start on $(hostname) driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
bash /workspace/veritor/verity_vllm_adapter/pod/bootstrap_w3c_cu129.sh || { echo "VENV FAILED"; exit 1; }
echo "[$(date -u +%H:%M:%SZ)] venv-cu129 ok"
/workspace/venv-cu129/bin/uv pip install --python /workspace/venv-cu129/bin/python -q zstandard pytest || exit 1
/workspace/venv-cu129/bin/python - <<'EOF'
import hashlib, json, sys
from pathlib import Path
from huggingface_hub import snapshot_download
ck = json.load(open('/workspace/veritor/manifests/checkpoints.json'))
bad = 0
for e in ck['checkpoints']:
    if not e['downloaded'] or not (e['role'] == 'B0' or e['role'].startswith('B1')):
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

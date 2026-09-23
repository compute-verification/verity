#!/usr/bin/env bash
# Pod-side (started by pod_up.sh in the background): after the pinned vLLM env reports VLLM-ENV-OK,
#   1. create the torch-free Python 3.12 venv the CPU half needs (/workspace/venv-py312: numpy, safetensors, pytest) — the vLLM venv is
#      Python 3.11 and `verity.ir` uses 3.12 syntax (M6: compare/replay/gates run under this venv on the pod), and
#   2. seed every checkpoint role in $@ from the shipped manifest (manifests/checkpoints.json: repo, revision, local_path,
#      safetensors_sha256) into the pod's HF cache (/workspace/hf) with sha256 verification.  Gated repos need a token and are NOT
#      supported here (credentials never leave the laptop, D55): register an ungated mirror instead.
#
#   bash /workspace/cp/pod_env_extra.sh B0 B1 TINYLLAMA      -> /workspace/cp/env-extra.log lines: PY312-OK, SEED-OK <case> | SEED-FAIL <case>,
#                                                               HIDDEN-GPU-OK <sha256> (R12 X-00: the GPU chunk-tree extension), ENV-EXTRA-DONE
set -uo pipefail
WORK=${WORK:-/workspace}; VENV=$WORK/venv-cu129; PY=$VENV/bin/python; PY312=$WORK/venv-py312/bin/python
SRC=${SRC:-$WORK/cp/veritor}; MANIFEST=$SRC/manifests/checkpoints.json
export HF_HOME=$WORK/hf
log() { echo "[env-extra] $(date -u +%FT%TZ) $*"; }

for i in $(seq 1 240); do grep -q VLLM-ENV-OK $WORK/cp/vllm-env.log 2>/dev/null && break; sleep 10; done
grep -q VLLM-ENV-OK $WORK/cp/vllm-env.log 2>/dev/null || { log "vLLM env not OK after 40 min; continuing with the seeds anyway (system python)"; }

# 1. Python 3.12 venv (uv downloads a managed interpreter when the image lacks 3.12)
if [ ! -x "$PY312" ]; then
  UV=$VENV/bin/uv; [ -x "$UV" ] || { python3 -m pip install -q uv && UV=uv; }
  $UV venv $WORK/venv-py312 --python 3.12 >/dev/null 2>&1 || $UV venv $WORK/venv-py312 --python 3.12
  $UV pip install --python $PY312 -q numpy safetensors pytest huggingface_hub
fi
$PY312 -c "import sys, numpy, safetensors; assert sys.version_info[:2] == (3, 12), sys.version; print('py312', sys.version.split()[0], 'numpy', numpy.__version__)" && log PY312-OK || log PY312-FAIL

# 2. checkpoint seeds
SEEDPY=$PY; [ -x "$SEEDPY" ] && $SEEDPY -c "import huggingface_hub" 2>/dev/null || SEEDPY=$PY312
for CASE in "$@"; do
  ENTRY=$($SEEDPY - "$CASE" "$MANIFEST" <<'EOF'
import json, sys
case, manifest = sys.argv[1], sys.argv[2]
doc = json.load(open(manifest))
cases = {"B0": "HuggingFaceTB/SmolLM2-135M", "B1": "Qwen/Qwen2.5-1.5B", "QWEN05": "Qwen/Qwen2.5-0.5B", "SMOL360": "HuggingFaceTB/SmolLM2-360M", "B7": "Qwen/Qwen2.5-7B"}
ent = None
for c in doc["checkpoints"]:
    if c.get("role") == case or (case in cases and c["repo"] == cases[case] and c.get("local_path")):
        ent = c; break
if ent is None:
    sys.exit(f"no checkpoints.json entry for case {case}")
print(json.dumps({"repo": ent["repo"], "revision": ent["revision"], "local_path": ent["local_path"], "sha": ent.get("safetensors_sha256", {})}))
EOF
  ) || { log "SEED-FAIL $CASE (manifest)"; continue; }
  $SEEDPY - "$ENTRY" <<'EOF' && log "SEED-OK $CASE" || log "SEED-FAIL $CASE"
import hashlib, json, os, sys
from pathlib import Path
e = json.loads(sys.argv[1])
from huggingface_hub import snapshot_download
local = Path(e["local_path"])
if not all((local / f).exists() for f in e["sha"]) or not (local / "config.json").exists():
    p = snapshot_download(e["repo"], revision=e["revision"], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py"],
                          cache_dir=os.path.join(os.environ["HF_HOME"], "hub"))
    print("downloaded", p, flush=True)
ok = True
for f, sha in e["sha"].items():
    h = hashlib.sha256()
    with open(local / f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    good = h.hexdigest() == sha
    ok &= good
    print(("OK  " if good else "BAD ") + f, flush=True)
sys.exit(0 if ok else 1)
EOF
done
rm -rf /root/.cache/uv /root/.cache/pip 2>/dev/null; log "uv/pip wheel caches removed (11 GB dead weight on a 60 GB worker; the venvs are installed)"

# 3. [R12 X-00] the GPU-resident chunk-tree extension `hidden_gpu_tree` that every Commit (`commit_delta.py --gpu-tree`, native_collect_v2b)
#    imports in the worker: JIT-built here for this pod's GPU (~1-2 min) so no lane hits `ModuleNotFoundError: No module named 'hidden_gpu'`.
if [ -f "$SRC/verity_capture/sweep/pod_hidden_gpu.sh" ]; then
  (cd "$SRC" && VENV=$VENV bash verity_capture/sweep/pod_hidden_gpu.sh 2>&1 | grep -E "^\[hidden_gpu\]|^HIDDEN-GPU" | while read -r l; do log "$l"; done)
else
  log "HIDDEN-GPU-SKIP (verity_capture/sweep/pod_hidden_gpu.sh not in the shipped tree)"
fi
# 4. [R13 REL-03] the FA2 matReq hidden-value tap, built from the shipped sources (verity_capture/sweep/fa2_tap_src) for this pod
#    (~5-6 min: clone of the pinned vllm-flash-attn + nvcc) -> FA2-TAP-OK <sha256> <path> arch= hdims=.  Until R13 every pod got the .so by hand.
if [ -f "$SRC/verity_capture/sweep/pod_fa2_tap.sh" ]; then
  (cd "$SRC" && VENV=$VENV bash verity_capture/sweep/pod_fa2_tap.sh 2>&1 | grep -E "^\[fa2_tap\]|^FA2-TAP" | while read -r l; do log "$l"; done)
else
  log "FA2-TAP-SKIP (verity_capture/sweep/pod_fa2_tap.sh not in the shipped tree)"
fi
log ENV-EXTRA-DONE

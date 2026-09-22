#!/usr/bin/env bash
# Pod runtime for the Verity x vLLM integration: ONE Python 3.12 environment (torch cu129, vLLM d9105ea80, numpy, safetensors),
# the native taps (hidden_gpu chunk tree, FA2 matReq tap; FA3 on sm_90), the pinned checkpoints, and a readiness record.
#
#   research run --on <machine> --project verity --source <verity checkout> -- bash -c "cd integrations/vllm && bash pod_bootstrap.sh [--cases B0[,B1]] [--out DIR]"
#
# Idempotent: every step checks before it builds.  Runs from this directory (the shipped tree's integrations/vllm).  Fails closed with
# a named class (BOOTSTRAP_FAIL_<step>) and exits 3; the readiness record (readiness.json) is written whether or not every check passed.
# Env: WORK (/workspace), VENV ($WORK/venv312), HF_HOME ($WORK/hf), FA2_TAP_HDIMS (64,96,128,256), MAX_JOBS (nproc), CUDA_HOME.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P); cd "$ROOT" || exit 3
CORE=$(cd "$ROOT/../../packages/verity/src" 2>/dev/null && pwd -P) || { echo "BOOTSTRAP_FAIL_TREE: no packages/verity/src beside $ROOT (run from the shipped Verity tree)" >&2; exit 3; }
RESEARCH=$(cd "$ROOT/../../tools/research/src" 2>/dev/null && pwd -P) || { echo "BOOTSTRAP_FAIL_TREE: no tools/research/src beside $ROOT" >&2; exit 3; }
WORK=${WORK:-/workspace}; VENV=${VENV:-$WORK/venv312}; PY=$VENV/bin/python
export HF_HOME=${HF_HOME:-$WORK/hf}
export PYTHONPATH=$ROOT:$CORE:$RESEARCH${PYTHONPATH:+:$PYTHONPATH}
export VERITOR_REPO=$ROOT
CASES=B0; OUT=$ROOT/out/bootstrap
while [ $# -gt 0 ]; do case "$1" in --cases) CASES=$2; shift 2;; --out) OUT=$2; shift 2;; *) echo "unknown arg $1" >&2; exit 2;; esac; done
mkdir -p "$OUT" "$WORK/wheels"
log() { echo "[bootstrap] $(date -u +%FT%TZ) $*"; }
fail() { echo "BOOTSTRAP_FAIL_$1: ${*:2}" | tee -a "$OUT/bootstrap.fail" >&2; exit 3; }

# ---- 1. Python 3.12 venv ---------------------------------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install -q uv 2>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
  export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
fi
command -v uv >/dev/null || fail UV "uv not installable"
if [ ! -x "$PY" ]; then
  log "creating $VENV (Python 3.12)"
  uv venv "$VENV" --python 3.12 >/dev/null || fail VENV "uv venv --python 3.12 failed"
fi
$PY -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' || fail PYTHON "$PY is not Python 3.12: $($PY -V 2>&1)"

# ---- 2. vLLM d9105ea80 (cu129 wheel, sha pinned) + torch cu129 + the Python deps ---------------------------------------------------
W="vllm-0.28.1rc1.dev472+gd9105ea80.cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
WSHA=7aa52ac71500caf74643135a340eb2d03ae0386d4282fc0b23846940029fbbb3
if ! $PY -c "import vllm, torch" 2>/dev/null; then
  [ -f "$WORK/wheels/$W" ] || curl -fLo "$WORK/wheels/$W" \
    "https://wheels.vllm.ai/d9105ea8001e0a6d77a96327d17515bb5791fb36/vllm-0.28.1rc1.dev472%2Bgd9105ea80.cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
    || fail WHEEL "vLLM wheel download"
  echo "$WSHA  $WORK/wheels/$W" | sha256sum -c >/dev/null || fail WHEEL_SHA "vLLM wheel sha256 != $WSHA"
  log "installing vLLM + torch cu129 into $VENV"
  # UV_HTTP_TIMEOUT: uv's default 30 s per response timed out twice on a 2x L40S pod fetching nvidia-cuda-cccl-cu12 (22 Sep, vyv-l40s2b,
  # 'operation timed out') while a sibling pod in the same DC installed in 30 s; one retry of the whole install before failing by name.
  export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-600}
  for _try in 1 2; do
    uv pip install --python "$PY" --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match \
      "$WORK/wheels/$W" "numpy>=1.26" safetensors huggingface_hub zstandard pytest ninja >"$OUT/pip.log" 2>&1 && break
    [ "$_try" = 2 ] && fail PIP "$(tail -3 "$OUT/pip.log")"
    log "pip install failed (attempt $_try): $(tail -1 "$OUT/pip.log" | cut -c1-160); retrying"
  done
fi
$PY -c "import ninja" 2>/dev/null || uv pip install --python "$PY" ninja >/dev/null 2>&1
rm -rf /root/.cache/uv /root/.cache/pip 2>/dev/null

# ---- 3. pinned checkpoints (manifests/checkpoints.json; sha256 of every safetensors) --------------------------------------------
IFS=, read -r -a CASE_LIST <<< "$CASES"
for CASE in "${CASE_LIST[@]}"; do
  $PY - "$CASE" manifests/checkpoints.json <<'EOF' >>"$OUT/checkpoints.log" 2>&1 && log "checkpoint OK $CASE" || fail CHECKPOINT "$CASE (see $OUT/checkpoints.log)"
import hashlib, json, os, sys
from pathlib import Path
case, manifest = sys.argv[1], sys.argv[2]
doc = json.load(open(manifest))
cases = {"B0": "HuggingFaceTB/SmolLM2-135M", "B1": "Qwen/Qwen2.5-1.5B"}   # every other case is a `role` in checkpoints.json
ent = next((c for c in doc["checkpoints"] if c.get("role") == case or (case in cases and c["repo"] == cases[case] and c.get("local_path"))), None)
if ent is None:
    sys.exit(f"no checkpoints.json entry for case {case}")
local = Path(ent["local_path"]); shas = ent.get("safetensors_sha256", {})
if not all((local / f).exists() for f in shas) or not (local / "config.json").exists():
    from huggingface_hub import snapshot_download
    p = snapshot_download(ent["repo"], revision=ent["revision"], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py"],
                          cache_dir=os.path.join(os.environ["HF_HOME"], "hub"))
    print("downloaded", p, flush=True)
ok = True
for f, sha in shas.items():
    h = hashlib.sha256()
    with open(local / f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    good = h.hexdigest() == sha; ok &= good
    print(("OK  " if good else "BAD ") + str(local / f), flush=True)
sys.exit(0 if ok else 1)
EOF
done

# ---- 4. native taps (built by the scripts the sweep already uses; VENV points them at the 3.12 environment) ----------------------
CC=$($PY -c "import torch; cc=torch.cuda.get_device_capability(0); print(f'{cc[0]}.{cc[1]}')" 2>/dev/null) || fail CUDA "torch sees no CUDA device ($($PY -c 'import torch;print(torch.__version__)' 2>&1))"
export MAX_JOBS=${MAX_JOBS:-$(nproc)}
VENV=$VENV bash verity_capture/sweep/pod_hidden_gpu.sh >"$OUT/hidden_gpu.log" 2>&1; grep -q '^HIDDEN-GPU-OK' "$OUT/hidden_gpu.log" || fail HIDDEN_GPU "$(tail -2 "$OUT/hidden_gpu.log")"
HG_LINE=$(grep '^HIDDEN-GPU-OK' "$OUT/hidden_gpu.log" | tail -1); log "$HG_LINE"
VENV=$VENV FA2_TAP_HDIMS=${FA2_TAP_HDIMS:-64,96,128,256} bash verity_capture/sweep/pod_fa2_tap.sh >"$OUT/fa2_tap.log" 2>&1; grep -q '^FA2-TAP-OK' "$OUT/fa2_tap.log" || fail FA2_TAP "$(tail -2 "$OUT/fa2_tap.log")"
log "$(grep '^FA2-TAP-OK' "$OUT/fa2_tap.log" | tail -1)"
if [ "$CC" = "9.0" ]; then
  VENV=$VENV bash verity_capture/sweep/pod_fa3_tap.sh >"$OUT/fa3_tap.log" 2>&1; grep -q '^FA3-TAP-OK' "$OUT/fa3_tap.log" || fail FA3_TAP "$(tail -2 "$OUT/fa3_tap.log")"
  log "$(grep '^FA3-TAP-OK' "$OUT/fa3_tap.log" | tail -1)"
fi

# ---- 5. readiness: prove the environment mechanically and record the identities --------------------------------------------------
HG_SO=$(echo "$HG_LINE" | awk '{print $3}')
$PY - "$ROOT" "$CORE" "$OUT/readiness.json" "$HG_SO" "$CC" "$RESEARCH" <<'EOF' || fail READINESS "see $OUT/readiness.json"
import hashlib, importlib, json, os, platform, subprocess, sys
root, core, out, hg_so, cc, research = sys.argv[1:7]
r = {"schema": "verity-vllm/readiness/v1", "checks": {}, "identities": {}}
def check(name, fn):
    try:
        r["checks"][name] = {"ok": True, "value": fn()}
    except Exception as e:  # noqa: BLE001
        r["checks"][name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
def sha(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read()); return h.hexdigest()
def under(mod, top):
    m = importlib.import_module(mod); f = os.path.realpath(m.__file__)
    if not f.startswith(os.path.realpath(top) + os.sep):
        raise RuntimeError(f"{mod} resolved from {f}, not from {top}")
    return f
check("python_3_12", lambda: (sys.version_info[:2] == (3, 12)) or (_ for _ in ()).throw(RuntimeError(sys.version)))
check("torch_cuda", lambda: {"torch": __import__("torch").__version__, "cuda": __import__("torch").version.cuda, "device": __import__("torch").cuda.get_device_name(0), "cc": cc})
check("vllm_import", lambda: __import__("vllm").__version__)
check("vllm_commit_d9105ea80", lambda: ("d9105ea80" in __import__("vllm").__version__) or (_ for _ in ()).throw(RuntimeError(__import__("vllm").__version__)))
check("verity_from_tree", lambda: under("verity.ir", core))
check("verity_vllm_from_tree", lambda: under("verity_vllm.registry", root))
check("verity_capture_from_tree", lambda: under("verity_capture", root))
check("research_telemetry_from_tree", lambda: under("research.telemetry", research))
check("numpy_import", lambda: __import__("numpy").__version__)
check("no_old_tree_on_path", lambda: [p for p in sys.path if "/workspace/cp/veritor" in p or p.endswith("/vllm-poc")] == [] or (_ for _ in ()).throw(RuntimeError(str(sys.path))))
check("hidden_gpu_so", lambda: {"path": hg_so, "sha256": sha(hg_so)})
fa2 = "/workspace/cp/fa2/build/matReq/verity_fa2_matReq.so"
def fa2_check():
    import torch
    torch.ops.load_library(fa2); ns = torch.ops.verity_fa2_matReq
    return {"path": fa2, "sha256": sha(fa2), "tap_mode": int(ns.verity_tap_mode()), "hdims_mask": int(ns.verity_tap_hdims())}
check("fa2_tap_so", fa2_check)
if cc == "9.0":
    fa3 = "/workspace/cp/fa2/build/fa3_matReq/verity_fa3_matReq.so"
    check("fa3_tap_so", lambda: {"path": fa3, "sha256": sha(fa3)})
def nvsmi(q):
    return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
r["identities"] = {"python": sys.version.split()[0], "executable": sys.executable, "platform": platform.platform(),
                   "torch": r["checks"]["torch_cuda"].get("value"), "vllm": r["checks"]["vllm_import"].get("value"),
                   "driver": nvsmi("driver_version"), "gpu": nvsmi("name"), "cuda_home": os.environ.get("CUDA_HOME"),
                   "nvcc": subprocess.run(["bash", "-lc", "nvcc --version | tail -1"], capture_output=True, text=True).stdout.strip(),
                   "tree": {"integration_root": root, "verity_core": core, "research": research},
                   "native": {k: r["checks"][k].get("value") for k in ("hidden_gpu_so", "fa2_tap_so", "fa3_tap_so") if k in r["checks"]}}
r["ok"] = all(c["ok"] for c in r["checks"].values())
# ---- measurements [r19-int-20]: the pod's non-GPU admission resources, MEASURED at bootstrap and recorded beside the checks.  Advisory:
# none fails the bootstrap (a slow or small pod can still compute); the launcher and the attempt's conditions read them.  Why each:
#   network  -- vyv-l40s2b (EUR-IS-2) moved 2.6 MB/s to /vault and 6-8 MiB/s to R2 while its neighbours did 84-287 MiB/s (F-r19-mirror-3;
#               R2 probe 16:30Z): a 30 GB row landed on the one pod that could not ship its evidence.  One 64 MiB download + one 16 MiB
#               upload against Cloudflare's public speed endpoint (a User-Agent is required: the default python-urllib UA is refused 403).
#   disk     -- the container overlay is an unmodelled admission resource (F-r19-int-14): l40s4 filled under two rows' captures.
#   memory   -- the cgroup ceiling, not the host's RAM, is what a Match / Commit is admitted against (351 vs 1511 GiB on tp2).
import shutil, time, urllib.request
UA = {"User-Agent": "verity-pod-readiness/1 (python-urllib)"}
def net(down_bytes=64 << 20, up_bytes=16 << 20):
    t = time.monotonic(); got = 0
    with urllib.request.urlopen(urllib.request.Request(f"https://speed.cloudflare.com/__down?bytes={down_bytes}", headers=UA), timeout=90) as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            got += len(b)
    down = got / (time.monotonic() - t) / 1e6
    data = os.urandom(up_bytes); t = time.monotonic()
    urllib.request.urlopen(urllib.request.Request("https://speed.cloudflare.com/__up", data=data, method="POST", headers=UA), timeout=90).read()
    up = up_bytes / (time.monotonic() - t) / 1e6
    return {"down_MBps": round(down, 1), "up_MBps": round(up, 1), "probe": "speed.cloudflare.com", "bytes": {"down": got, "up": up_bytes}}
def disk(path="/workspace"):
    u = shutil.disk_usage(path)
    return {"path": path, "total_bytes": u.total, "free_bytes": u.free}
def memory():
    m = {"total_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"), "cgroup": None, "cgroup_max_bytes": None}
    for p, k in (("/sys/fs/cgroup/memory.max", "v2"), ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "v1")):
        try:
            v = open(p).read().strip(); m["cgroup"] = k; m["cgroup_max_bytes"] = None if v == "max" else int(v); break
        except OSError:
            pass
    return m
r["measurements"] = {}
for name, fn in (("network", net), ("disk", disk), ("memory", memory)):
    try:
        r["measurements"][name] = fn()
    except Exception as e:  # noqa: BLE001
        r["measurements"][name] = {"error": f"{type(e).__name__}: {e}"}
n = r["measurements"]["network"]
if "error" not in n and (n["down_MBps"] < 20 or n["up_MBps"] < 20):
    print(f"WARN slow pod network: down {n['down_MBps']} MB/s up {n['up_MBps']} MB/s (a 30 GB capture ships in "
          f"{30000 / max(n['up_MBps'], 0.1) / 60:.0f} min at this uplink) -- advisory, recorded in readiness.json")
json.dump(r, open(out, "w"), indent=1)
print(json.dumps({k: v["ok"] for k, v in r["checks"].items()}))
print("measurements " + json.dumps(r["measurements"]))
sys.exit(0 if r["ok"] else 1)
EOF
log "READY $OUT/readiness.json"
echo BOOTSTRAP-OK

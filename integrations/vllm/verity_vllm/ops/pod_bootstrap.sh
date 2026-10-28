#!/usr/bin/env bash
# Pod runtime for the Verity x vLLM integration (research/infra; no Verity semantics): ONE Python 3.12 environment (torch cu129, vLLM
# d9105ea80, numpy, safetensors),
# the native taps (hidden_gpu chunk tree, FA2 matReq tap; FA3 on sm_90; the default-on workload taps), the pinned checkpoints, and a readiness record.
#
#   research run --on <machine> --project verity --source <verity checkout> -- bash -c "cd integrations/vllm && bash verity_vllm/ops/pod_bootstrap.sh [--cases B0[,B1]] [--out DIR] [--cpu|--gpu] [--dry-run]"
#
# Idempotent: every step checks before it builds.  Runs from this directory (the shipped tree's integrations/vllm).  Fails closed with
# a named class (BOOTSTRAP_FAIL_<step>) and exits 3; the readiness record (readiness.json) is written whether or not every check passed.
# Env: WORK (/workspace), VENV ($WORK/venv312), HF_HOME ($WORK/hf), FA2_TAP_HDIMS (64,96,128,256), MAX_JOBS (nproc), CUDA_HOME,
# ROW_TAPS (1: build the taps a workload runs by default -- the guarded-max FA2 / FA3 builds, the norm and router taps; 0 skips
# them, and a workload on this pod then needs GUARDED_MAX_TAP=0 NORM_TAP=0 ROUTER_TAP=0).
#
# PROTOCOL SUMMARY (pod readiness; runs once per pod, before any row_pod.sh)
#   Purpose: make a fresh pod able to run workloads from THIS shipped tree: venv312 (Python 3.12) + the sha256-pinned vLLM d9105ea80 cu129
#     wheel + torch cu129, the pinned checkpoints (manifests/checkpoints.json: sha256 per safetensors shard), the native taps
#     (hidden_gpu chunk tree, FA2 matReq tap, FA3 tap on cc 9.0) and a mechanical readiness record.
#   Inputs: --cases B0[,B1] (checkpoint roles of checkpoints.json), --out DIR, --cpu | --gpu (force a mode), --dry-run; the env above.
#   Modes: gpu (default when `nvidia-smi` is on PATH) runs every step.  cpu (`--cpu`, or auto-selected when `nvidia-smi` is absent)
#     runs steps 1-3 unchanged, skips the native taps (step 4), and writes the same readiness record with the
#     GPU checks (torch_cuda, hidden_gpu_so, fa2_tap_so, fa3_tap_so) replaced by torch_cpu, plus "mode": "cpu",
#     "cuda_available": <torch.cuda.is_available()> and a cpu{nproc, model} measurement; `ok` = AND of the CPU checks.  The gpu-mode
#     record is unchanged (no "mode" key = gpu).  `--dry-run` prints the resolved mode and the steps it would run, touches nothing, exits 0.
#   Outputs: $OUT/readiness.json (schema verity-vllm/readiness/v1: checks{ok, value|error}, identities, measurements network/disk/memory),
#     $OUT/{pip,checkpoints,hidden_gpu,fa2_tap,fa3_tap,fa2_tap_guarded,fa3_tap_guarded,norm_tap,router_tap}.log, $OUT/bootstrap.fail on a
#     failure, $OUT/staged-<case> for a checkpoint this run staged (and its `staged` label in the store); the last stdout line is BOOTSTRAP-OK.
#   Invariants: idempotent (every step checks before it builds); a failing step exits 3 with the named class BOOTSTRAP_FAIL_<STEP>;
#     readiness.json is written BEFORE its verdict is taken (a failed check still leaves the record; `ok` = AND of the checks);
#     every Verity package must import from this tree (verity_from_tree, verity_vllm_from_tree,
#     no_old_tree_on_path); the network/disk/cgroup-memory measurements are advisory and never fail the bootstrap.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P); cd "$ROOT" || exit 3
CORE=$(cd "$ROOT/../.." 2>/dev/null && pwd -P) || { echo "BOOTSTRAP_FAIL_TREE: no Verity tree above $ROOT (run from the shipped Verity tree)" >&2; exit 3; }
RESEARCH=$(cd "$ROOT/../../tools/research/src" 2>/dev/null && pwd -P) || { echo "BOOTSTRAP_FAIL_TREE: no tools/research/src beside $ROOT" >&2; exit 3; }
[ -d "$CORE/verity/core" ] || { echo "BOOTSTRAP_FAIL_TREE: no verity/core beside $ROOT" >&2; exit 3; }
WORK=${WORK:-/workspace}; VENV=${VENV:-$WORK/venv312}; PY=$VENV/bin/python
export HF_HOME=${HF_HOME:-$WORK/hf}
export PYTHONPATH=$ROOT:$CORE:$RESEARCH${PYTHONPATH:+:$PYTHONPATH}
export VERITY_TREE=$ROOT
CASES=B0; OUT=$ROOT/out/bootstrap; MODE=auto; DRY_RUN=0
while [ $# -gt 0 ]; do case "$1" in --cases) CASES=$2; shift 2;; --out) OUT=$2; shift 2;; --cpu) MODE=cpu; shift;; --gpu) MODE=gpu; shift;; --dry-run) DRY_RUN=1; shift;; *) echo "unknown arg $1" >&2; exit 2;; esac; done
if [ "$MODE" = auto ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then MODE=gpu; else MODE=cpu; echo "[bootstrap] no nvidia-smi on PATH: cpu mode (--gpu forces the GPU path, --cpu forces this one)"; fi
fi
if [ "$DRY_RUN" = 1 ]; then
  echo "mode=$MODE cases=$CASES out=$OUT venv=$VENV"
  echo "steps: 1 venv312; 2 vLLM d9105ea80 + torch cu129; 3 checkpoints $CASES; $([ "$MODE" = gpu ] && echo "4 native taps (hidden_gpu, fa2_tap, fa3_tap on cc 9.0; row taps ROW_TAPS=${ROW_TAPS:-1}); " || echo '4 native taps SKIPPED (cpu mode); ')5 readiness.json ($MODE)"
  echo BOOTSTRAP-DRY-RUN; exit 0
fi
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
# torchcodec from PyPI (0.17.0, built for another torch) breaks `import vllm` and cuteDSL on H200: the cu129 build
TORCHCODEC="torchcodec==0.16.0+cu129"
if ! $PY -c "import vllm, torch" 2>/dev/null; then
  [ -f "$WORK/wheels/$W" ] || curl -fLo "$WORK/wheels/$W" \
    "https://wheels.vllm.ai/d9105ea8001e0a6d77a96327d17515bb5791fb36/vllm-0.28.1rc1.dev472%2Bgd9105ea80.cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
    || fail WHEEL "vLLM wheel download"
  echo "$WSHA  $WORK/wheels/$W" | sha256sum -c >/dev/null || fail WHEEL_SHA "vLLM wheel sha256 != $WSHA"
  log "installing vLLM + torch cu129 into $VENV"
  # UV_HTTP_TIMEOUT: uv's default 30 s per response can time out on a slow pod fetching nvidia-cuda-cccl-cu12 ('operation timed out')
  # where a sibling pod installs in 30 s; one retry of the whole install before failing by name.
  export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-600}
  for _try in 1 2; do
    uv pip install --python "$PY" --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match \
      "$WORK/wheels/$W" "$TORCHCODEC" "numpy>=1.26" safetensors huggingface_hub zstandard pytest ninja >"$OUT/pip.log" 2>&1 && break
    [ "$_try" = 2 ] && fail PIP "$(tail -3 "$OUT/pip.log")"
    log "pip install failed (attempt $_try): $(tail -1 "$OUT/pip.log" | cut -c1-160); retrying"
  done
fi
$PY -c "import ninja" 2>/dev/null || uv pip install --python "$PY" ninja >/dev/null 2>&1
uv pip show --python "$PY" torchcodec 2>/dev/null | grep -qx "Version: ${TORCHCODEC#*==}" \
  || uv pip install --python "$PY" --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match "$TORCHCODEC" \
  >>"$OUT/pip.log" 2>&1 || fail PIP "$TORCHCODEC: $(tail -1 "$OUT/pip.log")"
# the `verity-vllm` launcher ([project.scripts] entry point); verity_vllm resolves from PYTHONPATH like every process on the pod
printf '#!%s\nimport sys\nfrom verity_vllm.pipeline.cli import main\nsys.exit(main())\n' "$PY" > "$VENV/bin/verity-vllm" \
  && chmod +x "$VENV/bin/verity-vllm" || fail VENV "cannot write $VENV/bin/verity-vllm"
rm -rf /root/.cache/uv /root/.cache/pip 2>/dev/null

# ---- 3. pinned checkpoints (manifests/checkpoints.json; sha256 of every safetensors) --------------------------------------------
# A checkpoint this run downloads and checks is staged: its staging event goes to the store (`verity-vllm records staged`) through
# the run's custody key; with no key, the log names the command that records it.
IFS=, read -r -a CASE_LIST <<< "$CASES"
for CASE in "${CASE_LIST[@]}"; do
  rm -f "$OUT/staged-$CASE"
  $PY - "$CASE" manifests/checkpoints.json "$OUT/staged-$CASE" <<'EOF' >>"$OUT/checkpoints.log" 2>&1 && log "checkpoint OK $CASE" || fail CHECKPOINT "$CASE (see $OUT/checkpoints.log)"
import hashlib, json, os, sys, time
from pathlib import Path
case, manifest, staged = sys.argv[1], sys.argv[2], Path(sys.argv[3])
from verity_vllm.config import checkpoint_entries   # the manifest and its sibling manifests (checkpoints-int8.json)
entries = checkpoint_entries(manifest)
cases = {"B0": "HuggingFaceTB/SmolLM2-135M", "B1": "Qwen/Qwen2.5-1.5B", "B7": "Qwen/Qwen2.5-7B"}   # every other case is a `role` in checkpoints.json
ent = next((c for c in entries if c.get("role") == case or (case in cases and c["repo"] == cases[case] and c.get("local_path"))), None)
if ent is None:
    sys.exit(f"no checkpoints.json entry for case {case}")
local = Path(ent["local_path"]); shas = ent.get("safetensors_sha256", {})
fetched = not all((local / f).exists() for f in shas) or not (local / "config.json").exists()
if fetched:
    from huggingface_hub import snapshot_download
    fetch = lambda e: snapshot_download(e["repo"], revision=e["revision"], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py"],
                                        cache_dir=os.path.join(os.environ["HF_HOME"], "hub"))
    if ent.get("recipe"):          # no hub repo holds a recipe pin: made here from its BF16 pin (engine/fp8_checkpoint.py, int8_checkpoint.py)
        from verity_vllm.engine import fp8_checkpoint as F, int8_checkpoint as I8
        recipe = {r.id: r for r in (F.FP8, *I8.RECIPES.values())}[ent["recipe"]["id"]]
        base = F.base_entry(ent, entries, recipe)
        if not all((Path(base["local_path"]) / f).exists() for f in base["safetensors_sha256"]):
            print("downloaded", fetch(base), flush=True)
        made, bad = F.remake(ent, entries, recipe=recipe)
        print("made", made["local_path"], "mismatch", bad, flush=True)
    else:
        print("downloaded", fetch(ent), flush=True)
ok = True
for f, sha in shas.items():
    h = hashlib.sha256()
    with open(local / f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    good = h.hexdigest() == sha; ok &= good
    print(("OK  " if good else "BAD ") + str(local / f), flush=True)
if ok and fetched:
    staged.write_text(f"{ent['repo']} {ent['revision']} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {local}\n")
sys.exit(0 if ok else 1)
EOF
  if [ -s "$OUT/staged-$CASE" ]; then
    read -r S_REPO S_REV S_AT S_DIR < "$OUT/staged-$CASE"
    S_REF=${RESEARCH_RUN_ID:-$(hostname):$S_DIR}; S_BY=${RESEARCH_LANE:-pod-bootstrap}
    C=${CUSTODY_DIR:-$(dirname "$(dirname "${RESEARCH_RUN_DIR:-/none/runs/x}")")/requests/${RESEARCH_RUN_ID:-none}/custody}
    if [ -f "$C/cred.json" ] && (export RESEARCH_STORE_CONFIG=$C/store.toml && eval "$("$VENV/bin/verity-vllm" epoch custody-env "$C")" \
        && "$VENV/bin/verity-vllm" records staged "$S_REPO" "$S_REV" --at "$S_AT" --ref "$S_REF" --by "$S_BY") >>"$OUT/checkpoints.log" 2>&1; then
      log "staged $CASE at $S_AT: recorded on ckpt:$S_REPO@$S_REV"
    else
      log "STAGED-UNRECORDED $CASE: verity-vllm records staged $S_REPO $S_REV --at $S_AT --ref $S_REF --by $S_BY"
    fi
  fi
done

# ---- 4. native taps (built by the scripts the sweep already uses; VENV points them at the 3.12 environment) ----------------------
if [ "$MODE" = gpu ]; then
CC=$($PY -c "import torch; cc=torch.cuda.get_device_capability(0); print(f'{cc[0]}.{cc[1]}')" 2>/dev/null) || fail CUDA "torch sees no CUDA device ($($PY -c 'import torch;print(torch.__version__)' 2>&1))"
export MAX_JOBS=${MAX_JOBS:-$(nproc)}
VENV=$VENV bash verity_vllm/ops/pod_hidden_gpu.sh >"$OUT/hidden_gpu.log" 2>&1; grep -q '^HIDDEN-GPU-OK' "$OUT/hidden_gpu.log" || fail HIDDEN_GPU "$(tail -2 "$OUT/hidden_gpu.log")"
HG_LINE=$(grep '^HIDDEN-GPU-OK' "$OUT/hidden_gpu.log" | tail -1); log "$HG_LINE"
VENV=$VENV FA2_TAP_HDIMS=${FA2_TAP_HDIMS:-64,96,128,256} bash verity_vllm/ops/pod_fa2_tap.sh >"$OUT/fa2_tap.log" 2>&1; grep -q '^FA2-TAP-OK' "$OUT/fa2_tap.log" || fail FA2_TAP "$(tail -2 "$OUT/fa2_tap.log")"
log "$(grep '^FA2-TAP-OK' "$OUT/fa2_tap.log" | tail -1)"
if [ "$CC" = "9.0" ]; then
  VENV=$VENV bash verity_vllm/ops/pod_fa3_tap.sh >"$OUT/fa3_tap.log" 2>&1; grep -q '^FA3-TAP-OK' "$OUT/fa3_tap.log" || fail FA3_TAP "$(tail -2 "$OUT/fa3_tap.log")"
  log "$(grep '^FA3-TAP-OK' "$OUT/fa3_tap.log" | tail -1)"
fi
if [ "${ROW_TAPS:-1}" = 1 ]; then       # the taps config.CommitConfig runs by default, at the paths cli.MACHINE names
  VENV=$VENV FA2_TAP_ROW_GUARD=1 FA2_TAP_HDIMS=${FA2_TAP_HDIMS:-64,96,128,256} bash verity_vllm/ops/pod_fa2_tap.sh >"$OUT/fa2_tap_guarded.log" 2>&1; grep -q '^FA2-TAP-OK' "$OUT/fa2_tap_guarded.log" || fail FA2_TAP_GUARDED "$(tail -2 "$OUT/fa2_tap_guarded.log")"
  log "$(grep '^FA2-TAP-OK' "$OUT/fa2_tap_guarded.log" | tail -1)"
  if [ "$CC" = "9.0" ]; then
    VENV=$VENV FA3_TAP_ROW_GUARD=1 bash verity_vllm/ops/pod_fa3_tap.sh >"$OUT/fa3_tap_guarded.log" 2>&1; grep -q '^FA3-TAP-OK' "$OUT/fa3_tap_guarded.log" || fail FA3_TAP_GUARDED "$(tail -2 "$OUT/fa3_tap_guarded.log")"
    log "$(grep '^FA3-TAP-OK' "$OUT/fa3_tap_guarded.log" | tail -1)"
  fi
  VENV=$VENV bash verity_vllm/ops/pod_norm_tap.sh >"$OUT/norm_tap.log" 2>&1; grep -q '^NORM-TAP-OK' "$OUT/norm_tap.log" || fail NORM_TAP "$(tail -2 "$OUT/norm_tap.log")"
  log "$(grep '^NORM-TAP-OK' "$OUT/norm_tap.log" | tail -1)"
  VENV=$VENV bash verity_vllm/ops/pod_router_tap.sh >"$OUT/router_tap.log" 2>&1; grep -q '^ROUTER-TAP-OK' "$OUT/router_tap.log" || fail ROUTER_TAP "$(tail -2 "$OUT/router_tap.log")"
  log "$(grep '^ROUTER-TAP-OK' "$OUT/router_tap.log" | tail -1)"
fi
else
CC=""; HG_LINE=""; log "cpu mode: native taps (hidden_gpu, fa2_tap, fa3_tap, row taps) skipped"
fi

# ---- 5. readiness: prove the environment mechanically and record the identities --------------------------------------------------
HG_SO=$(echo "$HG_LINE" | awk '{print $3}')
$PY - "$ROOT" "$CORE" "$OUT/readiness.json" "$HG_SO" "$CC" "$RESEARCH" "$MODE" "$CORE/verity/core" <<'EOF' || fail READINESS "see $OUT/readiness.json"
import hashlib, importlib, json, os, platform, subprocess, sys
root, core, out, hg_so, cc, research, mode, verifiers = sys.argv[1:9]
gpu = mode == "gpu"
r = {"schema": "verity-vllm/readiness/v1", "checks": {}, "identities": {}}
if not gpu:
    r["mode"] = mode
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
if gpu:
    check("torch_cuda", lambda: {"torch": __import__("torch").__version__, "cuda": __import__("torch").version.cuda, "device": __import__("torch").cuda.get_device_name(0), "cc": cc})
else:  # cpu mode: torch must import and run on the host; whether it also sees a CUDA device is recorded, not required
    check("torch_cpu", lambda: {"torch": __import__("torch").__version__, "cuda_available": bool(__import__("torch").cuda.is_available()), "threads": __import__("torch").get_num_threads()})
    r["cuda_available"] = bool((r["checks"]["torch_cpu"].get("value") or {}).get("cuda_available", False))
check("vllm_import", lambda: __import__("vllm").__version__)
check("vllm_commit_d9105ea80", lambda: ("d9105ea80" in __import__("vllm").__version__) or (_ for _ in ()).throw(RuntimeError(__import__("vllm").__version__)))
check("verity_from_tree", lambda: under("verity.ml.circuits", core))
check("verity_vllm_from_tree", lambda: under("verity_vllm.program", root))
check("research_telemetry_from_tree", lambda: under("research.telemetry", research))
check("sampled_proofs_from_tree", lambda: under("verity.core.protocols.sampled_proofs.one_stage.partition", verifiers))   # the program tests import it
def distributions_from_tree():
    # the catalog and the kernels, distributions of their own before the platform layout, and the verifiers, now in verity
    mods = {"catalog": "verity.ml.circuits.ops", "kernels": "verity.ml.kernels", "core": "verity.core"}
    return {d: under(m, os.path.join(core, *m.split("."))) for d, m in mods.items()}
check("distributions_from_tree", distributions_from_tree)
check("numpy_import", lambda: __import__("numpy").__version__)
check("no_old_tree_on_path", lambda: [p for p in sys.path if "/workspace/cp/veritor" in p or p.endswith("/vllm-poc")] == [] or (_ for _ in ()).throw(RuntimeError(str(sys.path))))
if gpu:
    check("hidden_gpu_so", lambda: {"path": hg_so, "sha256": sha(hg_so)})
fa2 = "/workspace/cp/fa2/build/matReq/verity_fa2_matReq.so"
def fa2_check():
    import torch
    torch.ops.load_library(fa2); ns = torch.ops.verity_fa2_matReq
    return {"path": fa2, "sha256": sha(fa2), "tap_mode": int(ns.verity_tap_mode()), "hdims_mask": int(ns.verity_tap_hdims())}
if gpu:
    check("fa2_tap_so", fa2_check)
if cc == "9.0":
    fa3 = "/workspace/cp/fa2/build/fa3_matReq/verity_fa3_matReq.so"
    check("fa3_tap_so", lambda: {"path": fa3, "sha256": sha(fa3)})
def nvsmi(q):
    if not gpu:  # cpu mode: a pod without nvidia-smi records null, it does not fail the readiness record
        try:
            return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None
    return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
r["identities"] = {"python": sys.version.split()[0], "executable": sys.executable, "platform": platform.platform(),
                   "torch": r["checks"]["torch_cuda" if gpu else "torch_cpu"].get("value"), "vllm": r["checks"]["vllm_import"].get("value"),
                   "driver": nvsmi("driver_version"), "gpu": nvsmi("name"), "cuda_home": os.environ.get("CUDA_HOME"),
                   "nvcc": subprocess.run(["bash", "-lc", "nvcc --version | tail -1"], capture_output=True, text=True).stdout.strip(),
                   "tree": {"integration_root": root, "verity_core": core, "research": research},
                   "native": {k: r["checks"][k].get("value") for k in ("hidden_gpu_so", "fa2_tap_so", "fa3_tap_so") if k in r["checks"]}}
r["ok"] = all(c["ok"] for c in r["checks"].values())
# ---- measurements: the pod's non-GPU admission resources, MEASURED at bootstrap and recorded beside the checks.  Advisory:
# none fails the bootstrap (a slow or small pod can still compute); the launcher and the attempt's conditions read them.  Why each:
#   network  -- one pod can move a few MB/s to /vault and R2 while its neighbours do hundreds, and a 30 GB workload placed there can't
#               ship its evidence.  One 64 MiB download + one 16 MiB
#               upload against Cloudflare's public speed endpoint (a User-Agent is required: the default python-urllib UA is refused 403).
#   disk     -- the container overlay is an unmodelled admission resource: two workloads' captures can fill it.
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
def cpu():
    model = subprocess.run(["bash", "-lc", "lscpu 2>/dev/null | grep 'Model name' | sed 's/.*: *//'"], capture_output=True, text=True).stdout.strip()
    return {"nproc": os.cpu_count(), "affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None, "model": model or None}
r["measurements"] = {}
for name, fn in (("network", net), ("disk", disk), ("memory", memory)) + ((("cpu", cpu),) if not gpu else ()):
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

#!/usr/bin/env bash
# A row's checkpoint staged on a GPU node from the hub (model.sh: REPO at REV into DST, LOCAL_CK by default): every file of the pin
# in checkpoints.json as a real file (`snapshot_download --local-dir`), each checked against its sha256 there (the shards' hub LFS
# records, the rest hashed from their bytes), then DST/.verified, which `ckpt_bind` (match_gpu.sh, commit.sh) serves at FS_CK, the
# pin's local_path.  A file already in DST at its size (the pin's, else the hub's at REV) and sha256 is kept, so a rerun after an
# eviction fetches only the rest.
# ONLY=GLOB stages just the matching files and writes no .verified: a rehearsal of the rate ($OUT/stage.json: GB, download and
# check seconds, MB/s).  Refused (exit 3) unless /workspace has room for what is left to fetch.
#   research run --on NODE --mem-gb 32 --timeout 4h --source . --cwd source --project verity -- env ROW=... bash integrations/vllm/manifests/glm_v2_tree/jobs/stage_ckpt.sh
# env: ROW, [DST=LOCAL_CK], [ONLY], [PY=/workspace/cp/venv312/bin/python], [JOBS=16 (files hashed at once)], [HF_MAX_WORKERS=16].
set -uo pipefail
: "${ROW:?ROW}"
TREE=$PWD
source "$TREE/integrations/vllm/manifests/glm_v2_tree/jobs/model.sh"
DST=${DST:-$LOCAL_CK}
PY=${PY:-/workspace/cp/venv312/bin/python}
[ -x "$PY" ] || { echo "no $PY: pod_bootstrap.sh makes it"; exit 3; }
mkdir -p "$DST" || exit 3
df -B1 --output=avail "$DST" | tail -1 >"$OUT/avail_bytes.txt"
HF_HUB_OFFLINE= HF_HUB_ENABLE_HF_TRANSFER= "$PY" - "$REPO" "$REV" "$DST" "$TREE/integrations/vllm/manifests/checkpoints.json" "${ONLY:-}" \
  "${JOBS:-16}" "$(cat "$OUT/avail_bytes.txt")" <<'EOF' | tee "$OUT/stage.log"
import concurrent.futures as cf, fnmatch, hashlib, json, os, sys, time
repo, rev, dst, pins, only, jobs, avail = sys.argv[1:8]
pin = [c for c in json.load(open(pins))["checkpoints"] if c["repo"] == repo and c["revision"] == rev][0]
files = {n: f for n, f in pin["files"].items() if not only or fnmatch.fnmatch(n, only)}
if not files:
    sys.exit(f"{repo}@{rev}: no file matches ONLY={only}")

def sha(n):
    h = hashlib.sha256()
    with open(os.path.join(dst, n), "rb") as fh:
        for c in iter(lambda: fh.read(1 << 24), b""):
            h.update(c)
    return n, h.hexdigest()

def check(names):
    with cf.ThreadPoolExecutor(int(jobs)) as ex:
        return dict(ex.map(sha, names))

size = {n: f["size"] for n, f in files.items() if "size" in f}
if len(size) < len(files):
    from huggingface_hub import HfApi
    hub = {s.rfilename: s.size for s in HfApi().model_info(repo, revision=rev, files_metadata=True).siblings}
    size.update({n: hub[n] for n in files if n not in size})
there = [n for n in files if os.path.isfile(os.path.join(dst, n)) and os.path.getsize(os.path.join(dst, n)) == size[n]]
good = {n for n, h in check(there).items() if h == files[n]["sha256"]}
todo = sorted(set(files) - good)
need = sum(size[n] for n in todo)
print(f"{len(good)} of {len(files)} files in {dst} already checkpoints.json's; fetching {len(todo)} ({need / 1e9:.1f} GB)", flush=True)
if need > int(avail) * 0.98:
    print(f"{dst}: {int(avail) / 1e9:.1f} GB free, {need / 1e9:.1f} GB to fetch"); sys.exit(3)
t0 = time.time()
if todo:
    from huggingface_hub import snapshot_download
    snapshot_download(repo, revision=rev, local_dir=dst, allow_patterns=todo, max_workers=int(os.environ.get("HF_MAX_WORKERS") or 16))
dl = time.time() - t0
got = check(todo)
bad = sorted(n for n, h in got.items() if h != files[n]["sha256"])
rec = {"repo": repo, "revision": rev, "dir": dst, "files": len(files), "fetched": len(todo), "fetched_gb": round(need / 1e9, 1),
       "download_s": round(dl), "MBps": round(need / 1e6 / dl) if todo and dl else None, "check_s": round(time.time() - t0 - dl), "bad": bad}
print(json.dumps(rec))
sys.exit(1 if bad else 0)
EOF
rc=${PIPESTATUS[0]}
tail -1 "$OUT/stage.log" >"$OUT/stage.json"
[ "$rc" = 0 ] || { echo "staging failed (rc $rc)"; exit "$rc"; }
if [ -z "${ONLY:-}" ]; then
  echo "$REPO $REV $(date -u +%FT%TZ) ${RESEARCH_RUN_ID:-} every file checkpoints.json's" >"$DST/.verified"
  [ "$DST" != "$LOCAL_CK" ] || ckpt_bind
fi

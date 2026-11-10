#!/usr/bin/env bash
# neutrality_tier.sh TIER OUT: `verity-vllm neutrality --tier TIER` from this tree on a node with no vLLM of its own, its report and
# each row's logs and build summary in OUT (`post_train.py` runs the fast tier after each landing).
#
#   bash tools/verity/check/neutrality_tier.sh fast "$out/neutrality"
#
# Made once a node, under a lock, in /workspace/tmp/neutrality-vllm: the pinned vLLM's venv
# (`pod_bootstrap.sh --cpu --cases ''`), each of the tier's rows' checkpoint metadata (a hub snapshot of every file but the
# safetensors shards: the derive runs on the meta device), and, on a node with no driver, libcuda.so.1 out of the driver package
# (vLLM's CUDA platform loads it at import; neutrality hides every device).  Each row's stored workload is fetched first.  The
# Builds go to a temporary directory (under $TMPDIR) and are removed.  Exit: neutrality's (0 when every pin is reproduced, 1 when a
# row moved or didn't build), 3 when the setup failed (OUT/setup.fail says which step).
set -uo pipefail
TIER=${1:?TIER}; OUT=${2:?OUT}; DRIVER_PKG=libnvidia-compute-595-server
TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P); ROOT=$TREE/integrations/vllm
cd "$ROOT" || exit 3
mkdir -p "$OUT" && OUT=$(cd "$OUT" && pwd -P) || exit 3
export WORK=/workspace/tmp/neutrality-vllm
export VENV=$WORK/venv312 HF_HOME=$WORK/hf
export PYTHONPATH=$ROOT:$TREE:$TREE/tools/research/src
PY=$VENV/bin/python DRV=$WORK/cuda-driver
mkdir -p "$WORK" "$HF_HOME" || exit 3
fail() { echo "$1: $2" | tee "$OUT/setup.fail" >&2; exit 3; }

ROWS=$(python3 - "$TIER" <<'EOF'
import json, sys
tier = sys.argv[1]
print(" ".join(sorted(r for r, e in json.load(open("manifests/records.json"))["rows"].items() if tier in ("all", e["tier"]))))
EOF
) || fail rows "manifests/records.json has no rows to read"
echo "neutrality_tier: $TIER: $ROWS"
for row in $ROWS; do
  (cd "$TREE" && uv run -q --project tools/research research data fetch-fixtures --repo "$TREE" "integrations/vllm/workloads/$row.json") \
    >>"$OUT/fixtures.log" 2>&1 || fail fixtures "$row's workload ($OUT/fixtures.log)"
done

(
  flock 9
  if [ -e "$VENV" ] && ! "$PY" -c 'import vllm, torch' >/dev/null 2>&1; then
    rm -rf "$VENV.stale" && mv "$VENV" "$VENV.stale"
  fi
  bash verity_vllm/ops/pod_bootstrap.sh --cpu --cases '' --out "$OUT/bootstrap" >"$OUT/bootstrap.log" 2>&1 \
    || fail bootstrap "$(tail -1 "$OUT/bootstrap.log")"
  HF_HUB_OFFLINE=0 "$PY" - $ROWS >"$OUT/metadata.log" 2>&1 <<'EOF' || fail metadata "a checkpoint's metadata ($OUT/metadata.log)"
import hashlib, os, sys
from pathlib import Path
from verity_vllm.config import checkpoint_dir, checkpoint_entries
from verity_vllm.pipeline import epoch
bad = []
for row in sys.argv[1:]:
    _role, repo, rev, _world = epoch.resolve(row)
    ent = next(c for c in checkpoint_entries("manifests/checkpoints.json") if c.get("repo") == repo and c.get("revision") == rev)
    have = checkpoint_dir(repo, rev, os.environ["HF_HOME"], "manifests/checkpoints.json")
    if not have or not (Path(have) / "config.json").is_file() or not Path(have).is_relative_to(os.environ["HF_HOME"]):
        from huggingface_hub import snapshot_download
        have = snapshot_download(repo, revision=rev, allow_patterns=["*.json", "*.py", "*.txt", "*.model", "*.tiktoken", "*.jinja"],
                                 cache_dir=os.path.join(os.environ["HF_HOME"], "hub"))
    for f, e in (ent.get("files") or {}).items():
        if f.endswith(".safetensors") or "sha256" not in e:
            continue
        p = Path(have) / f
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != e["sha256"]:
            bad.append(f"{repo}@{rev}/{f}")
    print(row, repo, rev, have, flush=True)
print("not the pinned bytes:", bad or "none")
sys.exit(1 if bad else 0)
EOF
  if ! ldconfig -p | grep -q 'libcuda\.so\.1 ' && [ ! -e "$DRV/libcuda.so.1" ]; then
    T=$(mktemp -d)
    { (cd "$T" && apt-get download "$DRIVER_PKG") && dpkg-deb -x "$T"/*.deb "$T/x" && mkdir -p "$DRV" \
      && cp -a "$T"/x/usr/lib/x86_64-linux-gnu/libcuda.so* "$DRV"/ \
      && { [ -e "$DRV/libcuda.so.1" ] || ln -sfn "$(basename "$(ls "$DRV"/libcuda.so.[0-9]*.* | head -1)")" "$DRV/libcuda.so.1"; }; } \
      >"$OUT/driver.log" 2>&1
    rm -rf "$T"
    [ -e "$DRV/libcuda.so.1" ] || fail driver "no libcuda.so.1 from $DRIVER_PKG ($OUT/driver.log)"
  fi
) 9>"$WORK.lock" || exit 3
[ -d "$DRV" ] && export LD_LIBRARY_PATH=$DRV${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

B=$(mktemp -d) || exit 3
trap 'rm -rf "$B"' EXIT
HF_HUB_OFFLINE=1 "$PY" -m verity_vllm.pipeline.cli neutrality --tier "$TIER" --out "$B" >"$OUT/neutrality.log" 2>&1
rc=$?
cp "$B/neutrality.json" "$OUT/" 2>/dev/null
for row in $ROWS; do
  mkdir -p "$OUT/$row" && (cd "$B/$row" 2>/dev/null && find . -maxdepth 2 \( -name '*.log' -o -name build_summary.json -o -name stages.txt \) -size -20M \
     -exec cp --parents {} "$OUT/$row/" \; ) 2>/dev/null || true
done
grep -E "^\[neutrality\]|^    " "$OUT/neutrality.log" | grep -v "row run" | tail -20
exit $rc

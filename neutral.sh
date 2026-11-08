#!/usr/bin/env bash
# neutral.sh ROW BUDGET_GB: `verity-vllm neutrality ROW --tier heavy --update` at this clone's HEAD, on a CPU node.
# The pinned vLLM's venv (pod_bootstrap.sh --cpu, steps 1-2 and readiness) and the checkpoint's metadata-only hub snapshot
# (every file but its safetensors shards; the derive runs on the meta device) are made once per node under a lock.
# The Build stays on the node under /workspace/tmp/circuits-neutrality/<run>; $OUT keeps its summaries, the index and the logs.
set -uo pipefail
ROW=$1; BUDGET=${2:-300}; DRIVER_PKG=${DRIVER_PKG:-libnvidia-compute-595-server}
RUN=$(basename "$RESEARCH_RUN_DIR")
OUT="$RESEARCH_RUN_DIR/out"; mkdir -p "$OUT"
SHA=$(git rev-parse HEAD); echo "$SHA" > "$OUT/head.txt"
TREE=$(pwd -P)
uv run -q --project tools/research research data fetch-fixtures --repo "$TREE" "integrations/vllm/workloads/$ROW.json" > "$OUT/fixtures.log" 2>&1
echo "fixtures rc=$? $(sha256sum "integrations/vllm/workloads/$ROW.json" 2>&1 | cut -c1-16)" >> "$OUT/summary.txt"
cd integrations/vllm || exit 3
VLLM_ROOT=$(pwd -P)
export WORK=/workspace/tmp/circuits-vllm
export VENV=$WORK/venv312 HF_HOME=$WORK/hf
mkdir -p "$WORK" "$HF_HOME"
PY=$VENV/bin/python
LOCK=/workspace/tmp/circuits-neutrality.lock
mkdir -p /workspace/tmp/circuits-neutrality
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/summary.txt"; }

say "row $ROW budget ${BUDGET}G head $SHA node $(hostname)"
( flock 9
  # pod_bootstrap.sh recreates the venv only when its python isn't executable (its interpreter was removed); replace that dead venv.
  UV_VENV_CLEAR=1 bash verity_vllm/ops/pod_bootstrap.sh --cpu --out "$OUT/bootstrap" > "$OUT/bootstrap.log" 2>&1
  echo "bootstrap rc=$?"
) 9>"$LOCK" | tee -a "$OUT/summary.txt"
"$PY" -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__)' >> "$OUT/summary.txt" 2>&1 || { say "no vLLM in $VENV"; exit 3; }

export PYTHONPATH=$VLLM_ROOT:$TREE:$TREE/tools/research/src
( flock 9
  HF_HUB_OFFLINE=0 "$PY" - "$ROW" <<'EOF'
import hashlib, json, os, sys
from pathlib import Path
from verity_vllm.pipeline import epoch
from verity_vllm.config import checkpoint_entries, checkpoint_dir
row = sys.argv[1]
role, repo, rev, world = epoch.resolve(row)
ent = next(c for c in checkpoint_entries("manifests/checkpoints.json") if c.get("repo") == repo and c.get("revision") == rev)
have = checkpoint_dir(repo, rev, os.environ["HF_HOME"], "manifests/checkpoints.json")
print("row", row, "role", role, "repo", repo, "rev", rev, "world", world, "have", have, flush=True)
if not have or not (Path(have) / "config.json").is_file():
    from huggingface_hub import snapshot_download
    have = snapshot_download(repo, revision=rev, allow_patterns=["*.json", "*.py", "*.txt", "*.model", "*.tiktoken", "*.jinja"],
                             cache_dir=os.path.join(os.environ["HF_HOME"], "hub"))
    print("downloaded metadata into", have, flush=True)
bad = []
for f, e in (ent.get("files") or {}).items():
    if f.endswith(".safetensors") or "sha256" not in e:
        continue
    p = Path(have) / f
    got = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    print(("OK  " if got == e["sha256"] else "BAD ") + f, flush=True)
    bad += [f] if got != e["sha256"] else []
print("files:", sorted(os.listdir(have)), flush=True)
sys.exit(1 if bad else 0)
EOF
  echo "metadata rc=$?"
) 9>"$LOCK" > "$OUT/metadata.log" 2>&1
tail -1 "$OUT/metadata.log" | tee -a "$OUT/summary.txt"
grep -q "metadata rc=0" "$OUT/metadata.log" || { say "metadata snapshot failed"; exit 3; }

# vLLM's CUDA platform loads libcuda.so.1 at import; a GPU-less pod takes the driver's copy, which reaches no GPU without
# /dev/nvidia* (research/pods/nebius/sky: /workspace/jobs/cuda-driver). This node has no driver, so the driver package's
# library is extracted into $WORK (apt-get download; nothing installed).
DRV=$WORK/cuda-driver
( flock 9
  if [ ! -e "$DRV/libcuda.so.1" ]; then
    T=$(mktemp -d)
    ( cd "$T" && apt-get download "$DRIVER_PKG" ) && dpkg-deb -x "$T"/*.deb "$T/x" && mkdir -p "$DRV" \
      && cp -a "$T"/x/usr/lib/x86_64-linux-gnu/libcuda.so* "$DRV"/ \
      && { [ -e "$DRV/libcuda.so.1" ] || ln -sfn "$(basename "$(ls "$DRV"/libcuda.so.[0-9]*.* | head -1)")" "$DRV/libcuda.so.1"; }
    rm -rf "$T"
  fi
  ls -la "$DRV"
) 9>"$LOCK" > "$OUT/driver.log" 2>&1
export LD_LIBRARY_PATH=$DRV${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
CUDA_VISIBLE_DEVICES= "$PY" -c 'import vllm._C_stable_libtorch, torch; print("libcuda ok; cuda available", torch.cuda.is_available())' >> "$OUT/summary.txt" 2>&1 \
  || { say "vLLM's CUDA platform does not import with $DRV"; tail -5 "$OUT/driver.log" >> "$OUT/summary.txt"; exit 3; }

N=/workspace/tmp/circuits-neutrality/$RUN; mkdir -p "$N"
say "neutrality start (out $N)"
{ time "$PY" -m verity_vllm.pipeline.cli neutrality "$ROW" --tier heavy --update --out "$N" \
    --row-flag=--build-ram-budget-gb="$BUDGET" --run-id "$RUN" --source-sha "$SHA"; } > "$OUT/neutrality.log" 2>&1
say "neutrality rc=$?"
cp "$N/neutrality.json" "$OUT/" 2>/dev/null
cp manifests/records.json "$OUT/records.json"
git diff --stat -- manifests/records.json >> "$OUT/summary.txt"
mkdir -p "$OUT/row"
( cd "$N/$ROW" 2>/dev/null && find . -maxdepth 2 \( -name '*.json' -o -name '*.txt' -o -name '*.log' -o -name '*.jsonl' \) -size -20M \
    -exec cp --parents {} "$OUT/row/" \; )
du -sh "$N/$ROW" >> "$OUT/summary.txt" 2>&1
grep -E "^\[neutrality\]" "$OUT/neutrality.log" | tail -20 >> "$OUT/summary.txt"
tail -3 "$OUT/neutrality.log" >> "$OUT/summary.txt"

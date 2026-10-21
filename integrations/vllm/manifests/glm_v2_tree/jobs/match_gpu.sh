#!/usr/bin/env bash
# A v2 row's Match GPU legs (MATCH_PHASE=gpu: the capture and the control) on an 8x H200 node (gpu_batch.sh), from this tree.  A
# model without a post-load hook (GLM-5.3) needs no Build on the node; one with it (model.sh POST_LOAD: DeepSeek-V4-Pro) needs
# the Build of record in the row dir (copy_rowdir.sh), every rank's request Programs recording POST_LOAD, which the capture
# registers as the rank's roots (`--build-rank`, `component_programs.post_load_tensors`): refused (exit 2) otherwise.  The legs
# take the node's lease themselves (gpu-lease 8 --wait, around each leg) once every GPU reads 0 MiB and no preemptible lease is
# held, the first also once the row dir's disk has the capture's room (DISK_NEED_GIB): queued behind the batch's other GPU jobs,
# never preempting.  ROW_ON=ROOT keeps the row dir on another disk at its path (model.sh `row_on`).  A match/ with no gpu_legs.json is an
# evicted job's and is moved aside (match.evicted-*); one whose legs ended is refused.  After the legs, with no lease, the job
# records each engine's gpu_memory_utilization read back from its vllm_config, the driver's and each rank's (the control, with no
# observer, the driver's alone; $OUT/gpu_memory_utilization.json; exit 12 unless every one is GPU_UTIL; each rank's own sizing
# line from vLLM's log, both legs, in $OUT/vllm_memory_sizing.txt), and hashes match/ and $SW/$ROW.tree (REF@SHA: every job of the row refuses another tree) into
# $OUT/match.sha256 (paths relative to $SW) for the copy to the CPU node that runs the Match's CPU phase (README "GPU batch").
#   research run --on vy-glm53-eu --mem-gb 400 --timeout 3h --source . --cwd source --project verity -- env ROW=... bash integrations/vllm/manifests/glm_v2_tree/jobs/match_gpu.sh
# env: ROW, [SW], [REF] (model.sh's defaults), [DISK_NEED_GIB: by row, below], [ROW_ON].
set -uo pipefail
: "${ROW:?ROW}"
TREE=$PWD
source "$TREE/integrations/vllm/manifests/glm_v2_tree/jobs/model.sh"
D=$SW/$ROW
mkdir -p "$SW" || exit 3
[ -z "${ROW_ON:-}" ] || row_on "$ROW_ON" || exit 3
mkdir -p "$D" || exit 3
[ ! -e "$D/match/gpu_legs.json" ] || { echo "$D/match/gpu_legs.json exists: this row dir already ran its Match GPU legs"; exit 3; }
if [ -n "$(ls -A "$D/match" 2>/dev/null)" ]; then
  mv "$D/match" "$D/match.evicted-$(date -u +%Y%m%dT%H%M%SZ)" && echo "$D/match without gpu_legs.json (an evicted leg's): moved aside"
fi
source "$TREE/integrations/vllm/manifests/glm_v2_tree/jobs/tree_guard.sh" && tree_guard "$ROW" "$SW" "$REF" first || exit 2
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee "$OUT/gpus_at_start.txt"
df -h "$D" | tail -1 | tee "$OUT/disk_at_start.txt"
C=$(dirname "$(dirname "$RESEARCH_RUN_DIR")")/requests/$RESEARCH_RUN_ID/custody
[ -f "$C/cred.json" ] || { echo "no custody key at $C"; exit 4; }
ckpt_bind
unset NCCL_P2P_DISABLE VLLM_USE_DEEP_GEMM
export MATCH_PHASE=gpu PY=/workspace/cp/venv312/bin/python HF_HOME=/workspace/cp/hf GPU_UTIL=0.92 COMMIT_GPU_UTIL=0.92
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
W=$(mktemp -d /tmp/$V2-match.XXXXXX); trap 'rm -rf "$W"' EXIT
# The capture's room on the row dir's disk, from the v1 legs' match/capture (B1 104 GiB, B8 greedy 312, B8 stochastic 341; B32's
# unmeasured, scaled from B8): the first leg also waits for it, read once the GPUs are free, so after the leg before it wrote.
case $ROW in *__b1__*) need=110 ;; *__b8__*__greedy__*) need=330 ;; *__b8__*) need=360 ;; *__b32__*) need=1100 ;; *) need=110 ;; esac
export DISK_NEED_GIB=${DISK_NEED_GIB:-$need} DISK_AT=$D
echo "the capture waits for $DISK_NEED_GIB GiB free on $(df --output=target "$D" | tail -1) ($(df -B1G --output=avail "$D" | tail -1 | tr -d ' ') GiB now)"
cat >"$W/gpu-lease" <<'EOF'
#!/usr/bin/env bash
while nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 != 0 {b = 1} END {exit !b}' \
  || /usr/local/bin/gpu-lease status | grep -q ' held .*preempt=1 ' \
  || { [ ! -e "$(dirname "$0")/disk-ok" ] && [ "$(df -B1G --output=avail "$DISK_AT" | tail -1)" -lt "$DISK_NEED_GIB" ]; }; do sleep 30; done
touch "$(dirname "$0")/disk-ok"
exec /usr/local/bin/gpu-lease "$@"
EOF
chmod +x "$W/gpu-lease"
export GPU_LEASE=$W/gpu-lease GPU_LEASE_N=8 GPU_LEASE_WHO=circuits-$V2 CUDA_VISIBLE_DEVICES=
export GUARD_PRESERVE_FILE=$SW/guard.preserve TMPDIR=$W
source /workspace/cp/venv312/bin/activate
export PATH=$VIRTUAL_ENV/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export PYTHONPATH=$TREE:$TREE/tools/research/src:$TREE/infra/cluster/src:$TREE/infra/ops:$TREE/integrations/vllm:$TREE/verity/ml/flock/python:$TREE/tools/verity/circuit_check/src:$TREE/tools/verity/cli:$TREE/verity/core/difftest/python
export RESEARCH_STORE_CONFIG=$C/store.toml
eval "$($PY -m verity_vllm.pipeline.cli epoch custody-env "$C")"
if [ -n "$POST_LOAD" ]; then
  $PY - "$D" "$ROW" "$POST_LOAD" <<'EOF2' || exit 2
import re, sys
from verity_vllm.check.component_programs import post_load_tensors
d, row, want = sys.argv[1], sys.argv[2], set(sys.argv[3].split(","))
tp = int(re.search(r"__tp(\d+)__", row).group(1))
got = post_load_tensors({r: f"{d}/build/rank{r}" for r in range(tp)})
bad = {r: sorted(want - set(n)) for r, n in got.items() if want - set(n)}
if bad:
    sys.exit(f"the Build in {d} does not record the post-load tensors the capture registers ({bad} missing by rank): copy the Build of record in first")
print(f"post-load tensors: {sorted(want)} on all {tp} ranks' Builds")
EOF2
fi
printf '#!/bin/sh\nexec %s -m research "$@"\n' "$PY" >"$W/research" && chmod +x "$W/research"
export RESEARCH_BIN=$W/research
"$RESEARCH_BIN" keep "$SW" --owner @circuits --expires 5d --priority normal \
  --why "$WHAT v2 rows ($REF): Match GPU legs ($RESEARCH_RUN_ID) for the CPU phase and the Commit" || echo "research keep failed (continuing)"
compgen -e | grep -E '^(VLLM_|NCCL_|CUDA_|TORCH_|GPU_LEASE|GPU_UTIL|MATCH_)' | sort | tr '\n' ' ' | tee "$OUT/engine_env_names.txt"; echo
$PY -m verity_vllm.pipeline.cli row chain --stages match --ref "$REF" --by circuits \
  "$ROW" "$ROLE" "$REPO" "$REV" --sweep-dir "$SW" >"$OUT/chain.out" 2>&1
for f in row.json stages.txt row.log effective_config.json gpu_lease.json match/gpu_legs.json; do cp "$D/$f" "$OUT/" 2>/dev/null; done
tail -20 "$OUT/chain.out"
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d["capture_rc"] == 0 and d["control_rc"] == 0 else 1)' \
  "$D/match/gpu_legs.json" 2>/dev/null || { echo "the GPU legs did not both pass ($D/match/gpu_legs.json)"; exit 11; }
grep -h "Desired GPU memory utilization" "$D/match/capture.log" "$D/match/control.log" >"$OUT/vllm_memory_sizing.txt"
python3 "$TREE/integrations/vllm/manifests/glm_v2_tree/jobs/gpu_util_read_back.py" "$OUT/gpu_memory_utilization.json" "$GPU_UTIL" \
  "$D/match/capture/topology.json" "driver:$D/match/control/topology.json" ||
  { echo "the legs passed, but an engine did not read back gpu_memory_utilization=$GPU_UTIL ($OUT/gpu_memory_utilization.json)"; exit 12; }
du -sh "$D/match" | tee "$OUT/match_du.txt"
(cd "$SW" && { echo "$ROW.tree"; find "$ROW/match" -type f; } | tr '\n' '\0' | xargs -0 -P 16 -n 256 sha256sum) | sort -k2 >"$OUT/match.sha256"
echo "match.sha256: $(wc -l <"$OUT/match.sha256") files"

#!/usr/bin/env bash
# A v2 row's Match GPU legs (MATCH_PHASE=gpu: the capture and the control) on vy-glm53-eu or vy-glm53-h8, from this tree; no Build
# needed.  The legs take the node's lease themselves (gpu-lease 8 --wait, around each leg) once every GPU reads 0 MiB and no
# preemptible lease is held: queued behind the batch's other GPU jobs, never preempting.  A match/ with no gpu_legs.json is an
# evicted job's and is moved aside (match.evicted-*); one whose legs ended is refused.  After the legs, with no lease, the job
# records each engine's gpu_memory_utilization read back from its vllm_config ($OUT/gpu_memory_utilization.json; exit 12 unless
# it is GPU_UTIL), and hashes match/ and $SW/$ROW.tree (REF@SHA: every job of the row refuses another tree) into
# $OUT/match.sha256 (paths relative to $SW) for the copy to the CPU node that runs the Match's CPU phase (README "GPU batch").
#   research run --on vy-glm53-eu --gpu-host --mem-gb 400 --timeout 3h --cwd source --project verity -- env ROW=... bash integrations/vllm/manifests/glm_v2_tree/jobs/match_gpu.sh
# env: ROW, [SW=/workspace/cp/circuits/glm-v2/rows], [REF=cursor/glm-v2-21f3].
set -uo pipefail
: "${ROW:?ROW}"
REV=aca966e4e02791568aa6a4ced368624b3d897f42
REF=${REF:-cursor/glm-v2-21f3}
SW=${SW:-/workspace/cp/circuits/glm-v2/rows}
TREE=$PWD
D=$SW/$ROW
mkdir -p "$SW" || exit 3
[ ! -e "$D/match/gpu_legs.json" ] || { echo "$D/match/gpu_legs.json exists: this row dir already ran its Match GPU legs"; exit 3; }
if [ -n "$(ls -A "$D/match" 2>/dev/null)" ]; then
  mv "$D/match" "$D/match.evicted-$(date -u +%Y%m%dT%H%M%SZ)" && echo "$D/match without gpu_legs.json (an evicted leg's): moved aside"
fi
source "$TREE/integrations/vllm/manifests/glm_v2_tree/jobs/tree_guard.sh" && tree_guard "$ROW" "$SW" "$REF" first || exit 2
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee "$OUT/gpus_at_start.txt"
df -h /workspace | tail -1 | tee "$OUT/disk_at_start.txt"
C=$(dirname "$(dirname "$RESEARCH_RUN_DIR")")/requests/$RESEARCH_RUN_ID/custody
[ -f "$C/cred.json" ] || { echo "no custody key at $C"; exit 4; }
FS_CK=/workspace/fs/glm53/$REV; LOCAL_CK=/workspace/cp/glm53-ckpt/$REV
if [ -e "$LOCAL_CK/.verified" ] && ! mountpoint -q "$FS_CK"; then sudo -n mount --bind "$LOCAL_CK" "$FS_CK"; fi
mountpoint -q "$FS_CK" && echo "checkpoint: local verified copy bound over $FS_CK" || echo "checkpoint: $FS_CK read over the shared fs"
unset NCCL_P2P_DISABLE VLLM_USE_DEEP_GEMM
export MATCH_PHASE=gpu PY=/workspace/cp/venv312/bin/python HF_HOME=/workspace/cp/hf GPU_UTIL=0.92 COMMIT_GPU_UTIL=0.92
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
W=$(mktemp -d /tmp/glm-v2-match.XXXXXX); trap 'rm -rf "$W"' EXIT
cat >"$W/gpu-lease" <<'EOF'
#!/usr/bin/env bash
while nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 != 0 {b = 1} END {exit !b}' \
  || /usr/local/bin/gpu-lease status | grep -q ' held .*preempt=1 '; do sleep 30; done
exec /usr/local/bin/gpu-lease "$@"
EOF
chmod +x "$W/gpu-lease"
export GPU_LEASE=$W/gpu-lease GPU_LEASE_N=8 GPU_LEASE_WHO=circuits-glm-v2 CUDA_VISIBLE_DEVICES=
export GUARD_PRESERVE_FILE=$SW/guard.preserve TMPDIR=$W
source /workspace/cp/venv312/bin/activate
export PATH=$VIRTUAL_ENV/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export PYTHONPATH=$TREE:$TREE/tools/research/src:$TREE/infra/cluster/src:$TREE/infra/ops:$TREE/integrations/vllm:$TREE/verity/ml/flock/python:$TREE/tools/verity/circuit_check/src:$TREE/tools/verity/cli:$TREE/verity/core/difftest/python
export RESEARCH_STORE_CONFIG=$C/store.toml
eval "$($PY -m verity_vllm.pipeline.cli epoch custody-env "$C")"
printf '#!/bin/sh\nexec %s -m research "$@"\n' "$PY" >"$W/research" && chmod +x "$W/research"
export RESEARCH_BIN=$W/research
"$RESEARCH_BIN" keep "$SW" --owner @circuits --expires 5d --priority normal \
  --why "GLM-5.3 v2 rows (cursor/glm-v2-21f3): Match GPU legs ($RESEARCH_RUN_ID) for the CPU phase and the Commit" || echo "research keep failed (continuing)"
compgen -e | grep -E '^(VLLM_|NCCL_|CUDA_|TORCH_|GPU_LEASE|GPU_UTIL|MATCH_)' | sort | tr '\n' ' ' | tee "$OUT/engine_env_names.txt"; echo
$PY -m verity_vllm.pipeline.cli row chain --stages match --ref "$REF" --by circuits \
  "$ROW" GLM53 zai-org/GLM-5.3 "$REV" --sweep-dir "$SW" >"$OUT/chain.out" 2>&1
for f in row.json stages.txt row.log effective_config.json gpu_lease.json match/gpu_legs.json; do cp "$D/$f" "$OUT/" 2>/dev/null; done
tail -20 "$OUT/chain.out"
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d["capture_rc"] == 0 and d["control_rc"] == 0 else 1)' \
  "$D/match/gpu_legs.json" 2>/dev/null || { echo "the GPU legs did not both pass ($D/match/gpu_legs.json)"; exit 11; }
python3 "$TREE/integrations/vllm/manifests/glm_v2_tree/jobs/gpu_util_read_back.py" "$OUT/gpu_memory_utilization.json" "$GPU_UTIL" \
  "$D/match/capture/topology.json" "$D/match/control/topology.json" ||
  { echo "the legs passed, but an engine did not read back gpu_memory_utilization=$GPU_UTIL ($OUT/gpu_memory_utilization.json)"; exit 12; }
du -sh "$D/match" | tee "$OUT/match_du.txt"
(cd "$SW" && { echo "$ROW.tree"; find "$ROW/match" -type f; } | tr '\n' '\0' | xargs -0 -P 16 -n 256 sha256sum) | sort -k2 >"$OUT/match.sha256"
echo "match.sha256: $(wc -l <"$OUT/match.sha256") files"

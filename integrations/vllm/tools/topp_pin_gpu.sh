#!/usr/bin/env bash
# The top-p split pin (engine.topp_pin) on one RTX PRO 6000: vLLM's split kernel pinned and unpinned at every drain batch
# (topp_pin_check kernel), Llama-3.2-1B serving seeded top_p=0.95 requests together and alone (topp_pin_check served), then the
# B8 and B1 stochastic workloads' Build, Match and Commit (ROWS; CHECKS=0 runs only the workloads).
#   research run --on vy-nebius-1 --project verity --cwd source --mem-gb 128 --timeout 3h \
#     --env GPU_LEASE_WHO=vllm-topp-pin -- bash integrations/vllm/tools/topp_pin_gpu.sh
# Needs on the node what examples/vllm/commit.sh needs: vLLM's venv, the checkpoint in /workspace/hf and gpu-lease.
set -uo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
ROWS=${ROWS:-"llama32-1b__bf16__rtxpro6000__tp1__b8__i256__o32__mixed__stoch-t0.8-p0.95__bi-eager llama32-1b__bf16__rtxpro6000__tp1__b1__i256__o32__mixed__stoch-t0.8-p0.95__bi-eager"}
ROLE=LLAMA32_1B REPO=unsloth/Llama-3.2-1B REV=9535bd9b1d1dea6acafbdc4813b728796aeb28da
TREE=$PWD
ROOT=${TOPP_ROOT:-/workspace/research/vllm-topp-pin}
BASE=$ROOT/${RESEARCH_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$BASE" "$ROOT/cache"
export PY=/workspace/jobs/venv312/bin/python PY312=/workspace/jobs/venv312/bin/python HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
export CUDA_HOME=/workspace/jobs/cuda-12.9 TORCH_EXTENSIONS_DIR=$ROOT/cache/torch-extensions
export PATH=/workspace/jobs/venv312/bin:$CUDA_HOME/bin:$PATH
export PYTHONPATH=$TREE:$TREE/integrations/vllm:$TREE/tools/research/src
export GPU_LEASE=${GPU_LEASE:-gpu-lease} GPU_LEASE_WHO=${GPU_LEASE_WHO:-vllm-topp-pin}
export TRITON_CACHE_DIR=$ROOT/cache/triton VLLM_CACHE_ROOT=$ROOT/cache/vllm
if grep -q staged_sources "$TREE/integrations/vllm/verity_vllm/commit/committer/native_jit.py"; then
  export HIDDEN_GPU_BUILD=$ROOT/cache/jit/hidden_gpu NATIVE_COLLECT_BUILD=$ROOT/cache/jit/nc
fi
rs() { "$PY" -m research "$@" 2>&1 || research "$@" 2>&1; }
{ id; hostname; git -C "$TREE" rev-parse HEAD 2>/dev/null; echo "base $BASE"; nvidia-smi -L; } | tee "$OUT/machine.txt"
rs keep "$ROOT" --owner @vllm --expires 7d --priority normal --why "the top-p split pin's rows (Build, Match, Commit)" | tail -2

step() {  # NAME CMD...: its output in $OUT/NAME.out
  local name=$1; shift
  echo "$(date -u +%H:%M:%SZ) $name: $*" | tee -a "$OUT/steps.txt"
  (cd "$TREE/integrations/vllm" && "$@") > "$OUT/$name.out" 2>&1
  local rc=$?
  echo "$(date -u +%H:%M:%SZ) $name rc=$rc" | tee -a "$OUT/steps.txt"
  tail -4 "$OUT/$name.out"
  return $rc
}
lease() { "$GPU_LEASE" 1 --wait --max-min "$1" --mem-gb 0 -- "${@:2}"; }

# the sampler Call at V = 128256 is one Q_word unit over the default gate limit, as on every top-p workload, so the limit is raised
# for it; a one-rank request Build states it as v3, v2 without its dead gates (program.prune.dead_free)
G=GumbelTopPTokenSelect_v2=113000000,GumbelTopPTokenSelect_v3=113000000
export VERITY_QWORD_MAX_GATES=${VERITY_QWORD_MAX_GATES:-$G} VERITY_QWORD_MAX_GATES_ALLOWED=${VERITY_QWORD_MAX_GATES_ALLOWED:-$G}

rc=0
if [ "${CHECKS:-1}" = 1 ]; then
  step kernel lease 15 "$PY" -m verity_vllm.pipeline.cli topp-pin-check kernel --out "$OUT" || rc=1
  step served lease 20 "$PY" -m verity_vllm.pipeline.cli topp-pin-check served --out "$OUT" \
    --workload "workloads/${ROWS%% *}.json" || rc=1
fi
for ROW in $ROWS; do
  B=$(sed 's/.*__b\([0-9]*\)__.*/\1/' <<<"$ROW")
  step "row-b$B" "$PY" -m verity_vllm.pipeline.cli row run "$ROW" "$ROLE" "$REPO" "$REV" --sweep-dir "$BASE" || rc=1
  D=$BASE/$ROW; mkdir -p "$OUT/b$B"
  for f in stages.txt row.log verdict.json build_summary.json config_record.json gpu_lease.json effective_config.json \
    commit.log match.log build.log; do cp "$D/$f" "$OUT/b$B/" 2>/dev/null; done
  for s in commit match; do
    mkdir -p "$OUT/b$B/$s"; find "$D/$s" -maxdepth 1 -type f -size -20M -exec cp {} "$OUT/b$B/$s/" \; 2>/dev/null
  done
  tail -6 "$D/stages.txt" 2>/dev/null
done
exit $rc

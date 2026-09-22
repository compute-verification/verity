#!/bin/bash
# RC: experiments 2 and 3 on the full-preparation control bundle, plus the control's witness accounting.
# usage: rc_post.sh   (after rc_chain.sh; GPU parts first, CPU parts after)
set -u
cd /workspace/veritor/vllm-poc
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
PY=${PY:-/workspace/venv-cu129/bin/python}
PY312=${PY312:-/workspace/venv312/bin/python}
R=/workspace/rc/results/B1
B=$R/control/commit
LOG=$R/post.log
echo "[$(date -u +%H:%M:%SZ)] POST start" | tee -a $LOG
# the control's verify, re-run with the verify.py fix (the reference verifier crashed assembling its result doc: `bindings.items()` on WB's Bindings object)
if ! grep -q '"result": "ok"' $R/control/verify-security/verify.json 2>/dev/null || grep -q "bundle refused" $R/control/verify-security/verify.json; then
  export VLLM_CACHE_ROOT=/workspace/cache/B1-control-verify TORCHINDUCTOR_CACHE_DIR=/workspace/cache/B1-control-verify/inductor TRITON_CACHE_DIR=/workspace/cache/B1-control-verify/triton
  rm -rf $R/control/verify-security.failed; mv $R/control/verify-security $R/control/verify-security.failed 2>/dev/null
  t0=$(date +%s.%N)
  PYTHONPATH=. $PY -m verity_vllm_adapter.verify $B --challenge security --q 1 --s 1/4096 --coverage-size 200 --no-negatives --production $R/control/production \
      --checkpoints manifests/checkpoints.json --noninterference $R/control/compare.json --adapter-revision ${REV:-3b30188} --case B1 \
      --semantic-profile manifests/semantic-profiles/qwen2-eager-batch-invariant-v1.json --out $R/control/verify-security > /workspace/logs/B1-control-verify-security-2.log 2>&1
  $PY - $R/control/control_timing.json "$($PY -c "import time; print(time.time()-$t0)")" <<'EOF2'
import json, sys
p, v = sys.argv[1:3]
d = json.load(open(p)); d["verify_wall_s_rerun"] = float(v); json.dump(d, open(p, "w"), indent=1)
EOF2
  echo "[$(date -u +%H:%M:%SZ)] control verify re-run: $(grep RESULT /workspace/logs/B1-control-verify-security-2.log | tail -1 | cut -c1-200)" | tee -a $LOG
fi
export VLLM_CACHE_ROOT=/workspace/cache/rc-post TORCHINDUCTOR_CACHE_DIR=/workspace/cache/rc-post/inductor TRITON_CACHE_DIR=/workspace/cache/rc-post/triton
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR $R/acq $R/plan
# experiment 2 (b): same-runtime relations on the GPU, 4 VUs per (family, phase)
PYTHONPATH=. $PY -m verity_vllm_adapter.acquisition $B --out $R/acq --mode same_runtime --per-stratum 4 > /workspace/logs/rc-acq-same-runtime.log 2>&1
echo "[$(date -u +%H:%M:%SZ)] acquisition same_runtime rc=$? $(tail -1 /workspace/logs/rc-acq-same-runtime.log | cut -c1-200)" | tee -a $LOG
# experiment 3, plans 1 and 2 on the GPU (slice: longest-prompt request, last decode step)
PYTHONPATH=. $PY -m verity_vllm_adapter.plan_compare $B --out $R/plan --mode gpu > /workspace/logs/rc-plan-gpu.log 2>&1
echo "[$(date -u +%H:%M:%SZ)] plan_compare gpu rc=$? $(tail -1 /workspace/logs/rc-plan-gpu.log | cut -c1-200)" | tee -a $LOG
# experiment 2 (c): independent CPU models on the same sample
PYTHONPATH=.:../src VERITY_ARCH=sm_80 $PY312 -m verity_vllm_adapter.acquisition $B --out $R/acq --mode independent --sample $R/acq/sample.json > /workspace/logs/rc-acq-independent.log 2>&1
echo "[$(date -u +%H:%M:%SZ)] acquisition independent rc=$? $(tail -1 /workspace/logs/rc-acq-independent.log | cut -c1-200)" | tee -a $LOG
# experiment 3, plans 3 (analytical) and 4 (coordinate sample) on the CPU, same slice
REQ=$($PY -c "import json; d=json.load(open('$R/plan/plan_compare_gpu.json')); print(d['slice']['request_id'], d['slice']['t'])")
set -- $REQ
PYTHONPATH=.:../src VERITY_ARCH=sm_80 $PY312 -m verity_vllm_adapter.plan_compare $B --out $R/plan --mode cpu --request $1 --t $2 --coordinates-per-family 8 > /workspace/logs/rc-plan-cpu.log 2>&1
echo "[$(date -u +%H:%M:%SZ)] plan_compare cpu rc=$? $(tail -1 /workspace/logs/rc-plan-cpu.log | cut -c1-200)" | tee -a $LOG
# the control's witness accounting (same columns as the selective rows)
PYTHONPATH=. $PY -m verity_vllm_adapter.witness_of $B $R/control/verify-security/verify.json --out $R/control/witness.json > /workspace/logs/rc-witness-control.log 2>&1
echo "[$(date -u +%H:%M:%SZ)] witness_of control rc=$? $(tail -1 /workspace/logs/rc-witness-control.log | cut -c1-200)" | tee -a $LOG
echo "[$(date -u +%H:%M:%SZ)] POST_DONE" | tee -a $LOG

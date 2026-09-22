#!/bin/bash
# RC: one genuine selective-preparation run on the pod (prover + verifier as two processes, file handshake).
# usage: run_selective.sh <K> <REV> [--tag TAG] [--production DIR] [--control DIR] [--s FRAC] [--coverage-size N]
#                          [--arena-gb N] [--weights-format 1|2] [--challenge security|diagnostic] [--seed N]
# env: PY (default /workspace/venv-cu129/bin/python), VERITY_CORE_PYTHON (default /workspace/venv312/bin/python)
set -u
K=$1; REV=$2; shift 2
TAG=""; PROD=/workspace/rc/results/B1/control/production; CONTROL=/workspace/rc/results/B1/control/commit; S=1/4096; COV=200; ARENA=48; WF=2
CHALLENGE=security; SEED=""
while [ $# -gt 0 ]; do case "$1" in
  --tag) TAG=$2; shift;; --production) PROD=$2; shift;; --control) CONTROL=$2; shift;; --s) S=$2; shift;;
  --coverage-size) COV=$2; shift;; --arena-gb) ARENA=$2; shift;; --weights-format) WF=$2; shift;;
  --challenge) CHALLENGE=$2; shift;; --seed) SEED=$2; shift;; esac; shift; done
TAG=${TAG:-k$K}
SEEDARG=""; [ -n "$SEED" ] && SEEDARG="--seed $SEED"
cd /workspace/veritor/vllm-poc
export PYTHONPATH=. HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
export VERITY_CORE_PYTHON=${VERITY_CORE_PYTHON:-/workspace/venv312/bin/python}
PY=${PY:-/workspace/venv-cu129/bin/python}
SP=manifests/semantic-profiles/qwen2-eager-batch-invariant-v1.json
R=/workspace/rc/results/B1; mkdir -p $R /workspace/logs /workspace/cache
RUN=$R/selective-$TAG; HS=$R/selective-$TAG.handshake; VOUT=$R/selective-$TAG.verify
rm -rf $RUN $HS $VOUT; mkdir -p $HS $VOUT
export VLLM_CACHE_ROOT=/workspace/cache/rc-selective-$TAG
export TORCHINDUCTOR_CACHE_DIR=$VLLM_CACHE_ROOT/inductor TRITON_CACHE_DIR=$VLLM_CACHE_ROOT/triton
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR
echo "[$(date -u +%H:%M:%SZ)] selective $TAG: K=$K s=$S coverage=$COV arena=$ARENA weights_format=$WF challenge=$CHALLENGE" | tee -a $R/selective.log
$PY -m verity_vllm_adapter.selective_prover --case B1 --checkpoints manifests/checkpoints.json --production $PROD --out $RUN --handshake $HS \
    --adapter-revision $REV --semantic-profile $SP --arena-gb $ARENA --weights-format $WF > /workspace/logs/rc-selective-$TAG-prover.log 2>&1 &
PROVER=$!
$PY -m verity_vllm_adapter.selective_verifier --run $RUN --handshake $HS --production $PROD --checkpoints manifests/checkpoints.json --case B1 \
    --semantic-profile $SP --k $K --s $S --coverage-size $COV --challenge $CHALLENGE $SEEDARG --adapter-revision $REV --out $VOUT > /workspace/logs/rc-selective-$TAG-verifier.log 2>&1
VRC=$?
wait $PROVER; PRC=$?
echo "[$(date -u +%H:%M:%SZ)] selective $TAG: prover rc=$PRC verifier rc=$VRC $(grep RESULT /workspace/logs/rc-selective-$TAG-verifier.log | tail -1)" | tee -a $R/selective.log
if [ -d "$CONTROL" ]; then
  $PY -m verity_vllm_adapter.compare_selected --selective $RUN --control $CONTROL --out $VOUT/compare_selected.json > /workspace/logs/rc-selective-$TAG-compare.log 2>&1
  echo "[$(date -u +%H:%M:%SZ)] selective $TAG: batch-invariance vs control: $(head -1 /workspace/logs/rc-selective-$TAG-compare.log)" | tee -a $R/selective.log
fi
du -sb $RUN/ru $RUN/weights $RUN 2>/dev/null > $VOUT/du.txt
echo "[$(date -u +%H:%M:%SZ)] SELECTIVE_DONE $TAG" | tee -a $R/selective.log

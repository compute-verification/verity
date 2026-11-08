#!/usr/bin/env bash
# The cross-node replay (PRF-461 item 1) on the Nebius 8xH200 NODE, from a clone of the replay ladder's tree (f095ae769f):
#   NODE=vy-dc-h200-N MODE=P|R|C bash launch_xnode.sh
# P  node_record.sh (the checkpoint against benchmarks/ppl/pins/glm-5.3.json) into out/node, then kernels' all-reduce probe
#    (fused, allreduce, probe, nccl_probe: MNNVL's words, the routes vLLM resolves at the served settings, this node's trees)
# R  72b6's served block (x72b6.tgz: its record arm, plan, served tokens and logprobs, engine env, from art:e4098b2b...) from an
#    empty engine with every hook on, held to the served tokens and logprob bits, the all-reduces summed in node A's tree
# C  R without node A's tree: run only when P's tree is not node A's
set -euo pipefail
export PATH=/agent/repos/verity/.git/verity-auth/bin:"$PATH"
: "${NODE:?NODE: the Nebius node}"; MODE=${MODE:?MODE: P, R or C}
F=/tmp/od8/send4; REV=aca966e4e02791568aa6a4ced368624b3d897f42
TREE='(((((((0,2),3),1),5),6),4),7)'
SRC='art:3ffc7fc21dc9978b0a071b8da7f6838594f356f9455ee5330d12f36de195f2e9 (kernels probe r20261107-192422-508b, vy-spot-glm53)'
VENV=${VENV:-/workspace/venv312}
case $MODE in
P)
  exec research run --on "$NODE" --project verity --source /tmp/od8/rc0 --cwd clone --timeout 2400 \
    --send "$F/node_record.sh" --send "$F/ar_capture.sh" -- bash -c "\
    OUT=\$RESEARCH_RUN_DIR/out; IN=\$RESEARCH_RUN_DIR/inputs; mkdir -p \$OUT; CK=; \
    for c in /workspace/cp/glm53-ckpt/$REV /workspace/fs/glm53/$REV; do [ -f \$c/config.json ] && { CK=\$c; break; }; done; \
    echo \"xnode: checkpoint \${CK:-none}\"; \
    VENV=$VENV CUDA_HOME=/usr/local/cuda-12.9 CK=\$CK PINS=benchmarks/ppl/pins/glm-5.3.json bash \$IN/node_record.sh \$OUT/node; \
    VENV=$VENV ONLY=fused,allreduce,probe,nccl_probe IDLE_MIN=5 bash \$IN/ar_capture.sh" ;;
R|C)
  SENDS=(); for f in serve_logprobs.patch replay_aborts.patch dsa_topk.py dsa_topk_kernels.py capture.py stage.py spot_job.sh spot_tp8.sh \
    gpu-lease sitecustomize.py long_rows.py spot_canonical.sh node_record.sh tree_allreduce.py x72b6.tgz; do SENDS+=(--send "$F/$f"); done
  T=; [ "$MODE" = R ] && T="TREE_AR='$TREE' TREE_AR_SOURCE='$SRC'"
  exec research run --on "$NODE" --project verity --source /tmp/od8/rc0 --cwd clone --timeout 4200 "${SENDS[@]}" -- bash -c "\
    X=\$RESEARCH_RUN_DIR/x72b6; mkdir -p \$X && tar xzf \$RESEARCH_RUN_DIR/inputs/x72b6.tgz -C \$X; \
    VENV=$VENV CUDA_HOME=/usr/local/cuda-12.9 bash \$RESEARCH_RUN_DIR/inputs/node_record.sh \$RESEARCH_RUN_DIR/out/node 2>&1 | tail -1; \
    VENV=$VENV SERVED_OUT=\$X/out OLD=/nonexistent IDLE_MIN=15 LONG=1 KEEP_TENSORS=opened SAMPLER=1 GATE_EVERY=0 STAGE=0 \
    ALLREDUCE_TREE='$TREE' ALLREDUCE_TREE_SOURCE='$SRC' $T bash \$RESEARCH_RUN_DIR/inputs/spot_canonical.sh" ;;
*) echo "MODE: P, R or C"; exit 2 ;;
esac

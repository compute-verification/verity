#!/usr/bin/env bash
# The node A pre-test of the cross-node replay: 72b6's served block (its record arm, served tokens and logprobs, engine env,
# read in place on node A) replayed from an empty engine with every hook on, the all-reduces past FlashInfer's and the custom
# route summed in node A's own NVSwitch tree by tree_allreduce.py. It must stay bit-exact, tokens and logprobs, as node A's
# replays 926b and c0f1 were without the plugin. Launch only once no other job holds glm53-sessions: this job's exit removes
# the hold file.
set -euo pipefail
export PATH=/agent/repos/verity/.git/verity-auth/bin:"$PATH"
F=/tmp/od8/send4; X=/workspace/research/runs/r20261107-171723-72b6/out
TREE='(((((((0,2),3),1),5),6),4),7)'
SRC='art:3ffc7fc21dc9978b0a071b8da7f6838594f356f9455ee5330d12f36de195f2e9 (kernels probe r20261107-192422-508b, vy-spot-glm53)'
SENDS=(); for f in serve_logprobs.patch replay_aborts.patch dsa_topk.py dsa_topk_kernels.py capture.py stage.py spot_job.sh spot_tp8.sh \
  gpu-lease sitecustomize.py long_rows.py spot_canonical.sh node_record.sh tree_allreduce.py; do SENDS+=(--send "$F/$f"); done
research run --on vy-spot-glm53 --project verity --source /tmp/od8/rc0 --cwd clone --timeout "${TIMEOUT:-4200}" "${SENDS[@]}" -- bash -c "\
  VENV=/workspace/venv312 CUDA_HOME=/usr/local/cuda-12.9 bash \"\$RESEARCH_RUN_DIR/inputs/node_record.sh\" \"\$RESEARCH_RUN_DIR/out/node-a\" 2>&1 | tail -1; \
  HOLD=/workspace/research/holds/glm53-sessions VENV=/workspace/venv312 SERVED_OUT=$X OLD=/nonexistent IDLE_MIN=15 LONG=1 \
  KEEP_TENSORS=opened SAMPLER=1 GATE_EVERY=0 STAGE=0 ALLREDUCE_TREE='$TREE' ALLREDUCE_TREE_SOURCE='$SRC' \
  TREE_AR='$TREE' TREE_AR_SOURCE='$SRC' bash \"\$RESEARCH_RUN_DIR/inputs/spot_canonical.sh\""

#!/usr/bin/env bash
# vllm's 128k serve r20261107-213820-c343, its whole block replayed from an empty engine on node A with capture.py at 275521d357
# (main), captured at step 5833 only (342 tokens, FA3 calls that don't split; run A captured it too, so the rank roots compare):
# one captured step writes about 13 GB of opened layers, so the job wants MIN_FREE_G=30 at its grant. Its end-of-block verdict
# is the token and logprob comparison against the served block. It queues for GPUs 0-7 on main's gpu_lease.sh behind LIFETIME's
# e6e3 as a --preemptible waiter, which preempts nothing (a plain waiter, once oldest, would stop e6e3's preemptible lease
# after its grace). It starts the replay only if granted by START_BY (17:30Z, the last 8-GPU start in node A's shared window),
# and TIMEOUT ends it before the window's 18:40Z close.
set -euo pipefail
export PATH=/agent/repos/verity/.git/verity-auth/bin:/workspace/.venv/bin:"$PATH"
F=/tmp/od8/send5; SERVED_OUT=/workspace/research/runs/r20261107-213820-c343/out
SENDS=(); for f in serve_logprobs.patch replay_aborts.patch dsa_topk.py dsa_topk_kernels.py capture.py stage.py spot_job.sh spot_tp8.sh \
    gpu-lease sitecustomize.py long_rows.py spot_canonical.sh gpu_lease.sh block_job.sh; do SENDS+=(--send "$F/$f"); done
research run --on vy-spot-glm53 --project verity --custody-ttl 9h --source /tmp/od8/rc0 --cwd clone --timeout "${TIMEOUT:?seconds until 18:35Z}" "${SENDS[@]}" -- bash -c "\
  echo \"QUEUED \$(date -u +%FT%TZ) DISK \$(df -BG --output=avail /workspace | tail -1)\"; \
  GPU_LEASE_WHO=proofs/overheaddef START_BY=${START_BY:-2026-11-08T17:30:00Z} MIN_FREE_G=${MIN_FREE_G:-30} \
  HOLD=/workspace/research/holds/glm53-sessions SERVED_OUT=$SERVED_OUT REC=$SERVED_OUT/boundary LOAD=saturated ENGINE_ENV_FILE=$SERVED_OUT/engine_env.txt \
  OLD=/nonexistent IDLE_MIN=5 STEPS=${STEPS:-5833} KEEP_TENSORS=opened MLA_TOKENS=8 SAMPLER=1 GATE_EVERY=0 STAGE=0 \
  ALLREDUCE_TREE='(((((((0,2),3),1),5),6),4),7)' ALLREDUCE_TREE_SOURCE='art:3ffc7fc21dc9978b0a071b8da7f6838594f356f9455ee5330d12f36de195f2e9 (kernels probe r20261107-192422-508b, vy-spot-glm53)' \
  bash \"\$RESEARCH_RUN_DIR/inputs/gpu_lease.sh\" 8 --on 0,1,2,3,4,5,6,7 --wait --preemptible --max-min ${MAX_MIN:-45} --mem-gb 0 -- bash \"\$RESEARCH_RUN_DIR/inputs/block_job.sh\""

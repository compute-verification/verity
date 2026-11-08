#!/usr/bin/env bash
# vllm's 128k serve r20261107-213820-c343, its whole block replayed from an empty engine on node A with capture.py at 275521d357
# (main), captured at step 4512 (at most 8 tokens) and step 5833 (FA3 calls that don't split), the steps run A
# (r20261108-092918-ebf8) captured, so the two replays' rank roots can be compared. Its end-of-block verdict is the token and
# logprob comparison against the served block. It takes GPUs 0-7 with main's gpu_lease.sh, first come first served behind the 72b6
# block (--wait, never preempting a lease not marked preemptible), and starts the replay only if granted by START_BY, so that it
# ends before 18:00Z (about 40 minutes); TIMEOUT stops it at 18:00Z in any case.
set -euo pipefail
export PATH=/agent/repos/verity/.git/verity-auth/bin:/workspace/.venv/bin:"$PATH"
F=/tmp/od8/send5; SERVED_OUT=/workspace/research/runs/r20261107-213820-c343/out
SENDS=(); for f in serve_logprobs.patch replay_aborts.patch dsa_topk.py dsa_topk_kernels.py capture.py stage.py spot_job.sh spot_tp8.sh \
    gpu-lease sitecustomize.py long_rows.py spot_canonical.sh gpu_lease.sh block_job.sh; do SENDS+=(--send "$F/$f"); done
research run --on vy-spot-glm53 --project verity --custody-ttl 9h --source /tmp/od8/rc0 --cwd clone --timeout "${TIMEOUT:?seconds until 18:00Z}" "${SENDS[@]}" -- bash -c "\
  echo \"QUEUED \$(date -u +%FT%TZ) DISK \$(df -BG --output=avail /workspace | tail -1)\"; \
  GPU_LEASE_WHO=proofs/overheaddef START_BY=${START_BY:-2026-11-08T17:10:00Z} \
  HOLD=/workspace/research/holds/glm53-sessions SERVED_OUT=$SERVED_OUT REC=$SERVED_OUT/boundary LOAD=saturated ENGINE_ENV_FILE=$SERVED_OUT/engine_env.txt \
  OLD=/nonexistent IDLE_MIN=5 STEPS=4512,5833 KEEP_TENSORS=opened MLA_TOKENS=8 SAMPLER=1 GATE_EVERY=0 STAGE=0 \
  ALLREDUCE_TREE='(((((((0,2),3),1),5),6),4),7)' ALLREDUCE_TREE_SOURCE='art:3ffc7fc21dc9978b0a071b8da7f6838594f356f9455ee5330d12f36de195f2e9 (kernels probe r20261107-192422-508b, vy-spot-glm53)' \
  bash \"\$RESEARCH_RUN_DIR/inputs/gpu_lease.sh\" 8 --on 0,1,2,3,4,5,6,7 --wait --max-min ${MAX_MIN:-60} --mem-gb 0 -- bash \"\$RESEARCH_RUN_DIR/inputs/block_job.sh\""

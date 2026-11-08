#!/usr/bin/env bash
# 72b6's whole served block replayed from an empty engine on node A with capture.py at 275521d357 (main), captured at step 163
# (PIECEWISE, 369 tokens) and step 1321 (at most 8 tokens: the router's ll_bf16_gemm). Its end-of-block verdict is the token and
# logprob comparison against the served block. Node A isn't a gpu-lease host for research run --gpus, so the workload takes GPUs
# 0-7 itself with main's gpu_lease.sh, first come first served (--wait, never preempting a lease not marked preemptible), for at
# most MAX_MIN minutes once granted. TIMEOUT: seconds until it must be done, the wait included.
set -euo pipefail
export PATH=/agent/repos/verity/.git/verity-auth/bin:/workspace/.venv/bin:"$PATH"
F=/tmp/od8/send5; X=/workspace/research/runs/r20261107-171723-72b6/out
SENDS=(); for f in serve_logprobs.patch replay_aborts.patch dsa_topk.py dsa_topk_kernels.py capture.py stage.py spot_job.sh spot_tp8.sh \
  gpu-lease sitecustomize.py long_rows.py spot_canonical.sh gpu_lease.sh block_job.sh; do SENDS+=(--send "$F/$f"); done
research run --on vy-spot-glm53 --project verity --custody-ttl 9h --source /tmp/od8/rc0 --cwd clone --timeout "${TIMEOUT:?seconds until done}" "${SENDS[@]}" -- bash -c "\
  echo \"QUEUED \$(date -u +%FT%TZ) DISK \$(df -BG --output=avail /workspace | tail -1)\"; \
  GPU_LEASE_WHO=proofs/overheaddef HOLD=/workspace/research/holds/glm53-sessions VENV=/workspace/venv312 SERVED_OUT=$X OLD=/nonexistent IDLE_MIN=5 STEPS=163,1321 \
  KEEP_TENSORS=opened SAMPLER=1 GATE_EVERY=0 STAGE=0 \
  ALLREDUCE_TREE='(((((((0,2),3),1),5),6),4),7)' ALLREDUCE_TREE_SOURCE='art:3ffc7fc21dc9978b0a071b8da7f6838594f356f9455ee5330d12f36de195f2e9 (kernels probe r20261107-192422-508b, vy-spot-glm53)' \
  bash \"\$RESEARCH_RUN_DIR/inputs/gpu_lease.sh\" 8 --on 0,1,2,3,4,5,6,7 --wait --max-min ${MAX_MIN:-45} --mem-gb 0 -- bash \"\$RESEARCH_RUN_DIR/inputs/block_job.sh\""

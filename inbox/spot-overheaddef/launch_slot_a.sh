#!/usr/bin/env bash
# Slot run A on node A: vllm's serve r20261107-213820-c343 replayed from an empty engine with capture.py at 275521d357, captured
# at step 4512 (at most 8 tokens: the router's ll_bf16_gemm) and step 5833 (where 9d0e's graph-off forward failed in FA3's fwd).
# No node record, no eviction, no gate, no staging. It is cancelled once step 5833's capture and commitment are written, so it
# replays no further than that and gives no whole-block verdict; the verdicts are each captured step's check.
set -euo pipefail
export PATH=/agent/repos/verity/.git/verity-auth/bin:/workspace/.venv/bin:"$PATH"
F=/tmp/od8/send5; SERVED_OUT=/workspace/research/runs/r20261107-213820-c343/out
SENDS=(); for f in serve_logprobs.patch replay_aborts.patch dsa_topk.py dsa_topk_kernels.py capture.py stage.py spot_job.sh spot_tp8.sh \
  gpu-lease sitecustomize.py long_rows.py spot_canonical.sh; do SENDS+=(--send "$F/$f"); done
research run --on vy-spot-glm53 --project verity --source /tmp/od8/rc0 --cwd clone --timeout "${TIMEOUT:-1800}" "${SENDS[@]}" -- bash -c "\
  echo \"DISK \$(df -BG --output=avail /workspace | tail -1)\"; \
  HOLD=/workspace/research/holds/glm53-sessions SERVED_OUT=$SERVED_OUT REC=$SERVED_OUT/boundary LOAD=saturated ENGINE_ENV_FILE=$SERVED_OUT/engine_env.txt \
  OLD=/nonexistent IDLE_MIN=5 STEPS=4512,5833 KEEP_TENSORS=opened MLA_TOKENS=8 SAMPLER=1 GATE_EVERY=0 STAGE=0 \
  ALLREDUCE_TREE='(((((((0,2),3),1),5),6),4),7)' ALLREDUCE_TREE_SOURCE='art:3ffc7fc21dc9978b0a071b8da7f6838594f356f9455ee5330d12f36de195f2e9 (kernels probe r20261107-192422-508b, vy-spot-glm53)' \
  bash \"\$RESEARCH_RUN_DIR/inputs/spot_canonical.sh\""

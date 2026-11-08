#!/usr/bin/env bash
# The 128k hooked replay of vllm's serve r20261107-213820-c343 on node A, over the whole replay block: every hook on, the
# opened layers kept at the coin-drawn steps only, no gate, no staging. It first evicts node A's store blobs that are
# preserved on the remote (approved), then replays the serve's client aborts at their steps (replay_aborts.patch), holds the
# replay to the client's tokens and the served schedule, records its own logprobs, and splits the result at 2,048 keys.
# Before the replay, node_record.sh records node A (OUT/node-a), the reference a replay on another node is compared with.
set -euo pipefail
export PATH=/agent/repos/verity/.git/verity-auth/bin:"$PATH"
F=/tmp/od8/send3a; SERVED_OUT=/workspace/research/runs/r20261107-213820-c343/out
SENDS=(); for f in serve_logprobs.patch replay_aborts.patch dsa_topk.py dsa_topk_kernels.py capture.py stage.py spot_job.sh spot_tp8.sh gpu-lease sitecustomize.py long_rows.py spot_canonical.sh node_record.sh; do SENDS+=(--send "$F/$f"); done
research run --on vy-spot-glm53 --project verity --source /tmp/od8/rc0 --cwd clone --timeout "${TIMEOUT:-3300}" "${SENDS[@]}" -- bash -c "\
  echo \"disk before: \$(df -BG --output=avail /workspace | tail -1)\"; research data evict --target-free-gb 285 2>&1 | tail -4; \
  echo \"DISK after evict: \$(df -BG --output=avail /workspace | tail -1)\"; \
  VENV=/workspace/venv312 CUDA_HOME=/usr/local/cuda-12.9 bash \"\$RESEARCH_RUN_DIR/inputs/node_record.sh\" \"\$RESEARCH_RUN_DIR/out/node-a\" 2>&1 | tail -2; \
  HOLD=/workspace/research/holds/glm53-sessions SERVED_OUT=$SERVED_OUT REC=$SERVED_OUT/boundary LOAD=saturated ENGINE_ENV_FILE=$SERVED_OUT/engine_env.txt \
  OLD=/nonexistent IDLE_MIN=15 LONG=1 LONG_KEYS=65536 LONG_TOKENS=1,384 LONG_PREFILL=0 BIG_MAX=1024 \
  KEEP_TENSORS=opened MLA_TOKENS=8 SAMPLER=1 GATE_EVERY=0 STAGE=0 \
  ALLREDUCE_TREE='(((((((0,2),3),1),5),6),4),7)' ALLREDUCE_TREE_SOURCE='art:3ffc7fc21dc9978b0a071b8da7f6838594f356f9455ee5330d12f36de195f2e9 (kernels probe r20261107-192422-508b, vy-spot-glm53)' \
  bash \"\$RESEARCH_RUN_DIR/inputs/spot_canonical.sh\""

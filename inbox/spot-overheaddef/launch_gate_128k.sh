#!/usr/bin/env bash
# usage: UNTIL=ISO launch_gate_128k.sh   (the 128k gate on node A, after the replay; CPU only)
set -euo pipefail
export PATH=/agent/repos/verity/.git/verity-auth/bin:"$PATH"
: "${UNTIL:?UNTIL: a UTC time}"
research run --on vy-spot-glm53 --project verity --source /tmp/od8/push --cwd source --timeout "${TIMEOUT:-5400}" \
  --send /tmp/od8/push/benchmarks/spot_proved/stage.py --send /tmp/od8/cputest/gate_128k.sh \
  -- bash -c "UNTIL=$UNTIL bash \"\$RESEARCH_RUN_DIR/inputs/gate_128k.sh\""

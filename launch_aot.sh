#!/usr/bin/env bash
# launch_aot.sh MACHINE NEUTRAL_RUN ROW CASE: declare a heavy re-pin's traced Build and compare it with the predictor, on the
# node that holds it.
cd /agent/repos/kwork/rebind || exit 2
research run --project verity --on "$1" --source . --cwd clone --custody-ttl 24h --timeout 7200 --mem-gb 96 --mem-wait 14400 \
  --send /agent/repos/kwork/jobs/aot.sh -- bash -lc "bash \"\$RESEARCH_RUN_DIR/inputs/aot.sh\" $2 $3 $4 8" 2>&1 | grep -v "commit(s) behind"

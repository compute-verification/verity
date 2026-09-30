#!/usr/bin/env bash
# The reference verifier on the window's two verify passes, CPU only, after the window: verify_run.py on each manifest under
# $WORK/passes/<window run id>, J processes each; the verdicts (verify.json) are copied into this run's directory.
#   research run --on vy-nebius-2 --project verity --campaign pouw --source <run tree> --cwd source \
#     --env GPU_LEASE_WHO=bc-dd22acf8 -- bash benchmarks/pouw/pearl_c_vllm/verify.sh <window run id> [J]
# Prints each pass's ACCEPT or REJECT line and exits non-zero unless the prefill and decode passes are accepted and the
# negative control's pass is rejected.
set -euo pipefail
WORK=${WORK:-/workspace/pouw/mvp-e2e}; PY=$WORK/venv312/bin/python
PASSES=$WORK/passes/${1:?the window run id}; JOBS=${2:-48}; OUT=${RESEARCH_RUN_DIR:-.}
export PYTHONPATH=$PWD/packages/verity/src:$PWD/protocols/pouw:$PWD/integrations/vllm:$PWD/benchmarks/pouw
status=0
for pass in prefill decode control; do
  "$PY" benchmarks/pouw/pearl_c_vllm/verify_run.py "$PASSES/$pass/manifest.json" --jobs "$JOBS" \
    $([ "$pass" = control ] && echo --expect-reject) || status=1
  cp "$PASSES/$pass/verify.json" "$OUT/$pass-verify.json" && cp "$PASSES/$pass/manifest.json" "$OUT/$pass-manifest.json"
done
exit $status

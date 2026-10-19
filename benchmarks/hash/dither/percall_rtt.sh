#!/usr/bin/env bash
# ASK-64's per-call round (percall_rtt.py): the challenger here, on the auditor's machine (vy-nebius-1, no GPU), the client on
# PEER (default vy-nebius-2, the serving node) through `ssh -R`.  The challenger module ships with the run (--send), at the
# head CHALLENGER_SHA names.  Extra arguments go to percall_rtt.py.
#   git show SHA:verity/core/protocols/sampled_proofs/challenger.py > /tmp/challenger.py
#   research run --on vy-nebius-1 --project verity --campaign dither-loop --source . --cwd source --send /tmp/challenger.py \
#     --env CHALLENGER_SHA=SHA -- bash benchmarks/hash/dither/percall_rtt.sh
set -uo pipefail
D=${RESEARCH_RUN_DIR:?}; OUT=$D/out; mkdir -p "$OUT"
git rev-parse HEAD 2>/dev/null | tee "$OUT/commit.txt"
echo "percall_rtt: challenger on $(hostname), peer ${PEER:-vy-nebius-2}, at $(date -u +%H:%M:%SZ)"
python3 benchmarks/hash/dither/percall_rtt.py --peer "${PEER:-vy-nebius-2}" --challenger "$D/inputs/challenger.py" \
  --challenger-sha "${CHALLENGER_SHA:-}" --out "$OUT/percall-rtt.json" "$@"
rc=$?
echo "percall_rtt: exited $rc at $(date -u +%H:%M:%SZ)"
exit "$rc"

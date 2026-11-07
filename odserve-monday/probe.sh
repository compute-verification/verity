#!/usr/bin/env bash
# The probe and the preflight on vy-dc-h200-2 at the run's tree, before the serve, CPU only:
#  1  probe/afa1_probe.sh: H1's window Program digests (program_probe.sh, at S 1, 4 and 16), whose S 1 must be
#     rr/circuit.txt's; the row's Build of record against the traced row (PREDICT-ROW, PREDICT-COMPARE exit 0); and the
#     installed settings file's sha, which must be the kit's settings/settings-monday.env's. Waited for up to 20 minutes;
#     a probe still running then is read again with PROBE_RID=ID bash probe.sh, which launches no second probe;
#  2  once the probe passes, serve.sh with PREFLIGHT_ONLY=1 and TIMEOUT=30m: remote_run.sh's checks (the gate's pin, H1's
#     registration, the row, the challenger key, the settings), ending "PREFLIGHT_ONLY: every check passed;
#     served_verdicts.sh not run" (runlog.sh h200-2 ID serve).
# Its lines and the runs' ids in LOGS/probe.txt and LOGS/preflight.txt.
#   [PROBE_RID=ID] [DRY=1] bash probe.sh
set -uo pipefail
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$H/env.sh" || exit 3
tree_ok || exit 3
want=$(sha256sum < "$H/settings/$SETTINGS_NAME" | cut -c1-16)
cmd=(research run --on "$SERVED" --project verity --source "$TREE" --cwd clone --timeout 30m --wait
  --send "$H/probe/afa1_probe.sh" --send "$H/probe/program_probe.sh" -- bash -c 'exec bash "$RESEARCH_RUN_DIR/inputs/afa1_probe.sh"')
[ -z "${DRY:-}" ] || { printf '%q ' "${cmd[@]}"; echo; PREFLIGHT_ONLY=1 TIMEOUT=30m DRY=1 bash "$H/serve.sh"; exit $?; }
rid=${PROBE_RID:-}
if [ -z "$rid" ]; then
  (cd "$TREE" && "${cmd[@]}") > "$LOGS/probe-launch.txt" 2>&1
  rc=$?
  rid=$(grep -o -E 'run r[0-9]{8}-[0-9]{6}-[0-9a-f]{4} launched' "$LOGS/probe-launch.txt" | head -1 | cut -d' ' -f2)
  echo "PROBE ${rid:-not launched} exit $rc"
  [ -n "$rid" ] || { tail -5 "$LOGS/probe-launch.txt"; exit 3; }
  [ "$rc" != 79 ] || { echo "PROBE still running after 20 minutes: PROBE_RID=$rid bash probe.sh once it ends"; exit 2; }
fi
bash "$H/runlog.sh" h200-2 "$rid" all > "$LOGS/probe.txt" 2>&1
cat "$LOGS/probe.txt"
echo "WANT: DIGEST S 1 $(head -c 16 "$H/rr/circuit.txt")..., PREDICT-COMPARE ... exit 0, SETTINGS $ONSITE/$SETTINGS_NAME $want"
grep -q "^DIGEST S 1 $(cat "$H/rr/circuit.txt")$" "$LOGS/probe.txt" && grep -q '^PREDICT-COMPARE .* exit 0$' "$LOGS/probe.txt" \
  && grep -q -F "SETTINGS $ONSITE/$SETTINGS_NAME $want" "$LOGS/probe.txt" || { echo "PROBE: FAIL"; exit 1; }
echo "PROBE: PASS"
PREFLIGHT_ONLY=1 TIMEOUT=30m bash "$H/serve.sh" > "$LOGS/preflight.txt" 2>&1
echo "PREFLIGHT launch exit $?: $(grep -o -E '^r[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$' "$LOGS/preflight.txt" | head -1) (bash runlog.sh h200-2 ID serve once it ends)"

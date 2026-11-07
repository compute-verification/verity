#!/usr/bin/env bash
# Monday's served run on vy-dc-h200-2: rr/launch.sh (H1 at S 1 under rr/circuit.txt's window Program, weights root
# H1_ROOT) at the run's tree, with the installed settings file (settings-monday.env: V3c 8,32,4,4,16 at 120m, OF_RECORD=1,
# untied, subset:1), the front's id form and the traced row. rr/main.txt, main's first-parent line from the challenger
# gate's pin, is written from the tree first. Its --timeout ends it by END (the node's lease end less the publish margin,
# an ISO time: required unless TIMEOUT is set) and within MAX_S (8 h, which also bounds how far research run moves the
# node's deadline), and it refuses with less than MIN_S (4 h: the serve, step 3's records at WC_TIMEOUT and the commit at
# V3C_TIMEOUT) left. It prints the run's id (a store run id, which custody publishes) and
# returns once the node has started it; its custody key lives CUSTODY_TTL (rr/launch.sh's 20h).
# PREFLIGHT_ONLY=1 stops at remote_run.sh's own checks (probe.sh runs it so).
#   END=2026-11-09T22:00:00Z [PREFLIGHT_ONLY=1] [TIMEOUT=30m] [DRY=1] bash serve.sh
set -euo pipefail
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$H/env.sh"
tree_ok
! grep -q -- '"--case-only"' "$TREE/benchmarks/network_traces/certifier_piece.py" \
  || echo "NOTE: this tree's certifier_piece.py takes --case-only; settings-monday.env runs the period in full, inside its 1h cap"
git -C "$TREE" cat-file -e "$SERVE_GATE_PIN^{commit}" \
  || { echo "REFUSED: the tree doesn't hold the gate's pin $SERVE_GATE_PIN"; exit 3; }
git -C "$TREE" rev-list --first-parent "$SERVE_GATE_PIN" > "$H/rr/main.txt"
SETTINGS=$ONSITE/$SETTINGS_NAME
if [ -z "${TIMEOUT:-}" ]; then
  [ -n "${END:-}" ] || { echo "REFUSED: set END, the lease end of the node less the publish margin (END=2026-11-09T22:00:00Z), or TIMEOUT"; exit 3; }
  end=$(date -u -d "$END" +%s) now=$(date -u +%s)
  TIMEOUT=$(( end - now ))
  [ "$TIMEOUT" -ge "${MIN_S:-14400}" ] || { echo "REFUSED: $TIMEOUT s to $END, under MIN_S ${MIN_S:-14400}"; exit 3; }
  [ "$TIMEOUT" -le "${MAX_S:-28800}" ] || TIMEOUT=${MAX_S:-28800}
fi
args=("$SERVED" "SETTINGS=$SETTINGS" "SERVE_EXTRA=--protocol ids --id-payload 503" "ROW=$ROW")
[ -z "${PREFLIGHT_ONLY:-}" ] || args+=("PREFLIGHT_ONLY=$PREFLIGHT_ONLY")
echo "TREE $(git -C "$TREE" rev-parse --short=11 HEAD) (= $REF), $SETTINGS_NAME $(sha256sum < "$H/settings/$SETTINGS_NAME" | cut -c1-16), main.txt $(wc -l < "$H/rr/main.txt") commits from $(head -c 11 "$H/rr/main.txt"), TIMEOUT $TIMEOUT${PREFLIGHT_ONLY:+, PREFLIGHT_ONLY}"
[ -z "${DRY:-}" ] || { printf 'H1_ROOT=%q SOURCE=%q TIMEOUT=%q bash %q' "$H1_ROOT" "$TREE" "$TIMEOUT" "$H/rr/launch.sh"; printf ' %q' "${args[@]}"; echo; exit 0; }
export H1_ROOT SOURCE=$TREE TIMEOUT
cd "$TREE"
exec bash "$H/rr/launch.sh" "${args[@]}"

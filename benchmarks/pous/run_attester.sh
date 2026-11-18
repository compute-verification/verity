#!/usr/bin/env bash
# The auditor's attesters of a development run, which served_piece.sh --run-keys starts in place of the gate's verbs
# (host.py's party_argv): PoUS's memory attester and PoUB's bandwidth party on this node, listening on loopback only,
# under the key made for the run in DIR/pous, with PY the tree's Python:
#
#   run_attester.sh challenger_party PY DIR
#   run_attester.sh bw_party PY DIR LEAN --minutes M --pous-session SID [--session SID --trusted-sha512 HEX]
#
# LEAN is the built poub-verifier. The bandwidth party keeps its answers in DIR/bw.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
USAGE="usage: run_attester.sh challenger_party|bw_party PY DIR [LEAN --minutes M ...]"
VERB=${1:?$USAGE} PY=${2:?$USAGE} DIR=${3:?$USAGE}
shift 3
A=verity.protocols.resource_accounting.memory.pous.attester
cd "$ROOT"
case $VERB in
  challenger_party)
    [ $# = 0 ] || { echo "run_attester.sh: challenger_party takes no flags, got $*" >&2; exit 2; }
    exec env -u PYTHONPATH "$PY" -m "$A.challenger_party" --device "$DIR/pous" --listen 127.0.0.1:47610 ;;
  bw_party)
    LEAN=${1:?$USAGE}
    shift
    [ "${1:-}" = --minutes ] && [ $# -ge 2 ] || { echo "run_attester.sh: bw_party's --minutes M comes first, got $*" >&2; exit 2; }
    m=$2
    shift 2
    args=()
    while [ $# -gt 1 ]; do
      case $1 in
        --pous-session) args+=(--pous-record "$DIR/pous" --pous-session "$2") ;;
        --session) args+=(--challenger "$DIR/pous" --session "$2") ;;
        --trusted-sha512) args+=(--trusted-sha512 "$2") ;;
        *) break ;;
      esac
      shift 2
    done
    [ $# = 0 ] || { echo "run_attester.sh: bw_party takes --pous-session SID, --session SID and --trusted-sha512 HEX, got $*" >&2; exit 2; }
    exec timeout --foreground "${m}m" env -u PYTHONPATH "$PY" -c "import sys
from $A import onsite
from verity.protocols.resource_accounting.memory.poub.verifier import p2_parties
sys.exit(p2_parties.main(sys.argv[1:], device_key=onsite.device_key))" \
      --device "$DIR/bw" --listen 127.0.0.1:0 --lean-exe "$LEAN" "${args[@]}" ;;
  *) echo "$USAGE" >&2; exit 2 ;;
esac

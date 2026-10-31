#!/usr/bin/env bash
# The wipe's piece of a served window (served_verdicts.sh): one streamed wipe (`demo.sh`) of the card the
# window is served on, before its engine loads, the fill taking the whole HBM. A piece as served_verdicts.py runs one, from
# the tree's root, writing its core answer (verity/wipe/v0's, `wipe_period.leg`'s) to DIR/answer.json:
#
#   WIPE_CMD='bash benchmarks/pous/streamed_wipe/served_wipe.sh "$1" "$2" "$3"'   # bash -c "$WIPE_CMD" _ SERVE_DIR WINDOW DIR
#
# Inside the serve's gpu-lease (GPU_LEASE_UUID set) it wipes that card, the one the engine then loads on; outside one it
# takes a card of its own (`gpu-lease 1 --wait`) and gives it back after. It refuses (exit 3, no answer, the reason in
# DIR/refused.txt) a card that isn't idle, any process on it or more than WIPE_IDLE_MIB (1024) of its memory in use, and
# a challenger command whose ssh key (-i) isn't on this node; it kills nothing. The memory challenger is vy-auditor-2's,
# through infra's wipe gate, as in Demo B (r20261027-052452-ff66, accepted on vy-dc-h200-2):
#   WIPE_CHALLENGER  its host (vy-auditor-2)
#   WIPE_ADDRESS     where the prover connects to it (10.96.0.36:47611)
#   WIPE_COMMAND     what starts its side (ssh -i ~/.ssh/vy-auditor2-wipe -o BatchMode=yes research@10.96.0.36)
#   WIPE_READS       the reads the gate's verb runs, which the leg holds the session to (5500)
#   WIPE_NODE        the wiped server's id (this host's name, as served_verdicts.py record's --node defaults to)
# Further arguments go to demo.sh (e.g. --no-lean). The Lean judge (wipe-protocol) is built before the card is checked, in a
# scratch copy of verity/core (WIPE_PROTOCOL_EXE names one built already), never in this tree's .lake, which
# served_verdicts.sh's Lean builds write meanwhile. DIR gets demo.sh's outputs (env.txt, gpu.txt, period.json, wipe.json,
# wipe.case.json, summary.json), lake-build.log, answer.json (wipe.json's answer, whatever its verdict) and served.json (the
# serve dir and window the wipe precedes, the server, the card, the lease, the run); its last line is `WIPE {...}`.
# Exit: demo.sh's (0 on accept, 1 on another verdict), 3 on a refusal, 7 when wipe-protocol doesn't build.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SELF=$HERE/$(basename "$0")
ROOT=$(cd "$HERE/../../.." && pwd)
USAGE="usage: served_wipe.sh SERVE_DIR WINDOW DIR [DEMO_OPTIONS]"
SERVE_DIR=${1:?$USAGE} WINDOW=${2?$USAGE} DIR=${3:?$USAGE}
shift 3
mkdir -p "$DIR"
DIR=$(cd "$DIR" && pwd)
CH=${WIPE_CHALLENGER:-vy-auditor-2} ADDR=${WIPE_ADDRESS:-10.96.0.36:47611} READS=${WIPE_READS:-5500}
CMD=${WIPE_COMMAND:-ssh -i $HOME/.ssh/vy-auditor2-wipe -o BatchMode=yes research@10.96.0.36}
NODE=${WIPE_NODE:-$(hostname)} IDLE_MIB=${WIPE_IDLE_MIB:-1024} DEMO=${WIPE_DEMO:-$HERE/demo.sh}
refuse() { echo "WIPE REFUSED: $*" | tee "$DIR/refused.txt"; exit 3; }

if [ -z "${WIPE_OWN_LEASE:-}" ]; then  # once, before any lease: the key, the Lean judge
  rm -f "$DIR"/{answer,served,summary,wipe,wipe.case}.json "$DIR/refused.txt"
  read -r -a C <<< "$CMD"
  KEY=
  for ((i = 0; i + 1 < ${#C[@]}; i++)); do [ "${C[$i]}" = -i ] && KEY=${C[$((i + 1))]}; done
  KEY=${KEY/#\~/$HOME}
  [ -z "$KEY" ] || [ -r "$KEY" ] || refuse "no wipe gate key $KEY on $NODE ($(hostname)): infra installs it, and its gate verb on $CH for this card"
  if [ -z "${WIPE_PROTOCOL_EXE:-}" ] && [[ " $* " != *" --no-lean "* ]]; then
    S=$(mktemp -d)
    trap 'rm -rf "$S"' EXIT
    L=verity/core/devices/memory_challenger/lean
    tar -C "$ROOT" -c --exclude=.lake --exclude=__pycache__ verity/core | tar -C "$S" -x
    export PATH="$HOME/.elan/bin:$PATH"
    command -v lake >/dev/null 2>&1 || bash "$ROOT/tools/verity/lean/setup.sh" "$S/$L" > "$DIR/lake-setup.log" 2>&1
    (cd "$S/$L" && lake build wipe-protocol) > "$DIR/lake-build.log" 2>&1 \
      || { tail -20 "$DIR/lake-build.log"; echo "WIPE FAILED: wipe-protocol doesn't build ($DIR/lake-build.log)"; exit 7; }
    export WIPE_PROTOCOL_EXE=$S/$L/.lake/build/bin/wipe-protocol
  fi
  if [ -z "${GPU_LEASE_UUID:-}" ]; then
    command -v gpu-lease >/dev/null 2>&1 || refuse "no gpu-lease on $NODE to hold the card by"
    rc=0
    WIPE_OWN_LEASE=1 GPU_LEASE_WHO=memory-accounting gpu-lease 1 --wait --max-min 30 -- \
      bash "$SELF" "$SERVE_DIR" "$WINDOW" "$DIR" "$@" || rc=$?
    exit "$rc"
  fi
fi

# under the lease: the card idle, then the wipe, then its answer
CARD=${GPU_LEASE_UUID%%,*}
APPS=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$CARD" 2>&1) \
  || refuse "nvidia-smi can't read the card $CARD: $APPS"
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CARD" 2>&1) \
  || refuse "nvidia-smi can't read the card $CARD: $USED"
[ -z "$APPS" ] || refuse "the card $CARD isn't idle: $(echo "$APPS" | sed 's/^/pid /' | paste -sd ';' -) on it; the wipe" \
  "fills the whole card and kills nothing"
[ "$USED" -le "$IDLE_MIB" ] 2>/dev/null || refuse "the card $CARD isn't idle: $USED MiB of its memory in use, past" \
  "WIPE_IDLE_MIB $IDLE_MIB; the wipe fills the whole card and kills nothing"
unset PYTHONPATH VIRTUAL_ENV
rc=0
OUT=$DIR bash "$DEMO" --node "$NODE" --challenger "$CH" --challenger-address "$ADDR" --challenger-command "$CMD" \
  --reads "$READS" ${WIPE_PROTOCOL_EXE:+--wipe-protocol "$WIPE_PROTOCOL_EXE"} "$@" || rc=$?
python3 - "$DIR" "$SERVE_DIR" "$WINDOW" "$NODE" "$CARD" "${WIPE_OWN_LEASE:+own}" "$rc" <<'EOF'
import json, os, sys
from pathlib import Path

d, serve_dir, window, node, card, own, rc = sys.argv[1:]
d = Path(d)
served = {"serve_dir": serve_dir, "window": window, "server": node, "card": card, "run": os.environ.get("RESEARCH_RUN_ID"),
          "lease": "its own (gpu-lease 1 --wait), given back before the serve takes one" if own else
          "the serve's: its engine loads on this card after the wipe"}
(d / "served.json").write_text(json.dumps(served, indent=1, sort_keys=True) + "\n")
try:
    answer = json.loads((d / "wipe.json").read_text())["answer"]
except (OSError, ValueError, KeyError) as e:
    print(f"WIPE FAILED: demo.sh (exit {rc}) wrote no wipe.json answer: {type(e).__name__}: {e}"[:600])
    sys.exit(0)
(d / "answer.json").write_text(json.dumps(answer, sort_keys=True) + "\n")
try:
    s = json.loads((d / "summary.json").read_text())
except (OSError, ValueError):
    s = {}
print("WIPE " + json.dumps({"verdict": answer["verdict"], "reason": answer["reason"], "protocol": answer["protocol"],
                            "judge": s.get("judge"), "k": s.get("k"), "segments": s.get("segments"),
                            "fill_GBps": (s.get("challenger") or {}).get("fill_GBps"), "t0": s.get("t0"), "t1": s.get("t1"),
                            "server": node, "card": card, "exit": int(rc)}, sort_keys=True), flush=True)
EOF
exit "$rc"

#!/usr/bin/env bash
# The wipe's launcher: the wipe of block 0, in a process of its own on the card a window is served on, before its engine
# loads (served_verdicts.sh's WIPE_CMD), or on a card of its own. It attaches the wipe scheme by name (`Exchange.attach`,
# `wipe_served.attach`) with the runtime's side, and writes the piece served_verdicts.py reads, from the tree's root:
#
#   WIPE_CMD='bash benchmarks/pous/streamed_wipe/served_wipe.sh "$1" "$2" "$3" [SETTINGS]'   # _ SERVE_DIR WINDOW DIR
#
# SETTINGS is the wipe's settings (`WIPE.settings`), one JSON object or a file holding one; by default the auditor's
# memory attester through infra's wipe gate (vy-auditor-2, `DEFAULT` below). A setting the application's configuration
# states (VERITY_PROTOCOLS_CONFIG's `wipe`) is the application's: SETTINGS stating it otherwise is refused. The control
# link is the node's own (`ip route`) unless SETTINGS names `links`; a `disk.paths` directory gets a scratch directory of
# the run's inside it, removed when the wipe ends. With `run_keys` the memory attester is this node's, under a key made for
# the run in SERVE_DIR/private/run-keys/wipe (0700), removed when the wipe ends: a rehearsal.
#   WIPE_KEY_SHA512    the memory attester's key the verifier trusts (the attach's `trusted`), in 128 lowercase hex
#                      digits; unset, the key the run's configuration pins for the wipe (VERITY_PINS); set empty, none:
#                      refused
#   WIPE_NODE          the wiped server's id (this host's name)
#   WIPE_CARD          the card's SKU the wipe is measured for (h200-sxm-141gb; `any` takes any), a cuda store's only
#   WIPE_PROTOCOL_EXE  the Lean judge built already; else built here, in a scratch copy of verity/ (never this tree's
#                      .lake), beside the lease and the wipe, which waits for it only to answer (`wipe_served.built`);
#                      one that doesn't build ends the piece after the wipe with its own error and exit 7
# Inside the serve's gpu-lease (GPU_LEASE_UUID) it wipes that card; outside one it takes a card of its own (`gpu-lease 1
# --wait`, else the tree's gpu_lease.sh) and gives it back after. DIR gets period.json, wipe.json, summary.json,
# answer.json (wipe.json's answer), served.json (the serve dir and window, the server, the card, the lease, the run, the
# key's pin), lake-build.log and stamps.txt (`UTC EVENT` lines). The case goes to SERVE_DIR/private/wipe (0700), and
# served_case.py writes the composite's piece beside it. Its last line is `WIPE {...}`.
# A refusal, the attach's or this launcher's, each by name, writes its one sentence to DIR/refused.txt and no
# answer.json, and exits 3. Exit 0 on accept, 1 on another verdict or a wipe that failed, 7 when the judge doesn't build.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SELF=$HERE/$(basename "$0")
ROOT=$(cd "$HERE/../../.." && pwd)
USAGE="usage: served_wipe.sh SERVE_DIR WINDOW DIR [SETTINGS]"
SERVE_DIR=${1:?$USAGE} WINDOW=${2?$USAGE} DIR=${3:?$USAGE} SETTINGS=${4:-}
mkdir -p "$DIR"
DIR=$(cd "$DIR" && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
stamp() { echo "$(date -u +%FT%TZ) $*" >> "$DIR/stamps.txt"; }
refuse() { echo "WIPE REFUSED: $*" > "$DIR/refused.txt"; refused; }
refused() { cat "$DIR/refused.txt"; stamp refused; exit 3; }
# `attach.py check|run`: the settings checked and the attach's refusals made before any lease, then the wipe itself
attach() { "$PY" - "$@" "$DIR" "$SERVE_DIR" "$WINDOW" "$SETTINGS" "${WIPE_NODE:-$(hostname)}" \
  "${WIPE_KEY_SHA512:-}" "${WIPE_PROTOCOL_EXE:-}" "${VERITY_PINS:-}" "${VERITY_CONFIGURATION:-}" \
  "${VERITY_PROTOCOLS_CONFIG:-}" "${GPU_LEASE_UUID:-}" "${WIPE_OWN_LEASE:+own}" "$CTRL" <<'EOF'
import json, os, shutil, sys, tempfile
from pathlib import Path
from verity import hooks as H
from verity.protocols.compose import schemes
from verity.protocols.resource_accounting import scheme as S
from verity.protocols.resource_accounting.memory.pous.prover import wipe_served as WS
from verity.runtime.stream import Side
from verity_vllm.messages import Exchange

DEFAULT = {"host": "vy-auditor-2", "address": "10.96.0.36:47611", "reads": 5500,
           "command": ["ssh", "-i", "~/.ssh/vy-auditor2-wipe", "-o", "BatchMode=yes", "research@10.96.0.36"]}
mode, d, serve_dir, window, given, node, trusted, judge, pins, configuration, app, lease, own, ctrl = sys.argv[1:]
d, wipe = Path(d), schemes.get("wipe")


def refused(why):
    (d / "refused.txt").write_text(f"WIPE REFUSED: {why}\n")
    sys.exit(3)


try:
    doc = DEFAULT if not given else json.loads(Path(given).read_text() if Path(given).is_file() else given)
    app = (json.loads(app) if app else {}).get(wipe.name) or {}
except (OSError, ValueError) as e:
    refused(f"the wipe's settings are one JSON object, or a file holding one: {e}")
if clash := sorted(k for k in set(doc) & set(app) if doc[k] != app[k]):
    refused(f"the application's configuration states {clash} for the wipe, and the launcher's settings state them "
            "otherwise: the application's stand")
doc = {**doc, **app}
if "links" not in doc and ctrl:
    dev, src, mbps = ctrl.split(" ")
    doc["links"] = [{"name": "control", "to": "outside", "via": f"{dev} {src}: the node's own NIC (ssh, research run)",
                     "mbps": int(mbps) if mbps.isdigit() and int(mbps) > 0 else None}]
pins = json.loads(pins or "[]")
ctx = {"process": H.ENGINE, "side_of": Side.of, "private": f"{serve_dir}/private", "pins": pins, "trusted": trusted or None,
       "node": node, "judge": judge or None}
try:
    settings = S.settings(wipe, doc)
    if why := WS.refusals(ctx | {"settings": settings}):
        raise ValueError(f"wipe: {'; '.join(why)}")
except ValueError as e:
    refused(str(e).removeprefix("wipe: "))
if mode == "check":
    print(settings["store"])
    sys.exit(0)
made = []
for i, p in enumerate((doc.get("disk") or {}).get("paths") or ()):
    if Path(p).is_dir():
        made.append(tempfile.mkdtemp(prefix="verity-wipe-disk.", dir=p))
        doc["disk"]["paths"][i] = made[-1]
    elif not Path(p).is_block_device():
        refused(f"the disk target {p} is neither a directory nor a block device")
try:
    got = Exchange().attach(wipe, doc, **ctx)
except ValueError as e:
    refused(str(e).removeprefix("wipe: "))
except Exception as e:
    print(f"WIPE FAILED: the wipe raised {type(e).__name__}: {e}"[:600], flush=True)
    sys.exit(1)
finally:
    for m in made:
        shutil.rmtree(m, ignore_errors=True)
private = Path(serve_dir) / "private" / wipe.name
shutil.copy(private / "period.json", d / "period.json")
(d / "wipe.json").write_text(json.dumps(got.record(), sort_keys=True, indent=1))
(d / "summary.json").write_text(json.dumps(s := got.summary(), indent=1, sort_keys=True))
answer, ran = got.answer, got.record()["served"]
card = lease.split(",")[0] or ran["card"]
(d / "served.json").write_text(json.dumps({
    "serve_dir": serve_dir, "window": window, "server": ran["server"], "card": card, "key_pin": ran["key_pin"],
    "run": os.environ.get("RESEARCH_RUN_ID"),
    "lease": "its own (gpu-lease 1 --wait), given back before the serve takes one" if own else
             "the serve's: its engine loads on this card after the wipe"}, indent=1, sort_keys=True) + "\n")
(d / "answer.json").write_text(json.dumps(answer, sort_keys=True) + "\n")
print("WIPE " + json.dumps({"verdict": answer["verdict"], "reason": answer["reason"], "protocol": answer["protocol"],
                            "decided_by": s.get("decided_by"), "k": s.get("k"), "segments": s.get("segments"),
                            "fill_GBps": (s.get("challenger") or {}).get("fill_GBps"), "t0": s.get("t0"),
                            "t1": s.get("t1"), "server": ran["server"], "card": card, "key_pin": ran["key_pin"],
                            "exit": 0 if answer["verdict"] == "accept" else 1}, sort_keys=True), flush=True)
sys.exit(0 if answer["verdict"] == "accept" else 1)
EOF
}

if [ -z "${WIPE_OWN_LEASE:-}" ]; then  # once, before any lease: the node's link, the settings, the refusals, the judge
  rm -f "$DIR"/{answer,served,summary,wipe,period}.json "$DIR/refused.txt" "$DIR/stamps.txt" \
    "$SERVE_DIR"/private/wipe/{answer,case,wipe.case}.json
  stamp start
  # served_table.py's form (`_wipe`), until it hands SETTINGS: `--run-keys`, or the bound attester's
  # WIPE_CHALLENGER, WIPE_ADDRESS and WIPE_COMMAND
  if [ "$SETTINGS" = --run-keys ]; then
    for v in WIPE_KEY_SHA512 WIPE_ADDRESS WIPE_CHALLENGER WIPE_COMMAND; do
      [ -z "${!v+set}" ] || refuse "--run-keys runs the memory attester on this node under a key made for the run," \
        "and $v is set: unset $v, or drop --run-keys and bind the auditor's memory attester"
    done
    SETTINGS='{"run_keys": true}'
  elif [ -z "$SETTINGS" ] && [ -n "${WIPE_COMMAND:-}" ]; then
    SETTINGS=$(python3 -c 'import json, shlex, sys; h, a, c = sys.argv[1:]
print(json.dumps({"host": h, "address": a, "command": shlex.split(c), "reads": 5500}))' \
      "${WIPE_CHALLENGER:-vy-auditor-2}" "${WIPE_ADDRESS:-10.96.0.36:47611}" "$WIPE_COMMAND")
  fi
  [ "${WIPE_KEY_SHA512-unset}" != "" ] || refuse "no challenger key is pinned: WIPE_KEY_SHA512 is set empty, which" \
    "pins none (served_table.py's command when the configuration pins no key for the wipe)"
  command -v ip > /dev/null || refuse "this host has no \`ip\` (iproute2), which names the node's links: apt-get install iproute2"
  read -r DEV SRC < <(ip -o route get 1.1.1.1 2>/dev/null | sed -nE 's/.* dev ([^ ]+).* src ([^ ]+).*/\1 \2/p') || true
  export WIPE_CTRL=${DEV:+$DEV $SRC $(cat "/sys/class/net/$DEV/speed" 2>/dev/null || echo -)}
  CTRL=$WIPE_CTRL
  attach check > /dev/null || { [ $? != 3 ] || refused; exit 1; }
  if [ -z "${WIPE_PROTOCOL_EXE:-}" ]; then
    S=$(mktemp -d)
    trap 'rm -rf "$S"' EXIT
    L=verity/protocols/resource_accounting/memory/pous/attester/lean
    # tar's exit 1 is only a file that changed as it read it: a process importing the tree's Python writes __pycache__
    # beside it meanwhile, as a check's other suites do; 2 or more is a copy that failed
    { tar -C "$ROOT" -c --exclude=.lake --exclude=__pycache__ --warning=no-file-changed verity || [ $? = 1 ]; } \
      | tar -C "$S" -x || { echo "WIPE FAILED: verity/ doesn't copy for wipe-protocol's build"; exit 7; }
    export PATH="$HOME/.elan/bin:$PATH"
    export WIPE_PROTOCOL_EXE=$S/$L/.lake/build/bin/wipe-protocol
    BUILT=${WIPE_PROTOCOL_EXE%/*}
    mkdir -p "$BUILT"
    # the lock is taken before the build starts and held by the build alone, so the answer waits for its end
    exec {LOCK}>"$BUILT/wipe-protocol.lock"
    flock -x "$LOCK"
    stamp "the Lean build starts"
    (
      command -v lake >/dev/null 2>&1 \
        || (exec {LOCK}>&- && bash "$ROOT/tools/verity/lean/setup.sh" "$S/$L") > "$DIR/lake-setup.log" 2>&1
      if (cd "$S/$L" && exec {LOCK}>&- && exec lake build wipe-protocol) > "$DIR/lake-build.log" 2>&1; then
        : > "$BUILT/wipe-protocol.ok"
        stamp "built wipe-protocol"
      else
        { tail -20 "$DIR/lake-build.log"; echo "WIPE FAILED: wipe-protocol doesn't build ($DIR/lake-build.log)"; } \
          > "$BUILT/failed"
        stamp "wipe-protocol didn't build"
      fi
    ) &
    BUILDING=$!
    exec {LOCK}>&-
    trap 'kill "$BUILDING" 2> /dev/null || :; rm -rf "$S"' EXIT
  fi
  if [ -z "${GPU_LEASE_UUID:-}" ]; then
    LEASE=(gpu-lease)
    command -v gpu-lease >/dev/null 2>&1 || LEASE=(bash "$ROOT/tools/research/src/research/pods/sh/gpu_lease.sh")
    rc=0
    stamp "asked for the card"
    WIPE_OWN_LEASE=1 GPU_LEASE_WHO=memory-accounting "${LEASE[@]}" 1 --wait --max-min 30 -- \
      bash "$SELF" "$SERVE_DIR" "$WINDOW" "$DIR" "$SETTINGS" || rc=$?
    exit "$rc"
  fi
fi

# under the lease: the card's SKU, then the wipe (the attach checks the card idle), then its piece
[ -z "${WIPE_OWN_LEASE:-}" ] || stamp "leased the card"
CTRL=${WIPE_CTRL:-}
CARD=${GPU_LEASE_UUID%%,*}
export CUDA_VISIBLE_DEVICES=${GPU_LEASE_UUID:-${CUDA_VISIBLE_DEVICES:-}}
STORE=$(attach check) || { [ $? != 3 ] || refused; exit 1; }
if [ "${WIPE_CARD:-h200-sxm-141gb}" != any ] && [ "$STORE" = cuda ]; then
  NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$CARD" 2>&1 | head -1) \
    || refuse "nvidia-smi can't read the card $CARD: $NAME"
  SKU=$("$PY" "$ROOT/benchmarks/kernels/machine.py" sku "$NAME" 2>&1) || refuse "the card $CARD: $SKU"
  [ "$SKU" = "${WIPE_CARD:-h200-sxm-141gb}" ] || refuse "the card $CARD is '$NAME', a $SKU, not a ${WIPE_CARD:-h200-sxm-141gb}"
fi
K=$SERVE_DIR/private/run-keys/wipe
rm -rf "$K" && (umask 077 && mkdir -p "$K") \
  || refuse "the serve dir's private directory $SERVE_DIR/private can't hold the run's keys and the wipe's case"
rc=0
SAID=$(TMPDIR=$K attach run) || rc=$?
rm -rf "$K"
[ "$rc" != 3 ] || refused
stamp "the wipe exit $rc"
if [ -e "${WIPE_PROTOCOL_EXE:-}.lock" ] \
   && ! bash "$ROOT/benchmarks/pous/lean_ready.sh" "${WIPE_PROTOCOL_EXE%/*}" wipe-protocol true 2> /dev/null; then
  echo "$SAID"
  cat "${WIPE_PROTOCOL_EXE%/*}/failed"
  stamp "the piece stops: wipe-protocol didn't build"
  exit 7
fi
P=$SERVE_DIR/private/wipe
if [ -f "$DIR/answer.json" ] && [ -f "$P/wipe.case.json" ]; then
  PYTHONPATH=$ROOT "$PY" "$ROOT/benchmarks/pous/served_case.py" --window "$DIR" --private "$P" \
    || echo "WIPE CASE: served_case.py exited $?: the composite has no wipe piece"
fi
stamp "exit $rc"
echo "$SAID"
exit "$rc"

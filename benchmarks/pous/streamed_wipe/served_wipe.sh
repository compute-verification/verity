#!/usr/bin/env bash
# The wipe's launcher: the wipe of block 0, in a process of its own on the card a window is served on, before its engine
# loads (served_verdicts.sh's WIPE_CMD), or on a card of its own. It attaches the wipe scheme by name (`Exchange.attach`,
# `wipe_served.attach`) with the runtime's side, and writes the piece served_verdicts.py reads, from the tree's root:
#
#   WIPE_CMD='bash benchmarks/pous/streamed_wipe/served_wipe.sh "$1" "$2" "$3" [SETTINGS]'   # _ SERVE_DIR WINDOW DIR
#
# SETTINGS is one JSON object, or a file holding one: the wipe's parameters (`WIPE.settings`, checked by
# `S.settings`) and this node's facts (`wipe_served.FACTS`, handed in the attach's context, the rest at their defaults),
# any other key refused; by default the auditor's memory attester through infra's wipe gate (vy-auditor-2, `DEFAULT`
# below). The pins and the configuration's hash are the attach's own reading (VERITY_PINS, VERITY_CONFIGURATION,
# `Exchange.attach`). A setting the application's configuration
# states (VERITY_PROTOCOLS_CONFIG's `wipe`) is the application's: SETTINGS stating it otherwise is refused. The control
# link is the node's own (`ip route`) unless SETTINGS names `links`; a `disk.paths` directory gets a scratch directory of
# the run's inside it, removed when the wipe ends. With `run_keys` the memory attester is this node's, under a key made for
# the run in SERVE_DIR/private/run-keys/wipe (0700), removed when the wipe ends: a rehearsal. A memory attester on this
# node streaming natively runs the streamer (wipe_stream.c) this launcher builds into a mktemp -d of its own, never under
# SERVE_DIR, which may be on a filesystem that runs no programs (/dev/shm mounted noexec), handed as the attach's
# `wipe_stream` and removed with the key directory; a cc that fails exits 7.
#   WIPE_KEY_SHA512    the memory attester's key the verifier trusts (the attach's `trusted`), in 128 lowercase hex
#                      digits; unset, the key the run's configuration pins for the wipe (VERITY_PINS); set empty, none:
#                      refused
#   WIPE_NODE          the wiped server's id (this host's name)
#   WIPE_CARD          the card's SKU the wipe is measured for (h200-sxm-141gb), a cuda store's only; `any` is refused
#                      by name: the SKU is checked, and the judge holds the fill to its card file's fill floor,
#                      store.wipe_fill
# The reads are the declaration's (`WIPE.settings`' reads, 0: the memory attester chooses them, wipe_party.READS on the
# auditor's party), never this launcher's. The Lean judge, wipe-protocol, is the wipe's declared executable
# (`Launch.executables`, declared.py), built here in a scratch copy of verity/ (never this tree's .lake) whose place
# follows from DIR alone, beside the lease and the wipe, which waits for it only to answer (`wipe_served.built`); no
# other program judges in its place, nothing a node sets names another (WIPE_OWN_LEASE says only that the launcher took
# a card of its own), and one that doesn't build ends the piece after the wipe with its own error and exit 7.
# Inside the serve's gpu-lease (GPU_LEASE_UUID) it wipes that card; outside one it takes a card of its own (`gpu-lease 1
# --wait`, else the tree's gpu_lease.sh) and gives it back after. DIR gets period.json, wipe.json, summary.json,
# answer.json (wipe.json's answer), served.json (the serve dir and window, the server, the fill floor, the card, the
# lease, the run, the key's pin), lake-build.log and stamps.txt (`UTC EVENT` lines). summary.json, served.json and
# the `WIPE` line state `uncovered_card_bytes` beside the answer (a core answer, which holds its own fields alone): the
# card file's store.dram less its fill floor, the card's memory no label fills, which the wipe does not cover (null for
# a store on no card). The node's other storage the wipe does not fill is not known here and is not stated. The case
# goes to SERVE_DIR/private/wipe (0700), and served_case.py writes the composite's piece beside it. Its last line is `WIPE {...}`.
# A refusal, the attach's or this launcher's, each by name, writes its one sentence to DIR/refused.txt and no
# answer.json, and exits 3. Exit 0 on accept, 1 on another verdict or a wipe that failed, 7 when the judge or the
# streamer doesn't build.
set -euo pipefail
export LEAN_ABORT_ON_PANIC=1  # one_stage/settings.py's ABORT_ON_PANIC, for every Lean verifier below and every process it starts
HERE=$(cd "$(dirname "$0")" && pwd)
SELF=$HERE/$(basename "$0")
ROOT=$(cd "$HERE/../../.." && pwd)
USAGE="usage: served_wipe.sh SERVE_DIR WINDOW DIR [SETTINGS]"
SERVE_DIR=${1:?$USAGE} WINDOW=${2?$USAGE} DIR=${3:?$USAGE} SETTINGS=${4:-}
mkdir -p "$DIR"
DIR=$(cd "$DIR" && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
STREAM= JUDGE=
stamp() { echo "$(date -u +%FT%TZ) $*" >> "$DIR/stamps.txt"; }
refuse() { echo "WIPE REFUSED: $*" > "$DIR/refused.txt"; refused; }
refused() { cat "$DIR/refused.txt"; stamp refused; exit 3; }
# `attach.py check|run`: the settings checked and the attach's refusals made before any lease, then the wipe itself
attach() { "$PY" - "$@" "$DIR" "$SERVE_DIR" "$WINDOW" "$SETTINGS" "${WIPE_NODE:-$(hostname)}" \
  "${WIPE_KEY_SHA512:-}" "$JUDGE" "${VERITY_PINS:-}" "${VERITY_CONFIGURATION:-}" \
  "${VERITY_PROTOCOLS_CONFIG:-}" "${GPU_LEASE_UUID:-}" "${WIPE_OWN_LEASE:+own}" "$CTRL" "${STREAM:-}" \
  "${WIPE_CARD:-h200-sxm-141gb}" "$ROOT/calibration/hardware/cards" <<'EOF'
import json, os, shutil, socket, sys, tempfile
from pathlib import Path
from verity import device, hooks as H
from verity.protocols.run import schemes
from verity.protocols.accounting import scheme as S
from verity.protocols.accounting.memory.pous.attester import memory_attester as mc
from verity.protocols.accounting.memory.pous.prover import wipe_served as WS
from verity.protocols.accounting.memory.pous.verifier import period
from verity.protocols.accounting.memory.pous.scheme import MEMORY_ATTESTER
from verity.runtime.stream import Side
from verity_vllm.messages import Exchange

DEFAULT = {"host": "vy-auditor-2", "address": "10.96.0.36:47611",
           "command": ["ssh", "-i", "~/.ssh/vy-auditor2-wipe", "-o", "BatchMode=yes", "research@10.96.0.36"]}
(mode, d, serve_dir, window, given, node, trusted, judge, pins, configuration, app, lease, own, ctrl, stream, sku,
 cards) = sys.argv[1:]
d, wipe = Path(d), schemes.get("wipe")


def unnamed(why):
    """``why`` with the machines' own names put by party (`memory_attester.unnamed`): this node's, and the challenger's
    host and address as the settings give them."""
    at = globals().get("doc") or {}
    address = str(at.get("address") or "")
    return mc.unnamed(why, (node, socket.gethostname()), (at.get("host"), address or None, address.rpartition(":")[0] or None))


def refused(why):
    (d / "refused.txt").write_text(f"WIPE REFUSED: {unnamed(why)}\n")
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
if unknown := sorted(set(doc) - set(wipe.settings) - set(WS.FACTS)):
    refused(f"unknown settings {unknown}: the wipe's parameters are {sorted(wipe.settings)}, and the launcher's facts "
            f"{sorted(WS.FACTS)}")
given = {k: v for k, v in doc.items() if k in wipe.settings}
facts = json.loads(json.dumps(WS.FACTS | {k: v for k, v in doc.items() if k in WS.FACTS}))
ctx = {"process": H.PIECE, "side_of": Side.of, "private": f"{serve_dir}/private", "trusted": trusted or None,
       "node": node, "judge": judge or None, "wipe_stream": stream or None, **facts}
uncovered = None
try:
    settings = S.settings(wipe, given)
    if settings["store"] == "cuda":
        # the verifier's fill floor: the card file's, never the node's settings
        store = device.load_card(sku, directory=Path(cards))["store"]
        if "wipe_fill" not in store:
            raise ValueError(f"wipe: the card file {sku} gives no store.wipe_fill, the fill floor the judge holds "
                             f"the wipe to")
        ctx["fill"] = int(device.number(store["wipe_fill"]))
        # the card's memory no label fills: the card file's store.dram less the floor, never the driver's report
        uncovered = int(device.number(store["dram"])) - ctx["fill"] if "dram" in store else None
        if uncovered is None or uncovered < 0:
            raise ValueError(f"wipe: the card file {sku}'s store.dram is no memory at or above its fill floor "
                             f"store.wipe_fill, so the memory the wipe leaves uncovered is not known")
    # the scheme's attach hands the party in (below); this check runs before it, so it hands the scheme's own
    if why := WS.refusals(ctx | {"settings": settings, "pins": json.loads(pins) if pins else None,
                                 "party": MEMORY_ATTESTER}):
        raise ValueError(f"wipe: {'; '.join(why)}")
except ValueError as e:
    refused(str(e).removeprefix("wipe: "))
if mode == "check":
    # the store, and whether the memory attester streams natively on this node, so needs the streamer built here
    print(settings["store"], "here" if facts["streamer"] == "native" and not facts["host"] else "away")
    sys.exit(0)
made = []
for i, p in enumerate(facts["disk"].get("paths") or ()):
    if Path(p).is_dir():
        made.append(tempfile.mkdtemp(prefix="verity-wipe-disk.", dir=p))
        facts["disk"]["paths"][i] = made[-1]
    elif not Path(p).is_block_device():
        refused(f"the disk target {p} is neither a directory nor a block device")
try:
    got = Exchange().attach(wipe, given, **ctx)
except ValueError as e:
    refused(str(e).removeprefix("wipe: "))
except Exception as e:
    print(unnamed(f"WIPE FAILED: the wipe raised {type(e).__name__}: {e}")[:600], flush=True)
    sys.exit(1)
finally:
    for m in made:
        shutil.rmtree(m, ignore_errors=True)
private = Path(serve_dir) / "private" / wipe.name
shutil.copy(private / "period.json", d / "period.json")
(d / "wipe.json").write_text(json.dumps(got.record(), sort_keys=True, indent=1))
(d / "summary.json").write_text(json.dumps(s := got.summary() | {"uncovered_card_bytes": uncovered}, indent=1,
                                          sort_keys=True))
answer, ran = got.answer, got.record()["served"]
card = lease.split(",")[0] or ran["card"]
(d / "served.json").write_text(json.dumps({
    "serve_dir": serve_dir, "window": window, "server": ran["server"], "fill": ran["fill"], "card": card,
    "uncovered_card_bytes": uncovered, "key_pin": ran["key_pin"],
    "run": os.environ.get("RESEARCH_RUN_ID"),
    "lease": "its own (gpu-lease 1 --wait), given back before the serve takes one" if own else
             "the serve's: its engine loads on this card after the wipe"}, indent=1, sort_keys=True) + "\n")
period.publish(d, answer, ran["development"])
print("WIPE " + json.dumps({"verdict": answer["verdict"], "reason": answer["reason"], "protocol": answer["protocol"],
                            "decided_by": s.get("decided_by"), "executables": s.get("executables"), "k": s.get("k"), "segments": s.get("segments"),
                            "fill_GBps": (s.get("challenger") or {}).get("fill_GBps"), "t0": s.get("t0"),
                            "t1": s.get("t1"), "server": mc.PROVER_MACHINE, "card": card, "key_pin": ran["key_pin"],
                            "uncovered_card_bytes": uncovered,
                            "exit": 0 if answer["verdict"] == "accept" else 1}, sort_keys=True), flush=True)
sys.exit(0 if answer["verdict"] == "accept" else 1)
EOF
}

# the judge's scratch copy of verity/: one place for DIR, so the pass under the card's own lease finds it by DIR alone
lean_root() { echo "/tmp/verity-wipe-lean.$(printf %s "$DIR" | sha256sum | cut -c1-16)"; }
judge() {
  PYTHONPATH=$ROOT "$PY" "$ROOT/benchmarks/pous/declared.py" --built wipe wipe-protocol "$(lean_root)" \
    || { echo "WIPE FAILED: the wipe's declared executable doesn't read (declared.py)"; exit 7; }
}
[ "${WIPE_CARD:-}" != any ] || refuse "WIPE_CARD=any: the card's SKU is checked, and the wipe is measured for" \
  "h200-sxm-141gb unless WIPE_CARD names another SKU"
if [ -z "${WIPE_OWN_LEASE:-}" ]; then  # once, before any lease: the node's link, the settings, the refusals, the judge
  rm -f "$DIR"/{answer,served,summary,wipe,period}.json "$DIR/refused.txt" "$DIR/stamps.txt" \
    "$SERVE_DIR"/private/wipe/{answer,case,wipe.case}.json
  stamp start
  # the wipe's launcher's form (`memory/pous/scheme.py`'s `launch_wipe`), until it hands SETTINGS: `--run-keys`, or the bound attester's
  # WIPE_CHALLENGER, WIPE_ADDRESS and WIPE_COMMAND
  if [ "$SETTINGS" = --run-keys ]; then
    for v in WIPE_KEY_SHA512 WIPE_ADDRESS WIPE_CHALLENGER WIPE_COMMAND; do
      [ -z "${!v+set}" ] || refuse "--run-keys runs the memory attester on this node under a key made for the run," \
        "and $v is set: unset $v, or drop --run-keys and bind the auditor's memory attester"
    done
    SETTINGS='{"run_keys": true}'
  elif [ -z "$SETTINGS" ] && [ -n "${WIPE_COMMAND:-}" ]; then
    SETTINGS=$(python3 -c 'import json, shlex, sys; h, a, c = sys.argv[1:]
print(json.dumps({"host": h, "address": a, "command": shlex.split(c)}))' \
      "${WIPE_CHALLENGER:-vy-auditor-2}" "${WIPE_ADDRESS:-10.96.0.36:47611}" "$WIPE_COMMAND")
  fi
  [ "${WIPE_KEY_SHA512-unset}" != "" ] || refuse "no challenger key is pinned: WIPE_KEY_SHA512 is set empty, which" \
    "pins none (the wipe's launcher's command when the configuration pins no key for the wipe)"
  command -v ip > /dev/null || refuse "this host has no \`ip\` (iproute2), which names the node's links: apt-get install iproute2"
  read -r DEV SRC < <(ip -o route get 1.1.1.1 2>/dev/null | sed -nE 's/.* dev ([^ ]+).* src ([^ ]+).*/\1 \2/p') || true
  export WIPE_CTRL=${DEV:+$DEV $SRC $(cat "/sys/class/net/$DEV/speed" 2>/dev/null || echo -)}
  CTRL=$WIPE_CTRL
  attach check > /dev/null || { [ $? != 3 ] || refused; exit 1; }
  S=$(lean_root)
  rm -rf "$S" && (umask 077 && mkdir -p "$S") || { echo "WIPE FAILED: $S can't hold wipe-protocol's build"; exit 7; }
  trap 'rm -rf "$S"' EXIT
  # tar's exit 1 is only a file that changed as it read it: a process importing the tree's Python writes __pycache__
  # beside it meanwhile, as a check's other suites do; 2 or more is a copy that failed
  { tar -C "$ROOT" -c --exclude=.lake --exclude=__pycache__ --warning=no-file-changed verity || [ $? = 1 ]; } \
    | tar -C "$S" -x || { echo "WIPE FAILED: verity/ doesn't copy for wipe-protocol's build"; exit 7; }
  export PATH="$HOME/.elan/bin:$PATH"
  JUDGE=$(judge) || { echo "$JUDGE"; exit 7; }
  BUILT=${JUDGE%/*} L=${JUDGE%/.lake/build/bin/*}
  mkdir -p "$BUILT"
  # the lock is taken before the build starts and held by the build alone, so the answer waits for its end
  exec {LOCK}>"$BUILT/wipe-protocol.lock"
  flock -x "$LOCK"
  stamp "the Lean build starts"
  (
    command -v lake >/dev/null 2>&1 \
      || (exec {LOCK}>&- && bash "$ROOT/tools/verity/lean/setup.sh" "$L") > "$DIR/lake-setup.log" 2>&1
    if (cd "$L" && exec {LOCK}>&- && exec lake build wipe-protocol) > "$DIR/lake-build.log" 2>&1; then
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
# under its own lease, the judge is the declared one the pass before the lease started building, in its scratch copy
if [ -n "${WIPE_OWN_LEASE:-}" ]; then
  JUDGE=$(judge) || { echo "$JUDGE"; exit 7; }
  stamp "leased the card"
fi
CTRL=${WIPE_CTRL:-}
CARD=${GPU_LEASE_UUID%%,*}
export CUDA_VISIBLE_DEVICES=${GPU_LEASE_UUID:-${CUDA_VISIBLE_DEVICES:-}}
CHECKED=$(attach check) || { [ $? != 3 ] || refused; exit 1; }
read -r STORE STREAMS <<< "$CHECKED"
if [ "$STORE" = cuda ]; then
  NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$CARD" 2>&1 | head -1) \
    || refuse "nvidia-smi can't read the card $CARD: $NAME"
  SKU=$("$PY" "$ROOT/benchmarks/kernels/machine.py" sku "$NAME" 2>&1) || refuse "the card $CARD: $SKU"
  [ "$SKU" = "${WIPE_CARD:-h200-sxm-141gb}" ] || refuse "the card $CARD is '$NAME', a $SKU, not a ${WIPE_CARD:-h200-sxm-141gb}"
fi
K=$SERVE_DIR/private/run-keys/wipe
rm -rf "$K" && (umask 077 && mkdir -p "$K") \
  || refuse "the serve dir's private directory $SERVE_DIR/private can't hold the run's keys and the wipe's case"
X=
if [ "$STREAMS" = here ]; then
  X=$(mktemp -d)
  cc -O3 -pthread "$ROOT/verity/protocols/accounting/memory/pous/attester/wipe_stream.c" -o "$X/wipe_stream" \
    || { rm -rf "$K" "$X"; echo "WIPE FAILED: wipe_stream doesn't build"; stamp "wipe_stream didn't build"; exit 7; }
fi
rc=0
SAID=$(TMPDIR=$K STREAM=${X:+$X/wipe_stream} attach run) || rc=$?
rm -rf "$K" ${X:+"$X"}
[ "$rc" != 3 ] || refused
stamp "the wipe exit $rc"
if ! bash "$ROOT/benchmarks/pous/lean_ready.sh" "${JUDGE%/*}" wipe-protocol true 2> /dev/null; then
  echo "$SAID"
  cat "${JUDGE%/*}/failed"
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

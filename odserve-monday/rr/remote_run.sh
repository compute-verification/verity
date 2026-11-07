#!/usr/bin/env bash
# The remote product on one served window of Llama 3.2 1B int8 (H1_MODEL, default LLAMA32_1B_INT8_K2048, at T H1_T and
# S WINDOW_STEPS), in a clone of the run branch on an H200 node: sampled proofs on the live challenger's coins, PoUW at
# K 2048 in the lanes form (served_pouw.sh: the window's live rounds and its drawn proof block proved, the piece and the
# period's case), the record, and remote's Lean composite. Parametric in the node: H1_PEER (the gate's name for it),
# ROW (else the newest traced B4 int8 row here that predict compare holds equal at this tree, else one re-traced here at
# this tree by retrace-row.sh, sent, CPU only; RETRACE=1 re-traces whatever is cached, as a tree whose Definition
# digests changed needs), cargo (rustup's RUST_TOOLCHAIN where missing), PP_CHALLENGER_KEY (the
# challenger's deployment key record: a file on the node, else inputs/challenger-key.json, sent). PREFLIGHT_ONLY=1 stops before served_verdicts.sh.
#  0. toolchains; H1_LANE's gate key (network-accounting's by default); the gate's pin, GATE_PIN, which main.txt (sent) starts at; H1_MODEL's registration
#     under the window Program's circuit H1_CIRCUIT, its weights root starting with H1_ROOT; with AFTER_PERIOD (another
#     run's period), a wait (until AFTER_BY) for H1_LANE's next reserve after that period's close, that run's sampled
#     proofs' session, so this run's period, opened at its serve, never takes it in;
#  1. served_verdicts.sh with PRODUCT=remote (its onsite commands empty, V3c only with V3C_CLASS) and POUW_CMD served_pouw.sh, the load's
#     counts in the private file SETTINGS names (on the node, never in the command); then the PORTS line (whether any
#     circuit file the run proved on names public input ports) and the SHARES line (served_shares.py: the declared
#     utilization and the accounted share at both clock readings, as measured; totals in out/shares.json);
#  2. scrub.sh on out/ (sent: what out/ mustn't publish withheld into the serve dir's private/withheld), and stated.txt's
#     STATED lines (sent) at the head of out/verdicts.txt.
set -uo pipefail
IN=$RESEARCH_RUN_DIR/inputs
export OUT=${OUT:-$RESEARCH_RUN_DIR/out}
mkdir -p "$OUT"
TREE=$PWD
note() { echo "$*" | tee -a "$OUT/verdicts.txt"; }
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.elan/bin:$PATH
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
export PATH=$PATH:$CUDA_HOME/bin
for f in main.txt stated.txt scrub.sh scan-out.py; do [ -s "$IN/$f" ] || { note "PREFLIGHT: no inputs/$f (--send it)"; exit 3; }; done
cat "$IN/stated.txt" >> "$OUT/verdicts.txt"
note "STATED tree: $(git rev-parse HEAD 2>/dev/null || echo "${RESEARCH_TREE_SHA:-unknown}")"

command -v lake > /dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
  | sh -s -- -y --default-toolchain none > "$OUT/elan.log" 2>&1
command -v cargo > /dev/null || { curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --no-modify-path --profile minimal --default-toolchain "${RUST_TOOLCHAIN:-1.99.0}" > "$OUT/rustup.log" 2>&1 \
  && note "RUST: $(cargo --version) installed by rustup"; } \
  || { note "PREFLIGHT: no cargo, and rustup didn't install it: $(tail -1 "$OUT/rustup.log")"; exit 3; }

# The gate takes the lane from the key: H1_GATE_KEY is H1_LANE's key here, onsite's own lane network-accounting's by default.
export H1_LANE=${H1_LANE:-network-accounting}
KEY=${H1_GATE_KEY:-$HOME/.ssh/vy-gate-network-accounting}
[ -f "$KEY" ] || { note "PREFLIGHT: no gate key for lane $H1_LANE on this node ($KEY)"; exit 3; }
GATE_HOST=${GATE_HOST:-204.12.178.84}
export H1_GATE="ssh -i $KEY -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 research@$GATE_HOST"
export H1_PEER=${H1_PEER:?the name the gate gives this node in its /etc/vy/peers}
export H1_MODEL=${H1_MODEL:-LLAMA32_1B_INT8_K2048} H1_DRAW_K=${H1_DRAW_K:-1}
CIRCUIT=${H1_CIRCUIT:?the circuit of the window Program the gate registers H1_MODEL under}

pin=$($H1_GATE pins | tail -n1 | python3 -c 'import json, sys; print(json.load(sys.stdin)["commit"])') \
  || { note "PREFLIGHT: the gate's pins log didn't read"; exit 3; }
[ "$pin" = "${GATE_PIN:?the commit the gate is pinned at}" ] \
  || { note "REFUSED: the gate is pinned at $pin, not $GATE_PIN: send main.txt of its first-parent line and recheck the STATED gate line"; exit 3; }
[ "$(head -1 "$IN/main.txt")" = "$GATE_PIN" ] || { note "PREFLIGHT: main.txt doesn't start at the gate's pin $GATE_PIN"; exit 3; }
note "GATE pinned at $pin, main.txt its first-parent line ($(grep -c . "$IN/main.txt") commits)"

WROOT=$($H1_GATE models | python3 -c '
import json, sys
model, circuit = sys.argv[1:3]
roots = [e["weights_root"] for e in map(json.loads, filter(str.strip, sys.stdin)) if e.get("model") == model and e.get("circuit") == circuit]
print(roots[-1] if roots else "")' "$H1_MODEL" "$CIRCUIT")
[ -n "$WROOT" ] || { note "REFUSED: the gate registers no $H1_MODEL under ${CIRCUIT:0:16}..."; exit 4; }
[[ "$WROOT" == "${H1_ROOT:?the first hex digits of the registration weights root}"* ]] \
  || { note "REFUSED: $H1_MODEL's registration has weights root ${WROOT:0:16}..., not $H1_ROOT... (registered again?)"; exit 4; }
note "REGISTERED: $H1_MODEL under the window Program ${CIRCUIT:0:16}..., weights root ${WROOT:0:16}..."

if [ -n "${AFTER_PERIOD:-}" ]; then
  aby=$(date -u -d "${AFTER_BY:?a UTC time}" +%s)
  note "WAIT: for lane $H1_LANE's next reserve after period $AFTER_PERIOD's close (its run's sampled proofs' session), from $(date -u +%H:%M:%SZ)"
  until $H1_GATE ledger | python3 -c '
import json, sys
name, lane, closed = sys.argv[1], sys.argv[2], False
for e in map(json.loads, filter(str.strip, sys.stdin)):
    if e["kind"] == "period-close" and e["body"].get("period") == name:
        closed = True
    elif closed and e["kind"] == "reserve" and (e["body"].get("for") or e["body"].get("by")) == lane:
        sys.exit(0)
sys.exit(1)' "$AFTER_PERIOD" "$H1_LANE"; do
    [ "$(date -u +%s)" -lt "$aby" ] || { note "REFUSED: no reserve of lane $H1_LANE after $AFTER_PERIOD's close by $AFTER_BY"; exit 6; }
    sleep 60
  done
  note "WAIT: done ($(date -u +%H:%M:%SZ))"
fi

VENV=${VENV:-/workspace/cp/venv312}
export PY=$VENV/bin/python HF_HOME=${HF_HOME:-/workspace/hf} HF_HUB_OFFLINE=1
export REV=${REV:-15b0976d400d796ee589e7737d03a919526bf800}
RH=${ROWS_HOME:-/workspace/research/vllm-composed-h200int8}
NAME=llama32-1b__int8__h200__tp1__b4__i256__o32__mixed__greedy__bi-eager
if [ -z "${ROW:-}" ] && [ "${RETRACE:-}" != 1 ]; then
  PW=$(mktemp -d)
  (export PYTHONPATH=$TREE:$TREE/integrations/vllm:$TREE/tools/research/src CUDA_VISIBLE_DEVICES=
   "$PY" -m verity_vllm.pipeline.cli predict row --row "$NAME" --out "$PW/pred" > "$PW/row.json" 2> "$PW/row.err"
   CONFIG=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["config"])' "$PW/pred/predicted.json") || exit 1
   for r in $(ls -td "$RH"/r*/"$NAME" 2>/dev/null); do
     "$PY" -m verity_vllm.pipeline.cli predict compare "$r" --row "$NAME" --config "$CONFIG" --predicted "$PW/pred" \
       --out "$PW/compare.json" > "$PW/compare.log" 2>&1 && { echo "$r"; exit 0; }
   done
   exit 1) > "$PW/pick" 2> "$PW/pick.err"
  ROW=$(head -1 "$PW/pick")
  rm -rf "$PW"
  [ -n "$ROW" ] || note "ROW: no traced B4 int8 row on this node equals this tree's Build of record"
fi
if [ -z "${ROW:-}" ]; then
  note "ROW: re-traced at this tree (retrace-row.sh, CPU only${RETRACE:+; RETRACE=1 skips the cached rows})"
  [ -s "$IN/retrace-row.sh" ] || { note "PREFLIGHT: no inputs/retrace-row.sh (--send it)"; exit 3; }
  (export OUT=$OUT/retrace; CUDA_VISIBLE_DEVICES='' bash "$IN/retrace-row.sh") > "$OUT/retrace.log" 2>&1
  rrc=$?
  grep -E '^(ROW-BUILD|ROW-MANIFEST|PREDICT-ROW|PREDICT-COMPARE)' "$OUT/retrace.log" | cut -c1-600 | tee -a "$OUT/verdicts.txt"
  [ "$rrc" = 0 ] || { note "REFUSED: the re-trace failed (retrace-row.sh exit $rrc)"; exit 17; }
  ROW=$RH/$RESEARCH_RUN_ID/$NAME
fi
export ROW
note "ROW: $(basename "$(dirname "$ROW")"), the traced Build predict compare holds this tree's Build of record to"

export PRODUCT=remote
export H1_T=${H1_T:-512} WINDOW_STEPS=${WINDOW_STEPS:-1} H1_LANES=${H1_LANES:-2048}
export H1_PROTOCOLS_CONFIG=${H1_PROTOCOLS_CONFIG:-'{"pouw": {"ks": [2048]}}'}
export SERVE_ROOT=${SERVE_ROOT:-/workspace/research/vllm-served-verdicts}
export H1_CIRCUITS=${H1_CIRCUITS:-$SERVE_ROOT/h1-circuits-k${H1_LANES}-lanes-t${H1_T}s${WINDOW_STEPS}}
export SERVE_ARGS="--h1-receipts --h1-model $H1_MODEL --h1-weights-root $WROOT${SERVE_EXTRA:+ $SERVE_EXTRA}"
export H1_CHECKPOINT=${H1_CHECKPOINT:-$HF_HOME/hub/models--verity-int8--Llama-3.2-1B-INT8/snapshots/$REV}
[ -d "$H1_CHECKPOINT" ] || { note "PREFLIGHT: no checkpoint at $H1_CHECKPOINT"; exit 3; }
export H1_DRAWN_MEM_GB=${H1_DRAWN_MEM_GB:-400}
export WIPE_CMD= POUS_CMD= BANDWIDTH_CMD= CERTIFIER_CMD= CERTIFIER_CHAIN_CMD= CERTIFIER_PERIOD_CMD= CERTIFIER_FRONT= V3C_CLASS=${V3C_CLASS-}
# shellcheck disable=SC2016
export POUW_CMD='bash benchmarks/one_stage/served_pouw.sh "$1" "$2" "$3"' POUW_TIMEOUT=${POUW_TIMEOUT:-14h}
export PP_CHALLENGER=gate:research@$GATE_HOST PP_PEER=$H1_PEER PP_GATE_KEY=$KEY
ck=/workspace/research/vllm-served-verdicts/settings-rerunA/challenger-key.json
[ -s "$ck" ] || ck=$IN/challenger-key.json
export PP_CHALLENGER_KEY=${PP_CHALLENGER_KEY:-$ck}
[ -s "$PP_CHALLENGER_KEY" ] || { note "PREFLIGHT: no challenger key record at $PP_CHALLENGER_KEY"; exit 3; }
note "CHALLENGER KEY: $(python3 -c 'import json, sys; d = json.load(open(sys.argv[1])); print(d.get("schema"), d.get("host"), str(d.get("pubkey_sha512"))[:16] + "...")' "$PP_CHALLENGER_KEY" 2>&1 | cut -c1-200)"
[ "${PP_KEEP_WINDOWS:-1}" = 1 ] && PP_KEEP_WINDOWS=$SERVE_ROOT/kept-windows/$RESEARCH_RUN_ID
case $PP_KEEP_WINDOWS in /*) ;; *) note "PREFLIGHT: PP_KEEP_WINDOWS ($PP_KEEP_WINDOWS) is not an absolute directory, and run.sh copies the windows there from the clone, which is removed on exit"; exit 3;; esac
export PP_KEEP_WINDOWS LOAD=${LOAD:-varied}
note "KEPT WINDOWS: $PP_KEEP_WINDOWS on this node, private (a later job's WINDOW=$PP_KEEP_WINDOWS/served-windows/rank0)"
export SETTINGS=${SETTINGS:?a private file of the load counts on this node}
[ -r "$SETTINGS" ] || { note "PREFLIGHT: SETTINGS $SETTINGS isn't readable"; exit 3; }
export MAIN_FILE=$IN/main.txt
export HZ_CACHE=${HZ_CACHE:-$SERVE_ROOT/hz-cache}
export TMPDIR=$SERVE_ROOT/tmp
export LEASE_MIN=${LEASE_MIN:-180} DISK_FLOOR_GB=${DISK_FLOOR_GB:-64} WC_RECORD_TIMEOUT=${WC_RECORD_TIMEOUT:-1m}
export GPU_LEASE_WHO=${GPU_LEASE_WHO:-vllm-remoterun}
(umask 077; mkdir -p "$TMPDIR")
for f in "$ROW/manifest.json" "$ROW/build_summary.json" "${HIDDEN_SO:-$RH/fa2/build/fa3_matReqG/verity_fa3_matReqG.so}" \
  "${NORM_TAP_SO:-$RH/norm_tap/build/verity_norm_tap.so}" "$PY"; do
  [ -e "$f" ] || { note "PREFLIGHT: no $f"; exit 3; }
done
command -v gpu-lease > /dev/null || { note "PREFLIGHT: no gpu-lease on this node"; exit 3; }
if [ "${PREFLIGHT_ONLY:-0}" = 1 ]; then
  note "PREFLIGHT_ONLY: every check passed; served_verdicts.sh not run"
  exit 0
fi

(umask 077; mkdir -p "$PP_KEEP_WINDOWS")
t0=$(date +%s)
bash benchmarks/one_stage/served_verdicts.sh
rc=$?
# every circuit file the run proved on (sampled proofs' members, PoUW's lanes, combine and live rounds): whether its META
# names public input ports, a form the Lean verifier refuses (`ProvedScope.statementRules`, `HmIn.checkOwn`)
python3 - "$SERVE_ROOT/$RESEARCH_RUN_ID" "$OUT" "$H1_CIRCUITS" <<'EOF' 2>&1 | tee -a "$OUT/verdicts.txt"
import json, os, sys
from pathlib import Path
seen, ports, bad = {}, [], []
for root in map(Path, sys.argv[1:]):
    for p in root.rglob("*circuit.txt"):
        try:
            s = p.stat()
            if (s.st_dev, s.st_ino) in seen or not s.st_size:
                continue
            with p.open() as f:
                f.readline()
                head = f.readline()
            meta = json.loads(head[5:]) if head.startswith("META ") else None
        except (OSError, ValueError) as e:
            bad.append(f"{p}: {type(e).__name__}")
            continue
        if meta is None:
            bad.append(f"{p}: no META line")
            continue
        seen[(s.st_dev, s.st_ino)] = p
        if "public_inputs" in meta or "leaves_public" in meta:
            ports.append(str(p))
d = Path(sys.argv[1]) / "private"
if d.is_dir():
    (d / "ports.txt").write_text("".join(f"{p}\n" for p in seen.values()) + "".join(f"PORTS {p}\n" for p in ports)
                                 + "".join(f"UNREAD {x}\n" for x in bad))
print(f"PORTS: {len(seen)} circuit files the run proved on (the serve dir's, out/'s and H1_CIRCUITS'), {len(ports)} naming public input "
      f"ports (META public_inputs), {len(bad)} unread (each in the serve dir's private/ports.txt)")
EOF
# the run's declared utilization and accounted share at both clock readings, as measured (served_shares.py over the serve
# dir's step trace and the period credit's answer); the full report stays in the serve dir's private/, out/ gets the totals
SD=$SERVE_ROOT/$RESEARCH_RUN_ID PA=$SERVE_ROOT/$RESEARCH_RUN_ID/private/pouw/period/answer.json SPY=.venv/bin/python
[ -x "$SPY" ] || SPY=python3
credit=(); [ -s "$PA" ] && credit=(--credit "$PA")
if (umask 077; "$SPY" benchmarks/one_stage/served_shares.py "$SD" "${credit[@]}" > "$SD/private/shares.json" 2> "$SD/private/shares.err"); then
  "$SPY" - "$SD/private/shares.json" "$OUT/shares.json" "${#credit[@]}" <<'EOF' 2>&1 | tee -a "$OUT/verdicts.txt"
import json, sys
s = json.load(open(sys.argv[1]))
pub = {k: s[k] for k in ("schema", "model", "wall_s", "macs", "credit", "readings")}
pub["loads"] = len(s["loads"])
json.dump(pub, open(sys.argv[2], "w"), indent=1)
r, c = s["readings"], s["credit"]
pct = lambda k: " / ".join(f"{100 * r[n][k]:.4g}% at {r[n]['clock_mhz']:,} MHz ({n})" for n in ("tensor", "boost"))
src = "the period credit's answer" if sys.argv[3] != "0" else "no period answer, so 0"
print(f"SHARES (served_shares.py, as measured): wall {s['wall_s']:.6g} s over {len(s['loads'])} load(s); declared utilization "
      f"{pct('declared')}; accounted share (floor {c['floor']}) {pct('accounted')}, credited {pct('credited')}, "
      f"from {src} (verdict {c['verdict']}, credited work {c['work']:,} MACs); totals in out/shares.json")
EOF
else
  note "SHARES: served_shares.py exit $? ($(tail -1 "$SD/private/shares.err" | cut -c1-400))"
fi
bash "$IN/scrub.sh" "$OUT" "$SERVE_ROOT/$RESEARCH_RUN_ID" "$IN/scan-out.py" 2>&1 | tee -a "$OUT/scrub.txt"
note "RUN: served_verdicts.sh exit $rc after $(( ($(date +%s) - t0) / 60 )) min"
exit "$rc"

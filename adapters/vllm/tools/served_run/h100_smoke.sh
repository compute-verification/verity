#!/usr/bin/env bash
# Does: serves ONE window of the vLLM adapter's statement (`verity_vllm.inputs.serves`: its model, revision, precision and window
#   class), the input dataset at the timing APPLICATION names (default applications/remote.toml), through its door, through serve-openai with its rows-sha512 Commit, no scheme, and seals it, each
#   window's close planning its class circuit and leaving the card's part of its record (the acquired units' strings,
#   made where they are committed, under the window's key: close_record.Record.handoff) and, with RECORD_IN_LEASE=1, the
#   record too (RECORD=0: the plan alone); then, the lease released, on this host's processors: opens the sealed bundle
#   against its record, makes the window's record from the handoff (verity-vllm close-record: the record, the close with the
#   card's strings, the write), and commits the fixed load's window as its class circuit's units
#   (`benchmarks/one_stage/window_commit.py window-commit`, as served_verdicts.sh's window_commit() calls it).
# Does NOT show: an audit, a proof, or a traced Build on the H100.
# Needs: PY (default the node's /workspace/venv312/bin/python), gpu-lease (it leases one card for the serve alone, at most
#   LEASE_MIN minutes, default 30), HF_HOME (default /workspace/hf) holding the model; CUDA_HOME (CUDA $TOOLKIT, default
#   12.9, machine.py's; TOOLKIT=13.0 on the H200, as served_verdicts.sh takes there),
#   CARD_SKU (machine.py sku), PYTHONPATH and VERITY_PROTOCOLS_CONFIG are set here. APPLICATION (applications/NAME.toml),
#   read once, gives the dataset, the timing and the class `verity run` gives. WIPE_CMD, set only where the application
#   lists the wipe, and refused without APPLICATION, wipes the leased card before serve-openai starts, its answer in the serve's pieces/wipe (unset: no wipe).
# Writes: the serve's directory, the private directory (keys, salts) and every log under SR, outside the run's directory
#   (named private in the run's private-paths.json); under $OUT/window, named private too, the window's registration.json
#   and circuit.json and the deployment registration (it carries the Program digest); under $OUT smoke.txt: the reader's
#   counts, the units by source, the class's bits, then the window's plan, its strings (card and host, the twin's
#   comparison) and its record beside window-commit's.
#   research run --on vy-h100-1 --cwd source -- taskset -c 0-9 bash adapters/vllm/tools/served_run/h100_smoke.sh   (DRY=1: off the card)
set -euo pipefail
[ -z "${PEARLPLUS_TIE:-}" ] || [ "$PEARLPLUS_TIE" = 0 ] || { echo "h100_smoke.sh: PEARLPLUS_TIE is retired (the window commit takes no tie); unset it" >&2; exit 3; }
[ -z "${CLASS:-}" ] || { echo "h100_smoke.sh: CLASS is retired (the class is the adapter's statement's, verity_vllm.inputs.serves); unset it" >&2; exit 3; }
[ -z "${DATASET:-}" ] || { echo "h100_smoke.sh: DATASET is retired (the dataset is APPLICATION's, applications/NAME.toml); unset it" >&2; exit 3; }
[ -z "${SCHEME:-}" ] || { echo "h100_smoke.sh: SCHEME is retired (the smoke serves with no scheme; Pearl+ is served_verdicts.sh's POUW=pearlplus); unset it" >&2; exit 3; }
SELF=adapters/vllm/tools/served_run/h100_smoke.sh
TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd); cd "$TREE"
ROW=llama32-1b__bf16__h100__tp1__b4__i256__o32__mixed__greedy__bi-eager
DRY=${DRY:-0} PY=${PY:-/workspace/venv312/bin/python} LEASE_MIN=${LEASE_MIN:-30} LEASE_MEM_GB=${LEASE_MEM_GB:-64}
# the sampling design is the auditor's and not public: a printed line says only that one is registered
DESIGN="(the registered sampling design)"
# DEVELOPMENT_DESIGN=1: LAW is a development sampling design (default DEV_LAW, a uniform draw of 4 of class DEV_CLASS's
#   proof blocks, refused on another class: give LAW), which each record says, with what an accept of it establishes at δ
DEVELOPMENT_DESIGN=${DEVELOPMENT_DESIGN:-0} DEV_CLASS=4,27,2 DEV_LAW=rate:4/3654200
[ "$DEVELOPMENT_DESIGN" = 1 ] && LAW=${LAW:-$DEV_LAW}
MNBT=${MNBT:-512} LAW=${LAW:-rate:1/3481648} WC_TIMEOUT=${WC_TIMEOUT:-90m} FIN_TIMEOUT=${FIN_TIMEOUT:-120m}
RECORD_IN_LEASE=${RECORD_IN_LEASE:-0}
export PYTHONPATH=$TREE:$TREE/library/datasets:$TREE/adapters/vllm:$TREE/adapters/torch:$TREE/tools/config:$TREE/verity/protocols/proof/flock/prover/python:$TREE/tools/research/src
[ -z "${WIPE_CMD:-}" ] || [ -n "${APPLICATION:-}" ] || { echo "SMOKE REFUSED application: WIPE_CMD is set where an application" \
  "lists the wipe, so name it: APPLICATION=applications/NAME.toml"; exit 3; }
APPLICATION=${APPLICATION:-applications/remote.toml}
# the window class (S,P,O: slots, prompt ids, outputs) the serve's arena is sized from, and the model, from the adapter's
# statement; the dataset and its timing from APPLICATION, which the door reads through before the lease
SAID=$("$PY" adapters/vllm/tools/served_run/inputs.py application "$APPLICATION") || { echo "SMOKE REFUSED dataset: $SAID"; exit 3; }
read -r MODEL REV PRECISION CLASS DATASET TIMING CAP_BITS <<< "${SAID%%$'\n'*}"
CAP=${CAP:-$CAP_BITS}
if [ -n "${WIPE_CMD:-}" ]; then  # the wipe reads the application's pins and configuration as the road hands them to a piece
  WIPE_ENV=$(PYTHONPATH=$TREE/tools/verity/cli:$PYTHONPATH "$PY" - "$APPLICATION" <<'PINS'
import shlex, sys
from pathlib import Path
from verity_cli import run, served_table
for k, v in served_table.pins_and_configuration(run.load(Path(sys.argv[1]), run.registry())).items():
    print(f"export {k}={shlex.quote(v)}")
PINS
  ) || { echo "SMOKE REFUSED application: $APPLICATION's pins and configuration can't be read for the wipe"; exit 3; }
  eval "$WIPE_ENV"
fi
echo "${SAID#*$'\n'}"
IFS=, read -r SLOTS PROMPT OUTPUTS _ <<< "$CLASS"
[ "$DEVELOPMENT_DESIGN" != 1 ] || [ "$LAW" != "$DEV_LAW" ] || [ "$CLASS" = "$DEV_CLASS" ] || { echo "SMOKE REFUSED class:" \
  "DEV_LAW draws 4 of class $DEV_CLASS's proof blocks, not of CLASS=$CLASS's: pass LAW=rate:K/N, N its proof blocks"; exit 3; }
[ $((SLOTS * PROMPT)) -le "$MNBT" ] || { echo "SMOKE REFUSED class: CLASS=$CLASS's $SLOTS prompts of $PROMPT ids are $((SLOTS * PROMPT))" \
  "tokens, past the engine's $MNBT a step (MNBT): its requests would not start in the window's first step together"; exit 3; }
SERVER= SCRATCH=
stop() {
  if [ -n "$SERVER" ] && kill -0 "$SERVER" 2> /dev/null; then kill -INT "$SERVER"; wait "$SERVER" || true; fi
  if [ "$DRY" = 1 ] && [ -n "$SCRATCH" ]; then rm -rf "$SCRATCH"; fi
}
trap stop EXIT; trap 'exit 130' INT; trap 'exit 143' TERM
if [ "$DRY" = 1 ]; then
  SCRATCH=$(mktemp -d) SR=$SCRATCH/smoke
else
  SR=/workspace/research/vllm-servedproof/serve-root/smoke-${RESEARCH_RUN_ID:-local}
  command -v gpu-lease > /dev/null || { echo "SMOKE REFUSED lease: this machine has no gpu-lease, which the serve's card is leased under"; exit 3; }
fi
NOKIT= TOOLKIT=${TOOLKIT:-12.9}
CUDA_HOME=$("$PY" benchmarks/kernels/machine.py toolkit "$TOOLKIT") || if [ "$DRY" = 1 ]; then
  NOKIT=1 CUDA_HOME=$SCRATCH/no-toolkit; echo "DRY: this machine lists no CUDA $TOOLKIT, so the committer's native build is not run"
else echo "SMOKE REFUSED toolkit: neither CUDA_HOME nor the machine gives CUDA $TOOLKIT (machine.py, above)"; exit 3; fi
if [ "$DRY" = 1 ] && [ -n "${CARD_SKU:-}" ]; then echo "DRY: CARD_SKU $CARD_SKU as given (off the card, no card to read it from)"
else CARD_SKU=$("$PY" benchmarks/kernels/machine.py sku); fi || { echo "SMOKE REFUSED card: the machine's record names no card sku for this host (machine.py sku, above)"; exit 3; }
export CUDA_HOME CARD_SKU HF_HOME=${HF_HOME:-/workspace/hf} HF_HUB_OFFLINE=1 GPU_LEASE_WHO=${GPU_LEASE_WHO:-vllm-served-run}
export PATH=$(dirname "$PY"):$CUDA_HOME/bin:$PATH
SNAP=$HF_HOME/hub/models--${MODEL//\//--}/snapshots/$REV
{ [ -s "$SNAP/config.json" ] && ls "$SNAP"/*.safetensors > /dev/null 2>&1; } || { [ "$DRY" = 1 ] && echo "DRY: $MODEL at $REV is not in $HF_HOME (the dry run loads no model)"; } || {
  echo "SMOKE REFUSED model: $MODEL at $REV is not in $HF_HOME; fetch it outside the lease:" \
    "HF_HOME=$HF_HOME $PY -c \"from huggingface_hub import snapshot_download as s; s('$MODEL', revision='$REV')\""; exit 3; }
OUT=${OUT:-${RESEARCH_RUN_DIR:-$SR}/out}
# REGISTRATION_LANE: the lane the auditor's challenger signs the deployment's registration for (a lane takes one, ever),
#   the lane the epochs reserve on (served_verdicts.sh's H1_LANE), with no default: the first request waits, at most
#   REGISTRATION_HOLD_S seconds, until the challenger's ledger REGISTRATION_LEDGER signs the registration the serve wrote,
#   every entry tested under the pinned key record REGISTRATION_KEY (else PP_CHALLENGER_KEY) and the chain
#   (`served_verdicts.py registration-hold`); the period-open line printed before it names REGISTRATION_CAP (no
#   default) and, as --circuit, the epoch's Program digest computed before the serve from the class (fixed_circuit.py),
#   which the closed epoch's docket must name. With PP_CHALLENGER=gate:USER@HOST (and PP_GATE_KEY) the ledger is fetched
#   from the gate's `ledger` at each look into REGISTRATION_LEDGER (ledger_fetch.py). Unset: the development path.
HOLD=(benchmarks/one_stage/served_verdicts.py registration-hold) FETCH=()
if [ -n "${REGISTRATION_LANE:-}" ]; then
  [ -z "${H1_LANE:-}" ] || [ "$H1_LANE" = "$REGISTRATION_LANE" ] || { echo "SMOKE REFUSED lane: REGISTRATION_LANE" \
    "$REGISTRATION_LANE is not H1_LANE $H1_LANE, the lane the epochs reserve on"; exit 3; }
  [[ "${REGISTRATION_CAP:-}" =~ ^[1-9][0-9]*$ ]] || { echo "SMOKE REFUSED registration: REGISTRATION_LANE's period" \
    "opens with --cap REGISTRATION_CAP, the challenger sessions the run opens on the lane, a positive integer with no" \
    "default"; exit 3; }
  [[ "${REGISTRATION_HOLD_S:-}" =~ ^[1-9][0-9]*$ ]] || { echo "SMOKE REFUSED registration: REGISTRATION_LANE holds the" \
    "first request at most REGISTRATION_HOLD_S seconds, a positive integer with no default"; exit 3; }
  [ -r "${REGISTRATION_KEY:-${PP_CHALLENGER_KEY:-}}" ] || { echo "SMOKE REFUSED registration: REGISTRATION_LANE needs" \
    "the challenger's pinned key record, REGISTRATION_KEY or PP_CHALLENGER_KEY"; exit 3; }
  [ -n "${REGISTRATION_LEDGER:-}" ] || { echo "SMOKE REFUSED registration: REGISTRATION_LANE needs REGISTRATION_LEDGER"; exit 3; }
  if [[ "${PP_CHALLENGER:-}" = gate:* ]]; then
    g=(ssh -o BatchMode=yes); [ -z "${PP_GATE_KEY:-}" ] || g+=(-o IdentitiesOnly=yes -i "$PP_GATE_KEY")
    FETCH=(adapters/vllm/tools/served_run/ledger_fetch.py --gate "$(printf '%q ' "${g[@]}" "${PP_CHALLENGER#gate:}")" \
      --out "$REGISTRATION_LEDGER")
  else
    [ -r "$REGISTRATION_LEDGER" ] || { echo "SMOKE REFUSED registration: REGISTRATION_LEDGER is unreadable and no" \
      "PP_CHALLENGER=gate:USER@HOST fetches it"; exit 3; }
  fi
  HOLD+=(--lane "$REGISTRATION_LANE" --ledger "$REGISTRATION_LEDGER" --key "${REGISTRATION_KEY:-${PP_CHALLENGER_KEY:-}}")
elif [ "${1:-}" != card ]; then
  "$PY" "${HOLD[@]}" /dev/null
fi
D=$SR/serve PRED=$SR/predicted PORT=${PORT:-18282}
mkdir -p "$SR" "$OUT"
export VERITY_PROTOCOLS_CONFIG="{\"private\": \"$SR/private\"}"
WL=adapters/vllm/workloads/$ROW.json
say() { echo "$1"; [ "$DRY" = 1 ] || echo "$1" > "$OUT/smoke.txt"; }

# the load: DATASET at TIMING, released under a release run in the network attester's release order, so at once its
# requests start in the window's first step (served_verdicts.sh's LOAD=dataset)
RELEASE=${RELEASE:-$(tr -d - < /proc/sys/kernel/random/uuid)}
LOAD=(adapters/vllm/tools/served_run/inputs.py send --port "$PORT" --dataset "$DATASET" --timing "$TIMING" --release "$RELEASE" --out "$SR/load.jsonl")
REC_ARGS=()  # RECORD=0: each window's close plans its circuit and makes no record
[ "${RECORD:-1}" = 0 ] || REC_ARGS=(--sampling-design "$LAW" --checkpoint-dir "$SNAP")
[ "${RECORD:-1}" = 0 ] || [ "$RECORD_IN_LEASE" != 1 ] || REC_ARGS+=(--record-in-lease)
[ "${RECORD:-1}" = 0 ] || [ "$DEVELOPMENT_DESIGN" != 1 ] || REC_ARGS+=(--development-design)
# the fixed load's requests start in the window's first step only in the network attester's release order, which the serve
# reads under the name of the scheme that provides admission
ADMIT=$("$PY" - <<'EOF'
import json
from verity.protocols.run import schemes
print(json.dumps({schemes.named("provides", "admission"): {"order": "release-order"}}))
EOF
) || { echo "SMOKE REFUSED admission: no one declared scheme provides admission (verity.protocols.run.schemes, above)"; exit 3; }
SERVE=(serve-openai --model "$MODEL" --revision "$REV" --out "$D" --port "$PORT" --precision "$PRECISION"
  --run-id "smoke-${RESEARCH_RUN_ID:-local}" --seal --max-num-batched-tokens "$MNBT" --window-steps "$OUTPUTS"
  --row-build "$PRED" --window-requests "$SLOTS" --window-class "$CLASS" --max-num-seqs "$SLOTS"
  --max-model-len "$((PROMPT + OUTPUTS))" --window-gb 4 --disk-floor-gb 5 --seal-hold-gb 8
  ${REC_ARGS[@]+"${REC_ARGS[@]}"}
  --protocols-config "$ADMIT")
LEASE=(gpu-lease 1 --wait --max-min "$LEASE_MIN" --mem-gb "$LEASE_MEM_GB" -- bash "$TREE/$SELF" card)
WREG=$D/weights_registration.json DREG=$D/deployment-registration.json VW=$SR/window
FIN=(-m verity_vllm.pipeline.cli close-record --class "$CLASS" --sampling-design "$LAW" --checkpoint-dir "$SNAP")
[ "$DEVELOPMENT_DESIGN" != 1 ] || FIN+=(--development-design)
window_commit_of() {  # WC: window $1's commit as its class circuit's units, served_verdicts.sh's window_commit() command
  WC=(benchmarks/one_stage/window_commit.py window-commit --serve-run "$D" --window "$1" --row-dir "$D/windows/w$1"
    --advice "$D/advice/w$1/private/advice.bin" --class "$CLASS" --registration "$WREG" --checkpoint-dir "$SNAP"
    --sampling-design "$LAW" --out "$VW")
}

plan_lines() {  # the serve's close of window $K as plans.jsonl holds it: its plan, its strings and its record
  "$PY" - "$D/plans.jsonl" "$K" "$SR/window-commit.json" "$SR/record.json" "$RECORD_IN_LEASE" <<'EOF'
import json, sys
from pathlib import Path
from verity_vllm import messages as M
f, k, wc, fin, in_lease = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5] == "1"
got = [p for p in (json.loads(b) for b in (M.lines(f, "Run") if f.is_file() else [])) if p.get("window") == k]
p = got[-1] if got else {"error": "no plan line"}
why = p.get("refused") or p.get("error") or p.get("pending")
print(f"WINDOW-PLAN window {k}: " + (f"none, {str(why)[:160]}" if why else f"units {p.get('units')}, feeds {p.get('feeds')}, fed rows {p.get('fed_rows')}, seconds {json.dumps(p.get('seconds'))}, "
      "unprovable " + ("; ".join(f"{why} {len(u['templates'])} templates, {u['units']} units"
                                 for why, u in (p.get('unprovable') or {}).items()) or "none named")))
h = p.get("handoff") or {"error": "no handoff"}
print(f"WINDOW-HANDOFF window {k}: " + (f"none, {str(h['error'])[:200]}" if "error" in h else
      f"{h['ports']} ports, {h['units']} units hashed by {h['where']} in {h['s']} s under the lease, files {' '.join(h['files'])}"))
r = p.get("record") or {"error": "no record"}
if "pending" in r:
    r = json.loads(fin.read_text()) if fin.is_file() and fin.stat().st_size else {"error": f"not made after the lease ({fin} absent)"}
print("WHERE: the serve, the plan and the card's strings on the card under the lease; the record, the close and the write "
      + ("at the serve's close, under the lease" if in_lease else "after the lease, on this host's processors (verity-vllm close-record)"))
if "error" in r:
    print(f"WINDOW-STRINGS window {k}: none\nWINDOW-RECORD window {k}: none, {str(r['error'])[:200]}")
    sys.exit(0)
s, t = r["strings"], r["strings"].get("twin") or {}
# handed strings (the card's, made at its close) carry no acquired_s: their time is the handoff's, under the lease
card_s = f" in {s['acquired_s']} s" if "acquired_s" in s else ""
print(f"WINDOW-STRINGS window {k}: {s['ports']} ports, card ({s['where']}) {s['acquired']} units{card_s}, host "
      f"{s['host']} units in {s['host_s']} s; twin on {t.get('ports')} of {t.get('of')} fed ports ({t.get('units')} units): "
      + ("EQUAL" if t.get("equal") else f"NOT EQUAL {t.get('unequal')}"))
wc = json.loads(wc.read_text()) if wc.is_file() and wc.stat().st_size else None
vs = "no window-commit to compare" if wc is None else (
    f"{'equal to' if (r['members'], r['population']) == (wc.get('members'), wc.get('population')) else 'NOT equal to'} "
    f"window-commit's {wc.get('members')}, {wc.get('population')}")
print(f"WINDOW-RECORD window {k} at the serve's close: members {r['members']}, population {r['population']} ({vs}); "
      f"seconds {json.dumps(r['seconds'])}")
if d := r.get("design", {}).get("development"):
    print(f"a development sampling design is registered: {d['draws']} draws of the window's {d['of']} proof blocks, "
          f"not the rate of record")
EOF
}
# SR holds the keys and salts, and $OUT/window the circuit's description, which is the auditor's secret, with the
# window's plan and handoff: custody's statement names them private
if [ "$DRY" != 1 ] && [ -d "${RESEARCH_RUN_DIR:-}" ]; then
  "$PY" - "$RESEARCH_RUN_DIR/private-paths.json" "$SR" <<'EOF'
import json, os, pathlib, sys
f, sr = pathlib.Path(sys.argv[1]), os.path.realpath(sys.argv[2])
doc = json.loads(f.read_text()) if f.is_file() else {}
names = {*doc.get("names", []), "window/", "plans.jsonl", "handoff/", "handoff.json", "circuit.json"}
f.write_text(json.dumps({"schema": "private-paths/v1", "private": sorted({*doc.get("private", []), sr}),
                         "names": sorted(names)}, indent=1) + "\n")
EOF
fi
if [ "${1:-}" != card ]; then  # the row's Build of record, which the serve binds, before the lease
  "$PY" -m research data fetch-fixtures --repo "$TREE" --json "$TREE/$WL" > /dev/null
  want=$(awk -v n="$ROW.json" '$2==n{print $1}' adapters/vllm/workloads/SHA256SUMS)
  [ "$(sha256sum < "$WL" | cut -c1-64)" = "$want" ] || { say "SMOKE FAILED workload: $WL is not SHA256SUMS's"; exit 1; }
  (cd adapters/vllm && "$PY" -m verity_vllm.pipeline.cli predict row --row "$ROW" --out "$PRED") > "$SR/predict.log" 2>&1 \
    || { say "SMOKE FAILED predict ($SR/predict.log)"; exit 1; }
  # the close's plan and record as far as they build before the serve (serve.preflight), before the lease
  (cd adapters/vllm && CUDA_VISIBLE_DEVICES= "$PY" - "$DRY" "${SERVE[@]}" <<'EOF'
import json, os, sys
from verity_vllm.pipeline import cli
from verity_vllm.program.predict import serve
opts = cli.parse(sys.argv[2], sys.argv[3:], environ=os.environ)
print("PLAN PREFLIGHT PASSED: " + json.dumps(serve.preflight(opts, card=sys.argv[1] != "1")))
EOF
  ) > "$SR/preflight.log" 2>&1 || { tail -3 "$SR/preflight.log"; say "SMOKE REFUSED plan preflight ($SR/preflight.log)"; exit 3; }
  tail -1 "$SR/preflight.log"
fi
if [ "$DRY" = 1 ]; then
  "$PY" - "${SERVE[@]}" <<'EOF'
import os, sys
from verity_vllm.pipeline import cli, serve_openai as so
opts = cli.parse(sys.argv[1], sys.argv[2:], environ=os.environ)
so.preflight(opts, which=lambda _: "/stand-in/ninja")
print("SERVE PARSED AND PREFLIGHT PASSED:", opts.precision, opts.scheme, "seal", opts.seal, "window steps", opts.window_steps,
      "window requests", opts.window_requests, "window class", opts.window_class)
if run := os.environ.get("RESEARCH_RUN_DIR"):
    try:
        so.preflight(cli.parse(sys.argv[1], [*sys.argv[2:], "--out", run + "/serve"], environ=os.environ), which=lambda _: "/stand-in/ninja")
        sys.exit("SMOKE FAILED: an --out inside the run dir was not refused")
    except cli.UsageError as e:
        print("INSIDE THE RUN DIR REFUSED:", str(e)[:60])
EOF
  [ -n "$NOKIT" ] || CUDA_VISIBLE_DEVICES= "$PY" - <<'EOF'
import os
from verity_vllm import settings
from verity_vllm.commit.committer import native_ext, native_jit
from verity_vllm.config import cuda_arch
sku, home = settings.card_sku(), settings.cuda_home()
ext = native_ext.ext()
assert not native_jit.missing_bindings(ext, os.path.join(native_ext.HERE, "native_collect.cpp"))
print(f"COMMITTER BUILT ITS NATIVE COLLECTOR: CARD_SKU {sku}, arch {cuda_arch(sku)}, CUDA_HOME {home}, every binding present")
EOF
  echo "LOAD: ${LOAD[*]}"
  echo "LEASE: ${LEASE[*]}"
  [ -z "${WIPE_CMD:-}" ] || echo "WIPE under the lease before serve-openai: $WIPE_CMD $D all $D/pieces/wipe"
  window_commit_of 0
  CUDA_VISIBLE_DEVICES= "$PY" - "${WC[@]}" <<'EOF'
import runpy, sys
g = runpy.run_path(sys.argv[1], run_name="window_commit")
pp = g["prove_path"]
pp.window_commit_of_advice = lambda a, **_: {"parsed": sorted(k for k, v in vars(a).items() if v is not None)}
sys.exit(pp.main(sys.argv[2:], kept_ingress=g["attester_piece"].kept_ingress, scan_state_file=g["SCAN_STATE_FILE"]))
EOF
  echo "WINDOW-COMMIT PARSED (at window 0; the run's is the fixed load's): ${WC[*]//"$LAW"/$DESIGN}"
  (cd adapters/vllm && "$PY" "${FIN[@]}" --serve-run "$D" --window 0 --out "$SR/record.json" --parse-only)
  echo "RECORD AFTER THE LEASE PARSED (RECORD_IN_LEASE=$RECORD_IN_LEASE): ${FIN[*]//"$LAW"/$DESIGN} --serve-run $D --window K --out $SR/record.json"
  echo "SMOKE DRY OK${NOKIT:+ (no toolkit here: the native collector not built)}; not exercised off the card: the model load, the committer's construction over the engine, the request, the seal, the card's strings, the open, the record after the lease and the window commit; no proof is made, and Lean verifies nothing"
  exit 0
fi

if [ "${1:-}" = card ]; then  # under the lease: the wipe where the application lists it, the serve, the load and the server's stop
  if [ -n "${WIPE_CMD:-}" ]; then  # the wipe's command as served_verdicts.sh's serve() calls it, before the engine loads
    mkdir -p "$D/pieces/wipe"; rc=0
    timeout -k 1m "${WIPE_TIMEOUT:-20m}" bash -c "$WIPE_CMD" _ "$D" all "$D/pieces/wipe" > "$SR/wipe.log" 2>&1 || rc=$?
    printf '{"limit": "%s", "rc": %d}\n' "${WIPE_TIMEOUT:-20m}" "$rc" > "$D/pieces/wipe/piece.json"
  fi
  (cd adapters/vllm && exec "$PY" -m verity_vllm.pipeline.cli "${SERVE[@]}") > "$SR/serve.log" 2>&1 &
  SERVER=$!
  listened=0
  for _ in $(seq 1 120); do
    kill -0 "$SERVER" 2> /dev/null || { say "SMOKE FAILED serve: exited before it listened (tail $SR/serve.log)"; exit 1; }
    curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1 && { listened=1; break; }
    sleep 5
  done
  [ "$listened" = 1 ] || { say "SMOKE FAILED serve: the server did not listen on port $PORT within 10 minutes (tail $SR/serve.log)"; exit 1; }
  # the deployment's one registration, before the first request: every epoch's close reads it and none writes one
  "$PY" benchmarks/prove_path/prove_path.py weights-registration --serve-run "$D" --checkpoint-dir "$SNAP" --out "$WREG" \
    --deployment-record "$DREG" > "$SR/weights_registration.log" 2>&1 \
    || { say "SMOKE FAILED weights-registration ($SR/weights_registration.log)"; exit 1; }
  if [ -n "${REGISTRATION_LANE:-}" ]; then
    FIXED=$("$PY" adapters/vllm/tools/served_run/fixed_circuit.py --serve-run "$D" --class "$CLASS" 2> "$SR/fixed_circuit.err" \
      | "$PY" -c 'import json, sys; print(json.load(sys.stdin)["program"])') \
      || { say "SMOKE REFUSED registration: no epoch's circuit fixed before the serve: $(tail -1 "$SR/fixed_circuit.err")"; exit 3; }
    echo "$FIXED" > "$SR/fixed_program"
    echo "REGISTRATION PERIOD-OPEN on the challenger's host: vy-challenger-gate period-open --name $REGISTRATION_LANE" \
      "--lane $REGISTRATION_LANE --cap $REGISTRATION_CAP --circuit $FIXED" \
      "--registration $("$PY" -c 'import json, sys; from verity.commitments import docket_digest as DD; print(DD.digest(json.load(open(sys.argv[1]))))' "$DREG")"
  fi
  if [ ${#FETCH[@]} -gt 0 ]; then "$PY" "${FETCH[@]}" > "$SR/ledger_fetch.log" 2>&1 & FETCHER=$!; fi
  rc=0; ${REGISTRATION_LANE:+timeout -k 10s "$REGISTRATION_HOLD_S"} "$PY" "${HOLD[@]}" "$DREG" 2>&1 \
    | tee "$SR/registration_hold.log" || rc=$?
  [ -z "${FETCHER:-}" ] || { kill "$FETCHER" 2> /dev/null; wait "$FETCHER" 2> /dev/null || true; }
  [ "$rc" = 0 ] || { say "SMOKE FAILED registration hold, exit $rc: $(tail -1 "$SR/registration_hold.log")"; exit 1; }
  "$PY" adapters/vllm/tools/served_run/inputs.py tokenize --port "$PORT" --dataset "$DATASET" > "$SR/tokenize.log" 2>&1 \
    || echo "TOKENIZE FAILED ($SR/tokenize.log): $(tail -1 "$SR/tokenize.log")" >> "$SR/tokenize.log"
  "$PY" "${LOAD[@]}" > "$SR/load.log" 2>&1 || { say "SMOKE FAILED request ($SR/load.log)"; exit 1; }
  kill -INT "$SERVER"; s=0; wait "$SERVER" || s=$?; SERVER=
  [ "$s" = 0 ] || { say "SMOKE FAILED stop: the server's exit ($SR/serve.log)"; exit 1; }
  exit 0
fi

"${LEASE[@]}" || { [ -s "$OUT/smoke.txt" ] || say "SMOKE FAILED lease: the card phase under gpu-lease failed ($SR/serve.log)"; exit 1; }
export CUDA_VISIBLE_DEVICES=
said=$("$PY" - "$D/windows" <<'EOF'
import json, sys
from pathlib import Path
from verity_vllm.pipeline import deferred_rows as DR
for w in sorted(Path(sys.argv[1]).glob("w*")):
    r = DR.of_record(json.load(open(w / "commit/verdict.json"))["replay_deferred"]["bundles"][0]["path"], str(w))
print(f"SMOKE SEALED AND OPENED window {w.name} with no card visible: {r.commitment}, run root {bytes(r.run.run_root).hex()[:16]}")
EOF
) || { say "SMOKE FAILED open: the sealed bundle against its record ($D/windows)"; exit 1; }
[ -s "$WREG" ] && [ -s "$DREG" ] || { say "$said; SMOKE FAILED: no deployment registration ($WREG, $DREG), which is written before the first request"; exit 1; }
said+="; WEIGHTS REGISTRATION the deployment's, written before the first request"
CA=$("$PY" benchmarks/prove_path/prove_path.py compiler --serve-run "$D" --cap-bits "$CAP" \
  | "$PY" -c "import json, sys; c = json.load(sys.stdin)['inner']; print('--compiler-name', c['name'], '--compiler-digest', c['digest'])") \
  || { say "$said; SMOKE FAILED compiler (prove_path.py compiler)"; exit 1; }
K=$("$PY" benchmarks/one_stage/served_verdicts.py window "$D" --slots "$SLOTS" --of-load "$SR/load.jsonl" 2> "$SR/window.err" \
  | "$PY" -c 'import json, sys; print(json.load(sys.stdin)["window"])') \
  || { say "$said; SMOKE FAILED window: no one sealed window holds the fixed load ($SR/window.err)"; exit 1; }
if [ "${RECORD:-1}" = 0 ]; then echo "WINDOW-RECORD after the lease: not run (RECORD=0: the plan alone)"
elif [ "$RECORD_IN_LEASE" = 1 ]; then echo "WINDOW-RECORD after the lease: not run (RECORD_IN_LEASE=1: the serve's close made it)"
else  # the serve's close left the card's strings; the record is made here, from the advice its close committed
  (cd adapters/vllm && timeout -k 1m "$FIN_TIMEOUT" "$PY" "${FIN[@]}" --serve-run "$D" --window "$K" --out "$SR/record.json") \
    > "$SR/record.log" 2>&1 || { e=$?; say "$said"$'\n'"$(plan_lines)"$'\n'"SMOKE FAILED record after the lease of window $K, exit $e: $(tail -1 "$SR/record.log" | cut -c1-300) ($SR/record.log)"; exit 1; }
  mkdir -p "$OUT/window" && cp "$SR/record.json" "$D/windows/w$K/handoff/handoff.json" "$OUT/window/"
fi
# shellcheck disable=SC2086
(cd adapters/vllm && "$PY" -m verity_vllm.pipeline.cli advice "$D" --window "$K" --cap "$CAP" $CA --weights-of-record "$WREG" \
  --out "$D/advice/.w$K") > "$SR/advice.log" 2>&1 \
  || { say "$said; SMOKE FAILED advice of window $K ($SR/advice.log)"; exit 1; }
if [ -d "$D/advice/w$K" ]; then  # the serve's close compiled its circuit from this advice, and its record is bound to its opening
  cmp -s "$D/advice/.w$K/w$K/private/advice.bin" "$D/advice/w$K/private/advice.bin" \
    || { say "$said; SMOKE FAILED advice of window $K: not the advice the serve's close committed ($D/advice)"; exit 1; }
  rm -rf "$D/advice/serve-w$K" && mv "$D/advice/w$K" "$D/advice/serve-w$K"
fi
mv "$D/advice/.w$K/w$K" "$D/advice/w$K"
window_commit_of "$K"
timeout -k 1m "$WC_TIMEOUT" "$PY" "${WC[@]}" > "$SR/window-commit.json" 2> "$SR/window-commit.err" \
  || { e=$?; say "$said"$'\n'"$(plan_lines)"$'\n'"SMOKE FAILED window-commit of window $K, exit $e ($SR/window-commit.err)"; exit 1; }
mkdir -p "$OUT/window"
cp "$VW/registration.json" "$VW/circuit.json" "$DREG" "$OUT/window/"
if [ -s "$SR/fixed_program" ]; then
  got=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["program"])' "$VW/registration.json")
  [ "$got" = "$(cat "$SR/fixed_program")" ] || { say "$said; SMOKE FAILED circuit: the closed epoch's docket names Program" \
    "$got, not the one fixed before the serve and opened as the period's --circuit, $(cat "$SR/fixed_program")"; exit 1; }
  said+="; CIRCUIT the closed epoch's docket names the Program fixed before the serve, the period's --circuit"
fi
said+="; $("$PY" - "$SR/window-commit.err" "$VW/registration.json" "$K" "$CLASS" "$SR" "$MNBT" <<'EOF'
import json, math, sys
from verity.commitments import docket_digest as DD
err, reg, k, cls, sr, mnbt = sys.argv[1:]
rows = json.loads([x for x in open(err) if x.startswith("WINDOW-ROWS ")][-1].removeprefix("WINDOW-ROWS "))
o, u = rows["opened"] or {}, rows["units"]
print(f"WINDOW-COMMIT window {k} class {cls}: reads {o.get('reads')}, failed {o.get('failed')}, not retained {o.get('not_retained')}; "
      f"units served {u['served']}, computed {u['computed']}, evaluated {u['evaluated']}; registration written, its digest kept in SR (it binds the circuit's own digest)")
s = int(cls.split(",")[0])
print(f"CLASS {cls}: {math.log2(int(mnbt) // s):.2f} bits leave with it (its prompt length, one of the {int(mnbt) // s} "
      f"that fit {s} slots in {mnbt} tokens a step; its slots and outputs are the application's)")
open(f"{sr}/registration.digest", "w").write(DD.digest(json.load(open(reg))) + "\n")
EOF
)" || { say "$said; SMOKE FAILED counts: no WINDOW-ROWS line or registration ($SR/window-commit.err)"; exit 1; }
said+=$'\n'"$(plan_lines)"
said+=$'\n'"$(grep '^TOKENIZE' "$SR/tokenize.log" | tail -1)"
said+=$'\n'"SMOKE DONE window $K: served, planned, its strings made on the card and its record written, its circuit registered by the prover; no proof is made, and Lean verifies nothing; no sampled verdict, so no δ or ε; the circuit digest is kept private and NO public value fixes it before the coins"
say "$said"

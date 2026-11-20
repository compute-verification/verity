#!/usr/bin/env bash
# Does: serves ONE window of bf16 Llama 3.2 1B (unsloth at 9535bd9b) on the H100 at the window class CLASS (default 4,27,2: 4
#   slots of 27-token prompts, 2 outputs) through serve-openai with its rows-sha512 Commit, no scheme, and seals it, each
#   window's close planning its class circuit and leaving the card's part of its record (the acquired units' strings,
#   made where they are committed, under the window's key: close_record.Record.handoff) and, with RECORD_IN_LEASE=1, the
#   record too (RECORD=0: the plan alone); then, the lease released, on this host's processors: opens the sealed bundle
#   against its record, makes the window's record from the handoff (close_record's finish: the record, the close with the
#   card's strings, the write), and commits the fixed load's window as its class circuit's units
#   (`benchmarks/one_stage/window_commit.py window-commit`, as served_verdicts.sh's window_commit() calls it).
# Does NOT show: an audit, a proof, or a traced Build on the H100.
# Needs: PY (default the node's /workspace/venv312/bin/python), gpu-lease (it leases one card for the serve alone, at most
#   LEASE_MIN minutes, default 30), HF_HOME (default /workspace/hf) holding the model; CUDA_HOME (12.9, machine.py's),
#   CARD_SKU (machine.py sku), PYTHONPATH and VERITY_PROTOCOLS_CONFIG are set here. WIPE_CMD, set only where the application
#   lists the wipe, wipes the leased card before serve-openai starts, its answer in the serve's pieces/wipe (unset: no wipe).
# Writes: the serve's directory, the private directory (keys, salts) and every log under SR, outside the run's directory
#   (named private in the run's private-paths.json); under $OUT the window's registration.json and circuit.json, the
#   deployment registration and smoke.txt: the reader's counts, the units by source and the registration's digest, then the
#   window's plan, its strings (card and host, the twin's comparison) and its record beside window-commit's.
#   research run --on vy-h100-1 --cwd source -- taskset -c 0-9 bash adapters/vllm/tools/served_run/h100_smoke.sh   (DRY=1: off the card)
set -euo pipefail
[ -z "${PEARLPLUS_TIE:-}" ] || [ "$PEARLPLUS_TIE" = 0 ] || { echo "h100_smoke.sh: PEARLPLUS_TIE is retired (the window commit takes no tie); unset it" >&2; exit 3; }
[ -z "${SCHEME:-}" ] || { echo "h100_smoke.sh: SCHEME is retired (the smoke serves with no scheme; Pearl+ is served_verdicts.sh's POUW=pearlplus); unset it" >&2; exit 3; }
SELF=adapters/vllm/tools/served_run/h100_smoke.sh
TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd); cd "$TREE"
ROW=llama32-1b__bf16__h100__tp1__b4__i256__o32__mixed__greedy__bi-eager
MODEL=unsloth/Llama-3.2-1B REV=9535bd9b1d1dea6acafbdc4813b728796aeb28da
DRY=${DRY:-0} PY=${PY:-/workspace/venv312/bin/python} CLASS=${CLASS:-4,27,2} LEASE_MIN=${LEASE_MIN:-30} LEASE_MEM_GB=${LEASE_MEM_GB:-64}
# the sampling design is the auditor's and not public: a printed line says only that one is registered
DESIGN="(the registered sampling design)"
MNBT=${MNBT:-512} CAP=${CAP:-8388608} LAW=${LAW:-rate:1/3481648} WC_TIMEOUT=${WC_TIMEOUT:-90m} FIN_TIMEOUT=${FIN_TIMEOUT:-120m}
RECORD_IN_LEASE=${RECORD_IN_LEASE:-0}
# the window class (S,P,O[,LANES,EVERY]: slots, prompt ids, outputs) the serve's arena is sized from and the load fills
[[ $CLASS =~ ^[1-9][0-9]*,[1-9][0-9]*,[1-9][0-9]*(,[1-9][0-9]*,[1-9][0-9]*)?$ ]] \
  || { echo "SMOKE REFUSED class: CLASS is S,P,O[,LANES,EVERY] of positive integers, not $CLASS"; exit 3; }
IFS=, read -r SLOTS PROMPT OUTPUTS _ <<< "$CLASS"
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
NOKIT=
CUDA_HOME=$("$PY" benchmarks/kernels/machine.py toolkit 12.9) || if [ "$DRY" = 1 ]; then
  NOKIT=1 CUDA_HOME=$SCRATCH/no-toolkit; echo "DRY: this machine lists no CUDA 12.9, so the committer's native build is not run"
else echo "SMOKE REFUSED toolkit: the machine lists no CUDA 12.9 (machine.py, above)"; exit 3; fi
if [ "$DRY" = 1 ] && [ -n "${CARD_SKU:-}" ]; then echo "DRY: CARD_SKU $CARD_SKU as given (off the card, no card to read it from)"
else CARD_SKU=$("$PY" benchmarks/kernels/machine.py sku); fi || { echo "SMOKE REFUSED card: the machine's record names no card sku for this host (machine.py sku, above)"; exit 3; }
export CUDA_HOME CARD_SKU HF_HOME=${HF_HOME:-/workspace/hf} HF_HUB_OFFLINE=1 GPU_LEASE_WHO=${GPU_LEASE_WHO:-vllm-served-run}
export PATH=$(dirname "$PY"):$CUDA_HOME/bin:$PATH
SNAP=$HF_HOME/hub/models--unsloth--Llama-3.2-1B/snapshots/$REV
{ [ -s "$SNAP/config.json" ] && ls "$SNAP"/*.safetensors > /dev/null 2>&1; } || { [ "$DRY" = 1 ] && echo "DRY: $MODEL at $REV is not in $HF_HOME (the dry run loads no model)"; } || {
  echo "SMOKE REFUSED model: $MODEL at $REV is not in $HF_HOME; fetch it outside the lease:" \
    "HF_HOME=$HF_HOME $PY -c \"from huggingface_hub import snapshot_download as s; s('$MODEL', revision='$REV')\""; exit 3; }
OUT=${OUT:-${RESEARCH_RUN_DIR:-$SR}/out}
D=$SR/serve PRED=$SR/predicted PORT=${PORT:-18282}
mkdir -p "$SR" "$OUT"
export PYTHONPATH=$TREE:$TREE/adapters/vllm:$TREE/adapters/torch:$TREE/tools/config:$TREE/verity/protocols/proofs/flock/prover/python:$TREE/tools/research/src
export VERITY_PROTOCOLS_CONFIG="{\"private\": \"$SR/private\"}"
WL=adapters/vllm/workloads/$ROW.json
say() { echo "$1"; [ "$DRY" = 1 ] || echo "$1" > "$OUT/smoke.txt"; }

# the fixed load: S requests of P prompt ids and O outputs, released together under a release run in the network attester's
# release order, so they start in the window's first step and fill one window of O steps (served_verdicts.sh's LOAD=fixed)
RELEASE=${RELEASE:-$(tr -d - < /proc/sys/kernel/random/uuid)}
LOAD=(adapters/vllm/tools/served_run/serve_load.py --port "$PORT" --model "$MODEL" --fixed "$SLOTS" --window-steps "$OUTPUTS"
  --prompt-range "$PROMPT:$PROMPT" --release "$RELEASE" --seed 1 --out "$SR/load.jsonl")
REC_ARGS=()  # RECORD=0: each window's close plans its circuit and makes no record
[ "${RECORD:-1}" = 0 ] || REC_ARGS=(--sampling-design "$LAW" --checkpoint-dir "$SNAP")
[ "${RECORD:-1}" = 0 ] || [ "$RECORD_IN_LEASE" != 1 ] || REC_ARGS+=(--record-in-lease)
SERVE=(serve-openai --model "$MODEL" --revision "$REV" --out "$D" --port "$PORT" --precision bfloat16
  --run-id "smoke-${RESEARCH_RUN_ID:-local}" --seal --max-num-batched-tokens "$MNBT" --window-steps "$OUTPUTS"
  --row-build "$PRED" --window-requests "$SLOTS" --window-class "$CLASS" --max-num-seqs "$SLOTS"
  --max-model-len "$((PROMPT + OUTPUTS))" --window-gb 4 --disk-floor-gb 5 --seal-hold-gb 8
  ${REC_ARGS[@]+"${REC_ARGS[@]}"}
  --protocols-config '{"network-attester": {"order": "release-order"}}')
LEASE=(gpu-lease 1 --wait --max-min "$LEASE_MIN" --mem-gb "$LEASE_MEM_GB" -- bash "$TREE/$SELF" card)
WREG=$D/weights_registration.json DREG=$D/deployment-registration.json VW=$SR/window
FIN=(-m verity_vllm.program.predict.close_record --class "$CLASS" --sampling-design "$LAW" --checkpoint-dir "$SNAP")
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
      + ("at the serve's close, under the lease" if in_lease else "after the lease, on this host's processors (close_record's finish)"))
if "error" in r:
    print(f"WINDOW-STRINGS window {k}: none\nWINDOW-RECORD window {k}: none, {str(r['error'])[:200]}")
    sys.exit(0)
s, t = r["strings"], r["strings"].get("twin") or {}
print(f"WINDOW-STRINGS window {k}: {s['ports']} ports, card ({s['where']}) {s['acquired']} units in {s['acquired_s']} s, host "
      f"{s['host']} units in {s['host_s']} s; twin on {t.get('ports')} of {t.get('of')} fed ports ({t.get('units')} units): "
      + ("EQUAL" if t.get("equal") else f"NOT EQUAL {t.get('unequal')}"))
wc = json.loads(wc.read_text()) if wc.is_file() and wc.stat().st_size else None
vs = "no window-commit to compare" if wc is None else (
    f"{'equal to' if (r['members'], r['population']) == (wc.get('members'), wc.get('population')) else 'NOT equal to'} "
    f"window-commit's {wc.get('members')}, {wc.get('population')}")
print(f"WINDOW-RECORD window {k} at the serve's close: members {r['members']}, population {r['population']} ({vs}); "
      f"seconds {json.dumps(r['seconds'])}")
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
  "$PY" - "$CLASS" "${LOAD[@]}" <<'EOF'
import runpy, sys
S, P, O = (int(x) for x in sys.argv[1].split(",")[:3])
a = runpy.run_path(sys.argv[2])["parser"]().parse_args(sys.argv[3:])
assert (a.fixed, a.window_steps, a.prompt_range, a.varied) == (S, O, f"{P}:{P}", 0) and len(a.release) == 32, a
print("LOAD PARSED: fixed", a.fixed, "request(s) of", a.prompt_range, "tokens over", a.window_steps, "window steps, release run", a.release[:8])
EOF
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
if [ -s "$WREG" ]; then  # the serve's first close registered the weights (close_record.Record.roots)
  said+="; WEIGHTS REGISTRATION the serve's, made at its first close"
else
  "$PY" benchmarks/prove_path/prove_path.py weights-registration --serve-run "$D" --checkpoint-dir "$SNAP" --out "$WREG" \
    --deployment-record "$DREG" > "$SR/weights_registration.log" 2>&1 \
    || { say "$said; SMOKE FAILED weights-registration ($SR/weights_registration.log)"; exit 1; }
fi
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
cp "$VW/registration.json" "$VW/circuit.json" "$OUT/window/" && cp "$DREG" "$OUT/"
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
said+=$'\n'"SMOKE DONE window $K: served, planned, its strings made on the card and its record written; no proof is made, and Lean verifies nothing; no sampled verdict, so no δ or ε; the circuit digest is kept private and NO public value fixes it before the coins"
say "$said"

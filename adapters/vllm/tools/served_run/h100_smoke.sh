#!/usr/bin/env bash
# Does: serves ONE window of bf16 Llama 3.2 1B (unsloth at 9535bd9b) on the H100 at the window class CLASS (default 4,27,2: 4
#   slots of 27-token prompts, 2 outputs) through serve-openai with its rows-sha512 Commit, no scheme, and seals it; then, the
#   lease released, opens the sealed bundle against its record and commits the fixed load's window as its class circuit's units
#   (`benchmarks/one_stage/window_commit.py window-commit`, as served_verdicts.sh's window_commit() calls it), on the CPU.
# Does NOT show: an audit, a proof, or a traced Build on the H100.
# Needs: PY (default the node's /workspace/venv312/bin/python), gpu-lease (it leases one card for the serve alone, at most
#   LEASE_MIN minutes, default 30), HF_HOME (default /workspace/hf) holding the model; CUDA_HOME (12.9, machine.py's),
#   CARD_SKU (machine.py sku), PYTHONPATH and VERITY_PROTOCOLS_CONFIG are set here.
# Writes: the serve's directory, the private directory (keys, salts) and every log under SR, outside the run's directory
#   (named private in the run's private-paths.json); under $OUT the window's registration.json and circuit.json, the
#   deployment registration and smoke.txt, the last line: the reader's counts, the units by source and the registration's digest.
#   research run --on vy-h100-1 --cwd source -- taskset -c 0-9 bash adapters/vllm/tools/served_run/h100_smoke.sh   (DRY=1: off the card)
set -euo pipefail
[ -z "${PEARLPLUS_TIE:-}" ] || [ "$PEARLPLUS_TIE" = 0 ] || { echo "h100_smoke.sh: PEARLPLUS_TIE is retired (the window commit takes no tie); unset it" >&2; exit 3; }
[ -z "${SCHEME:-}" ] || { echo "h100_smoke.sh: SCHEME is retired (the smoke serves with no scheme; Pearl+ is served_verdicts.sh's POUW=pearlplus); unset it" >&2; exit 3; }
SELF=adapters/vllm/tools/served_run/h100_smoke.sh
TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd); cd "$TREE"
ROW=llama32-1b__bf16__h100__tp1__b4__i256__o32__mixed__greedy__bi-eager
MODEL=unsloth/Llama-3.2-1B REV=9535bd9b1d1dea6acafbdc4813b728796aeb28da
DRY=${DRY:-0} PY=${PY:-/workspace/venv312/bin/python} CLASS=${CLASS:-4,27,2} LEASE_MIN=${LEASE_MIN:-30} LEASE_MEM_GB=${LEASE_MEM_GB:-64}
MNBT=${MNBT:-512} CAP=${CAP:-8388608} LAW=${LAW:-rate:1/3481648} WC_TIMEOUT=${WC_TIMEOUT:-15m}
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
SERVE=(serve-openai --model "$MODEL" --revision "$REV" --out "$D" --port "$PORT" --precision bfloat16
  --run-id "smoke-${RESEARCH_RUN_ID:-local}" --seal --max-num-batched-tokens "$MNBT" --window-steps "$OUTPUTS"
  --row-build "$PRED" --window-requests "$SLOTS" --window-class "$CLASS" --max-num-seqs "$SLOTS"
  --max-model-len "$((PROMPT + OUTPUTS))" --window-gb 4 --disk-floor-gb 5 --seal-hold-gb 8
  --protocols-config '{"network-attester": {"order": "release-order"}}')
LEASE=(gpu-lease 1 --wait --max-min "$LEASE_MIN" --mem-gb "$LEASE_MEM_GB" -- bash "$TREE/$SELF" card)
WREG=$D/weights_registration.json DREG=$D/deployment-registration.json VW=$SR/window
window_commit_of() {  # WC: window $1's commit as its class circuit's units, served_verdicts.sh's window_commit() command
  WC=(benchmarks/one_stage/window_commit.py window-commit --serve-run "$D" --window "$1" --row-dir "$D/windows/w$1"
    --advice "$D/advice/w$1/private/advice.bin" --class "$CLASS" --registration "$WREG" --checkpoint-dir "$SNAP"
    --sampling-design "$LAW" --out "$VW")
}

if [ "$DRY" != 1 ] && [ -d "${RESEARCH_RUN_DIR:-}" ]; then  # SR holds the keys and salts: custody's statement names it private
  "$PY" - "$RESEARCH_RUN_DIR/private-paths.json" "$SR" <<'EOF'
import json, os, pathlib, sys
f, sr = pathlib.Path(sys.argv[1]), os.path.realpath(sys.argv[2])
doc = json.loads(f.read_text()) if f.is_file() else {}
f.write_text(json.dumps({"schema": "private-paths/v1", "private": sorted({*doc.get("private", []), sr}),
                         "names": list(doc.get("names", []))}, indent=1) + "\n")
EOF
fi
if [ "${1:-}" != card ]; then  # the row's Build of record, which the serve binds, before the lease
  "$PY" -m research data fetch-fixtures --repo "$TREE" --json "$TREE/$WL" > /dev/null
  want=$(awk -v n="$ROW.json" '$2==n{print $1}' adapters/vllm/workloads/SHA256SUMS)
  [ "$(sha256sum < "$WL" | cut -c1-64)" = "$want" ] || { say "SMOKE FAILED workload: $WL is not SHA256SUMS's"; exit 1; }
  (cd adapters/vllm && "$PY" -m verity_vllm.pipeline.cli predict row --row "$ROW" --out "$PRED") > "$SR/predict.log" 2>&1 \
    || { say "SMOKE FAILED predict ($SR/predict.log)"; exit 1; }
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
  window_commit_of 0
  CUDA_VISIBLE_DEVICES= "$PY" - "${WC[@]}" <<'EOF'
import runpy, sys
g = runpy.run_path(sys.argv[1], run_name="window_commit")
pp = g["prove_path"]
pp.window_commit_of_advice = lambda a, **_: {"parsed": sorted(k for k, v in vars(a).items() if v is not None)}
sys.exit(pp.main(sys.argv[2:], kept_ingress=g["attester_piece"].kept_ingress, scan_state_file=g["SCAN_STATE_FILE"]))
EOF
  echo "WINDOW-COMMIT PARSED (at window 0; the run's is the fixed load's): ${WC[*]}"
  echo "SMOKE DRY OK${NOKIT:+ (no toolkit here: the native collector not built)}; not exercised off the card: the model load, the committer's construction over the engine, the request, the seal, the open and the window commit"
  exit 0
fi

if [ "${1:-}" = card ]; then  # under the lease: the serve, the load and the server's stop, nothing else
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
"$PY" benchmarks/prove_path/prove_path.py weights-registration --serve-run "$D" --checkpoint-dir "$SNAP" --out "$WREG" \
  --deployment-record "$DREG" > "$SR/weights_registration.log" 2>&1 \
  || { say "$said; SMOKE FAILED weights-registration ($SR/weights_registration.log)"; exit 1; }
CA=$("$PY" benchmarks/prove_path/prove_path.py compiler --serve-run "$D" --cap-bits "$CAP" \
  | "$PY" -c "import json, sys; c = json.load(sys.stdin)['inner']; print('--compiler-name', c['name'], '--compiler-digest', c['digest'])") \
  || { say "$said; SMOKE FAILED compiler (prove_path.py compiler)"; exit 1; }
K=$("$PY" benchmarks/one_stage/served_verdicts.py window "$D" --slots "$SLOTS" --of-load "$SR/load.jsonl" 2> "$SR/window.err" \
  | "$PY" -c 'import json, sys; print(json.load(sys.stdin)["window"])') \
  || { say "$said; SMOKE FAILED window: no one sealed window holds the fixed load ($SR/window.err)"; exit 1; }
# shellcheck disable=SC2086
(cd adapters/vllm && "$PY" -m verity_vllm.pipeline.cli advice "$D" --window "$K" --cap "$CAP" $CA --weights-of-record "$WREG" \
  --out "$D/advice/.w$K") > "$SR/advice.log" 2>&1 \
  || { say "$said; SMOKE FAILED advice of window $K ($SR/advice.log)"; exit 1; }
if [ -d "$D/advice/w$K" ]; then  # the serve's close compiled its circuit from this advice; its docket binds no weights of record
  cmp -s "$D/advice/.w$K/w$K/private/advice.bin" "$D/advice/w$K/private/advice.bin" \
    || { say "$said; SMOKE FAILED advice of window $K: not the advice the serve's close committed ($D/advice)"; exit 1; }
  rm -rf "$D/advice/w$K"
fi
mv "$D/advice/.w$K/w$K" "$D/advice/w$K"
window_commit_of "$K"
timeout -k 1m "$WC_TIMEOUT" "$PY" "${WC[@]}" > "$SR/window-commit.json" 2> "$SR/window-commit.err" \
  || { say "$said; SMOKE FAILED window-commit of window $K, exit $? ($SR/window-commit.err)"; exit 1; }
mkdir -p "$OUT/window"
cp "$VW/registration.json" "$VW/circuit.json" "$OUT/window/" && cp "$DREG" "$OUT/"
said+="; $("$PY" - "$SR/window-commit.err" "$VW/registration.json" "$K" "$CLASS" <<'EOF'
import json, sys
from verity.commitments import docket_digest as DD
err, reg, k, cls = sys.argv[1:]
rows = json.loads([x for x in open(err) if x.startswith("WINDOW-ROWS ")][-1].removeprefix("WINDOW-ROWS "))
o, u = rows["opened"] or {}, rows["units"]
print(f"WINDOW-COMMIT window {k} class {cls}: reads {o.get('reads')}, failed {o.get('failed')}, not retained {o.get('not_retained')}; "
      f"units served {u['served']}, computed {u['computed']}, evaluated {u['evaluated']}; registration {DD.digest(json.load(open(reg)))}")
EOF
)" || { say "$said; SMOKE FAILED counts: no WINDOW-ROWS line or registration ($SR/window-commit.err)"; exit 1; }
said+="; $("$PY" - "$D/plans.jsonl" "$K" <<'EOF'
import json, sys
from pathlib import Path
from verity_vllm import messages as M
f, k = Path(sys.argv[1]), int(sys.argv[2])
got = [p for p in (json.loads(b) for b in (M.lines(f, "Run") if f.is_file() else [])) if p.get("window") == k]
p = got[-1] if got else {"error": "no plan line"}
why = p.get("refused") or p.get("error") or p.get("pending")
print(f"WINDOW-PLAN window {k}: " + (f"none, {str(why)[:160]}" if why else f"units {p.get('units')}, feeds {p.get('feeds')}, fed rows {p.get('fed_rows')}"))
EOF
)"
say "$said"

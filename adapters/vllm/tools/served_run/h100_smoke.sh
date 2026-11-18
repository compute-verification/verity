#!/usr/bin/env bash
# Does: serves ONE window of bf16 Llama 3.2 1B (unsloth at 9535bd9b) on the H100 through serve-openai with its rows-sha512 Commit and seals it, then opens the sealed bundle against its record on the CPU.
# Does NOT show: a recompute on a processor (the replay is the sealed bundle opened against its record), an audit, or a traced Build on the H100.
# Needs: PY (default the node's /workspace/venv312/bin/python, both branches), CUDA_HOME (12.9, machine.py's), CARD_SKU (machine.py sku), HF_HOME (/workspace/hf), PYTHONPATH, VERITY_PROTOCOLS_CONFIG (all set here), CUDA_VISIBLE_DEVICES from the lease; the norm tap (no hidden tap: the row's manifest names no hidden class).
# Writes: the serve's directory, the private directory (keys, salts) and every log under SR, outside the run's directory; under $OUT only smoke.txt, the last line.
#   research run --on vy-h100-1 --cwd source -- taskset -c 0-9 bash adapters/vllm/tools/served_run/h100_smoke.sh   (DRY=1: off the card)
# SCHEME=pearl-plus: the same window served under Pearl+ by its name (serve-openai --protocols pearl-plus), at Pearl+'s tie
#   (scheme.py's tie, class 4,27,2: 4 slots of 27-token prompts, 2 window steps, fixed load), its --protocols-config the
#   file `served_passes.py arm` writes from PEARLC_KERNEL and PEARLC_CUBIN (benchmarks/harness/setup/pearl_c_sm90.sh, CUDA
#   13.0, built outside the lease; PEARLC_BF16_ROWS default 0); its salt in-process unless PEARLC_CHALLENGER_GATE,
#   PEARLC_CHALLENGER_PEER and PEARLC_CHALLENGER_LANE name a live session; after the open, `served_passes.py verify` over
#   the serve dir's pearl/w0/ with no card visible ($OUT/pearlplus-verify.json); it keeps pearl/w0/ (passes, coins.json,
#   served.json) and schemes/pearl-plus/ (its report) beside the sealed window. The verify runs once the server has
#   stopped, --fast --jobs VJOBS (16), and holds the passes to no window record; PEARLPLUS_TIE=1 adds the tie as a second
#   step after it accepts: `served_passes.py retain --window-record W` (pearl/window-record.json), then `verify --tie W
#   --class 4,27,2` ($OUT/pearlplus-verify-tie.json).
set -euo pipefail
TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd); cd "$TREE"
ROW=llama32-1b__bf16__h100__tp1__b4__i256__o32__mixed__greedy__bi-eager
MODEL=unsloth/Llama-3.2-1B REV=9535bd9b1d1dea6acafbdc4813b728796aeb28da
DRY=${DRY:-0} PY=${PY:-/workspace/venv312/bin/python} SCHEME=${SCHEME:-} PEARLPLUS_TIE=${PEARLPLUS_TIE:-0} VJOBS=${VJOBS:-16}
case $SCHEME in ""|pearl-plus) ;; *) echo "SMOKE REFUSED scheme: SCHEME is empty or pearl-plus, not $SCHEME"; exit 3 ;; esac
NSO=/workspace/cp/norm_tap/build/verity_norm_tap.so
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
fi
NOKIT=
CUDA_HOME=$("$PY" benchmarks/kernels/machine.py toolkit 12.9) || if [ "$DRY" = 1 ]; then
  NOKIT=1 CUDA_HOME=$SCRATCH/no-toolkit; echo "DRY: this machine lists no CUDA 12.9, so the committer's native build is not run"
else echo "SMOKE REFUSED toolkit: the machine lists no CUDA 12.9 (machine.py, above)"; exit 3; fi
if [ "$DRY" = 1 ] && [ -n "${CARD_SKU:-}" ]; then echo "DRY: CARD_SKU $CARD_SKU as given (off the card, no card to read it from)"
else CARD_SKU=$("$PY" benchmarks/kernels/machine.py sku); fi || { echo "SMOKE REFUSED card: the machine's record names no card sku for this host (machine.py sku, above)"; exit 3; }
export CUDA_HOME CARD_SKU HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
export PATH=$(dirname "$PY"):$CUDA_HOME/bin:$PATH
SNAP=$HF_HOME/hub/models--unsloth--Llama-3.2-1B/snapshots/$REV
if [ "$DRY" = 1 ] && [ ! -s "$NSO" ]; then NSO=$SCRATCH/stand-in-norm-tap.so; echo x > "$NSO"; echo "DRY: no norm tap here, a stand-in path"; fi
[ -s "$NSO" ] || { echo "SMOKE REFUSED norm tap: no $NSO; build it outside the lease (no card):" \
  "cd $TREE/adapters/vllm && CUDA_HOME=$CUDA_HOME CARD_SKU=$CARD_SKU bash verity_vllm/ops/pod_norm_tap.sh"; exit 3; }
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

"$PY" -m research data fetch-fixtures --repo "$TREE" --json "$TREE/$WL" > /dev/null
want=$(awk -v n="$ROW.json" '$2==n{print $1}' adapters/vllm/workloads/SHA256SUMS)
[ "$(sha256sum < "$WL" | cut -c1-64)" = "$want" ] || { say "SMOKE FAILED workload: $WL is not SHA256SUMS's"; exit 1; }
(cd adapters/vllm && "$PY" -m verity_vllm.pipeline.cli predict row --row "$ROW" --out "$PRED") > "$SR/predict.log" 2>&1 \
  || { say "SMOKE FAILED predict ($SR/predict.log)"; exit 1; }
"$PY" -c 'import json, sys; from verity_vllm.check.commit_rules import hidden_classes_required as h
sys.exit(h(json.load(open(sys.argv[1]))) is not False)' "$PRED/manifest.json" \
  || { say "SMOKE FAILED manifest: it names the hidden class, and this smoke attaches no hidden tap"; exit 1; }

STEPS=256 REQS=1 PP=()
LOAD=(adapters/vllm/tools/served_run/serve_load.py --port "$PORT" --model "$MODEL" --varied 1 --rate 1000 --max-tokens 4
  --prompt-range 16:24 --seed 1 --out "$SR/load.jsonl")
if [ "$SCHEME" = pearl-plus ]; then
  if [ "$DRY" = 1 ] && [ -z "${PEARLC_KERNEL:-}" ]; then
    PEARLC_KERNEL=$TREE/verity/protocols/resource_accounting/compute/pouw/prover/kernels/pearl_c_sm90 PEARLC_CUBIN=$SCRATCH/stand-in.cubin
    : > "$PEARLC_CUBIN"; echo "DRY: PEARLC_KERNEL and PEARLC_CUBIN unset, a stand-in cubin (the serve's preflight reads no cubin)"
  fi
  [ -d "${PEARLC_KERNEL:-}" ] && [ -s "${PEARLC_CUBIN:-}" ] || [ "$DRY" = 1 ] || { echo "SMOKE REFUSED pearl-plus: PEARLC_KERNEL" \
    "and PEARLC_CUBIN name no kernel and cubin; build them outside the lease (no card): RESEARCH_RUN_DIR=DIR bash -c '. benchmarks/harness/env.sh" \
    "&& . benchmarks/harness/setup/pearl_c_sm90.sh && echo \$PEARLC_KERNEL \$PEARLC_CUBIN'"; exit 3; }
  IFS=, read -r slots prompt STEPS < <("$PY" -c 'from verity.protocols.resource_accounting.compute.pouw.scheme import PEARL_PLUS as s; print(s.tie["class"])')
  CLASS=$slots,$prompt,$STEPS
  [ -n "${STEPS:-}" ] || { echo "SMOKE REFUSED pearl-plus: its scheme states no tie class (scheme.py's PEARL_PLUS.tie, above)"; exit 3; }
  REQS=$slots
  export PEARLC_KERNEL PEARLC_CUBIN PEARLC_BF16_ROWS=${PEARLC_BF16_ROWS:-0}
  SP=benchmarks/pouw/pearl_c_vllm/served_passes.py SPP=$PYTHONPATH:$TREE/benchmarks/harness
  PYTHONPATH=$SPP "$PY" "$SP" arm --spec llama32-1b-1gpu-pouw-protocol-served@h100 --out "$SR/pearlplus-config.json" > "$SR/pearlplus-arm.log" 2>&1 \
    || { say "SMOKE FAILED pearl-plus arm: served_passes.py arm ($SR/pearlplus-arm.log)"; cat "$SR/pearlplus-arm.log"; exit 1; }
  echo "PEARL+ CONFIG: $(tail -1 "$SR/pearlplus-arm.log")"
  PP=(--protocols pearl-plus --protocols-config "$SR/pearlplus-config.json")
  [ -z "${PEARLC_CHALLENGER_GATE:-}" ] || PP+=(--challenger-gate "$PEARLC_CHALLENGER_GATE"
    --challenger-peer "$PEARLC_CHALLENGER_PEER" --challenger-lane "$PEARLC_CHALLENGER_LANE")
  LOAD=(adapters/vllm/tools/served_run/serve_load.py --port "$PORT" --model "$MODEL" --fixed "$slots" --window-steps "$STEPS"
    --prompt-range "$prompt:$prompt" --seed 1 --out "$SR/load.jsonl")
fi
SERVE=(serve-openai --model "$MODEL" --revision "$REV" --out "$D" --port "$PORT" --precision bfloat16
  --run-id "smoke-${RESEARCH_RUN_ID:-local}" --seal --max-num-batched-tokens 512 --window-steps "$STEPS"
  --required-manifest "$PRED/manifest.json" --norm-tap-so "$NSO" --window-requests "$REQS"
  --window-gb 4 --disk-floor-gb 5 --seal-hold-gb 8 "${PP[@]}")

if [ "$DRY" = 1 ]; then
  "$PY" - "${SERVE[@]}" <<'EOF'
import os, sys
from verity_vllm.pipeline import cli, serve_openai as so
opts = cli.parse(sys.argv[1], sys.argv[2:], environ=os.environ)
so.preflight(opts, which=lambda _: "/stand-in/ninja")
print("SERVE PARSED AND PREFLIGHT PASSED:", opts.precision, opts.scheme, "seal", opts.seal, "protocols", opts.protocols or "none",
      "window steps", opts.window_steps, "window requests", opts.window_requests)
if opts.protocols:
    found = so.listed(opts, so.Path(opts.out))
    print("SCHEMES ATTACHED BY NAME:", [(s.name, {k: g[k] for k in ("coins", "challenger", "transcript_dir") if k in g}) for s, g in found])
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
  SCHEME=$SCHEME "$PY" - "${LOAD[@]}" <<'EOF'
import os, runpy, sys
a = runpy.run_path(sys.argv[1])["parser"]().parse_args(sys.argv[2:])
if os.environ["SCHEME"] == "pearl-plus":
    assert (a.fixed, a.window_steps, a.prompt_range, a.varied) == (4, 2, "27:27", 0), a
    print("LOAD PARSED: fixed", a.fixed, "request(s) of", a.prompt_range, "tokens over", a.window_steps, "window steps (Pearl+'s tie)")
else:
    assert a.varied == 1 and not a.fixed, a
    print("LOAD PARSED:", a.varied, "request(s), max_tokens", a.max_tokens)
EOF
  if [ "$SCHEME" = pearl-plus ]; then
    mkdir -p "$SCRATCH/no-passes"
    if PYTHONPATH=$SPP "$PY" "$SP" verify --passes "$SCRATCH/no-passes" --out "$SCRATCH/v.json" > "$SR/pearlplus-verify.log" 2>&1; then
      echo "SMOKE FAILED: served_passes.py verify passed a directory of no passes"; exit 1; fi
    echo "PEARL+ VERIFY REFUSES A DIRECTORY OF NO PASSES: $(grep -v '^ ' "$SR/pearlplus-verify.log" | tail -1 | cut -c1-160)"
  fi
  echo "SMOKE DRY OK${NOKIT:+ (no toolkit here: the native collector not built)}; not exercised off the card: the model load, the committer's construction over the engine, the request, the seal and the open"
  exit 0
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
said=$(CUDA_VISIBLE_DEVICES= "$PY" - "$D/windows/w0" <<'EOF'
import json, os, sys
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "the opening must run with no card visible"
from verity_vllm.pipeline import deferred_rows as DR
w = sys.argv[1]
r = DR.of_record(json.load(open(f"{w}/commit/verdict.json"))["replay_deferred"]["bundles"][0]["path"], w)
print(f"SMOKE SEALED AND OPENED window 0 with no card visible: {r.commitment}, run root {bytes(r.run.run_root).hex()[:16]}")
EOF
) || { say "SMOKE FAILED open: the sealed bundle against its record ($D/windows/w0)"; exit 1; }
if [ "$SCHEME" = pearl-plus ]; then
  said+=$("$PY" - "$D" <<'EOF'
import json, os, sys
d = sys.argv[1]
c = json.load(open(f"{d}/schemes/pearl-plus/commitment.json"))
passes = [f for f in os.listdir(f"{d}/pearl/w0") if f not in ("coins.json", "served.json")]
assert passes and os.path.exists(f"{d}/pearl/w0/coins.json"), f"{d}/pearl/w0 holds no passes or no coins.json"
print(f"; PEARL+ epoch 0 kept {len(passes)} pass file(s) under pearl/w0, its commitment {json.dumps(c, default=str)[:120]}")
EOF
) || { say "SMOKE FAILED pearl-plus: no report or no passes ($D/schemes/pearl-plus, $D/pearl/w0)"; exit 1; }
  CUDA_VISIBLE_DEVICES= PYTHONPATH=$SPP "$PY" "$SP" verify --passes "$D/pearl/w0" --fast --jobs "$VJOBS" \
    --out "$OUT/pearlplus-verify.json" > "$SR/pearlplus-verify.log" 2>&1 \
    || { say "SMOKE FAILED pearl-plus verify: served_passes.py verify over pearl/w0 ($SR/pearlplus-verify.log)"; exit 1; }
  said+="; PEARL+ VERIFY: $(tail -1 "$SR/pearlplus-verify.log" | cut -c1-200)"
  if [ "$PEARLPLUS_TIE" = 1 ]; then
    W=$D/pearl/window-record.json
    CUDA_VISIBLE_DEVICES= PYTHONPATH=$SPP "$PY" "$SP" retain --passes "$D/pearl/w0" --window-record "$W" > "$SR/pearlplus-retain.log" 2>&1 \
      || { say "$said; SMOKE FAILED pearl-plus tie: served_passes.py retain --window-record ($SR/pearlplus-retain.log)"; exit 1; }
    CUDA_VISIBLE_DEVICES= PYTHONPATH=$SPP "$PY" "$SP" verify --passes "$D/pearl/w0" --tie "$W" --class "$CLASS" --fast --jobs "$VJOBS" \
      --out "$OUT/pearlplus-verify-tie.json" > "$SR/pearlplus-verify-tie.log" 2>&1 \
      || { say "$said; SMOKE FAILED pearl-plus tie: verify --tie --class $CLASS ($SR/pearlplus-verify-tie.log)"; exit 1; }
    said+="; PEARL+ VERIFY TIED ($CLASS): $(tail -1 "$SR/pearlplus-verify-tie.log" | cut -c1-200)"
  fi
fi
say "$said"

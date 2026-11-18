#!/usr/bin/env bash
# Does: serves ONE window of bf16 Llama 3.2 1B (unsloth at 9535bd9b) on the H100 through serve-openai with its rows-sha512 Commit and seals it, then opens the sealed bundle against its record on the CPU.
# Does NOT show: a recompute on a processor (the replay is the sealed bundle opened against its record), an audit, or a traced Build on the H100.
# Needs: PY (default the node's /workspace/venv312/bin/python, both branches), CUDA_HOME (12.9, machine.py's), CARD_SKU (machine.py sku), HF_HOME (/workspace/hf), PYTHONPATH, VERITY_PROTOCOLS_CONFIG (all set here), CUDA_VISIBLE_DEVICES from the lease; the norm tap (no hidden tap: the row's manifest names no hidden class).
# Writes: the serve's directory, the private directory (keys, salts) and every log under SR, outside the run's directory; under $OUT only smoke.txt, the last line.
#   research run --on vy-h100-1 --cwd source -- taskset -c 0-9 bash adapters/vllm/verity_vllm/ops/h100_smoke.sh   (DRY=1: off the card)
set -euo pipefail
TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd); cd "$TREE"
ROW=llama32-1b__bf16__h100__tp1__b4__i256__o32__mixed__greedy__bi-eager
MODEL=unsloth/Llama-3.2-1B REV=9535bd9b1d1dea6acafbdc4813b728796aeb28da
DRY=${DRY:-0} PY=${PY:-/workspace/venv312/bin/python}
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
CUDA_HOME=$("$PY" benchmarks/kernels/machine.py toolkit 12.9) || { echo "SMOKE REFUSED toolkit: the machine lists no CUDA 12.9 (machine.py, above)"; exit 3; }
CARD_SKU=$("$PY" benchmarks/kernels/machine.py sku) || { echo "SMOKE REFUSED card: the machine's record names no card sku for this host (machine.py sku, above)"; exit 3; }
export CUDA_HOME CARD_SKU HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
export PATH=$(dirname "$PY"):$CUDA_HOME/bin:$PATH
SNAP=$HF_HOME/hub/models--unsloth--Llama-3.2-1B/snapshots/$REV
[ -s "$NSO" ] || { echo "SMOKE REFUSED norm tap: no $NSO; build it outside the lease (no card):" \
  "cd $TREE/adapters/vllm && CUDA_HOME=$CUDA_HOME CARD_SKU=$CARD_SKU bash verity_vllm/ops/pod_norm_tap.sh"; exit 3; }
{ [ -s "$SNAP/config.json" ] && ls "$SNAP"/*.safetensors > /dev/null 2>&1; } || {
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

SERVE=(serve-openai --model "$MODEL" --revision "$REV" --out "$D" --port "$PORT" --precision bfloat16
  --run-id "smoke-${RESEARCH_RUN_ID:-local}" --scheme rows-sha512 --seal --max-num-batched-tokens 512 --window-steps 256
  --required-manifest "$PRED/manifest.json" --norm-tap-so "$NSO" --window-requests 1
  --max-model-len 512 --window-gb 4 --disk-floor-gb 5 --seal-hold-gb 8 --seal-warm-gb 0)
LOAD=(adapters/vllm/tools/served_run/serve_load.py --port "$PORT" --model "$MODEL" --varied 1 --rate 1000 --max-tokens 4
  --prompt-range 16:24 --seed 1 --out "$SR/load.jsonl")

if [ "$DRY" = 1 ]; then
  "$PY" - "${SERVE[@]}" <<'EOF'
import os, sys
from verity_vllm.pipeline import cli, serve_openai as so
opts = cli.parse(sys.argv[1], sys.argv[2:], environ=os.environ)
so.preflight(opts, which=lambda _: "/stand-in/ninja")
print("SERVE PARSED AND PREFLIGHT PASSED:", opts.precision, opts.scheme, "seal", opts.seal)
if run := os.environ.get("RESEARCH_RUN_DIR"):
    try:
        so.preflight(cli.parse(sys.argv[1], [*sys.argv[2:], "--out", run + "/serve"], environ=os.environ), which=lambda _: "/stand-in/ninja")
        sys.exit("SMOKE FAILED: an --out inside the run dir was not refused")
    except cli.UsageError as e:
        print("INSIDE THE RUN DIR REFUSED:", str(e)[:60])
EOF
  CUDA_VISIBLE_DEVICES= "$PY" - <<'EOF'
import os
from verity_vllm import settings
from verity_vllm.commit.committer import native_ext, native_jit
from verity_vllm.config import cuda_arch
sku, home = settings.card_sku(), settings.cuda_home()
missing = [p for p in native_jit.source_paths(native_ext.HERE) if not os.path.exists(p)]
assert not missing, missing
print(f"COMMITTER SETTINGS READ: CARD_SKU {sku}, arch {cuda_arch(sku)}, CUDA_HOME {home}, the native collector's sources present")
EOF
  "$PY" - "${LOAD[@]}" <<'EOF'
import runpy, sys
a = runpy.run_path(sys.argv[1])["parser"]().parse_args(sys.argv[2:])
assert a.varied == 1 and not a.fixed, a
print("LOAD PARSED:", a.varied, "request(s), max_tokens", a.max_tokens)
EOF
  echo "SMOKE DRY OK; not exercised off the card: the model load, the committer's construction and its native collector's JIT build, the request, the seal and the open"
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
said=$("$PY" - "$D/windows/w0" <<'EOF'
import json, sys
from verity_vllm.pipeline import deferred_rows as DR
w = sys.argv[1]
r = DR.of_record(json.load(open(f"{w}/commit/verdict.json"))["replay_deferred"]["bundles"][0]["path"], w)
print(f"SMOKE SEALED AND OPENED window 0: {r.commitment}, run root {bytes(r.run.run_root).hex()[:16]}")
EOF
) || { say "SMOKE FAILED open: the sealed bundle against its record ($D/windows/w0)"; exit 1; }
say "$said"

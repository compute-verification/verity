#!/usr/bin/env bash
# The two-stage variant on the smallest served window: one request, one token, its one span sampled whole. Builds
# flock-circuit and the Lean verifiers by two_stage_small.sh's steps (two_stage_build.sh); then, under one leased card,
# serve-openai with --boundary-rows (the first stage at the serve: the class's edges, its ids and the select's logits) and no Commit,
# one greedy request of PROMPT ids for one token, and the server's stop; then on the host: `two_stage.py first` on a
# local challenger session, the capture from the logits the serve kept (two_stage_served.py: no replay), and
# `two_stage.py second` (the interior, the second stage on a session of its own, a --zk session per member on the CPU,
# Lean's verdicts, round 2 and the stream's outcome). The variant runs unproved: no outcome here quotes a guarantee.
#   research run --on vy-h100-1 --cwd clone --cpus 16 --mem-gb 64 --timeout 180m -- bash benchmarks/one_stage/two_stage_served.sh
# Needs on the node: PY (default /workspace/venv312/bin/python, with vLLM), gpu-lease, HF_HOME (default /workspace/hf)
# holding MODEL at REV (default the application's, applications/remote.toml), curl. DRY=1 runs everything but the card:
# the builds, the serve's parse and preflight, and `first` and `second`'s --help.
# Writes under $OUT the builds' logs, the binaries' hashes, the class's commit.json, the logs and two_stage.py's public
# records (two_stage_small.sh's selection); the serve's directory, the logits' rows and salts and the capture stay in
# PRIV (outside the run dir, 0700), which $OUT/private.txt names.
set -euo pipefail
PUBF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/public_files.py
REPO=$(git rev-parse --show-toplevel); cd "$REPO"
O=${OUT:?research run sets OUT}; mkdir -p "$O"
W=$(mktemp -d); SERVER=
trap '[ -z "$SERVER" ] || kill -INT "$SERVER" 2>/dev/null || true; rm -rf "$W"' EXIT
DRY=${DRY:-0} PY=${PY:-/workspace/venv312/bin/python} LEASE_MIN=${LEASE_MIN:-30} PORT=${PORT:-18292}
MODEL=${MODEL:-unsloth/Llama-3.2-1B} REV=${REV:-9535bd9b1d1dea6acafbdc4813b728796aeb28da} V=${V:-128256} PROMPT=${PROMPT:-8}
CHECK="GreedySelect_v1{V=$V,N=$V}"
PRIV=${PRIV_ROOT:-/workspace/research/two-stage-served}/${RESEARCH_RUN_ID:-local}
RD=${RESEARCH_RUN_DIR:-/-}
case $PRIV in "$RD"|"$RD"/*) echo "the private dir $PRIV is in the run dir" >&2; exit 2 ;; esac
(umask 077; mkdir -p "$PRIV") && echo "$PRIV" > "$O/private.txt"
D=$PRIV/serve
BOUNDARY="{\"check\":\"$CHECK\",\"inline\":true,\"per_prompt\":0,\"per_output\":1,\"ids\":true,\"logits\":{\"private\":\"$PRIV/kept\"}}"
SERVE=(serve-openai --model "$MODEL" --revision "$REV" --out "$D" --port "$PORT" --precision bfloat16 --no-commit
  --run-id "two-stage-${RESEARCH_RUN_ID:-local}" --max-num-seqs 1 --max-model-len $((PROMPT + 1)) --boundary-rows "$BOUNDARY")

if [ "${1:-}" = card ]; then  # under the lease: the serve, its one request, the server's stop
  export HF_HOME=${HF_HOME:-/workspace/hf} HF_HUB_OFFLINE=1 PYTHONPATH=$REPO:$REPO/adapters/vllm:$REPO/adapters/torch:$REPO/tools/config
  (cd adapters/vllm && exec "$PY" -m verity_vllm.pipeline.cli "${SERVE[@]}") > "$PRIV/serve.log" 2>&1 &
  SERVER=$!
  for _ in $(seq 1 120); do
    kill -0 "$SERVER" 2>/dev/null || { echo "the serve exited before it listened ($PRIV/serve.log)"; exit 1; }
    curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1 && break
    sleep 5
  done
  ids=$(seq -s, 1000 $((1000 + PROMPT - 1)))
  curl -sf "http://127.0.0.1:$PORT/v1/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":[$ids],\"max_tokens\":1,\"temperature\":0,\"stream\":false}" > "$PRIV/response.json" \
    || { echo "the request failed ($PRIV/serve.log)"; exit 1; }
  kill -INT "$SERVER"; s=0; wait "$SERVER" || s=$?; SERVER=
  exit "$s"
fi

. benchmarks/one_stage/two_stage_build.sh
UPY=(env PYTHONPATH="$M0PATH" uv run --no-sync python)

if [ "$DRY" = 1 ]; then
  PYTHONPATH=$M0PATH:$REPO/tools/config "${UPY[@]}" - "${SERVE[@]}" <<'EOF' | tee "$O/serve-dry.txt"
import os, sys
from verity_vllm.pipeline import cli, serve_openai as so
opts = cli.parse(sys.argv[1], sys.argv[2:], environ=os.environ)
so.preflight(opts, which=lambda _: "/stand-in/ninja")
print("SERVE PARSED AND PREFLIGHT PASSED; first stage:", so.boundary_config(opts))
EOF
  "${UPY[@]}" benchmarks/one_stage/two_stage.py first --help > /dev/null && "${UPY[@]}" benchmarks/one_stage/two_stage.py second --help > /dev/null
  echo "DRY OK: built; serve-openai --boundary-rows parsed and preflighted; two_stage.py first and second import. Not run off the card: the serve, its request, first, the capture, second" | tee "$O/dry.txt"
  exit 0
fi

GPU_LEASE_WHO=${GPU_LEASE_WHO:-two-stage-served/${RESEARCH_RUN_ID:-local}} gpu-lease 1 --wait --max-min "$LEASE_MIN" -- bash "$0" card \
  || { echo "the serve under the lease failed ($PRIV/serve.log)"; tail -5 "$PRIV/serve.log" | cut -c1-300; exit 1; }
grep -v '"error"' "$D/boundary.jsonl" > /dev/null || { echo "the first stage committed no class: $(cut -c1-300 "$D/boundary.jsonl")"; exit 1; }
CLASS=$(ls -d "$D"/boundary/w*/p*-o* | head -1)
mkdir -p "$O/boundary" && cp "$CLASS/commit.json" "$O/boundary/"
"${UPY[@]}" benchmarks/one_stage/two_stage.py first --boundary "$CLASS" --check "$CHECK" --replay rate:1/1 --k 1 \
  --challenger local --main "$O/main.txt" --out "$W/first" > "$O/first.log" 2>&1 || { tail -20 "$O/first.log"; exit 3; }
tail -3 "$O/first.log"
"${UPY[@]}" benchmarks/one_stage/two_stage_served.py capture --boundary "$CLASS" --check "$CHECK" --kept "$PRIV/kept" \
  --out "$PRIV/capture" > "$PRIV/capture.log" 2>&1 || { tail -5 "$PRIV/capture.log"; exit 3; }
rc=0
"${UPY[@]}" benchmarks/one_stage/two_stage.py second --first-stage "$W/first/$(basename "$CLASS")/stage1.json" --boundary "$CLASS" \
  --capture "$PRIV/capture" --challenger local --challenger-host "$W/first/challenger-host" --main "$O/main.txt" \
  --m0-pythonpath "$M0PATH" --bin "$BIN" --lean "$LEAN" --kept "$PRIV/kept" --out "$W/second" > "$PRIV/second.log" 2>&1 || rc=$?
grep '^TWO ' "$PRIV/second.log" | tee "$O/two.txt" || tail -30 "$PRIV/second.log"
echo "second rc=$rc" | tee -a "$O/machine.txt"
RECORDS=(-type f ! -name circuit.txt ! -name 'inst-*' ! -name 'm*-inst-*' ! -name index.json
         ! -path './*/archive/*' ! -path '*-work/*' ! -path './challenger-host/*')
for d in first second; do
  mkdir -p "$O/$d"
  (cd "$W/$d" && find . "${RECORDS[@]}" -size -64M -print0 | python3 "$PUBF" filter | xargs -0 -r -I{} cp --parents {} "$O/$d/") \
    || true
done
exit "$rc"

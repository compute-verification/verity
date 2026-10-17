#!/usr/bin/env bash
# proof_throughput's verifier point: the compiled Lean verifier (`flock-verify verify --coins os`) on every session of one prover
# point's bundle (job.sh's out/bundle.json), each under /usr/bin/time, then the point's contract result from the prover's records
# and these verdicts. The verifier's `--partition` is its own: derived from the workload's program and checked against the
# digest the circuit binds (`partition`), never read from the bundle. Runs on a CPU node (vy-nebius-cpu-2 or -cpu-3), never on the prover's:
#   research run --on vy-nebius-cpu-3 --project verity --source . --cwd source --mem-gb 64 \
#     --tool verity_proof_throughput.tool:VERIFIER_POINT -- bash benchmarks/proof_throughput/verify.sh BUNDLE=art:... PROVE_RUN=r...
# env: BUNDLE=art:... PROVE_RUN= (the prover point's run id, recorded in the result)  BUILD_ONLY=1 (build the verifier, stop)
#      JOBS=1 (sessions verified at once: more share the node's memory bandwidth, so the timings are one session's alone
#      only at 1)
#      PT_WORK=/workspace/cache/proof-throughput (the Lean build, keyed by verity/core/service/flock's files, and the venv;
#      the CPU nodes' /workspace is root's, its cache/ the research user's)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
[ "${BUILD_ONLY:-0}" = 1 ] || : "${BUNDLE:?BUNDLE=art:... is required}"
REPO=$(pwd); O=${OUT:-${RESEARCH_RUN_DIR:-/tmp/ptv-$$}/out}; mkdir -p "$O"
W=${PT_WORK:-/workspace/cache/proof-throughput}; mkdir -p "$W" || { echo "cannot make $W (PT_WORK=)" >&2; exit 1; }
PYTHONPATH=$REPO/tools/research/src python3 -m research keep "$W" --owner @proofs --expires 14d --priority low \
  --regenerate "bash benchmarks/proof_throughput/verify.sh BUILD_ONLY=1" \
  --why "proof_throughput's Lean verifier builds and venv: a cache, rebuilt by the next point" > "$O/keep.json" 2>&1 \
  || echo "no retention record on $W: $(tail -1 "$O/keep.json")" >&2
S=$(mktemp -d "$W/scratch.XXXXXX"); trap 'rm -rf "$S"' EXIT
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$O/job.log"; }
command -v lake > /dev/null || export PATH="$HOME/.elan/bin:$PATH"

# the Lean verifier, built once per tree of verity/core/service/flock (as benchmarks/pouw/hidden_zk/job.sh builds it)
KEY=$(cd verity/core/service/flock && find . -path ./.lake -prune -o -type f -print | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)
L=$W/lean-$KEY
(
  exec 8>"$W/lean.lock"; flock 8
  if [ ! -x "$L/.lake/build/bin/flock-verify" ]; then
    rm -rf "$L.tmp" && mkdir -p "$L.tmp" && (cd verity/core/service/flock && tar --exclude=.lake -cf - .) | tar -xf - -C "$L.tmp" \
      && (cd "$L.tmp" && lake build flock-verify) > "$O/lake-build.log" 2>&1 && rm -rf "$L" && mv "$L.tmp" "$L"
  fi
)
LEAN=$L/.lake/build/bin/flock-verify
[ -x "$LEAN" ] || { tail -20 "$O/lake-build.log"; log "Lean build failed"; exit 1; }
sha256sum "$LEAN" | cut -d' ' -f1 > "$O/lean.sha256"; echo "$KEY" > "$O/lean.key"
PYBIN=$W/py/bin/python3
if [ ! -x "$PYBIN" ]; then
  UV=$(command -v uv || echo "$HOME/.local/bin/uv")
  (exec 8>"$W/venv.lock"; flock 8; [ -x "$PYBIN" ] || { "$UV" venv -q --python 3.12 "$W/py" && "$UV" pip install -q --python "$PYBIN" numpy blake3; })
fi
export PYTHONPATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/ml/flock/python:$REPO/integrations/vllm:$REPO/tools/verity/circuit_check/src:$REPO/benchmarks/proof_throughput
[ "${BUILD_ONLY:-0}" = 1 ] && { log "BUILD_ONLY: $LEAN ($(cat "$O/lean.sha256"))"; exit 0; }

log "fetching $BUNDLE"
"$PYBIN" -m verity_proof_throughput fetch-bundle "$BUNDLE" --to "$S/b" > "$O/fetch.txt" 2>&1 || { tail -5 "$O/fetch.txt"; log "fetch failed"; exit 1; }
BD=$(dirname "$(find "$S/b" -name circuit.txt | head -1)")
[ -f "$BD/circuit.txt" ] || { log "the bundle has no circuit.txt"; exit 1; }
for f in "$BD"/*; do [ -f "$f" ] && case $f in *.txt|*.bin) ;; *) cp "$f" "$O/" ;; esac; done
cp "$BD/gpu-mem.txt" "$O/" 2>/dev/null
WORKLOAD=$("$PYBIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["workload"])' "$BD/point.json")
B=$("$PYBIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["B"])' "$BD/point.json")
PUB=$(ls "$BD"/pub-*.bin | head -1)
# the verifier's own copy of the partition object the circuit binds, derived from the workload's program, not the bundle's
PART=$S/partition.json
"$PYBIN" -m verity_proof_throughput partition --workload "$WORKLOAD" --circuit "$BD/circuit.txt" --out "$PART" > "$O/partition.txt" 2>&1 \
  || { tail -5 "$O/partition.txt"; log "the workload's partition object is not the circuit's"; exit 1; }
cp "$PART" "$O/partition.json"
log "verifying $WORKLOAD B=$B: $(ls -d "$BD"/sessions/*/ | wc -l) sessions, circuit $(stat -c %s "$BD/circuit.txt") B, partition $(tail -1 "$O/partition.txt" | cut -c1-16)"

verify_one() {   # verify_one SESSION_DIR: one VERDICT line with the session's time and peak memory
  local s=${1%/} t; t=$(mktemp "$S/time.XXXXXX")
  /usr/bin/time -f '%e %U %S %M' -o "$t" "$LEAN" verify --statement verity/flock-circuit --coins os --circuit "$BD/circuit.txt" \
    --public "$PUB" --partition "$PART" --session "$s" > "$t.out" 2> "$t.err"
  local rc=$?
  "$PYBIN" -m verity_proof_throughput lean-verdict --session "$(basename "$s")" --rc $rc --time "$t" --stdout "$t.out" --stderr "$t.err"
}
export -f verify_one; export LEAN BD PUB PART S PYBIN
ls -d "$BD"/sessions/*/ | xargs -P "${JOBS:-1}" -I{} bash -c 'verify_one {}' > "$O/lean.jsonl"
log "Lean: $(grep -c '"accepted": true' "$O/lean.jsonl") of $(wc -l < "$O/lean.jsonl") sessions accepted"
"$PYBIN" -m verity_proof_throughput point-result --workload "$WORKLOAD" --B "$B" --dir "$O" --lean "$O/lean.jsonl" \
  --prove-run "${PROVE_RUN:-}" --bundle "$BUNDLE" --out "${RESEARCH_RUN_DIR:-$O}/result.json"

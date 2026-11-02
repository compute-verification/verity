#!/usr/bin/env bash
# proof_throughput's verifier point: the compiled Lean verifier (`flock-verify verify --coins os`) on every session of one prover
# point's bundle (job.sh's out/bundle.json), each under /usr/bin/time, then the point's contract result from the prover's records
# and these verdicts. The verifier's `--partition` is its own: derived from the workload's program and checked against the
# digest the circuit binds (`partition`), never read from the bundle. Runs in a proofs-pool slot of vy-nebius-cpu-3 or -cpu-2,
# or as a queued CPU job of its own on vy-nebius-1 (never inside a prover's run); `verify --prove-run RUN [--node N]` prints
# and launches it (plan.verifier_command), e.g.
#   research run --on vy-nebius-cpu-3 --pool proofs --as proofs --cwd clone --project verity --source . --mem-gb 16 \
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
# the verifier's own copies of the partition object the circuit binds and of its program, derived from the workload, not the
# bundle's; from them and its own K = B it derives the stratified law each session's sample must be by (U2)
PART=$S/partition.json PROG=$S/program.json
"$PYBIN" -m verity_proof_throughput partition --workload "$WORKLOAD" --circuit "$BD/circuit.txt" --out "$PART" --program-out "$PROG" \
  > "$O/partition.txt" 2>&1 || { tail -5 "$O/partition.txt"; log "the workload's partition object is not the circuit's"; exit 1; }
cp "$PART" "$O/partition.json"
DRAW_FLAGS="--program $PROG --stratified $B"
log "verifying $WORKLOAD B=$B: $(ls -d "$BD"/sessions/*/ | wc -l) sessions, circuit $(stat -c %s "$BD/circuit.txt") B, partition $(tail -1 "$O/partition.txt" | cut -c1-16)"

verify_one() {   # verify_one SESSION_DIR: one VERDICT line with the session's time and peak memory
  local s=${1%/} t; t=$(mktemp "$S/time.XXXXXX")
  # shellcheck disable=SC2086
  /usr/bin/time -f '%e %U %S %M' -o "$t" "$LEAN" verify --statement verity/flock-circuit --coins os --circuit "$BD/circuit.txt" \
    --public "$PUB" --partition "$PART" $DRAW_FLAGS --session "$s" > "$t.out" 2> "$t.err"
  local rc=$?
  "$PYBIN" -m verity_proof_throughput lean-verdict --session "$(basename "$s")" --rc $rc --time "$t" --stdout "$t.out" --stderr "$t.err"
}
if [ "${PROFILE:-0}" = 1 ]; then
  # one timed session under perf (DWARF call graphs: Lean's C keeps no frame pointers), with the verifier's own phase times
  s=$(ls -d "$BD"/sessions/*/ | grep -v warm | head -1); s=${s%/}
  log "profiling $(basename "$s")"
  # shellcheck disable=SC2086
  /usr/bin/time -f '%e %U %S %M' -o "$O/profile.time" perf record -F "${PERF_HZ:-49}" --call-graph dwarf,32768 -o "$S/perf.data" \
    "$LEAN" verify --statement verity/flock-circuit --coins os --circuit "$BD/circuit.txt" --public "$PUB" --partition "$PART" \
    $DRAW_FLAGS --session "$s" > "$O/profile.stdout" 2> "$O/profile.stderr"
  log "perf: exit $?, $(cat "$O/profile.time")"
  perf report -i "$S/perf.data" --no-children --sort symbol --stdio -g none --percent-limit 0.2 2>> "$O/perf.err" | head -400 > "$O/perf-self.txt"
  perf report -i "$S/perf.data" --children --sort symbol --stdio -g none --percent-limit 0.5 2>> "$O/perf.err" | head -600 > "$O/perf-children.txt"
  # the address field too: with call chains, `sym` alone prints no frames
  perf script -i "$S/perf.data" -F comm,ip,sym 2>> "$O/perf.err" | gzip > "$O/perf-script.txt.gz"
  "$PYBIN" -m verity_proof_throughput profile-phases --script "$O/perf-script.txt.gz" --time "$O/profile.time" \
    --stderr "$O/profile.stderr" --stdout "$O/profile.stdout" --out "${RESEARCH_RUN_DIR:-$O}/result.json" --bundle "$BUNDLE" --prove-run "${PROVE_RUN:-}"
  exit $?
fi
export -f verify_one; export LEAN BD PUB PART S PYBIN DRAW_FLAGS
ls -d "$BD"/sessions/*/ | xargs -P "${JOBS:-1}" -I{} bash -c 'verify_one {}' > "$O/lean.jsonl"
log "Lean: $(grep -c '"accepted": true' "$O/lean.jsonl") of $(wc -l < "$O/lean.jsonl") sessions accepted"
"$PYBIN" -m verity_proof_throughput point-result --workload "$WORKLOAD" --B "$B" --dir "$O" --lean "$O/lean.jsonl" \
  --prove-run "${PROVE_RUN:-}" --bundle "$BUNDLE" --out "${RESEARCH_RUN_DIR:-$O}/result.json"

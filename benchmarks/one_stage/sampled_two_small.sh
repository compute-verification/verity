#!/usr/bin/env bash
# The two-stage form's accept test's honest play: two_stage_small.sh's run on one challenger whose key is at the
# deployed LMS parameters (--deployed-key), its second stage's sessions' coin trees on the stage's own session under
# the second stage's family (--interior-trees, FLOCK_CHALLENGER=1), and the form's case of the run (--form-case,
# sampled_two_case.py), then the form's check on it: `sampled-two-on check` (the switch on) must accept, and
# `sampled-two check` (the shipped switch) must refuse by the switch alone. Exit 0 when both hold. The case's
# parameters, commitment and evidence are kept under $OUT/form with its blobs (blobs/, their SHA-512 and sizes in
# blobs.txt): the directory the form's accept test reads (SAMPLED_TWO_CASE).
#   research run --on auto --cwd clone --cpus 16 --mem-gb 64 --max-min 120 -- bash benchmarks/one_stage/sampled_two_small.sh
#   SMALL_ARGS, SMALL_FIRST and SMALL_SECOND pass two_stage_small.py serve, `first` and `second` more.
set -euo pipefail
REPO=$(git rev-parse --show-toplevel); cd "$REPO"
O=${OUT:?research run sets OUT}; mkdir -p "$O"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.elan/bin:$PATH
{ nproc; free -g | head -2; git rev-parse HEAD; } | tee "$O/machine.txt"
uv sync --all-packages --extra torch-cpu > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 5; }

{ (cd verity/protocols/flock/verifier && lake build flock-verify flock-firewall) \
    && (cd verity/protocols/spot/verifier/lean && lake build sampled-proofs-verifier sampled-proofs-difftest) \
    && (cd verity/protocols/recursion/verifier/recursive && lake build sampled-two sampled-two-on); } > "$O/build-lean.log" 2>&1 &
LP=$!
mkdir -p "$O/m0-build"
RESEARCH_RUN_DIR=$O/m0-build FLOCK_WORK=$W MODE=build GPU=0 bash verity/protocols/flock/pod/60-circuit.sh > "$O/m0-build/build.log" 2>&1 \
  || { tail -40 "$O/m0-build/build.log"; exit 6; }
BIN=$W/flock-gpu-link/flock/target/release/flock-circuit
wait $LP || { tail -40 "$O/build-lean.log"; exit 7; }
LEAN=$REPO/verity/protocols/flock/verifier/.lake/build/bin/flock-verify
FORM_BIN=$REPO/verity/protocols/recursion/verifier/recursive/.lake/build/bin
sha256sum "$BIN" "$LEAN" "$FORM_BIN/sampled-two" "$FORM_BIN/sampled-two-on" | tee "$O/binaries.sha256"

M0PATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/protocols/flock/prover/python:$REPO/adapters/vllm
export RAYON_NUM_THREADS=$(nproc) PYTHONPATH=$M0PATH FLOCK_CHALLENGER=1
UNSHALLOW=(); [ "$(git rev-parse --is-shallow-repository)" = true ] && UNSHALLOW=(--unshallow)
{ git fetch -q "${UNSHALLOW[@]}" origin +refs/heads/main:refs/remotes/origin/main \
    || git fetch -q "${UNSHALLOW[@]}" origin +refs/remotes/origin/main:refs/remotes/origin/main; } > "$O/fetch-main.log" 2>&1 \
  && git rev-list --first-parent refs/remotes/origin/main > "$O/main.txt" || { tail -5 "$O/fetch-main.log"; exit 4; }

PY=(uv run --no-sync python)
# shellcheck disable=SC2086
"${PY[@]}" benchmarks/one_stage/two_stage_small.py serve --out "$W/serve" ${SMALL_ARGS:-} | tee "$O/serve.json"
CHECK=$("${PY[@]}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["check"])' "$O/serve.json")
CLASS=$("${PY[@]}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["class"])' "$O/serve.json")
# shellcheck disable=SC2086
"${PY[@]}" benchmarks/one_stage/two_stage.py first --boundary "$CLASS" --check "$CHECK" --replay rate:2/6 --k 1 \
  --challenger local --deployed-key --main "$O/main.txt" --out "$W/first" ${SMALL_FIRST:-} > "$O/first.log" 2>&1 \
  || { tail -20 "$O/first.log"; exit 3; }
tail -3 "$O/first.log"
rc=0
# shellcheck disable=SC2086
"${PY[@]}" benchmarks/one_stage/two_stage.py second --stage1 "$W/first/$(basename "$CLASS")/stage1.json" --boundary "$CLASS" \
  --capture "$W/serve/capture" --challenger local --challenger-host "$W/first/challenger-host" --deployed-key \
  --interior-trees --form-case --main "$O/main.txt" --m0-pythonpath "$M0PATH" --bin "$BIN" --lean "$LEAN" \
  --out "$W/second" ${SMALL_SECOND:-} > "$O/second.log" 2>&1 || rc=$?
grep '^TWO ' "$O/second.log" || tail -30 "$O/second.log"
echo "second rc=$rc" | tee -a "$O/machine.txt"
"${PY[@]}" -c 'import json,sys; s=json.load(open(sys.argv[1])); print("INTERIOR " + json.dumps(s.get("interior", {}).get("members")))' \
  "$W/second/summary.json" || true
for f in summary.json run-input.json coins.json statements.json audit/summary.json audit/case-v2-members.json \
         audit/case-v2-members-error.txt audit/case-v2-error.txt; do
  [ -f "$W/second/$f" ] && { mkdir -p "$O/second/$(dirname "$f")"; cp "$W/second/$f" "$O/second/$f"; }
done
[ -f "$W/second/form/params.json" ] || { echo "FORM no case"; exit 2; }
mkdir -p "$O/form"
cp "$W/second/form/params.json" "$W/second/form/commitment.json" "$W/second/form/evidence.json" "$O/form/"
mapfile -t BLOBS < "$W/second/form/blobs.txt"
mkdir -p "$O/form/blobs"; i=0
for b in "${BLOBS[@]}"; do
  printf '%s  %s  %s\n' "$(sha512sum < "$b" | cut -d' ' -f1)" "$(stat -c %s "$b")" "$b"
  cp "$b" "$O/form/blobs/$(printf %03d $i)-$(basename "$b")"; i=$((i + 1))
done > "$O/form/blobs.txt"
FILES=("$W/second/form/params.json" "$W/second/form/commitment.json" "$W/second/form/evidence.json" "${BLOBS[@]}")
on=0; "$FORM_BIN/sampled-two-on" check "${FILES[@]}" > "$O/form/on.json" || on=$?
off=0; "$FORM_BIN/sampled-two" check "${FILES[@]}" > "$O/form/off.json" || off=$?
echo "FORM_ON rc=$on $(head -c 2000 "$O/form/on.json")"
echo "FORM_OFF rc=$off $(head -c 2000 "$O/form/off.json")"
[ "$on" -eq 0 ] && [ "$off" -eq 1 ]

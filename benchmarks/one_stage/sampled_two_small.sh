#!/usr/bin/env bash
# The two-stage form's accept test's honest play: two_stage_small.sh's run on one challenger whose key is at the
# deployed LMS parameters (--deployed-key), its second stage's sessions' coin trees on the stage's own session under
# the second stage's family (--interior-trees, FLOCK_CHALLENGER=1), and the form's case of the run (--form-case,
# sampled_two_case.py), then the form's check on it: `sampled-two-on check` (the switch on) must accept, and
# `sampled-two check` (the shipped switch) must refuse by the switch alone. The case's parameters, commitment and
# evidence are kept under $OUT/form with its blobs (blobs/, their SHA-512 and sizes in blobs.txt): a case as the
# form's accept test reads one, its kept case such a run's and VERITY_SAMPLED_TWO_CASE naming another in its place.
# Then a negative play: the same first stage with a coin tree sampled on
# its session (first --first-tree) and its own second stage, whose case `sampled-two-on check` must refuse by the
# first session's coin tree's line (kept under $OUT/form-tree). Before either, each kept play's case KEPT_CASES names
# (directories like $OUT/form, space separated) is checked again: accepted with the switch on, refused by the shipped
# switch. Exit 0 when all hold; 1 when the honest play's verdicts do not, 8 when a kept play's do not, 9 when the
# negative play is not refused by that line.
#   research run --on auto --cwd clone --cpus 16 --mem-gb 64 --max-min 120 -- bash benchmarks/one_stage/sampled_two_small.sh
#   With no SMALL_ARGS the serve is the play the form accepts: the class inlined as the greedy select in one unit, its
#   record committing the select's inputs beside the ids (two_stage_small.py serve --inline --units 1 --logits). The
#   record names the directory the serve kept them in and both `second` runs take it as --kept: the interior then
#   registers no value of the prover's. SMALL_ARGS, where given, is the serve's arguments instead; SMALL_FIRST and
#   SMALL_SECOND pass `first` and `second` more. A serve without --logits leaves the select's inputs a value the
#   prover registers after the first stage's sample, which the second stage refuses (second-inputs): that play's case
#   is not one the form accepts. ROUND_TWO prints the second stage's statement check as the run's summary has it.
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

kept=0; n=0
for d in ${KEPT_CASES:-}; do
  mapfile -t KB < <(find "$d/blobs" -maxdepth 1 -type f | sort)
  KF=("$d/params.json" "$d/commitment.json" "$d/evidence.json" "${KB[@]}")
  mkdir -p "$O/kept/$n"
  on=0; "$FORM_BIN/sampled-two-on" check "${KF[@]}" > "$O/kept/$n/on.json" || on=$?
  off=0; "$FORM_BIN/sampled-two" check "${KF[@]}" > "$O/kept/$n/off.json" || off=$?
  echo "KEPT_ON rc=$on $d $(head -c 600 "$O/kept/$n/on.json")"
  echo "KEPT_OFF rc=$off $d $(head -c 600 "$O/kept/$n/off.json")"
  { [ "$on" -eq 0 ] && [ "$off" -eq 1 ]; } || kept=1
  n=$((n + 1))
done

M0PATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/protocols/flock/prover/python:$REPO/adapters/vllm
export RAYON_NUM_THREADS=$(nproc) PYTHONPATH=$M0PATH FLOCK_CHALLENGER=1
UNSHALLOW=(); [ "$(git rev-parse --is-shallow-repository)" = true ] && UNSHALLOW=(--unshallow)
{ git fetch -q "${UNSHALLOW[@]}" origin +refs/heads/main:refs/remotes/origin/main \
    || git fetch -q "${UNSHALLOW[@]}" origin +refs/remotes/origin/main:refs/remotes/origin/main; } > "$O/fetch-main.log" 2>&1 \
  && git rev-list --first-parent refs/remotes/origin/main > "$O/main.txt" || { tail -5 "$O/fetch-main.log"; exit 4; }

PY=(uv run --no-sync python)
# shellcheck disable=SC2086
"${PY[@]}" benchmarks/one_stage/two_stage_small.py serve --out "$W/serve" ${SMALL_ARGS:---inline --units 1 --logits} | tee "$O/serve.json"
CHECK=$("${PY[@]}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["check"])' "$O/serve.json")
CLASS=$("${PY[@]}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["class"])' "$O/serve.json")
LOGITS=$("${PY[@]}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("logits", ""))' "$O/serve.json")
KEPT=(); [ -z "$LOGITS" ] || KEPT=(--kept "$LOGITS")
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
  --out "$W/second" "${KEPT[@]}" ${SMALL_SECOND:-} > "$O/second.log" 2>&1 || rc=$?
grep '^TWO ' "$O/second.log" || tail -30 "$O/second.log"
echo "second rc=$rc" | tee -a "$O/machine.txt"
"${PY[@]}" -c 'import json,sys; s=json.load(open(sys.argv[1])); print("INTERIOR " + json.dumps(s.get("interior", {}).get("members")))' \
  "$W/second/summary.json" || true
"${PY[@]}" -c 'import json,sys; s=json.load(open(sys.argv[1])); print("ROUND_TWO " + json.dumps(s.get("round_two"), sort_keys=True))' \
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
{ [ "$on" -eq 0 ] && [ "$off" -eq 1 ]; } || exit 1
[ "$kept" -eq 0 ] || exit 8

# shellcheck disable=SC2086
"${PY[@]}" benchmarks/one_stage/two_stage.py first --boundary "$CLASS" --check "$CHECK" --replay rate:2/6 --k 1 \
  --challenger local --deployed-key --first-tree --main "$O/main.txt" --out "$W/first-tree" ${SMALL_FIRST:-} \
  > "$O/first-tree.log" 2>&1 || { tail -20 "$O/first-tree.log"; exit 9; }
rc=0
# shellcheck disable=SC2086
"${PY[@]}" benchmarks/one_stage/two_stage.py second --stage1 "$W/first-tree/$(basename "$CLASS")/stage1.json" \
  --boundary "$CLASS" --capture "$W/serve/capture" --challenger local --challenger-host "$W/first-tree/challenger-host" \
  --deployed-key --interior-trees --form-case --main "$O/main.txt" --m0-pythonpath "$M0PATH" --bin "$BIN" --lean "$LEAN" \
  --out "$W/second-tree" "${KEPT[@]}" ${SMALL_SECOND:-} > "$O/second-tree.log" 2>&1 || rc=$?
echo "second-tree rc=$rc" | tee -a "$O/machine.txt"
[ -f "$W/second-tree/form/params.json" ] || { tail -30 "$O/second-tree.log"; echo "FORM_TREE no case"; exit 9; }
mkdir -p "$O/form-tree"
cp "$W/second-tree/form/params.json" "$W/second-tree/form/commitment.json" "$W/second-tree/form/evidence.json" "$O/form-tree/"
mapfile -t TBLOBS < "$W/second-tree/form/blobs.txt"
TFILES=("$W/second-tree/form/params.json" "$W/second-tree/form/commitment.json" "$W/second-tree/form/evidence.json" "${TBLOBS[@]}")
tree=0; "$FORM_BIN/sampled-two-on" check "${TFILES[@]}" > "$O/form-tree/on.json" || tree=$?
echo "FORM_TREE rc=$tree $(head -c 2000 "$O/form-tree/on.json")"
[ "$tree" -eq 1 ] && grep -q "coin tree's line" "$O/form-tree/on.json" || exit 9

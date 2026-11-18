#!/usr/bin/env bash
# The proof-block census of a deferred Commit's Build: every rank's Programs of record over every request step
# (`m1_families.py --steps all`), each rank's families sized (`--estimate`), then `block_census.py` over all ranks.
# A tensor-parallel row's ranks are its build/rank<r>/; a single-card row is rank 0 alone, its build_request*/ at its top
# level. A row whose requests are of several shapes gets one census a request, under $OUT/<request>/.
# Run on the machine holding the row directory (its build and commit/), as a research run of a clone:
#   research run --on MACHINE --project verity --source . --cwd clone --store-read --mem-gb 96 -- \
#     bash benchmarks/one_stage/block_census.sh ROW_DIR
#   BC_JOBS=4 (ranks at once, about 11 GB each)
# The row is an a5_llama1b_h200.sh run's (A5_KEEP=1) on the kept bootstrap, which kept_bootstrap.sh makes or reuses
# before the card's lease and names on its last line, DIR: VENV=DIR/venv312 HF_HOME=DIR/hf A5_TAPS=DIR/cp, under flock -s DIR.lock.
set -uo pipefail
ROW=${1:?the row directory: build/rank<r>/ (tensor-parallel) or build_request*/ (single-card), and commit/}
REPO=$(pwd); O=${OUT:-$RESEARCH_RUN_DIR/out}; mkdir -p "$O"; JOBS=${BC_JOBS:-4}
WL=adapters/vllm/workloads/$(basename "$ROW").json
if compgen -G "$ROW/build/rank*" > /dev/null; then
  BUILD=$ROW/build; RANKS=$(cd "$BUILD" && ls -d rank* | sed 's/rank//' | sort -n); BUILT='build/rank*'
elif compgen -G "$ROW/build_request*/" > /dev/null; then
  BUILD=$ROW; RANKS=0; BUILT='build_step build_request* build_workload'
else
  echo "block_census.sh: $ROW holds neither build/rank<r>/ (a tensor-parallel row) nor build_request*/ (a single-card row):" \
    "pass the row directory, the one holding commit/" >&2
  exit 2
fi
export PATH=$HOME/.local/bin:$PATH
{ nproc; free -g | head -2; git rev-parse HEAD; echo "$ROW"; } | tee "$O/machine.txt"
uv sync --all-packages --extra torch-cpu > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 5; }
PY=$REPO/.venv/bin/python
export PYTHONPATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/protocols/proofs/flock/prover/python:$REPO/adapters/vllm:$REPO/adapters/torch
(cd "$ROW" && find commit -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum) > "$O/commit.sha256"
# shellcheck disable=SC2086  # BUILT's globs expand in the row directory
(cd "$ROW" && find $BUILT -type f -print0 2>/dev/null | sort -z | xargs -0 sha256sum) > "$O/build.sha256"
cp "$ROW/commit/versions.json" "$O/" 2>/dev/null
$PY -c "from verity_vllm.pipeline.workload_file import workload_file; print(workload_file('$WL'))" || exit 4
rank() {
  $PY benchmarks/one_stage/m1_families.py --commit "$ROW/commit" --build-root "$BUILD" --workload "$WL" --rank "$1" \
    --steps all --out "$O/census-rank$1.json" > "$O/census-rank$1.log" 2>&1 || return 6
  $PY benchmarks/one_stage/m1_families.py --estimate "$O/census-rank$1.json" --out "$O/sized-rank$1.json" \
    > "$O/sized-rank$1.log" 2>&1 || return 7
}
pids=()
for r in $RANKS; do
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
  rank "$r" & pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=$?; done
[ $rc -eq 0 ] || { for f in "$O"/*-rank*.log; do echo "== $f"; tail -5 "$f"; done; exit $rc; }
SIZED=$(for r in $RANKS; do echo "$O/sized-rank$r.json"; done)
REQS=$($PY -c "import json, sys; print(*json.load(open(sys.argv[1]))['requests'])" "$O/sized-rank$(head -1 <<< "$RANKS").json")
if [ "$(wc -w <<< "$REQS")" -le 1 ]; then
  $PY benchmarks/one_stage/block_census.py --sized $SIZED --out "$O" 2>&1 | tee "$O/block_census.log"
  exit "${PIPESTATUS[0]}"
fi
rc=0
for q in $REQS; do
  $PY benchmarks/one_stage/block_census.py --sized $SIZED --out "$O/$q" --request "$q" 2>&1 | tee "$O/block_census-$q.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || rc=8
done
exit $rc
